#! /usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from history_matching.gpc import GPC

WORK_DIR = Path(__file__).parent.absolute()

# WARNING: FIXING RANDOM SEED!
np.random.seed(10)

# N = 20+30+10
N = 100
x = np.linspace(0, 2 * np.pi, N)
# f = (np.sin(x) + 1)/2.
f = 1 / (1 + np.exp(-3 * (x - np.pi)))
y = 2 * (np.random.rand(N) < f) - 1

"""
x = np.sort(np.concatenate([
    np.random.normal(loc=-6, scale=0.8, size=20),
    np.random.normal(loc=0, scale=0.8, size=30),
    np.random.normal(loc=2, scale=0.8, size=10),
]))


y = np.concatenate([
    np.ones(shape=20),
    np.negative(np.ones(shape=30)),
    np.ones(shape=10)
])
"""

data = pd.DataFrame({"x": x, "y": y})
data.index.name = "Sample"

param_info = pd.DataFrame(
    {
        "Name": ["x"],
        "Min": [0],  # [-9], #[0],
        "Max": [2 * np.pi],  # [5], #[2*np.pi]
    }
).set_index("Name")

print("Creating GPC...")

g = GPC(
    ["x"],
    "y",
    data,
    param_info,
    kernel_mode="RBF",
    # kernel_params = [0.001, 0.04],
    kernel_params=[40, 0.14],  # Sigma_f^2 and lengthscale^2
    verbose=False,
    debug=False,
)

### Test find posterior mode against eq 3.17
theta = [4, 0.2]
print("Calling g.find_posterior_mode(...)...")
ret = g.find_posterior_mode(theta, f_guess=None, tol_grad=1e-6, maxiter=10000)
assert np.allclose(
    ret["f_hat"], np.dot(ret["K"], ret["d_df_log_p_y_given_f"]), atol=1e-5
)

print("assert(...) passed")

# g.expectation_propagation(theta)

# exit()

##########################################


# s2_range = (1, 15)
# l2_range = (0.01, 0.15)
s2_range = (20, 80)
l2_range = (0.10, 0.75)

optim = None
if False:
    print("BEGIN: optimize_hyperparameters")
    optim = g.optimize_hyperparameters(
        x0=[40, 0.45],  # [4, 0.1],
        bounds=(s2_range, l2_range),
        eps=1e-3,
        disp=True,
        maxiter=15000,
    )
    print("DONE optimize_hyperparameters")


p = pd.DataFrame({"x": np.linspace(-2 * np.pi, 4 * np.pi, 1000)})
# p = pd.DataFrame({'x':np.linspace(-9, 5, 250)})
print("Calling g.evaluate(...)...")
prediction = g.evaluate(p)

# PLOT
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12), dpi=300)
ax1.plot(x, f, "r-")
ax1.scatter(x, (y + 1) / 2.0, s=100, color="g", marker="+")
ax1.plot(p["x"], prediction["Mean"], "b-")  # TODO: Var
ax1.plot(p["x"], prediction["Mean"] + 2 * np.sqrt(prediction["Var"]), "b--")
ax1.plot(p["x"], prediction["Mean"] - 2 * np.sqrt(prediction["Var"]), "b--")

transformed_mean_var = "Mean-Transformed"
transformed_var_var = "Var-Transformed"
ax2.plot(p["x"], prediction[transformed_mean_var], "b-")
ax2.plot(
    p["x"],
    prediction[transformed_mean_var] + 2 * np.sqrt(prediction[transformed_var_var]),
    "b--",
)
ax2.plot(
    p["x"],
    prediction[transformed_mean_var] - 2 * np.sqrt(prediction[transformed_var_var]),
    "b--",
)

##################################
print("Saving plots...")
fig.savefig(WORK_DIR / "gpc_test.png")
print("Displaying plots...")
plt.show()
print("Exiting.")
exit()
##################################
################### 1D
# Slice across sigma2
from matplotlib import cm

fig = plt.figure(figsize=(16,12), dpi=300)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# Make data.
l2 = 0.04
sigma2_vec = np.linspace(
    s2_range[0], s2_range[1], 100
)  # np.logspace(np.log10(s2_range[0]), np.log10(s2_range[1]), 25)
NLML = np.zeros_like(sigma2_vec)
d_dsigma2_NLML = np.zeros_like(sigma2_vec)

f_hat = None
for i, s2 in enumerate(sigma2_vec):
    nlml, df, f_hat = g.negative_log_marginal_likelihood_and_gradient(
        np.array([s2, l2]), f_hat
    )
    NLML[i] = nlml
    d_dsigma2_NLML[i] = df[0]


i_star = NLML.argmin()

# Plot
ax1.plot(sigma2_vec, NLML, "b")
ax1.plot(sigma2_vec[i_star], NLML[i_star], "b+")
ax2.plot(sigma2_vec, np.zeros_like(sigma2_vec), "k:")

num_diff = np.diff(NLML) / np.diff(sigma2_vec)
if i_star < len(num_diff):
    ax2.plot(sigma2_vec[i_star], num_diff[i_star], "b+")
ax2.plot(0.5 * (sigma2_vec[:-1] + sigma2_vec[1:]), num_diff, "b")

j = np.abs(d_dsigma2_NLML).argmin()
ax2.plot(sigma2_vec, d_dsigma2_NLML, "r:")
ax2.plot(sigma2_vec[j], d_dsigma2_NLML[j], "r.")


ax1.set_xlabel("sigma2_f")
ax1.set_ylabel("NLML")
ax1.set_xlim([np.min(sigma2_vec), np.max(sigma2_vec)])
ax1.set_title("Negative log marginal likelihood wrt s2, l2=%f" % l2)


################### 1D
# Slice across l2
from matplotlib import cm

fig = plt.figure(figsize=(16,12), dpi=300)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# Make data.
sigma2_f = 4
l2_vec = np.linspace(l2_range[0], l2_range[1], 100)

NLML = np.zeros_like(l2_vec)
d_dl2_NLML = np.zeros_like(l2_vec)

f_hat = None
for i, l2 in enumerate(l2_vec):
    nlml, df, f_hat = g.negative_log_marginal_likelihood_and_gradient(
        np.array([s2, l2]), f_hat
    )
    NLML[i] = nlml
    d_dl2_NLML[i] = df[1]

i_star = NLML.argmin()

# Plot
ax1.plot(l2_vec, NLML, "b")
ax1.plot(l2_vec[i_star], NLML[i_star], "b+")
ax2.plot(l2_vec, np.zeros_like(l2_vec), "k:")

num_diff = np.diff(NLML) / np.diff(l2_vec)
if i_star < len(num_diff):
    ax2.plot(l2_vec[i_star], num_diff[i_star], "b+")
ax2.plot(0.5 * (l2_vec[:-1] + l2_vec[1:]), num_diff, "b")

j = np.abs(d_dl2_NLML).argmin()
ax2.plot(l2_vec, d_dl2_NLML, "r:")
ax2.plot(l2_vec[j], d_dl2_NLML[j], "r.")

ax1.set_xlabel("l2")
ax1.set_ylabel("NLML")
ax1.set_xlim([np.min(l2_vec), np.max(l2_vec)])
ax1.set_title("Negative log marginal likelihood wrt l2, s2=%f" % sigma2_f)


############### 2D

# THETA PLOT
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure(figsize=(16,12), dpi=300)
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2)  # , projection='3d')

# Make data.
sigma2_vec = np.linspace(
    s2_range[0], s2_range[1], 10
)  # np.logspace(np.log10(s2_range[0]), np.log10(s2_range[1]), 25)
l2_vec = np.linspace(l2_range[0], l2_range[1], 10)
S2 = np.zeros([len(sigma2_vec), len(l2_vec)])
L2 = np.zeros_like(S2)
NLML = np.zeros_like(S2)
# NormDF = np.zeros_like(S2)
dS2 = np.zeros_like(S2)
dL2 = np.zeros_like(S2)


f_hat = None
for i, s2 in enumerate(sigma2_vec):
    for j, l2 in enumerate(l2_vec):
        S2[i, j] = s2
        L2[i, j] = l2
        nlml, df, f_hat = g.negative_log_marginal_likelihood_and_gradient(
            np.array([s2, l2]), f_hat
        )
        NLML[i, j] = nlml
        dS2[i, j] = df[0]
        dL2[i, j] = df[1]
        # NormDF[i,j] = np.linalg.norm(df)

# Correct for range
dS2 = dS2 * (s2_range[1] - s2_range[0]) ** 2
dL2 = dL2 * (l2_range[1] - l2_range[0]) ** 2

amin = NLML.argmin()
i_star, j_star = np.unravel_index(amin, NLML.shape)
print("MIN:", NLML[i_star, j_star])
print("S2:", S2[i_star, j_star])
print("L2:", L2[i_star, j_star])

# Plot the surface.
surf1 = ax1.plot_surface(
    S2, L2, NLML, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5
)
q1 = ax2.quiver(S2, L2, dS2, dL2, angles="xy", units="xy")  # , scale=1 scale_units='xy'

if optim:
    ax1.scatter(optim.x[0], optim.x[1], optim.fun, c="m", s=500, marker="*")
    ax2.scatter(optim.x[0], optim.x[1], c="m", marker="*", s=50)

ax1.scatter(
    S2[i_star, j_star],
    L2[i_star, j_star],
    NLML[i_star, j_star],
    c="r",
    s=500,
    marker=".",
)
ax2.scatter(S2[i_star, j_star], L2[i_star, j_star], c="r", marker=".", s=50)
# surf2 = ax2.plot_surface(S2, L2, NormDF, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax1.set_xlabel("sigma2_f")
ax1.set_ylabel("l^2")
ax1.set_xlim([np.min(sigma2_vec), np.max(sigma2_vec)])
ax1.set_ylim([np.min(l2_vec), np.max(l2_vec)])
ax1.set_title("Negative log marginal likelihood")
ax2.set_xlabel("sigma2_f")
ax2.set_ylabel("l^2")
ax2.set_xlim([np.min(sigma2_vec), np.max(sigma2_vec)])
ax2.set_ylim([np.min(l2_vec), np.max(l2_vec)])
# plt.axis('equal')

# ax1.view_init(azim=0, elev=90)
# plt.zlabel('Negative Log Marginal Likelihood')
# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
# plt.add_colorbar(surf1, shrink=0.5, aspect=5)

plt.show()
