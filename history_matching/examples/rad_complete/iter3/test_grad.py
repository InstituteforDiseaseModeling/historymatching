import os, json
import numpy as np
from history_matching.gpr import GPR
import matplotlib.pyplot as plt

res = 25 # Points per dimension

gpr_model_fn = os.path.join('Cuts', 'RadiusShouldBe15', 'GPR', 'model.json')
gpr_model = GPR.from_config(gpr_model_fn)
theta0 = gpr_model.theta

train_mean = gpr_model.training_data.reset_index().groupby('Sample_Id').mean()
X = gpr_model.basis.generate_dmatrix( train_mean, scaleX = True).values
Y = gpr_model.training_data.reset_index().groupby('Sample_Id').apply(gpr_model.assign_rep).pivot('Sample_Id', 'Replicate', gpr_model.Ycol).values
ll_base, grad_base = gpr_model.cross_validation_with_grad(gpr_model.theta, X, Y, optimize_sigma2_n=True, log_transform=False)

print('Base model:')
print(ll_base)
print(grad_base)

params = {
    'sigma2_f': {
        'min': 2,
        'max': 20,
        'idx': 0
    },
    'sigma2_n': {
        'min': 0.01,
        'max': 0.25,
        'idx': 1
    },
    'l2_0': {
        'min': 0.01,
        'max': 0.25,
        'idx': 2
    },
    'l2_1': {
        'min': 0.001,
        'max': 0.02,
        'idx': 3
    },
}

fig, ax_vec = plt.subplots(2, len(params))

for j, (p,v) in enumerate(params.items()):

    ax_f = ax_vec[0, j]
    ax_g = ax_vec[1, j]

    delta = np.linspace(v['min'], v['max'], res)
    ll_vec = np.zeros_like(delta)
    grad_vec = np.zeros_like(delta)
    for i, new_value in enumerate(delta):
        theta = np.copy(theta0)
        theta[v['idx']] = new_value
        gpr_model.set_theta(theta)

        train_mean = gpr_model.training_data.reset_index().groupby('Sample_Id').mean()
        X = gpr_model.basis.generate_dmatrix( train_mean, scaleX = True).values
        Y = gpr_model.training_data.reset_index().groupby('Sample_Id').apply(gpr_model.assign_rep).pivot('Sample_Id', 'Replicate', gpr_model.Ycol).values
        ll, grad = gpr_model.cross_validation_with_grad(gpr_model.theta, X, Y, optimize_sigma2_n=True, log_transform=False)

        ll_vec[i] = ll
        grad_vec[i] = grad[v['idx']]

    ax_f.plot(delta, ll_vec)
    ax_f.plot(theta0[v['idx']], ll_base, marker='+')
    #ax_f.set_xlabel(p)
    ax_f.set_ylabel('LL')

    ax_g.plot(delta, grad_vec)
    ax_g.plot(theta0[v['idx']], grad_base[v['idx']], marker='+')
    ax_g.set_xlabel(p)
    ax_g.set_ylabel('dLL/d%s'%p)

plt.show()

