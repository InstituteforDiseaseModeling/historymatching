import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import copy
import json

from multiprocessing import Pool
from functools import partial
#from normalizer import UserStandardize

import scipy.optimize as spo
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from string import Template
import skcuda.misc as misc

plt.rcParams['image.cmap'] = 'jet'

# NOTE theta = [sigma_f^2, sigma_n^2, l_1^2, l_2^2, ..., l_D^2]
# Ack https://github.com/lebedov/scikit-cuda/blob/master/demos/indexing_2d_demo.py

class GPR():

    def __init__(self, Xcols, Ycol, training_data, param_info,
            kernel_mode = 'RBF',
            kernel_params = None,
            #is_poisson = False,
            verbose = False,
            debug = False,
            **kwargs
        ):

        #sns.set_style("whitegrid")

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.kernel_fn = os.path.join(cur_dir, 'kernel.c')

        self.training_data = training_data.copy()
        self.param_info = param_info.copy()
        self.Xcols = Xcols
        self.Ycol = Ycol

        self.kernel_mode = kernel_mode

        if 'normalizer_mean' in kwargs and 'normalizer_std' in kwargs:
            self.normalizer_mean = kwargs['normalizer_mean']
            self.normalizer_std = kwargs['normalizer_std']
        else:
            Y = np.ma.masked_invalid(self.training_data[self.Ycol], copy=False)
            self.normalizer_mean = Y.mean(0).view(np.ndarray)
            self.normalizer_std  = Y.std(0).view(np.ndarray)

        # Normalize training data, change Ycol
        self.training_data[self.Ycol+'_normalized'] = self.normalize(self.training_data[self.Ycol])
        self.Ycol_orig = self.Ycol
        self.Ycol = self.Ycol+'_normalized'

        self.Xcols_scaled = []
        for xc in self.Xcols:
            xc_new = xc+' (scaled)'
            self.Xcols_scaled.append(xc_new)
            self.training_data[xc+' (scaled)'] = (self.training_data[xc] - self.param_info.loc[xc,'Min'])/(self.param_info.loc[xc,'Max']-self.param_info.loc[xc,'Min'])


        self.normalizer = True #UserStandardize(mean=self.normalizer_mean, std=self.normalizer_std)
        self.poisson = False #is_poisson
        self.verbose = verbose
        self.debug = debug
        self.D = len(self.Xcols)

        self.theta = None # Kernel/model hyperparameters
        self.kernel_xx_gpu = None

        self.kernel_params = kernel_params
        self.define_kernel(self.kernel_params)


    @classmethod
    def from_config(cls, config_fn):
        try:
            print "from_config:", config_fn
            with open(os.path.join(config_fn)) as data_file:
                config = json.load( data_file )

                return cls(
                    config['Xcols'],
                    config['Ycol'],
                    training_data = pd.read_json( config['Training_Data'], orient='split' ).set_index('Sample'),
                    param_info = pd.read_json( config['Param_Info'], orient='split' ).set_index('Name'),
                    kernel_mode = config['Kernel_Mode'],
                    kernel_params = np.array(config['Kernel_Params']),
                    normalizer_mean = config['Normalizer_Mean'],
                    normalizer_std = config['Normalizer_Std']
                )
        except EnvironmentError:
            print "Unable to load GPR from_config file", config_fn
            raise

    def set_training_data(self, new_training_data):
        self.training_data = new_training_data.copy()
        self.define_kernel(self.kernel_params)

        # Normalize training data as in __init__
        self.training_data[self.Ycol] = self.normalize(self.training_data[self.Ycol_orig])

        for xc in self.Xcols:
            xc_new = xc+' (scaled)'
            self.training_data[xc+' (scaled)'] = (self.training_data[xc] - self.param_info.loc[xc,'Min'])/(self.param_info.loc[xc,'Max']-self.param_info.loc[xc,'Min'])


    def save(self, save_to):
        with open(save_to, 'w') as fout:
            print self.normalizer_mean
            print self.normalizer_std

            json.dump(
                {
                    'Xcols'         : self.Xcols,
                    'Ycol'          : self.Ycol_orig,
                    'Kernel_Mode'   : self.kernel_mode,
                    'Kernel_Params' : self.theta.tolist(),
                    'Normalizer_Mean': self.normalizer_mean,
                    'Normalizer_Std': self.normalizer_std,
                    'Training_Data' : self.training_data.reset_index().to_json(orient='split'), # [self.Xcols + [self.Ycol]]
                    'Param_Info'        : self.param_info.reset_index().to_json(orient='split')
                }, fout, indent=4)

    def normalize(self, data):
        return (data - self.normalizer_mean)/self.normalizer_std
        ###print 'WARNING: NOT NORMALIZING!!!'
        ###return data

    def inverse_normalize_mean(self, data):
        return data*self.normalizer_std + self.normalizer_mean
        ###print 'WARNING: NOT INVERSE NORMALIZING!!!'
        ###return data

    def inverse_normalize_var(self, data):
        return data * (self.normalizer_std**2)
        ###print 'WARNING: NOT INVERSE NORMALIZING VAR!!!'
        ###return data

    def define_kernel(self, params):
        if self.kernel_mode == 'RBF':
            Nx = self.training_data.shape[0]

            with open(self.kernel_fn, 'r') as f:
                kernel_code_template = Template(f.read())

            max_threads_per_block, max_block_dim, max_grid_dim = misc.get_dev_attrs(pycuda.autoinit.device)
            block_dim, grid_dim = misc.select_block_grid_sizes(pycuda.autoinit.device, (Nx, Nx))
            max_blocks_per_grid = max(max_grid_dim)

            if self.verbose:
                print "max_threads_per_block", max_threads_per_block
                print "max_block_dim", max_block_dim
                print "max_grid_dim", max_grid_dim
                print "max_blocks_per_grid", max_blocks_per_grid
                print "block_dim", block_dim
                print "grid_dim", grid_dim

            # Substitute in template to get kernel code
            kernel_code = kernel_code_template.substitute(
                max_threads_per_block   = max_threads_per_block,
                max_blocks_per_grid     = max_blocks_per_grid,
                B = Nx)

            # Compile the kernel
            mod = compiler.SourceModule(kernel_code)

            # retrieve the kernel functions
            self.kernel_xx_gpu = mod.get_function("kernel_xx")
            self.kernel_xp_gpu = mod.get_function("kernel_xp")

        else:
            print 'Bad kernel mode, kernel_mode=%s'%self.kernel_mode
            raise

        if params is not None:
            assert( len(params) == 2+self.D )
            self.theta = params

    def kernel_xx(self, X, theta, add_sigma2_n):
        # NOTE: Slow, use GPU acceleration instead.
        sigma2_f = theta[0]
        sigma2_n = theta[1]

        N = X.shape[0]

        kxx = np.zeros([N,N], dtype=np.float32)
        for i in range(N):
            # Off-diagonal
            for j in range(i+1,N):
                dX = X[i,:]-X[j,:]
                r2 = 0
                for d in range(self.D):
                    r2 += dX[d] * dX[d]/theta[2+d]
                kxx[i,j] = sigma2_f * np.exp( -r2 / 2. )
                kxx[j,i] = kxx[i,j]
            # Diagonal:
            dX = X[i,:]-X[i,:]
            r2 = 0
            for d in range(self.D):
                r2 += dX[d] * dX[d]/theta[2+d]
            kxx[i,i] = sigma2_f * np.exp( -r2 / 2. )

            if add_sigma2_n:
                kxx[i,i] += sigma2_n

        return kxx


    def kernel_xp(self, X, P, theta):
        # NOTE: Slow, use GPU acceleration instead.
        sigma2_f = theta[0]

        Nx = X.shape[0]
        Np = P.shape[0]
        D = X.shape[1]

        kxp = np.zeros([Nx,Np])
        for i in range(Nx):
            for j in range(Np):
                dX = X[i,:]-P[j,:]
                r2 = 0
                for d in range(D):
                    r2 += dX[d] * dX[d]/theta[2+d]
                kxp[i,j] = sigma2_f * np.exp( -r2 / 2. )

        return kxp


    def kxx_gpu_wrapper(self, X, theta, add_sigma2_n = True):
        Nx = X.shape[0]
        # Use from before...?
        block_dim, grid_dim = misc.select_block_grid_sizes(pycuda.autoinit.device, (Nx, Nx))

        X_gpu = gpuarray.to_gpu(X.astype(np.float32))

        theta_gpu = gpuarray.to_gpu(theta.astype(np.float32))

        # create empty gpu array for the result
        Kxx_gpu = gpuarray.empty((Nx, Nx), np.float32)

        # call the kernel on the card
        self.kernel_xx_gpu(
            Kxx_gpu,            # <-- Output
            X_gpu, theta_gpu,   # <-- Inputs
            np.uint32(Nx),      # <-- N
            np.uint32(self.D),  # <-- D
            np.uint8(X.flags.f_contiguous), # FORTRAN (column) contiguous
            block = block_dim,
            grid = grid_dim
        )

        Kxx = Kxx_gpu.get()

        if add_sigma2_n:
            # Add sigma_n^2 to the diagonal, observation noise
            Kxx[np.diag_indices(Nx)] += theta[1]

        if self.debug:
            Kxx_cpu = self.kernel_xx(X.astype(np.float32), theta.astype(np.float32), add_sigma2_n)
            if not np.allclose(Kxx_cpu, Kxx):
                print 'kxx_gpu_wrapper(CPU):\n', Kxx_cpu
                print 'kxx_gpu_wrapper(GPU):\n', Kxx
                raise

        return Kxx


    def kxp_gpu_wrapper(self, X, P, theta):
        Nx = X.shape[0]
        Np = P.shape[0]
        block_dim, grid_dim = misc.select_block_grid_sizes(pycuda.autoinit.device, (Nx, Np))

        if X.flags.f_contiguous: # Fortran column-major
            # Convert to C contiguous (row major)
            X_gpu = gpuarray.to_gpu(np.ascontiguousarray(X).astype(np.float32))
        else:
            X_gpu = gpuarray.to_gpu(X.astype(np.float32))

        if P.flags.f_contiguous:
            # Convert to C contiguous (row major)
            P_gpu = gpuarray.to_gpu(np.ascontiguousarray(P).astype(np.float32))
        else:
            P_gpu = gpuarray.to_gpu(P.astype(np.float32))

        theta_gpu = gpuarray.to_gpu(theta.astype(np.float32))

        # create empty gpu array for the result
        Kxp_gpu = gpuarray.empty((Nx, Np), np.float32)

        # call the kernel on the card
        self.kernel_xp_gpu(
            Kxp_gpu,                   # <-- Output
            X_gpu, P_gpu, theta_gpu,   # <-- Inputs
            np.uint32(Nx),   # <-- Nx
            np.uint32(Np),   # <-- Nx
            np.uint32(self.D),  # <-- D
            block = block_dim,
            grid = grid_dim
        )

        if self.debug:
            Kxp_cpu = self.kernel_xp(X, P, theta)
            if not np.allclose(Kxp_cpu, Kxp_gpu.get()):
                print 'kxp_gpu_wrapper(CPU):\n', Kxp_cpu
                print 'kxp_gpu_wrapper(GPU):\n', Kxp_gpu.get()
                raise

        return Kxp_gpu.get()


    def cross_validation(self, theta, X, Y, P):
        num_partitions = int(max(P)+1)
        num_points = len(P)

        Y_mean = np.nanmean(Y, axis=1)

        KXX = self.kxx_gpu_wrapper(X, theta, add_sigma2_n = True) # Want predictive distribution
        if self.debug:
            KXX_cpu = self.kernel_xx(X, theta, add_sigma2_n = True)
            if not np.allclose(KXX_cpu, KXX):
                print 'loo_cross_validation(CPU XX):\n', KXX_cpu
                print 'loo_cross_validation(GPU XX):\n', KXX
                raise

        ll = 0
        for partition in range(num_partitions):
            train_inds = [k for k in range(num_points) if P[k]!=partition]
            train_array = np.array(train_inds, dtype=np.intp)

            test_inds = [k for k in range(num_points) if P[k]==partition]
            test_array = np.array(test_inds, dtype=np.intp)

            Kxx = KXX[train_array[:,np.newaxis], train_array] # kernel_xx(x, theta)

            #y = Y[train_inds, :]
            y_mean = Y_mean[train_inds]
            yp = Y[test_inds, :]

            Kxp = KXX[train_array[:,np.newaxis], test_inds] # kernel_xp(x, xp, theta)
            if self.debug:
                x_cpu = X[train_inds,]
                xp_cpu = X[test_inds,:][np.newaxis,:]
                Kxp_cpu = self.kernel_xp(x_cpu, xp_cpu, theta)
                if not np.allclose(Kxp_cpu, Kxp):
                    print 'loo_cross_validation(CPU Xp):\n', Kxp_cpu
                    print 'loo_cross_validation(GPU Xp):\n', Kxp
                    raise

            Kpp = KXX[test_array[:,np.newaxis], test_inds] #kernel_xx(xp, theta)
            if self.debug:
                Kpp_cpu = self.kernel_xx(xp_cpu, theta, add_sigma2_n = True) # PREDICTIVE
                if not np.allclose(Kpp_cpu, Kpp):
                    print 'loo_cross_validation(CPU pp):\n', Kpp_cpu
                    print 'loo_cross_validation(GPU pp):\n', Kpp
                    raise

            f = np.dot(Kxp.T, np.linalg.solve(Kxx, y_mean)) # NOTE: Using mean here
            covf = Kpp - np.dot(Kxp.T, np.linalg.solve(Kxx, Kxp))

            err = yp-np.repeat(f[:,np.newaxis], yp.shape[1], axis=1)
            for row in range(yp.shape[0]):
                err_row = err[row,:]
                err_row = err_row[~np.isnan(err_row)]
                ll += np.sum(-0.5*err_row**2/covf[row,row]) -0.5*np.log(2*np.pi*covf[row,row]) * len(err_row)

            #UNIVARIATE:
            #ll += -0.5*np.dot((yp-f).T, (yp-f))/covf -0.5*np.log(2*np.pi*covf)

            #(_, logdet) = np.linalg.slogdet(covf)
            #ll += -self.D/2.0*np.log(2*np.pi) - 0.5 * logdet -0.5*np.dot(yp-f, np.linalg.solve(covf, yp-f))

        if self.verbose:
            print theta, '-->', -ll

        return np.array([-ll])


    def assign_rep(self, sample):
        sample = sample.drop('index', axis=1).reset_index()
        sample.index.name='Replicate'
        sample.reset_index(inplace=True)
        return sample


    def optimize_hyperparameters(self, x0, bounds, K=-1, eps=1e-2, disp=True, maxiter=15000):
        # x0 like np.array([2, 0.10, 0.14641288665436947, 0.12166006573919039, 0.05, 0.05, 0.08055223671416605, 7.026854485434267 ])
        # bounds like ((0.005,10),)+((0.01,10),) + tuple((5e-5,10) for i in range(self.D))
        # K=None is leave one out cross validation, otherwise make K groups
        idx = self.training_data.index.names    # Save index
        self.training_data.reset_index(inplace=True)

        samples = self.training_data['Sample'].unique()
        for i,s in enumerate(samples):
            self.training_data.loc[ self.training_data['Sample']==s, 'Sample_Index'] = i

        if K <=1:
            # Identity partition (LOO)
            self.training_data['Partition'] = self.training_data['Sample_Index']
        else:
            assert(K<=len(samples))
            self.training_data['Partition'] = np.floor(self.training_data['Sample_Index']%K).astype(int)

        num_params = 2 + self.D # sigma_n, sigma_f, lengthscale 1, lengthscale_2, ..., lengthscale_D

        train_mean = self.training_data.reset_index().groupby('Sample').mean()
        X = train_mean[self.Xcols_scaled].values
        P = train_mean['Partition'].values
        Y = self.training_data.reset_index().groupby('Sample').apply(self.assign_rep).pivot('Sample', 'Replicate', self.Ycol).values

        # Maximize LOO cross-validation error
        ret = spo.minimize(
            self.cross_validation,
            args=(X,Y,P),
            x0 = x0,
            method='L-BFGS-B',
            bounds = bounds, # Constrain values
            jac=None, hess=None, hessp=None,
            constraints=(), tol=None, callback=None,
            options= {
                'maxiter':maxiter,
                'disp':disp,
                'eps':eps # eps: Step size used for numerical approximation of the jacobian (1e-3).
            }
        )

        print 'OPTIMIZATION RETURNED:\n', ret

        # Restore original index
        self.training_data.set_index(idx, inplace=True)

        self.theta = ret.x # Length scales now on 0-1 range


    def evaluate(self, data):
        # Predict at test and training points, store mean and variance in self.data

        # Normalize data
        for xc in self.Xcols:
            xc_new = xc+' (scaled)'
            data[xc+' (scaled)'] = (data[xc] - self.param_info.loc[xc,'Min'])/(self.param_info.loc[xc,'Max']-self.param_info.loc[xc,'Min'])

        X = self.training_data[self.Xcols_scaled].values
        Y = self.training_data[self.Ycol].values
        P = data[self.Xcols_scaled].values

        if self.debug:
            print 'X',X.shape,' flags:\n', X.flags
            print 'Y',Y.shape,' flags:\n', Y.flags
            print 'P',P.shape,' flags:\n', P.flags

        # TODO: Save Kxx, just compute Kxp and Kpp!
        Kxx = self.kxx_gpu_wrapper(X, self.theta, add_sigma2_n = True)  # Y is noisy
        if self.debug:
            Kxx_cpu = self.kernel_xx(X, self.theta, add_sigma2_n = True)
            if not np.allclose(Kxx_cpu, Kxx):
                print 'evaluate(CPU XX):\n', Kxx_cpu
                print 'evaluate(GPU XX):\n', Kxx
                raise

        Kxp = self.kxp_gpu_wrapper(X, P, self.theta)
        if self.debug:
            Kxp_cpu = self.kernel_xp(X, P, self.theta)
            if not np.allclose(Kxp_cpu, Kxp):
                print 'evaluate(CPU XP):\n', Kxp_cpu
                print 'evaluate(GPU XP):\n', Kxp
                raise

        Kpp = self.kxx_gpu_wrapper(P, self.theta, add_sigma2_n = False) # For latent distribution
        if self.debug:
            Kpp_cpu = self.kernel_xx(P, self.theta, add_sigma2_n = False)
            if not np.allclose(Kpp_cpu, Kpp):
                print 'evaluate(CPU PP):\n', Kpp_cpu
                print 'evaluate(GPU PP):\n', Kpp
                raise

        f = np.dot(Kxp.T, np.linalg.solve(Kxx, Y))
        # Print, just want diagonal elements!
        covf = Kpp - np.dot(Kxp.T, np.linalg.solve(Kxx, Kxp))
        stdf = np.sqrt(np.diag(covf))

        fig = None
        if self.D == 1: # One D
            print '1D'
            fig = plt.figure()
            plt.plot(X[:,0], Y, 'o')
            #plt.plot(P[:,0], f, '|-')
            plt.errorbar(P[:,0], f, yerr=2*stdf, lw=1)
        elif self.D == 2:
            print '2D'
            fig = plt.figure()
            plt.scatter(X[:,0], X[:,1], s=Y, c=Y, edgecolor='k', linewidth=2, cmap='jet')
            plt.scatter(P[:,0], P[:,1], s=f, c=f, linewidth=0, cmap='jet')

        # Note inverse normalize
        return {    'Mean': self.inverse_normalize_mean(f),
                    'Var_Latent': self.inverse_normalize_var(np.diag(covf)),
                    'Var_Predictive': self.inverse_normalize_var(np.diag(covf) + self.theta[1]*np.ones(P.shape[0])),
                    'Fig': fig      }

    def plot_data(self, samples_to_circle=[]):
        scaled = 5 + 45*(self.training_data[self.Ycol] - self.training_data[self.Ycol].min()) / (self.training_data[self.Ycol].max() - self.training_data[self.Ycol].min())

        figs = {}

        for row in range(self.D):
            for col in range(self.D):
                if col > row:
                    #gs = gridspec.GridSpec(self.D-1, self.D-1)
                    #ax = fig.add_subplot(gs[col-1,row])
                    fn = '%s-%s.pdf' % (self.Xcols[row], self.Xcols[col])
                    figs[fn] = plt.figure(figsize=(6,6)) #GPy.plotting.plotting_library().figure()

                    x = self.training_data[ self.Xcols[row] ]
                    y = self.training_data[ self.Xcols[col] ]

                    plt.scatter(x, y, s=scaled, c=self.training_data[self.Ycol], cmap='jet', lw=0.1, alpha=0.5, edgecolors='k') #, s=area, c=colors, alpha=0.5)

                    # Circle some interesting samples
                    for s in samples_to_circle:
                        plt.scatter(self.training_data.loc[s][ self.Xcols[row] ], self.training_data.loc[s][ self.Xcols[col] ], s=10+scaled.loc[s], alpha=1, lw=1.0, facecolors="None", edgecolors='k') #, s=area, c=colors, alpha=0.5)

                    plt.autoscale(tight=True)
                    plt.xlabel( self.Xcols[row] )
                    plt.ylabel( self.Xcols[col] )
                    plt.tight_layout()

        return figs


    def plot_histogram(self):
        fig, ax = plt.subplots(nrows=1, ncols=1) # , figsize=(5,5), sharex='col', sharey='row')
        sns.distplot(self.training_data[self.Ycol], rug=True, ax = ax)

        return fig


    def plot(self, Xcenter, res=10):
        Xmu = np.repeat( np.array([Xcenter]), res*res, axis=0)

        fig = plt.figure(figsize=(4*(self.D-1),4*(self.D-1)))
        fig_std_latent = plt.figure(figsize=(4*(self.D-1),4*(self.D-1)))
        for row in range(self.D):
            for col in range(self.D):
                if col > row:
                    gs = gridspec.GridSpec(self.D-1, self.D-1)
                    ax = fig.add_subplot(gs[col-1,row]) # , projection='3d'
                    ax_std_latent = fig_std_latent.add_subplot(gs[col-1,row]) # , projection='3d'

                    fixed_inputs = [ (x,mean) for (i, (x,mean)) in enumerate(zip(range(self.D), Xcenter)) if row is not i and col is not i]
                    print row, col, row*self.D+col, fixed_inputs

                    # TODO: Real parameter ranges here, not just 0-1
                    (row_min, row_max) = (self.training_data[self.Xcols[row]].min(), self.training_data[self.Xcols[row]].max())
                    (col_min, col_max) = (self.training_data[self.Xcols[col]].min(), self.training_data[self.Xcols[col]].max())
                    # sim_cases_range = data.reset_index().groupby('Sample')['Sim_Cases'].agg({'Min':np.min, 'Max':np.max, 'Mean':np.mean})
                    x1 = np.linspace(row_min, row_max, res)
                    x2 = np.linspace(col_min, col_max, res)
                    X1, X2 = np.meshgrid(x1, x2)

                    X = Xmu.copy()
                    X[:,row] = X1.flatten()
                    X[:,col] = X2.flatten()

                    Xdf = pd.DataFrame(X, columns=self.Xcols)

                    self.debug=False;
                    #print 'WARNING: DEBUG!\n'
                    self.verbose=False

                    ret = self.evaluate( Xdf )

                    Y_mean = np.reshape(ret['Mean'], [res,res])
                    Y_std_latent = np.reshape( np.sqrt(ret['Var_Latent']), [res, res])
                    #Y_std_predictive = np.reshape( np.sqrt(ret['Var_Predictive']), [res, res])

                    try:
                        CS = ax.contour(X1, X2, Y_mean, zorder=100)
                        ax.clabel(CS, inline=1, fontsize=10, zorder=100)
                    except:
                        print 'Unable to plot mean contour'
                        pass

                    ax.scatter(self.training_data[self.Xcols[row]], self.training_data[self.Xcols[col]], c=self.training_data[self.Ycol], s=25)

                    try:
                        CS = ax_std_latent.contour(X1, X2, Y_std_latent, zorder=100)
                        ax_std_latent.clabel(CS, inline=1, fontsize=10, zorder=100)
                    except:
                        print 'Unable to plot std contour'
                        pass

                    if col == self.D-1:
                        ax.set_xlabel( self.Xcols[row] )
                    if row == 0:
                        ax.set_ylabel( self.Xcols[col] )
        #plt.tight_layout()
        return (fig, fig_std_latent)

    def plot_errors(self, train, test, mean_col, var_predictive_col, var_latent_col):

        train['Z_Predictive'] = (train[self.Ycol_orig] - train[mean_col]) / np.sqrt(train[var_predictive_col])
        train['Z_Latent'] = (train[self.Ycol_orig] - train[mean_col]) / np.sqrt(train[var_latent_col])
        test['Z_Predictive'] = (test[self.Ycol_orig] - test[mean_col]) / np.sqrt(test[var_predictive_col])
        test['Z_Latent'] = (test[self.Ycol_orig] - test[mean_col]) / np.sqrt(test[var_latent_col])

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex='col', figsize=(16,10)) # , sharex='col', sharey='row')

        ax = ax1
        ax.errorbar(x=test[self.Ycol_orig], y=test[mean_col], yerr=2*np.sqrt(test[var_predictive_col]), fmt='o', c='m', lw=0.5)
        ax.errorbar(x=train[self.Ycol_orig], y=train[mean_col], yerr=2*np.sqrt(train[var_predictive_col]), fmt='o', c='c', lw=0.5)
        ax.margins(x=0,y=0.05)
        xlim = ax.get_xlim()
        ax.plot( [xlim[0],xlim[1]], [xlim[0], xlim[1]], 'r-')
        ax.set_xlabel(self.Ycol_orig)
        ax.set_ylabel('Predicted')

        ax = ax2
        ax.scatter(x=train['Sample'], y=train[self.Ycol_orig], c='c', marker='_', s=25, alpha=1, linewidths=1, zorder=50)
        ax.scatter(x=test['Sample'], y=test[self.Ycol_orig], c='m', marker='_', s=25, alpha=1, linewidths=1, zorder=50)
        ax.errorbar(x=train['Sample'], y=train[mean_col], yerr=2*np.sqrt(train[var_predictive_col]), fmt='.', ms=5, linewidth=1, c='k')
        ax.errorbar(x=test['Sample'], y=test[mean_col], yerr=2*np.sqrt(test[var_predictive_col]), fmt='.', ms=5, linewidth=1, c='k')
        ax.margins(x=0,y=0.05)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(self.Ycol_orig)


        a=0.05
        ax = ax4
        ax.scatter(x=train['Sample'], y=train['Z_Predictive'], c='c', marker='_', alpha=0.5, linewidth=1)
        ax.scatter(x=test['Sample'], y=test['Z_Predictive'], c='m', marker='_', alpha=0.5, linewidth=1)

        ax.margins(x=0,y=0.05)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.add_patch( patches.Rectangle( (0, -2), xlim[1], 4, alpha=a, color='g' ) )
        ax.add_patch( patches.Rectangle( (0, -5), xlim[1], 3, alpha=a, color='#FFA500' ) )
        ax.add_patch( patches.Rectangle( (0, 2), xlim[1], 3, alpha=a, color='#FFA500' ) )
        ax.add_patch( patches.Rectangle( (0, ylim[0]), xlim[1], abs(ylim[0])-5, alpha=a, color='r' ) )
        ax.add_patch( patches.Rectangle( (0, 5), xlim[1], abs(ylim[1])-5, alpha=a, color='r' ) )
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Z-Score')

        ax = ax3
        ax.scatter(x=train[self.Ycol_orig], y=train['Z_Predictive'], facecolor='c', marker='.', lw=1, alpha=0.5, s=50)
        ax.scatter(x=test[self.Ycol_orig], y=test['Z_Predictive'], facecolor='m', marker='.', lw=1, alpha=0.5, s=50)
        ax.set_xlabel(self.Ycol_orig)
        ax.set_ylabel('Z-Score')
        ax.margins(x=0,y=0.05)

        plt.tight_layout()

        return fig
