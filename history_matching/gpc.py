import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import copy
import json
import time

from multiprocessing import Pool
from functools import partial
from collections import deque

import scipy.optimize as spo
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from string import Template
import skcuda.misc as misc
from scipy.stats import norm

plt.rcParams['image.cmap'] = 'jet'

# NOTE theta = [sigma_f^2, l_1^2, l_2^2, ..., l_D^2] # NOTE: no sigma_n^2
# Ack https://github.com/lebedov/scikit-cuda/blob/master/demos/indexing_2d_demo.py

#with pd.option_context('display.max_rows', None): # , 'display.max_columns', 3
#    print(ret)

class GPC():

    def __init__(self, Xcols, Ycol, training_data, param_info,
            kernel_mode = 'RBF',
            kernel_params = None,
            verbose = False,
            debug = False,
            **kwargs
        ):


        self.use_laplace_approximation = True

        #sns.set_style("whitegrid")

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.kernel_fn = os.path.join(cur_dir, 'kernel.c')

        self.training_data = training_data.copy()
        self.param_info = param_info.copy()
        self.Xcols = Xcols
        self.Ycol = Ycol

        self.kernel_mode = kernel_mode

        self.Xcols_scaled = []
        for xc in self.Xcols:
            if xc not in self.training_data.columns:
                print('Cannot find columns %s in the training data, which has columns:'%xs, self.training_data.columns)
                raise Exception('Missing column')
            xc_new = xc+' (scaled)'
            self.Xcols_scaled.append(xc_new)
            self.training_data[xc+' (scaled)'] = (self.training_data[xc] - self.param_info.loc[xc,'Min'])/(self.param_info.loc[xc,'Max']-self.param_info.loc[xc,'Min'])


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
            print("from_config:", config_fn)
            with open(os.path.join(config_fn)) as data_file:
                config = json.load( data_file )
                return GPC.from_dict(config)
        except EnvironmentError:
            print("Unable to load GPC from_config file", config_fn)
            raise


    @classmethod
    def from_dict(cls, config):
        return cls(
            config['Xcols'],
            config['Ycol'],
            training_data = pd.read_json( config['Training_Data'], orient='split' ).set_index('Sample'),
            param_info = pd.read_json( config['Param_Info'], orient='split' ).set_index('Name'),
            kernel_mode = config['Kernel_Mode'],
            kernel_params = np.array(config['Kernel_Params']),
        )


    def set_training_data(self, new_training_data):
        self.training_data = new_training_data.copy()
        self.define_kernel(self.kernel_params)

        for xc in self.Xcols:
            xc_new = xc+' (scaled)'
            self.training_data[xc+' (scaled)'] = (self.training_data[xc] - self.param_info.loc[xc,'Min'])/(self.param_info.loc[xc,'Max']-self.param_info.loc[xc,'Min'])


    def save(self, save_to = None):
        save_dict = {
                    'Xcols'         : self.Xcols,
                    'Ycol'          : self.Ycol,
                    'Kernel_Mode'   : self.kernel_mode,
                    'Kernel_Params' : self.theta.tolist(),
                    'Training_Data' : self.training_data.reset_index().to_json(orient='split'), # [self.Xcols + [self.Ycol]]
                    'Param_Info'        : self.param_info.reset_index().to_json(orient='split')
                }

        if save_to:
            with open(save_to, 'w') as fout:
                json.dump(save_dict, fout, indent=4)
        return save_dict


    def define_kernel(self, params):
        if self.kernel_mode == 'RBF':
            Nx = self.training_data.shape[0]

            with open(self.kernel_fn, 'r') as f:
                kernel_code_template = Template(f.read())

            max_threads_per_block, max_block_dim, max_grid_dim = misc.get_dev_attrs(pycuda.autoinit.device)
            block_dim, grid_dim = misc.select_block_grid_sizes(pycuda.autoinit.device, (Nx, Nx))
            max_blocks_per_grid = max(max_grid_dim)

            if self.verbose:
                print("max_threads_per_block", max_threads_per_block)
                print("max_block_dim", max_block_dim)
                print("max_grid_dim", max_grid_dim)
                print("max_blocks_per_grid", max_blocks_per_grid)
                print("block_dim", block_dim)
                print("grid_dim", grid_dim)

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
            print('Bad kernel mode, kernel_mode=%s'%self.kernel_mode)
            raise

        if params is not None:
            assert( len(params) == 1+self.D )
            if isinstance(params,list):
                params = np.array(params)
            self.theta = params


    def kernel_xx(self, X, theta):
        # NOTE: Slow, use GPU acceleration instead.
        sigma2_f = theta[0]

        N = X.shape[0]

        kxx = np.zeros([N,N], dtype=np.float32)
        for i in range(N):
            # Off-diagonal
            for j in range(i+1,N):
                dX = X[i,:]-X[j,:]
                r2 = 0
                for d in range(self.D):
                    r2 += dX[d] * dX[d]/theta[1+d]
                kxx[i,j] = sigma2_f * np.exp( -r2 / 2. )
                kxx[j,i] = kxx[i,j]

            # Diagonal:
            kxx[i,i] = sigma2_f

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
                    r2 += dX[d] * dX[d]/theta[1+d]
                kxp[i,j] = sigma2_f * np.exp( -r2 / 2. )

        return kxp


    def kxx_gpu_wrapper(self, X, theta, deriv=-1):
        Nx = X.shape[0]
        # Use from before...?
        block_dim, grid_dim = misc.select_block_grid_sizes(pycuda.autoinit.device, (Nx, Nx))

        X_gpu = gpuarray.to_gpu(X.astype(np.float32))

        theta_extended = np.concatenate( (np.array(theta[:1]), np.array([0]), np.array(theta[1:])) ) # Kernel code needs sigma2_n
        theta_gpu = gpuarray.to_gpu(theta_extended.astype(np.float32))

        # create empty gpu array for the result
        Kxx_gpu = gpuarray.empty((Nx, Nx), np.float32)

        # call the kernel on the card
        self.kernel_xx_gpu(
            Kxx_gpu,            # <-- Output
            X_gpu, theta_gpu,   # <-- Inputs
            np.uint32(Nx),      # <-- N
            np.uint32(self.D),  # <-- D
            np.uint32(deriv),   # <- for dK/dtheta_i.  Negative for no deriv.
            np.uint8(X.flags.f_contiguous), # FORTRAN (column) contiguous
            block = block_dim,
            grid = grid_dim
        )

        Kxx = Kxx_gpu.get()

        if self.debug:
            Kxx_cpu = self.kernel_xx(X.astype(np.float32), theta.astype(np.float32))
            if not np.allclose(Kxx_cpu, Kxx):
                print('kxx_gpu_wrapper(CPU):\n', Kxx_cpu)
                print('kxx_gpu_wrapper(GPU):\n', Kxx)
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

        theta_extended = np.concatenate( (np.array(theta[:1]), np.array([0]), np.array(theta[1:])) ) # Kernel code needs sigma2_n
        theta_gpu = gpuarray.to_gpu(theta_extended.astype(np.float32))

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
                print('kxp_gpu_wrapper(CPU):\n', Kxp_cpu)
                print('kxp_gpu_wrapper(GPU):\n', Kxp_gpu.get())
                raise

        return Kxp_gpu.get()


    def assign_rep(self, sample):
        sample = sample.drop('index', axis=1).reset_index()
        sample.index.name='Replicate'
        sample.reset_index(inplace=True)
        return sample

    def expectation_propagation(self, theta):
        # Algorithm 3.5: "Expectation Propagation for binary classification" from "Gaussian Process for Machine Learning", p58
        # TODO: WIP!

        # TODO: Pass in 
        mu_tol = 1e-6
        Sigma_tol = 1e-6

        y = self.training_data[self.Ycol].values
        if self.verbose:
            print('y:', y)
        N = len(y)

        X = self.training_data[self.Xcols_scaled].values
        K = self.kxx_gpu_wrapper(X, theta)  # This is for f, no sigma2_n

        # Initialize
        nu = np.zeros(N)
        tau = np.zeros(N)
        Sigma = K.copy()
        mu = np.zeros(N)

        tau_minus_i_vec = np.zeros_like(tau)
        mu_minus_i_vec = np.zeros_like(mu)

        done = False
        it = 0
        while not done:
            prev_mu = mu
            prev_Sigma = Sigma
            prev_tau = tau
            prev_nu = nu
            for i in range(N):
                #print(it, ' ', i,' ', '-'*80)
                sigma2_i = Sigma[i,i] # Not sure on this one
                tau_minus_i = 1/sigma2_i - tau[i]
                nu_minus_i = mu[i]/sigma2_i - nu[i]
                assert(tau_minus_i > 0)

                den = tau_minus_i * (tau_minus_i+1)
                sqrt_den = np.sqrt( den )
                zi = y[i]*nu_minus_i / sqrt_den
                N_zi = norm.pdf(zi)
                Z_hat_i = norm.cdf(zi)
                mu_hat_i = nu_minus_i/tau_minus_i + y[i]*N_zi / (Z_hat_i * sqrt_den )
                sigma2_hat_i = 1/tau_minus_i - N_zi / (Z_hat_i * den) * (zi + N_zi / Z_hat_i)

                delta_tau = 1/sigma2_hat_i - tau_minus_i - tau[i] # using sigmahat2_i
                tau[i] = tau[i] + delta_tau
                nu[i] = mu_hat_i/sigma2_hat_i - nu_minus_i # using muhat_u and sigmahat2_i

                si = Sigma[:,i]
                Sigma = Sigma - 1/( (1/delta_tau) + Sigma[i,i] ) * np.outer(si, si)
                mu = np.dot(Sigma, nu)

                tau_minus_i_vec[i] = tau_minus_i
                mu_minus_i_vec[i] = nu_minus_i / tau_minus_i

            Stilde = np.diag(tau)
            '''
            sqrtStilde = np.diag(np.sqrt(tau))
            B = np.eye(N) + np.dot(sqrtStilde, np.dot(K, sqrtStilde))
            L = np.linalg.cholesky(B)
            V = np.linalg.solve( np.transpose(L), np.dot(sqrtStilde,K) )
            print('V', V)
            print('K', K)
            Sigma = K - np.dot(np.transpose(V),V)
            print('Sigma after', Sigma)

            t = time.time()
            SigmaDIRECT = np.linalg.inv(np.linalg.inv(K) + Stilde)
            print('Sigma DIRECT %f'%(time.time()-t), SigmaDIRECT)
            '''

            #t = time.time()
            SigmaMIL = K - np.dot(K, np.linalg.solve( np.linalg.inv(Stilde)+K, K) )
            #print('Sigma MIL %f'%(time.time()-t), SigmaMIL)

            '''
            import scipy as sp
            t = time.time()
            a = np.linalg.inv(Stilde)+K
            ainvsqrt = sp.linalg.sqrtm(np.linalg.inv(a))
            Vdjk = np.dot(ainvsqrt, K)
            SigmaDJK = K - np.dot(np.transpose(Vdjk),Vdjk)
            print('Sigma DJK %f'%(time.time()-t), SigmaDJK)
            print('Vdjk', Vdjk)

            #Bdjk = np.linalg.inv(Stilde)+K
            #Ldjk = np.linalg.cholesky(Bdjk)
            #Vdjk2 = np.linalg.solve(np.transpose(Ldjk),K)
            #print('Vdjk2', Vdjk2)
            '''

            Sigma = SigmaMIL

            mu = np.dot(Sigma, nu)

            d_mu = np.linalg.norm(mu - prev_mu)
            d_Sigma = np.linalg.norm(Sigma - prev_Sigma)

            done = d_mu > mu_tol or d_Sigma > Sigma_tol

            it = it + 1


        # TEMP because long-cut in calculating Sigma above
        sqrtStilde = np.diag(np.sqrt(tau))
        B = np.eye(N) + np.dot(sqrtStilde, np.dot(K, sqrtStilde))
        L = np.linalg.cholesky(B)
        ##################################################

        logZep_terms_1_and_4 = 0.5 * sum( np.log(1+np.multiply(tau, np.reciprocal(tau_minus_i_vec))) - np.log(np.diag(L)) )

        T = np.diag(tau_minus_i_vec)
        big_matrix = K - np.dot(K, np.dot(sqrtStilde, np.linalg.solve(B, np.dot(sqrtStilde, K)))) - np.linalg.inv(T+Stilde)
        logZep_terms_5b_and_2 = 0.5*np.dot(np.transpose(nu), np.dot(big_matrix, nu))

        stuff = np.linalg.solve( Stilde+T, np.dot(Stilde, mu_minus_i_vec - 2*nu) )
        logZep_term_5a = 0.5 * np.dot(np.transpose(mu_minus_i_vec), np.dot(T, stuff))

        num = np.multiply(y, mu_minus_i_vec)
        den = np.sqrt(1+tau_minus_i_vec)
        logZep_term_3 = np.sum( np.log( norm.cdf( np.multiply(num, np.reciprocal(den)))) )

        #print('logZep_terms_1_and_4', logZep_terms_1_and_4)
        #print('logZep_terms_5b_and_2', logZep_terms_5b_and_2)
        #print('logZep_term_5a', logZep_term_5a)
        #print('logZep_term_3', logZep_term_3)

        logZep = logZep_terms_1_and_4 + logZep_terms_5b_and_2 + logZep_term_5a + logZep_term_3

        if self.verbose:
            print(theta, '-->', -logZep)

        return -logZep, nu, tau


    def find_posterior_mode(self, theta, f_guess=None, tol_grad=1e-6, maxiter=100):
        # Mode finding for Laplace GPC.  Algorithm 3.1 from "Gaussian Process for Machine Learning", p46

        y = self.training_data[self.Ycol].values
        if self.verbose:
            print('y:', y)
        N = len(y)
        if f_guess is not None:
            f_hat = f_guess
            assert( isinstance(f_hat, np.ndarray) )
            assert(f_hat.shape[0] == y.shape[0])
        else:
            f_hat = np.zeros_like( y )

        X = self.training_data[self.Xcols_scaled].values

        K = self.kxx_gpu_wrapper(X, theta)  # This is for f, no sigma2_n

        for i in range(maxiter):
            if self.verbose: print('---[ %d ]------------------------------------'%i)
            if self.verbose: print('f_hat:', f_hat)

            pi = 1.0/(1.0+np.exp(-f_hat))
            if self.verbose: print('pi:', pi)

            t = (y+1)/2.0
            if self.verbose: print('t:', t)
            if self.verbose: print('Computing W ...')

            d2_df2_log_p_y_given_f = -np.multiply(pi, 1-pi)

            W = np.diag( -d2_df2_log_p_y_given_f ) # NOTE: Using logit (3.15)
            sqrtW = np.diag( np.sqrt(-d2_df2_log_p_y_given_f) )

            if self.verbose: print('Computing B ...')
            #B = np.eye(N) + np.dot(sqrtW, np.dot(K, sqrtW))
            ### Dan's method for B:
            w = np.sqrt( -d2_df2_log_p_y_given_f )
            w_outer = np.outer(w,w)
            B = np.eye(N) + np.multiply(K, w_outer)
            ###

            if self.verbose: print('Computing L ...')
            L = np.linalg.cholesky(B)

            if self.verbose: print('Computing b ...')
            d_df_log_p_y_given_f = t - pi
            b = np.dot(W, f_hat) + d_df_log_p_y_given_f
            if self.verbose: print('b:', b)

            if self.verbose: print('Computing W12_K_b ...')
            W12_K_b = np.dot(sqrtW, np.dot(K,b))

            if self.verbose: print('Computing L_slash_W12_K_b ...')
            L_slash_W12_K_b = np.linalg.solve(L, W12_K_b)

            if self.verbose: print('Computing Lt_slash_L_slash_W12_K_b ...')
            Lt_slash_L_slash_W12_K_b = np.linalg.solve(np.transpose(L), L_slash_W12_K_b)

            if self.verbose: print('Computing a ...')
            a = b - np.dot(sqrtW, Lt_slash_L_slash_W12_K_b)

            if self.verbose: print('a:', a)
            if self.verbose: print('Computing f_hat ...')
            f_hat = np.dot(K, a)

            #####log_p_y_given_f = -np.log(1 + np.exp(-np.dot(y, f_hat)))
            log_p_y_given_f = np.sum(-np.log(1 + np.exp(-np.multiply(y, f_hat))))
            log_q_y_given_X_theta = -0.5 * np.dot(np.transpose(a), f_hat) + log_p_y_given_f - sum( np.log(np.diag(L)) )

            d_df_log_q_y_given_X_theta = d_df_log_p_y_given_f - np.linalg.solve(K, f_hat)
            # print '***', log_q_y_given_X_theta, np.linalg.norm(d_df_log_q_y_given_X_theta)
            norm_grad = np.linalg.norm(d_df_log_q_y_given_X_theta)
            if norm_grad < tol_grad:
                break

            if i == maxiter - 1:
                print('WARNING: out of iterations in find_posterior_mode, |grad| =', norm_grad)

        if self.verbose: print(theta, '--> log_q_y_given_X_theta: %f (%d f_hat-iterations)' % (log_q_y_given_X_theta, i))

        return {
            'f_hat': f_hat,
            'log_q_y_given_X_theta': log_q_y_given_X_theta,
            'L': L,
            'K': K,
            'W': W,
            'sqrtW': sqrtW,
            'pi': pi,
            'a': a,
            'd_df_log_p_y_given_f': d_df_log_p_y_given_f,
            'log_p_y_given_f': log_p_y_given_f
        }


    def negative_log_marginal_likelihood(self, theta):
        if self.use_laplace_approximation:
            log_q_y_given_X_theta = self.find_posterior_mode(theta)['log_q_y_given_X_theta']
            return -log_q_y_given_X_theta
        else:
            logZep, _, _  = self.expectation_propagation(theta)
            return logZep

    def negative_log_marginal_likelihood_and_gradient(self, theta, f_guess=None):
        # Rasmussen and Williams GPML p126 algo 5.1

        #if np.any(theta < 0):
        #    theta = np.abs(theta)
        #theta += 1e-6 # Keep away from 0

        mode_results_dict = self.find_posterior_mode(theta, f_guess)

        f_hat = mode_results_dict['f_hat']
        logZ = mode_results_dict['log_q_y_given_X_theta']
        L = mode_results_dict['L']
        K = mode_results_dict['K']
        W = mode_results_dict['W']
        sqrtW = mode_results_dict['sqrtW']
        pi = mode_results_dict['pi']
        a = mode_results_dict['a']
        d_df_log_p_y_given_f = mode_results_dict['d_df_log_p_y_given_f']
        log_p_y_given_f = mode_results_dict['log_p_y_given_f']

        X = self.training_data[self.Xcols_scaled].values

        L_slash_sqrtW = np.linalg.solve(L, sqrtW)
        R = np.dot(sqrtW, np.linalg.solve(np.transpose(L), L_slash_sqrtW)) # <-- good

        C = np.linalg.solve(L, np.dot(sqrtW, K))

        #print(np.diag(K - np.dot(np.transpose(C),C)) - np.diag(K - np.dot(K,np.linalg.solve(np.linalg.inv(W)+K,K))))
        #exit()

        N = f_hat.shape[0]
        # L is good

        s2_part1 = np.diag( np.diag(K) - np.diag(np.dot(np.transpose(C),C)) )
        d3_df3_log_p_y_given_f = pi * (1-pi) * (2*pi-1)
        s2 = -0.5 * np.dot(s2_part1, d3_df3_log_p_y_given_f)
        s2 = -s2 # TEMP - OMG, RW is WRONG!!!!!

        d_dtheta_logZ = np.zeros_like(theta)

        for j in range(len(theta)):
            # Compute dK/dtheta_j
            if j == 0:
                # theta_0 is sigma2_f
                C = K / theta[0]
            else:
                C = self.kxx_gpu_wrapper(X, theta, deriv=j+1) # +1 because no sigma2_n

            s1_part_1 = 0.5 * np.dot(np.transpose(a), np.dot(C, a))
            s1 = s1_part_1 - 0.5 * np.trace( np.dot(R,C) ) # <-- WARNING, INEFFICIENT!  COMPUTE DIAG OF MATRIX PROD ONLY!
            b = np.dot(C, d_df_log_p_y_given_f)
            s3 = b - np.dot(K, np.dot(R,b))
            d_dtheta_logZ[j] = s1 + np.dot(np.transpose(s2), s3) #s1 seems good, s2 is good


        if self.verbose: print('d_dtheta_logZ:', d_dtheta_logZ)

        return -logZ, -d_dtheta_logZ, f_hat # Careful with sign


    @staticmethod
    def func_wrapper(f, cache_size=100):
        # http://stackoverflow.com/questions/10712789/scipy-optimize-fmin-bfgs-single-function-computes-both-f-and-fprime
        evals = {}
        last_points = deque()

        def get(pt, which):
            s = pt.tostring() # get binary string of numpy array, to make it hashable
            if s not in evals:
                evals[s] = f(pt)
                last_points.append(s)
                if len(last_points) >= cache_size:
                    del evals[last_points.popleft()]
            return evals[s][which]

        return partial(get, which=0), partial(get, which=1)


    def laplace_predict(self, theta, f_hat, P):
        y = self.training_data[self.Ycol].values
        if self.verbose: print('y:', y)
        N = len(y)
        X = self.training_data[self.Xcols_scaled].values
        KXX = self.kxx_gpu_wrapper(X, theta)  # This is for f

        if self.verbose: print('---[ PREDICT ]------------------------------------')
        if self.verbose: print('f_hat:', f_hat)
        pi = 1.0/(1.0+np.exp(-f_hat))
        if self.verbose: print('pi:', pi)
        t = (y+1)/2.0
        if self.verbose: print('t:', t)

        d2_df2_log_p_y_given_f = -np.multiply(pi, 1-pi)
        sqrtW = np.diag( np.sqrt(-d2_df2_log_p_y_given_f) )

        if self.verbose: print('Computing B ...')
        ### Dan's method for B:
        w = np.sqrt( -d2_df2_log_p_y_given_f )
        w_outer = np.outer(w,w)
        B = np.eye(N) + np.multiply(KXX, w_outer)
        ###
        #Bslow = np.eye(N) + np.dot(sqrtW, np.dot(KXX, sqrtW))
        #print( np.allclose(B, Bslow) )
        #exit()
        ###

        L = np.linalg.cholesky(B)
        d_df_log_p_y_given_f = t-pi

        # p-specific code begins here:
        ret = pd.DataFrame(columns = ['Mean-Transformed', 'Var-Transformed', 'Mean', 'Var']) #, 'Trapz' 
        for idx, p_series in P.iterrows():
            if self.verbose: print(idx, 'x_star is', p_series['x (scaled)'])
            p = p_series.as_matrix()[np.newaxis,:]
            KXp = self.kxp_gpu_wrapper(X, p, theta)
            f_bar_star = np.dot(np.transpose(KXp), d_df_log_p_y_given_f) # MEAN (vector of length 1)

            v = np.linalg.solve(L, np.dot(sqrtW, KXp))
            Kpp = self.kxx_gpu_wrapper(p, theta) # For latent distribution, don't add sigma2_n
            V = Kpp - np.dot(np.transpose(v),v) # VARIANCE (matrix of size 1x1)

            def logistic(f):
                return 1.0 / (1 + np.exp(-f))

            mu = f_bar_star[0]
            sigma2 = V[0,0]

            import time
            tz = time.time()
            # Numerical integration ##### (Works, but need variance calculation)
            fstar = np.linspace(mu - 3*np.sqrt(sigma2), mu + 3*np.sqrt(sigma2), 100) # <-- should choose num points (100) wisely
            mean_integrand = np.multiply(logistic(fstar), np.exp(-(fstar-mu)**2/(2.0*sigma2)) / np.sqrt(2.0*np.pi*sigma2) )
            mean_trapz = np.trapz(mean_integrand, x=fstar) # Average prediction (better)
            var_integrand = np.multiply( (logistic(fstar) - mean_trapz)**2, np.exp(-(fstar-mu)**2/(2.0*sigma2)) / np.sqrt(2.0*np.pi*sigma2) )
            var_trapz = np.trapz(var_integrand, x=fstar) # Average prediction (better)
            #print('TRAPZ', time.time()-tz)

            '''
            ### Monte Carlo
            mc = time.time()
            pts = np.random.multivariate_normal(f_bar_star, V, size=10000)
            yy = logistic(pts)
            mean = np.mean(yy)
            var = np.var(yy)
            print('MC', time.time()-mc)
            ###
            '''

            #print('MC: (%f, %f)  VS  TRAPZ: (%f, %f)' % (mean, var, mean_trapz, var_trapz))

            logi = logistic(mu) # MAP prediction

            if self.verbose: print('MEAN:', mu)
            if self.verbose: print('VAR:', sigma2)
            if self.verbose: print('LOGIS:', logi)
            #if self.verbose: print('MONTE CARLO:', 'mean=%f, var=%f'%(mean, var))
            if self.verbose: print('TRAPZ:', 'mean=%f, var=%f'%(mean_trapz, var_trapz))

            ret = pd.concat([ret, pd.DataFrame({'Mean-Transformed':[mu], 'Var-Transformed':[sigma2], 'Mean': [mean_trapz], 'Var': [var_trapz]})])
        ret.index = P.index.copy()

        return ret


    def ep_predict(self, theta, P):
        logZep, nu, tau = self.expectation_propagation(theta)

        y = self.training_data[self.Ycol].values
        if self.verbose: print('y:', y)
        N = len(y)
        X = self.training_data[self.Xcols_scaled].values
        KXX = self.kxx_gpu_wrapper(X, theta)  # This is for f

        if self.verbose: print('---[ PREDICT ]------------------------------------')

        if self.verbose: print('Computing B ...')
        sqrtStilde = np.diag(np.sqrt(tau))
        B = np.eye(N) + np.dot(sqrtStilde, np.dot(KXX, sqrtStilde))
        L = np.linalg.cholesky(B)

        sqrtStilde_K_nu = np.dot(sqrtStilde, np.dot(KXX, nu))
        L_slash_sqrtStilde_K_nu = np.linalg.solve(L, sqrtStilde_K_nu)
        Lt_slash_L_slash_sqrtStilde_K_nu = np.linalg.solve(np.transpose(L), L_slash_sqrtStilde_K_nu)
        z = np.dot(sqrtStilde, Lt_slash_L_slash_sqrtStilde_K_nu)

        ret = pd.DataFrame(columns = ['Mean-Transformed', 'Var-Transformed', 'Mean', 'Var'])
        for idx, p_series in P.iterrows():
            if self.verbose: print(idx, 'x_star is', p_series['x (scaled)'])
            p = p_series.as_matrix()[np.newaxis,:]
            KXp = self.kxp_gpu_wrapper(X, p, theta)
            f_bar_star = np.dot(np.transpose(KXp), nu-z) # MEAN (vector of length 1)

            v = np.linalg.solve(L, np.dot(sqrtStilde, KXp))
            Kpp = self.kxx_gpu_wrapper(p, theta) # For latent distribution, don't add sigma2_n
            V = Kpp - np.dot(np.transpose(v),v) # VARIANCE (matrix of size 1x1)

            mu = f_bar_star[0]
            sigma2 = V[0,0]

            mean = norm.cdf( mu / np.sqrt(1+sigma2) )

            fstar = np.linspace(mu - 3*np.sqrt(sigma2), mu + 3*np.sqrt(sigma2), 100) # <-- should choose num points (100) wisely
            '''
            mean_integrand = np.multiply(norm.cdf(fstar), np.exp(-(fstar-mu)**2/(2.0*sigma2)) / np.sqrt(2.0*np.pi*sigma2) )
            mean_trapz = np.trapz(mean_integrand, x=fstar) # Average prediction (better)
            print(mean, mean_trapz)
            '''

            var_integrand = np.multiply( (norm.cdf(fstar) - mean)**2, np.exp(-(fstar-mu)**2/(2.0*sigma2)) / np.sqrt(2.0*np.pi*sigma2) )
            var_trapz = np.trapz(var_integrand, x=fstar) # TODO: Closed form!

            ret = pd.concat([ret, pd.DataFrame({'Mean-Transformed':[mu], 'Var-Transformed':[sigma2], 'Mean': [mean], 'Var': [var_trapz]})])
        ret.index = P.index.copy()

        return ret


    def optimize_hyperparameters(self, x0, bounds=(), K=-1, eps=1e-2, disp=True, maxiter=15000):
        # x0 like np.array([2, 0.10, 0.14641288665436947, 0.12166006573919039, 0.05, 0.05, 0.08055223671416605, 7.026854485434267 ])
        # bounds like ((0.005,10),)+((0.01,10),) + tuple((5e-5,10) for i in range(self.D))
        # K=None is leave one out cross validation, otherwise make K groups
        idx = self.training_data.index.names    # Save index
        self.training_data.reset_index(inplace=True)

        if self.use_laplace_approximation:
            f_, fprime = GPC.func_wrapper(self.negative_log_marginal_likelihood_and_gradient)
        else:
            f_ = self.negative_log_marginal_likelihood
            fprime = None

        '''
        # Truncated Newton Conjugate-Gradient
        ret = spo.minimize(
            fun = f_,
            x0 = x0,
            #args=(X,Y,P),
            jac = fprime,
            method='TNC',
            bounds = bounds, # Constrain values
            hess=None, hessp=None,
            constraints=(), tol=None, callback=None,
            options= {
                'maxiter': maxiter,
                'disp': disp,
                'eps': eps
            }
        )
        '''

        '''
        # BFGS
        ret = spo.minimize(
            fun = f_,
            #args=(X,Y,P),
            x0 = x0,
            method='L-BFGS-B',
            bounds = bounds, # Constrain values
            jac = fprime,
            hess=None, hessp=None,
            constraints=(), tol=None, callback=None,
            options= {
                'maxiter': maxiter,
                'disp': disp,
                'ftol': 1e-12,
                'gtol': 1e-12,
                'factr': 0.01,
                'eps': eps
            }
        )
        '''

        attempts = 0
        done = False
        while attempts < 3 and not done:
            # No jacobian
            ret = spo.minimize(
                fun = f_, #self.negative_log_marginal_likelihood,
                #args=(X,Y,P),
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

            print('OPTIMIZATION RETURNED:\n', ret)
            done = ret['success'] == True
            if not done:
                print('OPTIMIZATION FAILED, trying AGAIN!!!')
                x0 = 1.1*x0
            attempts = attempts + 1


        self.theta = ret.x # Length scales now on 0-1 range
        #self.theta = np.abs(ret.x) + 1e-6 # Length scales now on 0-1 range

        f_hat = self.find_posterior_mode(self.theta)['f_hat']
        np.savetxt('f_hat.csv', f_hat, delimiter=',')   # X is an array

        print('OPTIMIZATION RETURNED:\n', ret)

        # Restore original index
        if idx[0] is not None:
            self.training_data.set_index(idx, inplace=True)

        return ret


    def evaluate(self, data):
        # Predict at test and training points, store mean and variance in self.data

        # Normalize data
        for xc in self.Xcols:
            xc_new = xc+' (scaled)'
            data[xc+' (scaled)'] = (data[xc] - self.param_info.loc[xc,'Min'])/(self.param_info.loc[xc,'Max']-self.param_info.loc[xc,'Min'])

        # PREDICT:
        if self.use_laplace_approximation:
            if True:
                f_hat = self.find_posterior_mode(self.theta)['f_hat']
                np.savetxt('f_hat.csv', f_hat, delimiter=',')   # X is an array
            else:
                f_hat = np.genfromtxt('f_hat.csv', delimiter=',')

            ret = self.laplace_predict(self.theta, f_hat, data[self.Xcols_scaled])
        else:
            ret = self.ep_predict(self.theta, data[self.Xcols_scaled])

        #ret = data.merge(ret, left_index=True, right_index=True)

        return ret


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
                    print(row, col, row*self.D+col, fixed_inputs)

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
                    #print('WARNING: DEBUG!\n')
                    self.verbose=False

                    ret = self.evaluate( Xdf )

                    Y_mean = np.reshape(ret['Mean'], [res,res])
                    Y_std_latent = np.reshape( np.sqrt(ret['Var_Latent']), [res, res])
                    #Y_std_predictive = np.reshape( np.sqrt(ret['Var_Predictive']), [res, res])

                    try:
                        CS = ax.contour(X1, X2, Y_mean, zorder=100)
                        ax.clabel(CS, inline=1, fontsize=10, zorder=100)
                    except:
                        print('Unable to plot mean contour')
                        pass

                    ax.scatter(self.training_data[self.Xcols[row]], self.training_data[self.Xcols[col]], c=self.training_data[self.Ycol], s=25)

                    try:
                        CS = ax_std_latent.contour(X1, X2, Y_std_latent, zorder=100)
                        ax_std_latent.clabel(CS, inline=1, fontsize=10, zorder=100)
                    except:
                        print('Unable to plot std contour')
                        pass

                    if col == self.D-1:
                        ax.set_xlabel( self.Xcols[row] )
                    if row == 0:
                        ax.set_ylabel( self.Xcols[col] )
        #plt.tight_layout()
        return (fig, fig_std_latent)

    def plot_errors(self, train, test, mean_col, var_predictive_col, truth_col=None, figsize=(16,10)):
        if train is not None:
            train['Z'] = (train[self.Ycol] + 1)/2
        if test is not None:
            test['Z'] = (test[self.Ycol] + 1)/2
        Ycol = 'Z'

        if truth_col:
            if train is not None:
                train['ZTrue_Predictive'] = (train[truth_col] - train[mean_col]) / np.sqrt(train[var_predictive_col])
                train['Truth-Logit'] = np.log(train[truth_col]/(1-train[truth_col]))
                train['ZTrue_Predictive_Logit'] = (train['Truth-Logit'] - train['Mean-Transformed']) / np.sqrt(train['Var-Transformed'])

            if test is not None:
                test['ZTrue_Predictive'] = (test[truth_col] - test[mean_col]) / np.sqrt(test[var_predictive_col])
                test['Truth-Logit'] = np.log(test[truth_col]/(1-test[truth_col]))
                test['ZTrue_Predictive_Logit'] = (test['Truth-Logit'] - test['Mean-Transformed']) / np.sqrt(test['Var-Transformed'])


        fig, ((ax1, ax2)) = plt.subplots(nrows=2, ncols=1, sharex='col', figsize=figsize) # , sharex='col', sharey='row')

        ax = ax1
        if train is not None:
            train['Z_Predictive'] = (train[Ycol] - train[mean_col]) / np.sqrt(train[var_predictive_col])
            ax.scatter(x=train['Sample'], y=train[Ycol], c='c', marker='_', s=25, alpha=1, linewidths=1, zorder=50)
            ax.errorbar(x=train['Sample'], y=train[mean_col], yerr=2*np.sqrt(train[var_predictive_col]), fmt='.', ms=5, linewidth=1, c='k')
            if truth_col and truth_col in train:
                ax.plot(train['Sample'], train[truth_col], 'c.')

        if test is not None:
            test['Z_Predictive'] = (test[Ycol] - test[mean_col]) / np.sqrt(test[var_predictive_col])
            ax.scatter(x=test['Sample'], y=test[Ycol], c='m', marker='_', s=25, alpha=1, linewidths=1, zorder=50)
            ax.errorbar(x=test['Sample'], y=test[mean_col], yerr=2*np.sqrt(test[var_predictive_col]), fmt='.', ms=5, linewidth=1, c='k')
            if truth_col and truth_col in test:
                ax.plot(test['Sample'], test[truth_col], 'm.')

        ax.margins(x=0,y=0.05)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(Ycol)


        a=0.05
        ax = ax2
        if train is not None:
            ax.scatter(x=train['Sample'], y=train['Z_Predictive'], c='c', marker='_', alpha=0.5, linewidth=1)
            if truth_col and truth_col in train:
                ax.scatter(x=train['Sample'], y=train['ZTrue_Predictive'], c='c', marker='.', alpha=0.5, linewidth=1)

        if test is not None:
            ax.scatter(x=test['Sample'], y=test['Z_Predictive'], c='m', marker='_', alpha=0.5, linewidth=1)
            if truth_col and truth_col in test:
                ax.scatter(x=test['Sample'], y=test['ZTrue_Predictive'], c='m', marker='.', alpha=0.5, linewidth=1)

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

        plt.tight_layout()

        return fig
