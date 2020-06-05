from string import Template
import json
import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as spo
import seaborn as sns

from history_matching.basis import Basis
from hm2.error import *

try:
    from pycuda import compiler, gpuarray
    import pycuda.autoinit
    import skcuda.misc as misc
    import skcuda.linalg as linalg
except ImportError as e:
    print("Looks like you don't have CUDA, that's okay, we'll try using CPU but it will be SLOW!")
except RuntimeError as e:
    print("Runtime error starting cuda, message was:\n", e.message)
except Exception as e:
    print("Unknown CUDA error. Falling back to CPU",e)

# NOTE theta = [sigma_f^2, sigma_n^2, l_1^2, l_2^2, ..., l_D^2]
# Ack https://github.com/lebedov/scikit-cuda/blob/master/demos/indexing_2d_demo.py

class DanGPR():
    """Daniel Klein's Gaussian Process Regression implementation

    This class implementes Gaussian Process Regression with leave-one-out
    cross-validation for parameter fitting and NVidia-CUDA-based GPU
    acceleration for speed.
    """

    def __init__(
            self, 
            Ycol, 
            training_data, 
            param_info,
            theta = None,
            normalize_y = True,
            sigma2_n = None,
            **kwargs
    ):
        """Initialize the GPR class.

        Args:
            Ycol:  (str)
                The name of the column in training_data that contains the model output values.  Ycol must be a column in training_data
            training_data:  (Pandas dataframe)
                Columns must include:
                * Sample_Id: A unique string that identifies each sample.
                * Sim_Id: A unique string the identifies each simulation, typically the COMPS simulation ID.
                * Sample: (optional?) The sample index.
                * Exp_Id: (optional?) The name of the experiment.
                * PARAMETER NAMES: One column for each parameter name.
            param_info:  (Pandas dataframe)
                Columns include:
                * Name: The name of the parameter, must match column name in training_data.
                * Min: Minimum value of parameter.
                * Max: Maximum value of parameter.
                * MapTo: (optional) For use in commissioning script to assist in mapping the parameter to model input.
                * Source: (optional) Source from which parameter ranges came from
            theta: (1D ndarray, optional)
                Optionally specify the hyperparameters.  This should be a numpy array of length 2+D, where D is the number of parameters:
                1) sigma_f^2
                2) sigma_n^2
                3) Squared lengthscale of first dimension
                4) Squared lengthscale of second dimension
                ...
                D+2) Squared lengthscale of dimension D
            normalize_y: (boolean, optional with default True)
                If the responses should be normalized
            sigma_n: (None or instance of GPR)
                For typical homoscedastic GPR, leave as None.  The kernel hyperparamter, sigma2_n, will be optimized.  Alternatively for heteroscedastic GPR, provide an instance of a GPR with the same input dimensions for which the optut is the log of the variance.
            normalizer_mean (float, optional):  Allows specification or recovery of the mean of the Y-normalizer.  Must specify normalizer_mean and normalizer_std for this feature to work.  It is typically used when restoring a GPR from file.
            normalizer_std (float, optional): Allows specification or recovery of the std of the Y-normalizer.  Must specify normalizer_mean and normalizer_std for this feature to work.  It is typically used when restoring a GPR from file.
        """
        try:
            device = pycuda.autoinit.device
            print('Autoinit GPU device name:', device.name())
            self.use_gpu = True
        except Exception as _:
            self.use_gpu = False

        if self.use_gpu:
            # Read in the RBF kernel
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            self.kernel_fn = os.path.join(cur_dir, 'dan_gpr_kernel.c')

        self.training_data = training_data.copy()
        self.param_info = param_info.copy()
        self.D = self.basis.D
        self.Ycol = Ycol
        self.fig_type = fig_type

        self.kernel_mode = kernel_mode

        if 'normalizer_mean' in kwargs and 'normalizer_std' in kwargs:
            self.normalizer_mean = kwargs['normalizer_mean']
            self.normalizer_std = kwargs['normalizer_std']
        else:
            Y = np.ma.masked_invalid(self.training_data[self.Ycol], copy=False)
            self.normalizer_mean = Y.mean(0).view(np.ndarray)
            self.normalizer_std  = Y.std(0).view(np.ndarray)

        # Normalize training data, change Ycol
        self.normalize_y = normalize_y
        self.training_data[self.Ycol+'_normalized'] = self.normalize(self.training_data[self.Ycol])
        self.Ycol_orig = self.Ycol
        self.Ycol = self.Ycol+'_normalized'

        self.normalizer = True #UserStandardize(mean=self.normalizer_mean, std=self.normalizer_std)
        self.poisson = False #is_poisson

        self.verbose = verbose

        # Heteroscedastic GP setup
        self.fixed_sigma_n = True
        if isinstance(sigma2_n, GPR):
            print('User has configured GPR with noise coming from another GPR')
            self.sigma2_n = sigma2_n
            self.fixed_sigma_n = False

        if self.use_gpu:
            self.define_kernel()

        if theta is None:
            self.reset_theta()
        else:
            self.set_theta(theta)

    def set_training_data(self, new_training_data):
        """Set the training data for GPR, will normalize if needed

        Args:
            new_training_data: (Pandas DataFrame)
                As in __init__.
        """
        self.training_data = new_training_data.copy()
        if self.use_gpu:
            self.define_kernel()

        # Normalize training data as in __init__
        self.training_data[self.Ycol] = self.normalize(self.training_data[self.Ycol_orig])

        self.update_cache()

    def reset_theta(self):
        """Resets hyperparameters (theta).
        """
        # Set the kernel/model hyperparameters
        #TODO(dklein): This does more than just reset theta - why?
        self.theta = None
        self.Kxx_inv_Y = None
        self.Kxx_inv = None
        self.X = None
        self.Y = None

    def set_theta(self, theta):
        """Sets hyperparameters (theta).

        Args:
            theta: (1D numpy array)
                As in __init__.
        """
        if len(theta)!=2+self.D:
            raise HistoryMatchingError("Length of theta must be 2 greater than the dimension!")

        self.theta = theta
        self.update_cache()

    def update_cache(self):
        """Update the internal cache of X, Y, Kxx_inv, and Kxx_inv_Y.

        When evaluating many points, these somewhat-slow to calculate properties
        do not change, so we compute and cache them here.

        """
        train_mean = self.training_data.reset_index().groupby('Sample_Id').mean()
        self.X = self.basis.generate_dmatrix( train_mean, scaleX = True).values
        self.Y = train_mean[self.Ycol].values # Is there a way/need to use all results?

        if self.use_gpu:
            try:
                Kxx = self.kxx_gpu_wrapper(self.X, self.theta, add_sigma2_n = True)  # Y is noisy
            except pycuda._driver.MemoryError:
                print('Insufficient video memory for Kxx matrix of dimension', X.shape[0],', reverting to (slow) CPU computation.')

            Kxx_gpu = gpuarray.to_gpu(np.asarray(Kxx.copy(), np.float64))
            linalg.init()
            self.Kxx_inv = linalg.inv(Kxx_gpu, overwrite=True, lib='cusolver').get()
        else:
            Kxx = self.kernel_xx(self.X, self.theta, add_sigma2_n = True)
            self.Kxx_inv = np.linalg.inv(Kxx)

        self.Kxx_inv_Y = np.dot(self.Kxx_inv, self.Y) # TODO: GPU

    def normalize(self, data):
        """If normalize_y is True, normalize some data by subtracting the mean and dividing by the standard deviation.

        Args:
            data: (Pandas DataFrame) Data to normalize.
        """
        if self.normalize_y:
            return (data - self.normalizer_mean)/self.normalizer_std
        return data

    def inverse_normalize_mean(self, data):
        """Reverse the normalization calculation for the mean.

        Args:
            data: (Pandas DataFrame) Data to unnormalize.
        """
        if self.normalize_y:
            return data*self.normalizer_std + self.normalizer_mean
        return data

    def inverse_normalize_var(self, data):
        """Reverse the normalization calculation for the variance.

        Args:
            data: (Pandas DataFrame) Data to unnormalize.
        """
        if self.normalize_y:
            return data * (self.normalizer_std**2)
        return data

    def define_kernel(self):
        """Prepare the Kernel.  For now, only the `RBF` kernel_mode is supprted.
        """

        if self.kernel_mode == 'RBF':
            Nx = self.training_data.shape[0]

            with open(self.kernel_fn, 'r') as f:
                kernel_code_template = Template(f.read())

            max_threads_per_block, max_block_dim, max_grid_dim = misc.get_dev_attrs(pycuda.autoinit.device)
            device = pycuda.autoinit.device
            block_dim, grid_dim = misc.select_block_grid_sizes(device, (Nx, Nx))
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
            raise HistoryMatchingError(f'Bad kernel mode, kernel_mode = {self.kernel_mode}')


    def kernel_xx(self, X, theta, add_sigma2_n = True, deriv=-1):
        """Compute the Kxx kernel using (SLOW) CPU-based calculations.

        This function really only remains for computers that do not have access to an NVidia GPU and for testing GPU calculations.

        Args:
            X: (2D ndarray) points of dimension N x D
            theta: (1D ndarray) hyperparameters
            add_sigma2_n: (boolean) if True, add observation variance, sigma2_n, to the diagonal.
        """

        Nx = X.shape[0]

        # Do not add sigma2_n to sigma2_f deriv
        if deriv >= 0 and add_sigma2_n!=False:
            raise HistoryMatchingError("If deriv>=0, then add_sigma2_n must be False!")

        if deriv == 1: # Assuming add_sigma2_n is True when taking deriv wrt sigma2_n, otherwise it would be zeros(Nx) ...
            if self.fixed_sigma_n:
                return np.eye(Nx)
            else:
                # No deriv wrt sigma2_n if the user has specified sigma2_n via GPR
                return np.zeros((Nx,Nx))

        sigma2_f = theta[0]
        if deriv == 0:
            sigma2_f = 1

        Kxx = np.zeros([Nx,Nx], dtype=np.float32)
        for i in range(Nx):
            # Diagonal:
            if deriv <= 1:
                Kxx[i,i] = sigma2_f # theta[0] or 1, see above
            else:
                Kxx[i,i] = 0

            # Off-diagonal
            for j in range(i+1,Nx):
                dX = X[i,:]-X[j,:]
                r2 = 0
                for d in range(self.D):
                    r2 += dX[d] * dX[d]/theta[2+d]
                Kxx[i,j] = sigma2_f * np.exp( -r2 / 2. )

                if deriv > 1: # Lengthscale derivatives
                    d = deriv-2
                    Kxx[i,j] *= 0.5 * (dX[d] * dX[d]) / (theta[2+d] * theta[2+d])

                Kxx[j,i] = Kxx[i,j]

        if add_sigma2_n:
            if self.fixed_sigma_n:
                sigma2_n = theta[1]
            else:
                Xcols = self.basis.param_info.index.values

                Xdf = pd.DataFrame(data=np.array(X), index=range(X.shape[0]), columns=Xcols) # ['Beta'], basis.param_info.index.values.tolist()
                # TODO: Cache
                sigma2_n = np.exp( self.sigma2_n.evaluate(Xdf)['Mean']) # TODO: internalize untransform_var # TODO: Just mean, or mean plus K sigma?
                if self.normalize_y:
                    sigma2_n /= self.normalizer_std**2

            # Add sigma_n^2 to the diagonal, observation noise
            Kxx[np.diag_indices(Nx)] += sigma2_n

        return Kxx


    def kernel_xp(self, X, P, theta):
        """Compute the Kxp kernel using (SLOW) CPU-based calculations.

        This function really only remains for computers that do not have access to an NVidia GPU and for testing GPU calculations.

        Args:
            X: (2D ndarray) points of dimension N x D
            P: (2D ndarray) points of dimension P x D
            theta: (1D ndarray) hyperparameters
        """

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


    def kxx_gpu_wrapper(self, X, theta, add_sigma2_n = True, deriv=-1):
        """Compute the Kxx kernel or derivatives using (FAST) GPU-based calculations.

        Args:
            X: (2D ndarray) points of dimension N x D.
            theta: (1D ndarray, optional with default True) hyperparameters.
            add_sigma2_n: (boolean) if True, add observation variance, sigma2_n, to the diagonal.
            deriv: (int) if negative, return the kernel.  If positive between 0 and D-1, compute the partial derivative of the Kxx kernel with respect to the deriv^th hyperparameter.
        """

        Nx = X.shape[0]

        # Do not add sigma2_n to sigma2_f deriv
        if deriv == 0 and add_sigma2_n!=False:
            raise HistoryMatchingError("If deriv>=0, then add_sigma2_n must be False!")

        if deriv == 1: # Assuming add_sigma2_n is True when taking deriv wrt sigma2_n, otherwise it would be zeros(Nx) ...
            if self.fixed_sigma_n:
                return np.eye(Nx)
            # No deriv wrt sigma2_n if the user has specified sigma2_n via GPR
            return np.zeros((Nx,Nx))

        # TODO: Can use from before...?
        block_dim, grid_dim = misc.select_block_grid_sizes(pycuda.autoinit.device, (Nx, Nx))
        X_gpu = gpuarray.to_gpu(X.astype(np.float32))
        theta_gpu = gpuarray.to_gpu(theta.astype(np.float32))

        # Create empty gpu array for the result
        Kxx_gpu = gpuarray.empty((Nx, Nx), np.float32)

        # Call the kernel on the card
        self.kernel_xx_gpu(
            Kxx_gpu,            # <-- Output
            X_gpu, theta_gpu,   # <-- Inputs
            np.uint32(Nx),      # <-- N
            np.uint32(self.D),  # <-- D
            np.int32(deriv),      # <-- Negative for no derivative
            np.uint8(X.flags.f_contiguous), # FORTRAN (column) contiguous
            block = block_dim,
            grid = grid_dim
        )

        Kxx = Kxx_gpu.get()

        if add_sigma2_n:
            if self.fixed_sigma_n:
                sigma2_n = theta[1]
            else:
                Xcols = self.basis.param_info.index.values

                Xdf = pd.DataFrame(data=np.array(X), index=range(X.shape[0]), columns=Xcols) # ['Beta'], basis.param_info.index.values.tolist()
                # TODO: Cache
                sigma2_n = np.exp( self.sigma2_n.evaluate(Xdf)['Mean']) # TODO: internalize untransform_var # TODO: Just mean, or mean plus K sigma?
                if self.normalize_y:
                    sigma2_n /= self.normalizer_std**2

            # Add sigma_n^2 to the diagonal, observation noise
            Kxx[np.diag_indices(Nx)] += sigma2_n

        return Kxx


    def kxp_gpu_wrapper(self, X, P, theta):
        """Compute the Kxp kernel or derivatives using (FAST) GPU-based calculations.

        Args:
            X: (2D ndarray) points of dimension N x D.
            P: (2D ndarray) points of dimension P x D.
            theta: (1D ndarray, optional with default True) hyperparameters.
        """

        if self.use_gpu:
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

            return Kxp_gpu.get()

        Kxp_cpu = self.kernel_xp(X, P, theta)
        return Kxp_cpu

    def cross_validation_with_grad(self, theta, X, Y, optimize_sigma2_n, log_transform):
        """Compute the LOO cross validation score and gradient with respect to the hyperparameters.

        Args:
            theta: (1D ndarray, optional with default True) hyperparameters.
            X: (2D ndarray) points of dimension N x D.
            Y: (1D ndarray) outputs of dimension N x 1.
            optimize_sigma2_n: (bool) Set False to keep sigma2_n at the initial guess value.
            log_transform: (bool) If True, a ln transformation has been applied to contrain parameters to positive values.
        """

        if log_transform:
            theta_log = theta # TODO(dklein): This variable is unused. Do we need it?
            theta = np.maximum(np.minimum(theta, np.log(sys.float_info.max)), np.log(sys.float_info.min))
            theta = np.exp(theta) # TEMP

        D = len(theta)

        if self.use_gpu:
            KXX = self.kxx_gpu_wrapper(X, theta, add_sigma2_n = True) # Want predictive distribution, so add sigma2
        else:
            KXX = self.kernel_xx(X, theta, add_sigma2_n = True)

        if self.use_gpu:
            KXX_gpu = gpuarray.to_gpu(np.asarray(KXX.copy(), np.float64))
            linalg.init()
            KXX_inv_gpu = linalg.inv(KXX_gpu, overwrite=True, lib='cusolver')
            KXX_inv = KXX_inv_gpu.get()
        else:
            KXX_inv = np.linalg.inv(KXX) # self.?

        KXX_inv_Y = np.dot(KXX_inv, Y).squeeze()

        sigma2 = np.reciprocal(np.diag(KXX_inv))
        err = np.multiply(KXX_inv_Y, sigma2)

        ll = np.sum(-0.5* np.log(sigma2) - np.divide(np.square(err), 2*sigma2))
        ll -= 0.5*np.log(2*np.pi) * Y.shape[0]

        dLLOO_dtheta = np.empty_like(theta)
        if self.use_gpu:
            linalg.init()

        for j in range(D):
            # TODO: Could compute some from KXX without calling kxx_gpu_wrapper
            if self.use_gpu:
                dK_dthetaj = self.kxx_gpu_wrapper(X, theta, add_sigma2_n = False, deriv = j) # Do not want sigma2_n here!
                # Get these as gpu arrays from the kernel function
                dK_dthetaj_gpu = gpuarray.to_gpu(np.asarray(dK_dthetaj, np.float64))
                Zj = linalg.dot(KXX_inv_gpu, dK_dthetaj_gpu).get()
            else:
                dK_dthetaj = self.kernel_xx(X, theta, add_sigma2_n = False, deriv = j) # Do not want sigma2_n here!
                Zj = np.dot(KXX_inv, dK_dthetaj) # This is the slow part

            # Fancy Einstein summations to compute only the diagonal elements!
            dLLOO_dthetaj = np.multiply(KXX_inv_Y, np.dot(Zj, KXX_inv_Y))
            dLLOO_dthetaj -= 0.5 * np.multiply( \
                    (1 + np.divide(np.square(KXX_inv_Y), np.diag(KXX_inv))), \
                    np.einsum('ij,ji->i', Zj, KXX_inv)
            )
            dLLOO_dtheta[j] = np.sum( np.multiply(dLLOO_dthetaj, sigma2) )

        if log_transform:
            dLLOO_dtheta = np.multiply(dLLOO_dtheta, theta)

        if not optimize_sigma2_n:
            dLLOO_dtheta[1] = 0

        print('\n\tLL:', -ll, '\n\tTheta:', theta, '\n\tDeriv:', -dLLOO_dtheta)

        return -ll, -dLLOO_dtheta


    def assign_rep(self, sample):
        """Helper function to assign a unique replicate index to each point.
        """

        if 'index' in sample.columns:
            sample = sample.drop('index', axis=1).reset_index()
        else:
            sample = sample.reset_index()
        sample.index.name='Replicate'
        sample.reset_index(inplace=True)
        return sample

    def gpr(
            self,
            force_optimize_gpr = True,
            method = 'CrossValidation',
            verbose = False,
            plot = True,
            plot_data = False,
            sigma2_f_guess = 2,
            sigma2_f_bounds = (0.005,10),
            sigma2_n_guess = 0.10,
            sigma2_n_bounds = (0.01,10),
            lengthscale_guess = 0.1, # Note, lengthscale is in a scaled range, training data to [0,1] for each parameter
            lengthscale_bounds = (0.01,1),
            normalize_y = True,
            optimize_sigma2_n = True,
            log_transform = False,
            optimizer_options=None,
            **kwargs
    ):
        """Perform Gaussian Process Regression modeling.

        Note that the GLM will be performed on the mean of the training data if multiple replicates are provided for each Sample_Id.

        By default, the GPR will be configured to use the `RBF` kernel.

        Args:
            force_optimize_gpr: (bool) Set True to force optimization of the GPR parameters even when results from a previous optimization exist.
            plot: (bool) Set True if you want to see diagnostic plots.
            plot_data: (bool) Set True to produce many pairwise plots of the inputs and results.  Within the GPR folder, they will appear in `PairwiseResults.`
            sigma2_f_guess: (float) The guess value for the signal variance. Note that when normalizing Y, a value of 1 correspons to the variance of the results.
            sigma2_f_bounds: (tuple) Lower and upper bounds for sigma2_f, e.g. like (0.005,10).
            sigma2_n_guess: (float) Initial guess value for observation noise variance.  Normalized like sigma2_f.
            sigma2_n_bounds: (tuple) Lower and upper bounds for sigma2_n, e.g. like (0.01,10).
            lengthscale_guess: (float or ndarray)
                If supplying a float, this value representes the kernel lengthscale guess and will be used for all lengthscales.  Note, lengthscale is in a scaled range, training data to [0,1] for each parameter.
                Alternatively, you can provide a ndarray with one entry for each parameter.
            lengthscale_bounds: (tuple) Range for lengthscale, e.g. (0.01,1).
            normalize_y: (bool) Set True to normalize the outputs (recommended).
            method: (str) Must be 'CrossValidation' for now.
            verbose: (bool) Set True to see lots of output.
            optimizer_options: (dict) Dictionary to be passed to the optimization algorithm within the GPR code.
            kwargs: (dict) Additional arguments to pass to the GPR class.
        """
        methods = ['CrossValidation'] # Supporing only CV for now
        if method not in methods:
            raise HistoryMatchingError(f"method must be one of {methods}")

        if optimizer_options is None:
            optimizer_options = {}

        gpr_model_fn = os.path.join(self.gprdir, 'gpr.pickle')

        if plot_data:
            pairdir = os.path.join(self.gprdir, 'PairwiseResults')
            if not os.path.exists( pairdir):
                os.mkdir( pairdir )

        if not force_optimize_gpr and os.path.isfile(gpr_model_fn):
            print("Loading GPR from", gpr_model_fn)
            self.gpr_model = pickle.load(open(gpr_model_fn,'rb'))
            if plot_data:
                figs = self.gpr_model.plot_data(samples_to_circle=pd.DataFrame(), saveto_dir = pairdir, log_scale=True) # TODO(dklein): This is unused. Is that bad?
        else:
            if self.use_glm:
                Ycol = 'Yerr'
            else:
                Ycol = 'Sim_Result'

            self.gpr_model = GPR(
                basis = basis,
                Ycol = Ycol,
                training_data = self.training_data,
                param_info = self.param_info,
                kernel_mode = 'RBF',
                kernel_params = None,
                normalize_y = normalize_y,
                verbose = verbose,
                debug = False, # Debug is really for testing the code
                **kwargs)

            if isinstance(lengthscale_guess, (int,float)):
                lengthscale_guess = basis.D*[lengthscale_guess]
            elif not isinstance(lengthscale_guess,list):
                raise HistoryMatchingError("lengthscale_guess must be a list!")
            elif len(lengthscale_guess)!=basis.D:
                raise HistoryMatchingError("lengthscale_guess must be the same length as the basis dimension!")

            if os.path.isfile(gpr_model_fn):
                timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                backup_fn = os.path.join(self.gprdir, f'model_{timestamp}.json')
                print('Backing up gpr model to', backup_fn)
                copyfile(gpr_model_fn, backup_fn)

            #TODO: Check guess within bounds
            x0 = np.array([sigma2_f_guess, sigma2_n_guess] +  lengthscale_guess)
            self.gpr_model.theta = x0
            pickle.dump(self.gpr_model, open(gpr_model_fn, 'wb'))

            if plot_data:
                figs = self.gpr_model.plot_data(samples_to_circle=pd.DataFrame(), saveto_dir = pairdir, log_scale=True)

            print("Fitting the GPR")
            self.gpr_model.optimize_hyperparameters(
                x0 = x0,
                bounds = (sigma2_f_bounds,)+(sigma2_n_bounds,) + basis.D*(lengthscale_bounds,),
                optimize_sigma2_n = optimize_sigma2_n,
                log_transform = log_transform,
                optimizer_options = optimizer_options
            )
            #TODO(dklein): Why do we save both here and above?
            pickle.dump(self.gpr_model, open(gpr_model_fn, 'wb'))


        # Taking the mean prior to evaluation because it is unnecessary to evaluate each point more than once as the GP output will always be the same
        train_mean = self.training_data.reset_index().groupby(['Sample_Id']).mean()
        test_mean = self.test_data.reset_index().groupby(['Sample_Id']).mean()

        print('GPR evaluating training data')
        ret = self.gpr_model.evaluate(train_mean)
        train_mean['Mean_Err'] = ret['Mean']
        train_mean['Mean_Estimate'] = train_mean['Mean_Err']
        if self.use_glm:
            train_mean['Mean_Estimate'] += train_mean['Yglm']
        train_mean['Var_Err_Predictive'] = ret['Var_Predictive']
        train_mean['Var_Err_Latent'] = ret['Var_Latent']

        merge_cols = ['Mean_Err', 'Mean_Estimate', 'Var_Err_Predictive', 'Var_Err_Latent']
        if 'Mean_Err' in self.training_data:
            self.training_data.drop(merge_cols, axis=1, inplace=True)
        self.training_data = self.training_data.reset_index().join(train_mean[merge_cols], on='Sample_Id')
        self.training_data.set_index(['Sample_Id', 'Sim_Id'], inplace=True)

        print('GPR evaluating test data')
        ret = self.gpr_model.evaluate(test_mean)
        test_mean['Mean_Err'] = ret['Mean']
        test_mean['Mean_Estimate'] = test_mean['Mean_Err']
        if self.use_glm:
            test_mean['Mean_Estimate'] += test_mean['Yglm']
        test_mean['Var_Err_Predictive'] = ret['Var_Predictive']
        test_mean['Var_Err_Latent'] = ret['Var_Latent']
        if 'Mean_Err' in self.test_data:
            self.test_data.drop(merge_cols, axis=1, inplace=True)
        self.test_data = self.test_data.reset_index().join(test_mean[['Mean_Err', 'Mean_Estimate', 'Var_Err_Predictive', 'Var_Err_Latent']], on='Sample_Id')
        self.test_data.set_index(['Sample_Id', 'Sim_Id'], inplace=True)

        # Add test data to gpr training
        gpr_model_with_test_fn = os.path.join(self.gprdir, 'model_with_test_data.json')
        self.gpr_model.set_training_data(pd.concat([self.training_data, self.test_data]))
        pickle.dump(self.gpr_model, open(gpr_model_with_test_fn, 'wb'))

        if plot:
            print('Plotting')
            fig = self.gpr_model.plot_errors(self.training_data.reset_index(), self.test_data.reset_index(), 'Mean_Err', 'Var_Err_Predictive')
            fig.savefig(os.path.join(self.gprdir, f'gpr.{self.fig_type}'))
            plt.close(fig)

            '''' # Useful debugging
            if False:
                mu = self.training_data[self.Xcols_GPR].mean()
                #mu = train.loc[146][Xcols_GPR].mean(); print(mu)
                (fig_mean, fig_std_latent) = self.gpr_model.plot(mu, res=25);
                fig_mean.savefig( os.path.join(self.gprdir, 'plot_mean'+'.'+self.fig_type) );    plt.close(fig_mean) # SLOW
                fig_std_latent.savefig( os.path.join(self.gprdir, 'plot_std_latent'+'.'+self.fig_type) );    plt.close(fig_std_latent) # SLOW
            '''

            fig = self.gpr_model.plot_histogram()
            fig.savefig( os.path.join(self.gprdir, 'histogram'+'.'+self.fig_type) )
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(16,10))
            ax.errorbar(
                x=self.training_data['Sim_Result'],
                y=self.training_data['Mean_Err'] + self.training_data['Yglm'],
                yerr=2*np.sqrt(self.training_data['Var_Err_Predictive']),
                fmt='o', c='c', lw=0.5)
            ax.errorbar(
                x=self.test_data['Sim_Result'],
                y=self.test_data['Mean_Err'] + self.test_data['Yglm'],
                yerr=2*np.sqrt(self.test_data['Var_Err_Predictive']),
                fmt='o', c='m', lw=0.5)
            ax.margins(x=0,y=0.05)
            xlim = ax.get_xlim()
            ax.plot( [xlim[0],xlim[1]], [xlim[0], xlim[1]], 'r-')
            ax.set_xlabel('Simulation Result')
            ax.set_ylabel('Predicted')
            fig.savefig( os.path.join(self.gprdir, 'emulation'+'.'+self.fig_type) )
            plt.close(fig)

        return self.gpr_model

    def fit(self, train_x, train_y, stdev_y, bounds=None, optimize_sigma2_n=True, log_transform=False, optimizer_options=None):
        """Optimize the hyperparameter vector, theta, with respect to the training data.

        Note that while each input may be simulated multiple times, here the mean of the outputs for each Saimple_Id are used.

        Args:
            x0: (1D ndarray) guess values like np.array([2, 0.10, 0.14641288665436947, 0.12166006573919039, 0.05, 0.05, 0.08055223671416605, 7.026854485434267 ]).
            bounds: (tuple) ((0.005,10),)+((0.01,10),) + tuple((5e-5,10) for i in range(self.D)).
            optimize_sigma2_n: (bool) Set False to keep sigma2_n at the initial guess value.
            log_transform: (bool) Set True to apply a log transformation and use unconstrained Conjugate Gradient to optimize the hyperparameters.  Bounds will be ignored.
            optimizer_options: (dict) Options to pass along to the optimization algorithm.  Through scipy-optimize, you can see these options e.g. for l-bfgs-b via spo.show_options(solver='minimize', method='l-bfgs-b')
        """

        if optimizer_options is None:
            optimizer_options = {}

        if bounds is None:
            bounds = (sigma2_f_bounds,)+(sigma2_n_bounds,) + basis.D*(lengthscale_bounds,),


        idx = self.training_data.index.names    # Save index
        self.training_data.reset_index(inplace=True)

        samples = self.training_data['Sample_Id'].unique()
        for i,s in enumerate(samples):
            self.training_data.loc[ self.training_data['Sample_Id']==s, 'Sample_Index'] = i

        # Computing the mean here:
        train_mean = self.training_data.reset_index().groupby('Sample_Id').mean()

        X = self.basis.generate_dmatrix( train_mean, scaleX = True).values
        Y = self.training_data.reset_index().groupby('Sample_Id').apply(self.assign_rep).pivot('Sample_Id', 'Replicate', self.Ycol).values

        # TODO(dklein): Do we need the following block here?
        '''
        # Maximize LOO cross-validation error
        # Old way from before using jacobian
        P = train_mean['Partition'].values

        ret = spo.minimize(
            self.cross_validation,
            args=(X,Y,P),
            x0 = x0,
            method='L-BFGS-B',
            bounds = bounds, # Constrain values
            jac=None, hess=None, hessp=None,
            constraints=(), tol=None, callback=None,
            options = optimizer_options
        )
        '''

        method = 'L-BFGS-B'
        if log_transform:
            x0 = np.log(x0)
            method = 'CG'
            bounds = None,

        ret = spo.minimize(
            self.cross_validation_with_grad,
            args=(X,Y, optimize_sigma2_n, log_transform),
            x0 = x0,
            method=method,
            bounds = bounds, # Constrain values
            jac=True, hess=None, hessp=None,
            constraints=(), tol=None, callback=None,
            options = optimizer_options
        )

        print('OPTIMIZATION RETURNED:\n', ret)

        # Restore original index
        self.training_data.set_index(idx, inplace=True)

        # Set hyperparameters (theta) to optimal values

        if log_transform:
            x = np.maximum(np.minimum(ret.x, np.log(sys.float_info.max)), np.log(sys.float_info.min))
            self.set_theta(np.exp(x))
        else:
            self.set_theta(ret.x)


    def evaluate(self, data):
        """Predict output values at input points specified by data.

        Note: as in optimize_hyperparameters, uses mean of training data over replicates.

        Args:
            data: (Pandas DataFrame) Points at which to evaluate the output.

        Returns dictionay containing:
            Mean: Predicted mean
            Var_Latent: Variance of the latent function.  Does not include observation noise.
            Var_Predictive: Variance of the predictive function.  Includes observation noise.
        """

        if self.X is None or self.Y is None or self.Kxx_inv is None and self.Kxx_inv_Y is None: # if no cache
            if self.verbose:
                print('No cache for Kxx_inv or Kxx_inv_Y') # Does this happen?
            self.update_cache()

        P = self.basis.generate_dmatrix( data, scaleX = True).values

        Kxp = self.kxp_gpu_wrapper(self.X, P, self.theta)

        if self.use_gpu:
            Kpp = self.kxx_gpu_wrapper(P, self.theta, add_sigma2_n = False) # For latent distribution
        else:
            Kpp = self.kernel_xx(P, self.theta, add_sigma2_n = False)

        f = np.dot(Kxp.T, self.Kxx_inv_Y)

        # NOTE: Just computing diagonal elements of:
        #covf = Kpp - np.dot(Kxp.T, np.dot(self.Kxx_inv, Kxp))
        if self.use_gpu:
            # : Reuse gpu arrays to reduce to_gpu transfers
            Kxx_inv_gpu = gpuarray.to_gpu(np.asarray(self.Kxx_inv, np.float64))
            Kxp_gpu = gpuarray.to_gpu(np.asarray(Kxp, np.float64))
            tmp = linalg.dot(Kxx_inv_gpu, Kxp_gpu).get() # Need .copy() on gpu arrays?
        else:
            tmp = np.dot(self.Kxx_inv, Kxp)

        covf = np.diag(Kpp) - np.einsum('ji,ji->i', Kxp, tmp)

        # Add in observation noise
        if self.fixed_sigma_n:
            sigma2_n = self.theta[1]*np.ones(P.shape[0])
        else:
            # Evaluate the sigma2_n GPR and normalize
            sigma2_n = np.exp( self.sigma2_n.evaluate(data)['Mean']) # TODO: internalize untransform_var # TODO: Just mean, or mean plus K sigma?
            if self.normalize_y:
                sigma2_n /= self.normalizer_std**2

        # Note the inverse normalization
        return {    'Mean': self.inverse_normalize_mean(f),
                    'Var_Latent': self.inverse_normalize_var(covf),
                    'Var_Predictive': self.inverse_normalize_var(covf + sigma2_n) }
