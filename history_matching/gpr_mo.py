import json
import os
import sys

from history_matching.basis import Basis
from string import Template
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as spo
import seaborn as sns

try:
    from pycuda import compiler, gpuarray
    import pycuda.autoinit
    import skcuda.misc as misc
    import skcuda.linalg as linalg
except ImportError as e:
    print("Looks like you don't have CUDA, that's okay, we'll try using CPU but it will be SLOW!")
except RuntimeError as e:
    print("Runtime error starting cuda, message was:\n", e.message)

# Ack https://github.com/lebedov/scikit-cuda/blob/master/demos/indexing_2d_demo.py

class GPR_MO():
    """Gaussian Process Regression.

    This class implementes Gaussian Process Regression with leave-one-out cross-validation for parameter fitting and NVidia-CUDA-based GPU acceleration for speed.
    """

    def __init__(self, basis, Ycols, training_data, param_info,
            kernel_mode = 'RBF',
            sigma2_f = None,
            sigma2_n = None,
            lengthscales2 = None,
            b = None,
            #is_poisson = False, # Not currently supported
            normalize_y = True,
            verbose = False,
            debug = False,
            **kwargs
        ):
        """Initialize the GPR_MO class.

        Args:
            basis: (basis)
                Provide an instance of a basis class which determines the parameters for the GPR_MO.
            Ycols:  (str)
                The name of the columns in training_data that contains the model output values.
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
            kernel_mode:  (str, optional)
                Eventual support for various kernels, for now the only available option is `RBF`.
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
            sigma_n: (None or instance of GPR_MO)
                For typical homoscedastic GPR_MO, leave as None.  The kernel hyperparamter, sigma2_n, will be optimized.  Alternatively for heteroscedastic GPR_MO, provide an instance of a GPR_MO with the same input dimensions for which the optut is the log of the variance.
            verbose: (boolean, optional with default False)
            debug: (boolean, optional with default False)
            normalizer_mean (float, optional):  Allows specification or recovery of the mean of the Y-normalizer.  Must specify normalizer_mean and normalizer_std for this feature to work.  It is typically used when restoring a GPR_MO from file.
            normalizer_std (float, optional): Allows specification or recovery of the std of the Y-normalizer.  Must specify normalizer_mean and normalizer_std for this feature to work.  It is typically used when restoring a GPR_MO from file.
        """

        self.use_gpu = False
        '''
        try:
            device = pycuda.autoinit.device
            print('Autoinit GPU device name:', device.name())
            self.use_gpu = True
        except Exception as e:
            self.use_gpu = False
        '''

        if self.use_gpu:
            # Read in the RFB kernel
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            self.kernel_fn = os.path.join(cur_dir, 'kernel.c')

        self.training_data = training_data.copy()
        self.param_info = param_info.copy()
        self.basis = basis
        self.D = self.basis.D
        self.Ycols = Ycols
        self.R = len(self.Ycols)

        # TEMP
        self.B = self.makeB([1,0])

        self.kernel_mode = kernel_mode

        if 'normalizer_mean' in kwargs and 'normalizer_std' in kwargs:
            self.normalizer_mean = kwargs['normalizer_mean']
            self.normalizer_std = kwargs['normalizer_std']
        else:
            Y = np.ma.masked_invalid(self.training_data[self.Ycols], copy=False)
            self.normalizer_mean = Y.mean(0).view(np.ndarray)
            self.normalizer_std  = Y.std(0).view(np.ndarray)

        # Normalize training data, change Ycols
        self.normalize_y = normalize_y
        self.Ycols_orig = self.Ycols
        normalized = self.normalize(self.training_data[self.Ycols])
        for i, Ycol in enumerate(self.Ycols_orig):
            self.training_data[Ycol+'_normalized'] = normalized[Ycol]
            self.Ycols[i] = Ycol+'_normalized'

        self.normalizer = True #UserStandardize(mean=self.normalizer_mean, std=self.normalizer_std)
        self.poisson = False #is_poisson

        self.verbose = verbose
        self.debug = debug

        # Heteroscedastic GP setup
        self.fixed_sigma_n = True
        if isinstance(sigma2_n, GPR_MO):
            print('User has configured GPR_MO with noise coming from another GPR_MO')
            self.sigma2_n = sigma2_n
            self.fixed_sigma_n = False

        if self.use_gpu:
            self.define_kernel()

        if sigma2_f is None or sigma2_n is None or lengthscales2 is None or b is None:
            self.reset_hyperparameters()
        else:
            self.set_hyperparameters(sigma2_f, sigma2_n, lengthscales2, b)


    @classmethod
    def from_config(cls, config_fn):
        """Restore a GPR_MO instance from a saved configuration file.

        Args:
            config_fn: (str)
                Path to the configuration file.
        """

        try:
            print('from_config:', config_fn)
            with open(os.path.join(config_fn)) as data_file:
                config = json.load( data_file )

                if 'Basis' in config:
                    basis = Basis.deserialize(config['Basis'])
                else:
                    # Backwards compatibility
                    Xcols = config['Xcols']
                    basis = Basis.polynomial_basis(
                        params = Xcols,
                        intercept = False,
                        first_order = True,
                        second_order = False,
                        third_order = False,
                        fourth_order = False,
                        fifth_order = False,
                        higher_order = False,
                        param_info = pd.read_json( config['Param_Info'], orient='split' ).set_index('Name')
                    )

                return cls(
                    basis = basis,
                    Ycols = config['Ycols'],
                    training_data = pd.read_json( config['Training_Data'], orient='split' ).set_index('Sample_Id'),
                    param_info = pd.read_json( config['Param_Info'], orient='split' ).set_index('Name'),
                    kernel_mode = config['Kernel_Mode'],

                    sigma2_f = np.array(config['sigma2_f']),
                    sigma2_n = np.array(config['sigma2_n']),
                    lengthscales2 = np.array(config['lengthscales2']),
                    b = np.array(config['b']),

                    normalizer_mean = config['Normalizer_Mean'],
                    normalizer_std = config['Normalizer_Std'],
                    normalize_y = config['Normalize_Y'] if 'Normalize_Y' in config else True
                )
        except EnvironmentError:
            print('Unable to load GPR_MO from_config file', config_fn)
            raise


    def makeB(self, b):
        return np.array( [
            [b[0], b[0]*b[1]],
            [b[0]*b[1], 1]
        ])

    def set_training_data(self, new_training_data):
        """Set the training data for GPR_MO, will normalize if needed

        Args:
            new_training_data: (Pandas DataFrame)
                As in __init__.
        """
        self.training_data = new_training_data.copy()
        if self.use_gpu:
            self.define_kernel()

        # Normalize training data as in __init__
        self.training_data[self.Ycols] = self.normalize(self.training_data[self.Ycols_orig])

        self.update_cache()


    def reset_hyperparameters(self):
        """Resets hyperparameters (theta).
        """
        # Clear the kernel/model hyperparameters and cache
        self.sigma2_n = None
        self.sigma2_f = None
        self.lengthscales2 = None
        self.b = None

        self.Kxx_inv_Y = None
        self.Kxx_inv = None
        self.X = None
        self.Y = None


    def set_hyperparameters(self, sigma2_f, sigma2_n, lengthscales2, b):
        """Sets hyperparameters (theta).

        Args:
            hyperparameters:
        """

        assert( len(lengthscales2) == self.D )

        self.sigma2_f = sigma2_f
        self.sigma2_n = sigma2_n
        if ~isinstance(lengthscales2, list):
            lengthscales2 = lengthscales2.tolist()
        self.lengthscales2 = lengthscales2
        self.b = b

        self.B = self.makeB(self.b) #np.array( [[1, self.b[1]], [self.b[1], self.b[0]]] )

        self.update_cache()


    def update_cache(self):
        """Update the internal cache of X, Y, Kxx_inv, and Kxx_inv_Y.

        When evaluating many points, these somewhat-slow to calculate properties do not change, co we compute and cache them here.

        """

        if self.debug:
            print('Updating cache of Kxx_inv and Kxx_inv_Y')

        train_mean = self.training_data.reset_index().groupby('Sample_Id').mean()
        self.X = self.basis.generate_dmatrix( train_mean, scaleX = True).values
        self.Y = train_mean[self.Ycols].values # Is there a way/need to use all results?

        theta = [self.sigma2_f, self.sigma2_n] + self.lengthscales2
        print('THETA:', theta)

        add_sigma2_n = True if self.R == 1 else False
        if self.use_gpu:
            try:
                Kxx = self.kxx_gpu_wrapper(self.X, theta, add_sigma2_n = add_sigma2_n)  # Y is noisy
            except pycuda._driver.MemoryError:
                print('Insufficient video memory for Kxx matrix of dimension', X.shape[0],', reverting to (slow) CPU computation.')

            if self.R > 1:
                Kxx = np.kron(self.B, Kxx)
                Kxx[np.diag_indices(Kxx.shape[0])] += sigma2_n

            Kxx_gpu = gpuarray.to_gpu(np.asarray(Kxx.copy(), np.float64))
            linalg.init()
            self.Kxx_inv = linalg.inv(Kxx_gpu, overwrite=True, lib='cusolver').get()
        else:
            Kxx = self.kernel_xx(self.X, theta, add_sigma2_n = add_sigma2_n)
            if self.R > 1:
                Kxx = np.kron(self.B, Kxx)
                Kxx[np.diag_indices(Kxx.shape[0])] += theta[1]
            self.Kxx_inv = np.linalg.inv(Kxx)

        self.Kxx_inv_Y = np.dot(self.Kxx_inv, self.Y.flatten('F')) # TODO: GPU


    def save(self, save_to):
        """Save GPR_MO instance to file.

        Args:
            save_to: (str) Filename.
        """

        with open(save_to, 'w') as fout:
            json.dump(
                {
                    'Basis': self.basis.serialize(),
                    'Ycols': self.Ycols_orig,
                    'Kernel_Mode': self.kernel_mode,
                    'sigma2_f': self.sigma2_f.tolist(),
                    'sigma2_n': self.sigma2_n.tolist(),
                    'lengthscales2': self.lengthscales2.tolist(),
                    'b': self.b.tolist(),
                    'Normalizer_Mean': self.normalizer_mean,
                    'Normalizer_Std': self.normalizer_std,
                    'Normalize_Y': self.normalize_y,
                    'Training_Data': self.training_data.reset_index().to_json(orient='split'), # [self.Xcols + [self.Ycols]]
                    'Param_Info': self.param_info.reset_index().to_json(orient='split')
                }, fout, indent=4)

    def normalize(self, data):
        """If normalize_y is True, normalize some data by subtracting the mean and dividing by the standard deviation.

        Args:
            data: (Pandas DataFrame) Data to normalize.
        """
        if self.normalize_y:
            return (data - self.normalizer_mean)/self.normalizer_std
        else:
            return data

    def inverse_normalize_mean(self, data):
        """Reverse the normalization calculation for the mean.

        Args:
            data: (Pandas DataFrame) Data to unnormalize.
        """
        if self.normalize_y:
            return data*self.normalizer_std + self.normalizer_mean
        else:
            return data

    def inverse_normalize_var(self, data):
        """Reverse the normalization calculation for the variance.

        Args:
            data: (Pandas DataFrame) Data to unnormalize.
        """
        if self.normalize_y:
            return data * (self.normalizer_std**2)
        else:
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
            print('Bad kernel mode, kernel_mode =',self.kernel_mode)
            raise


    def kernel_xx(self, X, theta, add_sigma2_n = True, deriv=-1):
        """Compute the Kxx kernel using (SLOW) CPU-based calculations.

        This function really only remains for computers that do not have access to an NVidia GPU and for testing GPU calculations.

        Args:
            X: (2D ndarray) points of dimension N x D
            theta: (1D ndarray) hyperparameters
            add_sigma2_n: (boolean) if True, add observation variance, sigma2_n, to the diagonal.
        """

        Nx = X.shape[0]

        if deriv >= 0:
            assert(add_sigma2_n == False) # Do not add sigma2_n to sigma2_f deriv

        if deriv == 1: # Assuming add_sigma2_n is True when taking deriv wrt sigma2_n, otherwise it would be zeros(Nx) ...
            if self.fixed_sigma_n:
                return np.eye(Nx)
            else:
                # No deriv wrt sigma2_n if the user has specified sigma2_n via GPR_MO
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

                if (deriv > 1): # Lengthscale derivatives
                    d = deriv-2;
                    Kxx[i,j] *= 0.5 * (dX[d] * dX[d]) / (theta[2+d] * theta[2+d]);

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

        if deriv == 0:
            assert(add_sigma2_n == False) # Do not add sigma2_n to sigma2_f deriv

        if deriv == 1: # Assuming add_sigma2_n is True when taking deriv wrt sigma2_n, otherwise it would be zeros(Nx) ...
            if self.fixed_sigma_n:
                return np.eye(Nx)
            else:
                # No deriv wrt sigma2_n if the user has specified sigma2_n via GPR_MO
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
            np.uint32(deriv),      # <-- Negative for no derivative
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

        if self.debug:
            # Test on CPU
            Kxx_cpu = self.kernel_xx(X.astype(np.float32), theta.astype(np.float32), add_sigma2_n)
            if not np.allclose(Kxx_cpu, Kxx):
                print('Kxx_gpu_wrapper(CPU):\n', Kxx_cpu)
                print('Kxx_gpu_wrapper(GPU):\n', Kxx)
                raise

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

            if self.debug:
                # Test on CPU
                Kxp_cpu = self.kernel_xp(X, P, theta)
                if not np.allclose(Kxp_cpu, Kxp_gpu.get()):
                    print('kxp_gpu_wrapper(CPU):\n', Kxp_cpu)
                    print('kxp_gpu_wrapper(GPU):\n', Kxp_gpu.get())
                    raise

            return Kxp_gpu.get()

        Kxp_cpu = self.kernel_xp(X, P, theta)
        return Kxp_cpu


    def cross_validation_without_grad(self, theta_all, X, Y, optimize_sigma2_n, log_transform):
        """Compute the LOO cross validation score and gradient with respect to the hyperparameters.

        Args:
            theta: (1D ndarray, optional with default True) hyperparameters.
            X: (2D ndarray) points of dimension N x D.
            Y: (1D ndarray) outputs of dimension N x 1.
            optimize_sigma2_n: (bool) Set False to keep sigma2_n at the initial guess value.
            log_transform: (bool) If True, a ln transformation has been applied to contrain parameters to positive values.
        """

        theta = theta_all[:-2]
        b = theta_all[-2:]
        B = self.makeB(b) #np.array( [[1, b[1]], [b[1], b[0]]] )
        R = Y.shape[1]

        if log_transform:
            theta_log = theta
            theta = np.maximum(np.minimum(theta, np.log(sys.float_info.max)), np.log(sys.float_info.min))
            theta = np.log(theta)

        if self.use_gpu:
            Kxx = self.kxx_gpu_wrapper(X, theta, add_sigma2_n = True) # Want predictive distribution, so add sigma2
            if self.debug:
                # Compare to CPU
                Kxx_cpu = self.kernel_xx(X, theta, add_sigma2_n = True)
                if not np.allclose(Kxx_cpu, Kxx):
                    print('loo_cross_validation(CPU xx):\n', Kxx_cpu)
                    print('loo_cross_validation(GPU xx):\n', Kxx)
                    raise
        else:
            add_sigma2_n = True if R == 1 else False
            Kxx = self.kernel_xx(X, theta, add_sigma2_n = add_sigma2_n)

        if R > 1:
            Kxx = np.kron(B, Kxx)
            Kxx[np.diag_indices(Kxx.shape[0])] += theta[1] # Add sigma2_n after kron

        if self.use_gpu:
            Kxx_gpu = gpuarray.to_gpu(np.asarray(Kxx.copy(), np.float64))
            linalg.init()
            Kxx_inv_gpu = linalg.inv(Kxx_gpu, overwrite=True, lib='cusolver')
            Kxx_inv = Kxx_inv_gpu.get()
        else:
            Kxx_inv = np.linalg.inv(Kxx)

        #Kxx_inv_Y = np.dot(Kxx_inv, Y).squeeze()
        Yflat = Y.flatten('F')
        Kxx_inv_Y = np.dot(Kxx_inv, Yflat) # TODO: GPU

        sigma2 = np.reciprocal(np.diag(Kxx_inv))
        err = np.multiply(Kxx_inv_Y, sigma2)

        ll = np.sum(-0.5* np.log(sigma2) - np.divide(np.square(err), 2*sigma2))
        ll -= 0.5*np.log(2*np.pi) * Yflat.shape[0]

        print('\n\tLL:', -ll, '\n\tTheta:', theta, b)

        return -ll


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


    def optimize_hyperparameters(self, x0, bounds, optimize_sigma2_n=True, log_transform=False, optimizer_options={}):
        """Optimize the hyperparameter vector, theta, with respect to the training data.

        Note that while each input may be simulated multiple times, here the mean of the outputs for each Saimple_Id are used.

        Args:
            x0: (1D ndarray) guess values like np.array([2, 0.10, 0.14641288665436947, 0.12166006573919039, 0.05, 0.05, 0.08055223671416605, 7.026854485434267 ]).
            bounds: (tuple) ((0.005,10),)+((0.01,10),) + tuple((5e-5,10) for i in range(self.D)).
            optimize_sigma2_n: (bool) Set False to keep sigma2_n at the initial guess value.
            log_transform: (bool) Set True to apply a log transformation and use unconstrained Conjugate Gradient to optimize the hyperparameters.  Bounds will be ignored.
            optimizer_options: (dict) Options to pass along to the optimization algorithm.  Through scipy-optimize, you can see these options e.g. for l-bfgs-b via spo.show_options(solver='minimize', method='l-bfgs-b')
        """

        # Optimizer options for L-BFGS-B:
        '''
        print(spo.show_options(solver='minimize', method='l-bfgs-b'))

        Minimize a scalar function of one or more variables using the L-BFGS-B
        algorithm.

        Options
        -------
        disp : bool
           Set to True to print convergence messages.
        maxcor : int
            The maximum number of variable metric corrections used to
            define the limited memory matrix. (The limited memory BFGS
            method does not store the full hessian but uses this many terms
            in an approximation to it.)
        factr : float
            The iteration stops when ``(f^k -
            f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps``, where ``eps``
            is the machine precision, which is automatically generated by
            the code. Typical values for `factr` are: 1e12 for low
            accuracy; 1e7 for moderate accuracy; 10.0 for extremely high
            accuracy.
        ftol : float
            The iteration stops when ``(f^k -
            f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
        gtol : float
            The iteration will stop when ``max{|proj g_i | i = 1, ..., n}
            <= gtol`` where ``pg_i`` is the i-th component of the
            projected gradient.
        eps : float
            Step size used for numerical approximation of the jacobian.
        disp : int
            Set to True to print convergence messages.
        maxfun : int
            Maximum number of function evaluations.
        maxiter : int
            Maximum number of iterations.
        maxls : int, optional
            Maximum number of line search steps (per iteration). Default is 20.
        '''

        idx = self.training_data.index.names    # Save index
        self.training_data.reset_index(inplace=True)

        samples = self.training_data['Sample_Id'].unique()
        for i,s in enumerate(samples):
            self.training_data.loc[ self.training_data['Sample_Id']==s, 'Sample_Index'] = i

        '''
        # Old way from using K-fold cross-validation
        if K <= 1:
            # Identity partition (LOO)
            self.training_data['Partition'] = self.training_data['Sample_Index']
        else:
            assert( K <= len(samples) )
            self.training_data['Partition'] = np.floor(self.training_data['Sample_Index']%K).astype(int)
        '''

        num_params = 2 + self.D # sigma2_n, sigma2_f, lengthscale^2 0, lengthscale^2 1, ..., lengthscale^2 D-1

        # Computing the mean here:
        train_mean = self.training_data.reset_index().groupby('Sample_Id').mean()

        X = self.basis.generate_dmatrix( train_mean, scaleX = True).values
        Y = self.training_data.reset_index().groupby('Sample_Id').apply(self.assign_rep).pivot('Sample_Id', 'Replicate', self.Ycols).values

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
            self.cross_validation_without_grad,
            args=(X,Y, optimize_sigma2_n, log_transform),
            x0 = x0,
            method=method,
            bounds = bounds, # Constrain values
            jac=None, hess=None, hessp=None,
            constraints=(), tol=None, callback=None,
            options = optimizer_options
        )

        print('OPTIMIZATION RETURNED:\n', ret)

        # Restore original index
        self.training_data.set_index(idx, inplace=True)

        # Set hyperparameters (theta) to optimal values

        if log_transform:
            x = np.maximum(np.minimum(ret.x, np.log(sys.float_info.max)), np.log(sys.float_info.min))
            log_sigma2_f = ret.x[0]
            log_sigma2_n = ret.x[1]
            log_lengthscales2 = ret.x[2:-2]
            b = ret.x[-2:]
            self.set_hyperparameters( np.exp(log_sigma2_f), np.exp(log_sigma2_n), np.exp(log_lengthscales2), b )
        else:
            sigma2_f = ret.x[0]
            sigma2_n = ret.x[1]
            lengthscales2 = ret.x[2:-2]
            b = ret.x[-2:]
            self.set_hyperparameters( sigma2_f, sigma2_n, lengthscales2, b )


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

        if self.debug:
            print('X',self.X.shape,' flags:\n', self.X.flags)
            print('Y',self.Y.shape,' flags:\n', self.Y.flags)
            print('P',P.shape,' flags:\n', P.flags)

        theta = [self.sigma2_f, self.sigma2_n] + self.lengthscales2

        Kxp = self.kxp_gpu_wrapper(self.X, P, theta)
        if self.debug:
            Kxp_cpu = self.kernel_xp(self.X, P, theta)
            if not np.allclose(Kxp_cpu, Kxp):
                print('evaluate(CPU xp):\n', Kxp_cpu)
                print('evaluate(GPU xp):\n', Kxp)
                raise

        if self.use_gpu:
            Kpp = self.kxx_gpu_wrapper(P, theta, add_sigma2_n = False) # For latent distribution
            if self.debug:
                Kpp_cpu = self.kernel_xx(P, theta, add_sigma2_n = False)
                if not np.allclose(Kpp_cpu, Kpp):
                    print('evaluate(CPU pp):\n', Kpp_cpu)
                    print('evaluate(GPU pp):\n', Kpp)
                    raise
        else:
            Kpp = self.kernel_xx(P, theta, add_sigma2_n = False)

        if self.R > 1:
            Kxp = np.kron(self.B, Kxp)
            Kpp = np.kron(self.B, Kpp)
        f = np.dot(Kxp.T, self.Kxx_inv_Y)

        # NOTE: Just computing diagonal elements of:
        covf = Kpp - np.dot(Kxp.T, np.dot(self.Kxx_inv, Kxp)) # Full covariance
        # TODO: Generalize
        covf = np.multiply(covf, np.outer(self.normalizer_std, self.normalizer_std))

        if self.use_gpu:
            # : Reuse gpu arrays to reduce to_gpu transfers
            Kxx_inv_gpu = gpuarray.to_gpu(np.asarray(self.Kxx_inv, np.float64))
            Kxp_gpu = gpuarray.to_gpu(np.asarray(Kxp, np.float64))
            tmp = linalg.dot(Kxx_inv_gpu, Kxp_gpu).get() # Need .copy() on gpu arrays?
        else:
            tmp = np.dot(self.Kxx_inv, Kxp)

        varf = np.diag(Kpp) - np.einsum('ji,ji->i', Kxp, tmp)
        print('varf', varf)

        # Add in observation noise
        if self.fixed_sigma_n:
            sigma2_n = theta[1]*np.ones(P.shape[0])
        else:
            # Evaluate the sigma2_n GPR_MO and normalize
            sigma2_n = np.exp( self.sigma2_n.evaluate(data)['Mean']) # TODO: internalize untransform_var # TODO: Just mean, or mean plus K sigma?
            if self.normalize_y:
                sigma2_n /= self.normalizer_std**2


        if self.R > 1:
            f = f.reshape((data.shape[0], self.R), order='F')
            varf = varf.reshape((data.shape[0], self.R), order='F')
            sigma2_n = np.kron(np.ones((P.shape[0],1)), [theta[1], theta[1]]) # TODO: Separate sigma2_n per dimension!

        # Note the inverse normalization
        return {    'Mean': self.inverse_normalize_mean(f),
                    'Var_Latent': self.inverse_normalize_var(varf),
                    'Var_Predictive': self.inverse_normalize_var(varf + sigma2_n),
                    'Cov_Latent': covf,
                    'Cov_Predictive': covf + sigma2_n*np.eye(covf.shape[0]),
                }


    def plot_data(self, samples_to_circle=pd.DataFrame(), saveto_dir = None, log_scale = False):
        """Make pairwise plots of data.

        TODO: Make the scaling function a lambda.

        Size is maximum of 1 and 25*normalized_y_value.

        Args:
            samples_to_circle: (Pandas DataFrame, similar cols to training_data) Plot size 50 black x's, one for each row
            saveto_dir: (str, default None) Directory name where resulting figures should be saved.  None disables saving.
            log_scale:  (boolean, default is False) transforms size and color using log(10 * normalized_y_value + 1)

        Returns: Dictionary of matplotlib figure handle.  Keys are like VarName1-VarName2.pdf for each pair of variables.
        """

        scaled = (self.training_data[self.Ycols]-self.training_data[self.Ycols].min()) / (self.training_data[self.Ycols].max()-self.training_data[self.Ycols].min())
        if log_scale:
            scaled = np.log( 10*scaled+1 )

        figs = {}

        X = self.basis.generate_dmatrix( self.training_data, scaleX = True)
        Xcols = X.columns.tolist()

        if samples_to_circle.shape[0] > 0:
            samples_to_circle_dmat = self.basis.generate_dmatrix( samples_to_circle, scaleX = True)

        for row in range(self.D):
            for col in range(self.D):
                if col > row:
                    fn = '%s-%s.pdf' % (Xcols[row], Xcols[col])
                    fig = plt.figure(figsize=(6,6)) #GPy.plotting.plotting_library().figure()

                    x = X[Xcols[row]]
                    y = X[Xcols[col]]

                    plt.scatter(x, y, s=np.maximum(1, 25*scaled), c=scaled, cmap='jet', linewidths=0.1, alpha=0.5, edgecolors='k') #, s=area, c=colors, alpha=0.5)

                    # Circle some interesting samples
                    if samples_to_circle.shape[0] > 0:
                        for idx, pt in samples_to_circle_dmat.iterrows():
                            plt.scatter(pt[ Xcols[row] ], pt[ Xcols[col] ], s=50, c='k', alpha=1, linewidths=2.0, marker='x') #, s=area, c=colors, alpha=0.5)

                    plt.autoscale(tight=True)
                    plt.xlabel( Xcols[row] )
                    plt.ylabel( Xcols[col] )
                    plt.tight_layout()

                    if saveto_dir is not None:
                        fig.savefig( os.path.join(saveto_dir, fn) ); plt.close(fig)
                    else:
                        figs[fn] = fig

        return figs


    def plot_histogram(self):
        """Plots histograms of the training data using Seaborn's distplot routine.

        Returns: Matplotlib figure handle
        """

        fig, ax = plt.subplots(nrows=1, ncols=1) # , figsize=(5,5), sharex='col', sharey='row')
        sns.distplot(self.training_data[self.Ycols], rug=True, ax = ax)

        return fig


    def plot(self, Xcenter, res=10):
        """Plots 2D contour slices through the output GPR_MO.

        When evaluating sweeping two parameteres at a time, the other parameters are fixed at Xcenter.

        Args:
            Xcenter: (1D ndarray similar to x0) These are the 'baseline' values unless modified in a 2D sweep.
            res: (int) number of grid points per dimension.  res*res points will be evaluated to generate each pairwise plot.

        Returns: Tuple of matplotlib figure handles.  The first element is for the mean and the second is for the latent standard deviation.
        """
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

                    (row_min, row_max) = (self.training_data[self.Xcols[row]].min(), self.training_data[self.Xcols[row]].max())
                    (col_min, col_max) = (self.training_data[self.Xcols[col]].min(), self.training_data[self.Xcols[col]].max())
                    x1 = np.linspace(row_min, row_max, res)
                    x2 = np.linspace(col_min, col_max, res)
                    X1, X2 = np.meshgrid(x1, x2)

                    X = Xmu.copy()
                    X[:,row] = X1.flatten()
                    X[:,col] = X2.flatten()

                    Xdf = pd.DataFrame(X, columns=self.Xcols)

                    self.debug=False
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

                    ax.scatter(self.training_data[self.Xcols[row]], self.training_data[self.Xcols[col]], c=self.training_data[self.Ycols], s=25, cmap='jet')

                    try:
                        CS = ax_std_latent.contour(X1, X2, Y_std_latent, zorder=100)
                        ax_std_latent.clabel(CS, inline=1, fontsize=10, zorder=100)
                    except:
                        print('Unable to plot std contour')

                    if col == self.D-1:
                        ax.set_xlabel( self.Xcols[row] )
                    if row == 0:
                        ax.set_ylabel( self.Xcols[col] )
        #plt.tight_layout()
        return (fig, fig_std_latent)


    def plot_errors(self, train, test, mean_col, var_col):
        """Generates two plots on a single figure.

        The upper plot shows GP prediction on Y as a function of the true Y-values on X.  The lower panel shows Z-score on Y and the true Y-values on X.

        In both panels, training data is cyan and test data is magenta.

        Args:
            train: (Pandas DataFrame) training data like training_data.
            test: (Pandas DataFrame) test data like training_data.
            mean_col: (str) Column name of predicted mean in train and test dataframes.
            var_col: (str) Column name of variance (latent or predictive, you pick) in train and test dataframes.

        Returns: Matplotlib figure handle.
        """

        train['Z_Score'] = (train[self.Ycols_orig] - train[mean_col]) / np.sqrt(train[var_col])
        test['Z_Score'] = (test[self.Ycols_orig] - test[mean_col]) / np.sqrt(test[var_col])

        fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col', figsize=(16,10)) # , sharex='col', sharey='row')

        ax.errorbar(x=test[self.Ycols_orig], y=test[mean_col], yerr=2*np.sqrt(test[var_col]), fmt='o', c='m', lw=0.5)
        ax.errorbar(x=train[self.Ycols_orig], y=train[mean_col], yerr=2*np.sqrt(train[var_col]), fmt='o', c='c', lw=0.5)
        ax.margins(x=0,y=0.05)
        xlim = ax.get_xlim()
        ax.plot( [xlim[0],xlim[1]], [xlim[0], xlim[1]], 'r-')
        ax.set_xlabel(self.Ycols_orig)
        ax.set_ylabel('Predicted')

        '''
        ax = ax2
        ax.scatter(x=train[self.Ycols_orig], y=train['Z_Score'], facecolor='c', marker='.', lw=1, alpha=0.5, s=50)
        ax.scatter(x=test[self.Ycols_orig], y=test['Z_Score'], facecolor='m', marker='.', lw=1, alpha=0.5, s=50)
        ax.set_xlabel(self.Ycols_orig)
        ax.set_ylabel('Z-Score')
        ax.margins(x=0,y=0.05)
        '''

        plt.tight_layout()

        return fig
