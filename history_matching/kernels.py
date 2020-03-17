from string import Template
import functools
import logging
import os

import numpy as np

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



@functools.lru_cache(maxsize=None)
def _define_kernels(Nx):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    kernel_fn = os.path.join(cur_dir, 'kernel.c')

    with open(kernel_fn, 'r') as f:
        kernel_code_template = Template(f.read())

    max_threads_per_block, max_block_dim, max_grid_dim = misc.get_dev_attrs(pycuda.autoinit.device)
    block_dim, grid_dim = misc.select_block_grid_sizes(pycuda.autoinit.device, (Nx, Nx))
    max_blocks_per_grid = max(max_grid_dim)

    log = logging.getLogger("HistoryMatching")
    log.info(f"max_threads_per_block: {max_threads_per_block}")
    log.info(f"max_block_dim:         {max_block_dim}")
    log.info(f"max_grid_dim:          {max_grid_dim}")
    log.info(f"max_blocks_per_grid:   {max_blocks_per_grid}")
    log.info(f"block_dim:             {block_dim}")
    log.info(f"grid_dim:              {grid_dim}")

    # Substitute in template to get kernel code
    kernel_code = kernel_code_template.substitute(
        max_threads_per_block   = max_threads_per_block,
        max_blocks_per_grid     = max_blocks_per_grid,
        B = Nx)

    # Compile the kernel
    mod = compiler.SourceModule(kernel_code)

    # retrieve the kernel functions
    return {
      "kernel_xp": mod.get_function("kernel_xp2"),
      "kernel_xx": mod.get_function("kernel_xx")
    }



def _kernel_xp_gpu(X, P, theta, sigma2_f):
    """Compute the Kxp kernel or derivatives using (FAST) GPU-based calculations.

    Args:
        X: (2D ndarray) points of dimension N x D.
        P: (2D ndarray) points of dimension P x D.
        theta: (1D ndarray, optional with default True) hyperparameters.
    """
    assert len(X.shape)==2
    assert len(P.shape)==2
    assert len(theta.shape)==1
    assert X.shape[1]==P.shape[1]
    assert X.shape[1]==theta.shape[0]

    Nx = X.shape[0]
    Np = P.shape[0]
    D  = X.shape[1]
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
    _define_kernels(Nx)["kernel_xp"](
        Kxp_gpu,                   # <-- Output
        X_gpu, P_gpu, theta_gpu,   # <-- Inputs
        np.float32(sigma2_f),
        np.uint32(Nx),             # <-- Nx
        np.uint32(Np),             # <-- Nx
        np.uint32(D),              # <-- D
        block = block_dim,
        grid = grid_dim
    )

    return Kxp_gpu.get()



def _kernel_xp_cpu(X, P, theta, sigma2_f):
    """Compute the Kxp kernel using (SLOW) CPU-based calculations.

    This function really only remains for computers that do not have access to an NVidia GPU and for testing GPU calculations.

    Args:
        X: (2D ndarray) points of dimension N x D
        P: (2D ndarray) points of dimension P x D
        theta: (1D ndarray) hyperparameters
    """
    assert len(X.shape)==2
    assert len(P.shape)==2
    assert len(theta.shape)==1
    assert X.shape[1]==P.shape[1]
    assert X.shape[1]==theta.shape[0]

    Nx = X.shape[0]
    Np = P.shape[0]
    D  = X.shape[1]

    kxp = np.zeros([Nx,Np])
    for i in range(Nx):
        for j in range(Np):
            dX = X[i,:]-P[j,:]
            r2 = np.sum(dX*dX/theta)
            kxp[i,j] = sigma2_f * np.exp( -r2 / 2. )

    return kxp





def kernel_xp(X, P, theta, sigma2_f, mode="cpu"):
    """Compute the Kxp kernel using (SLOW) CPU-based calculations.

    This function really only remains for computers that do not have access to an NVidia GPU and for testing GPU calculations.

    Args:
        X: (2D ndarray) points of dimension N x D
        P: (2D ndarray) points of dimension P x D
        theta: (1D ndarray) hyperparameters
    """
    if mode=="cpu":
      return _kernel_xp_cpu(X,P,theta,sigma2_f)
    elif mode=="gpu":
      return _kernel_xp_gpu(X,P,theta,sigma2_f)