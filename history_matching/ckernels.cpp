#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <stdexcept>

namespace py = pybind11;

/**
Compute the Kxp kernel using (SLOW) CPU-based calculations.

This function really only remains for computers that do not have access to an NVidia GPU and for testing GPU calculations.

Args:
    X: (2D ndarray) points of dimension N x D
    P: (2D ndarray) points of dimension P x D
    theta: (1D ndarray) hyperparameters
*/
auto kernel_xp(
    const py::array_t<double> Xarr,
    const py::array_t<double> Parr,
    const py::array_t<double> theta_arr
){
    // for i in range(Nx):
    //     for j in range(Np):
    //         dX = X[i,:]-P[j,:]
    //         r2 = 0
    //         for d in range(D):
    //             r2 += dX[d] * dX[d]/theta[2+d]
    //         kxp[i,j] = sigma2_f * np.exp( -r2 / 2. )

    // return kxp

    const auto X = Xarr.unchecked<2>(); //2 dimensions
    const auto P = Parr.unchecked<2>(); //2 dimensions
    const auto theta = theta_arr.unchecked<1>(); //1 dimension

    if (X.shape(1)!=P.shape(1))
        throw std::runtime_error("Second dimension of X and P matrices must match!");

    const auto sigma2_f = theta[0];

    const auto Nx = X.shape(0);
    const auto Np = P.shape(0);
    const auto D  = X.shape(1);


    auto kxp_arr = py::array(py::buffer_info(
        nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
        sizeof(double),     /* Size of one item */
        py::format_descriptor<double>::value, /* Buffer format */
        2,          /* How many dimensions? */
        { Nx, Np },  /* Number of elements for each dimension */
        { Np*sizeof(double), sizeof(double) }  /* Strides for each dimension */
    ));

    auto kxp = kxp_arr.mutable_unchecked<double,2>();

    for(int i=0;i<Nx;i++)
    for(int j=0;j<Np;j++){
        double r2=0;
        for(int d=0;d<D;d++){
            auto dX = X(i,d)-P(j,d);
            r2 += dX*dX/theta[2+d];
        }
        kxp(i,j) = sigma2_f * std::exp(-r2/2.0);
    }

    return kxp_arr;
}

PYBIND11_MODULE(ckernels, m) {
    m.doc() = "Fast C++ Kernels";
    m.def("kernel_xp", &kernel_xp, "TODO");
}



/**
Compute the Kxx kernel using (SLOW) CPU-based calculations.

This function really only remains for computers that do not have access to an NVidia GPU and for testing GPU calculations.

Args:
    X: (2D ndarray) points of dimension N x D
    theta: (1D ndarray) hyperparameters
    add_sigma2_n: (boolean) if True, add observation variance, sigma2_n, to the diagonal.
*/

/*
void kernel_xx(
    py::array_t<double> X,
    py::array_t<double> theta,
    bool add_sigma2_n
){
    auto Kxx = np.zeros([Nx,Nx], dtype=np.float32);

    for(int i=0;i<Nx;i++){
        if(deriv<=1){
            Kxx[i,] = sigma2_f
        }
        for(int j=i+1;j<Nx;j++){

        }
    }    
}


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
*/