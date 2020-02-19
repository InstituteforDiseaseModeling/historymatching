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
    const auto X = Xarr.unchecked<2>(); //2 dimensions
    const auto P = Parr.unchecked<2>(); //2 dimensions
    const auto theta = theta_arr.unchecked<1>(); //1 dimension

    if(X.shape(1)!=P.shape(1))
        throw std::runtime_error("Second dimension of X and P matrices must match!");
    if(2+X.shape(1)!=theta.shape(0))
        throw std::runtime_error("Second dimension of X and P matrices must be 2 less than length of theta!");

    const auto sigma2_f = theta[0];

    const auto Nx = X.shape(0);
    const auto Np = P.shape(0);
    const auto D  = X.shape(1);

    auto kxp_arr = py::array(py::buffer_info(
        nullptr,                               // Pointer to data (nullptr -> ask NumPy to allocate!)
        sizeof(double),                        // Size of one item
        py::format_descriptor<double>::value,  // Buffer format
        2,                                     // How many dimensions?
        { Nx, Np },                            // Number of elements for each dimension
        { Np*sizeof(double), sizeof(double) }  // Strides for each dimension
    ));

    auto kxp = kxp_arr.mutable_unchecked<double,2>();

    for(int i=0;i<Nx;i++)
    for(int j=0;j<Np;j++){
        double r2=0;
        for(int d=0;d<D;d++){
            const auto dX = X(i,d)-P(j,d);
            r2 += dX*dX/theta(2+d);
        }
        kxp(i,j) = sigma2_f * std::exp(-r2/2.0);
    }

    return kxp_arr;
}



/**
Compute the Kxx kernel using (SLOW) CPU-based calculations.

This function really only remains for computers that do not have access to an NVidia GPU and for testing GPU calculations.

Args:
    X: (2D ndarray) points of dimension N x D
    theta: (1D ndarray) hyperparameters
    add_sigma2_n: (boolean) if True, add observation variance, sigma2_n, to the diagonal.
*/
auto kernel_xx(
    const py::array_t<double> Xarr,
    const py::array_t<double> theta_arr,
    const py::array_t<double> sigma2_n_arr,
    const bool add_sigma2_n,
    const int deriv
){
    const auto X        = Xarr.unchecked<2>();         //2 dimensions
    const auto theta    = theta_arr.unchecked<1>();    //1 dimension
    const auto sigma2_n = sigma2_n_arr.unchecked<1>(); //1 dimension

    if(2+X.shape(1)!=theta.shape(0))
        throw std::runtime_error("Second dimension of X and P matrices must be 2 less than length of theta!");
    if(X.shape(0)!=sigma2_n.shape(0))
        throw std::runtime_error("Length of sigma2_n must match that of X!");

    const auto sigma2_f = theta[0];

    const auto Nx = X.shape(0);
    const auto D  = X.shape(1);

    auto kxx_arr = py::array(py::buffer_info(
        nullptr,                               // Pointer to data (nullptr -> ask NumPy to allocate!)
        sizeof(double),                        // Size of one item
        py::format_descriptor<double>::value,  // Buffer format
        2,                                     // How many dimensions?
        { Nx, Nx },                            // Number of elements for each dimension
        { Nx*sizeof(double), sizeof(double) }  // Strides for each dimension
    ));

    auto kxx = kxx_arr.mutable_unchecked<double,2>();

    for(int i=0;i<Nx;i++){
        kxx(i,i)=0;
        if(deriv<=1){
            kxx(i,i) += sigma2_f;
        }

        for(int j=i+1;j<Nx;j++){
            double r2=0;
            for(int d=0;d<D;d++){
                const auto dX = X(i,d) - X(j,d);
                r2 += dX*dX/theta(2+d);
            }
            kxx(i,j) = sigma2_f * std::exp(-r2/2.0);

            if(deriv>1){
                const auto der=deriv-2;
                const auto dX = X(i,der) - X(j,der);
                kxx(i,j) *= 0.5 * (dX*dX) / (theta(2+der)*theta(2+der));
            }

            kxx(j,i) = kxx(i,j);
        }       
    }

    if(add_sigma2_n){
        for(int i=0;i<Nx;i++){
            kxx(i,i) += sigma2_n(i);
        }
    }

    return kxx_arr;
}

PYBIND11_MODULE(ckernels, m) {
    m.doc() = "Fast C++ Kernels";
    m.def("kernel_xp", &kernel_xp, "TODO");
    m.def("kernel_xx", &kernel_xx, "TODO");
}
