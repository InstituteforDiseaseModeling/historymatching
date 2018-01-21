// Macro for converting subscripts to linear index:
__global__ void kernel_xx(float *kxx, float *X, float *theta, unsigned int N, unsigned int D, int deriv, bool FORTRAN_CONTIGUOUS) {

    // Obtain the linear index corresponding to the current thread:
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

    if( idx < N*N ) {
        unsigned int i = idx/N;
        unsigned int j = idx%N;

	// if i == j! Special, including deriv!

        if (i <= j) {
            float sigma2_f = theta[0];
            if (deriv == 0) {
                sigma2_f = 1;
            }

            if (i == j) {
                kxx[N*i+j] = sigma2_f;
            } else {
                float r2 = 0;
                for( int d=0; d<D; d++) {
                    if (FORTRAN_CONTIGUOUS) {
                        // FORTRAN contiguous (row major)
                        r2 += (X[N*d+i]-X[N*d+j]) * (X[N*d+i]-X[N*d+j]) / theta[2+d];
                    } else {
                        // C contiguous (column major)
                        r2 += (X[D*i+d]-X[D*j+d]) * (X[D*i+d]-X[D*j+d]) / theta[2+d];
                    }
                }

                kxx[N*i+j] = sigma2_f * exp( -r2 / 2.0 );
            }


            if (deriv > 1) {    // Lengthscale derivatives
                int d = deriv-2;
                if (FORTRAN_CONTIGUOUS) {
                    // FORTRAN contiguous (row major)
                    kxx[N*i+j] *= 0.5 * (X[N*d+i]-X[N*d+j]) * (X[N*d+i]-X[N*d+j]) / (theta[2+d] * theta[2+d]);
                } else {
                    // C contiguous (column major)
                    kxx[N*i+j] *= 0.5 * (X[D*i+d]-X[D*j+d]) * (X[D*i+d]-X[D*j+d]) / (theta[2+d] * theta[2+d]);
                }
            }

            if(i < j) {
                kxx[N*j+i] = kxx[N*i+j];
            }
        }
    }
}

__global__ void kernel_xp(float *kxp, float *X, float *P, float *theta, unsigned int Nx, unsigned int Np, unsigned int D) {

    // Obtain the linear index corresponding to the current thread:
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

    if( idx < Nx*Np ) {
        // Convert the linear index to subscripts:
        unsigned int i = idx/Np;
        unsigned int j = idx%Np;
        float sigma2_f = theta[0];

        float r2 = 0;
        for( int d=0; d<D; d++) {
            // Assuming C contiguous (column major order)
            r2 += (X[D*i+d]-P[D*j+d]) * (X[D*i+d]-P[D*j+d]) / theta[2+d];
        }

        kxp[idx] = sigma2_f * exp( -r2 / 2.0 );
    }
}
