#include "utils.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <sys/stat.h>  // For mkdir...

/// Compute phi^(k+1} from phi^k, rho and h. For convenience we send hh=h^2 and invhh=1/h^2.
/// Here, `rho`, `phik` and `phik1` are row-major matrices, i.e. rho_{iy, ix} = rho[iy * N + ix].
__global__ void jacobiStepKernel(int N, double hh, double invhh,
                                 const double *rho, const double *phik, double *phik1) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= N || iy >= N)
        return;
    if (ix == 0 || iy == 0 || ix == N - 1 || iy == N - 1) {
        phik1[iy * N + ix] = 0.0;
        return;
    }

    double sum = phik[(iy + 1) * N + ix]
               + phik[(iy - 1) * N + ix]
               + phik[iy * N + (ix + 1)]
               + phik[iy * N + (ix - 1)];

    // double out = 1 / (-4 * invhh) * (-rho[iy * N + ix] - invhh * sum);
    double out = 0.25 * (hh * rho[iy * N + ix] + sum);

    phik1[iy * N + ix] = out;
}

void jacobiStep(int N, double h, const double *rhoDev, const double *phikDev, double *phik1Dev) {
    constexpr int threadsPerBlock = 32;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    CUDA_LAUNCH(jacobiStepKernel,
                dim3(numBlocks, numBlocks, 1),
                dim3(threadsPerBlock, threadsPerBlock, 1),
                N, h * h, 1 / (h * h), rhoDev, phikDev, phik1Dev);
}


__global__ void computeAphiKernel(int N, double invhh, const double *phi, double *Aphi) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= N || iy >= N)
        return;
    if (ix == 0 || iy == 0 || ix == N - 1 || iy == N - 1) {
        Aphi[iy * N + ix] = 0.0;
        return;
    }

    double out = invhh * (
              phi[(iy + 1) * N + ix]
            + phi[(iy - 1) * N + ix]
            + phi[iy * N + (ix + 1)]
            + phi[iy * N + (ix - 1)]
            - 4 * phi[iy * N + ix]);

    Aphi[iy * N + ix] = out;
}

void computeAphi(int N, double h, const double *phiDev, double *AphiDev) {
    constexpr int threadsPerBlock = 32;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    CUDA_LAUNCH(computeAphiKernel,
                dim3(numBlocks, numBlocks, 1),
                dim3(threadsPerBlock, threadsPerBlock, 1),
                N, 1 / (h * h), phiDev, AphiDev);
}

// Print L1 and L2 error. Do not edit!
void printL1L2(int iter, int N, const double *AphiHost, const double *rhoHost) {
    double L1 = 0.0;
    double L2 = 0.0;
    for (int iy = 1; iy < N - 1; ++iy)
    for (int ix = 1; ix < N - 1; ++ix) {
        int i = iy * N + ix;
        double error = std::fabs(AphiHost[i] - (-rhoHost[i]));
        L1 += error;
        L2 += error * error;
    }
    printf("%05d  Aphi - -rho  ==>  L1=%10.5g  L2=%10.5g\n", iter, L1, L2);
}

/*
 * Dump a vector x to a csv file for visualization (see usage below). Do not edit!
 *
 * Usage:
 *     dumpCSV(N, someHostVector, iterationNumber);
 * Note: This function is very slow compared to the kernels.
 *       Run on every 1000th time step.
 */
void dumpCSV(int N, const double *xHost, int iter) {
    char filename[64];
    sprintf(filename, "output/dump-%05d.csv", iter);
    mkdir("output", 0777);
    FILE *f = fopen(filename, "w");
    if (f == nullptr) {
        fprintf(stderr, "Error opening file \"%s\".", filename);
        exit(1);
    }

    for (int iy = 0; iy < N; ++iy)
        for (int ix = 0; ix < N; ++ix)
            fprintf(f, ix == N - 1 ? "%g\n" : "%g,", xHost[iy * N + ix]);

    fclose(f);
}

int main() {
    const int N = 400;
    const int numIterations = 50000;
    const double L = 1.0;
    const double h = L / N;

    double *rhoDev;
    double *phikDev;
    double *phik1Dev;
    double *rhoHost;
    double *tmpHost;
    double *tmpDev;
    CUDA_CHECK(cudaMalloc(&rhoDev,      N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&phikDev,     N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&phik1Dev,    N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&tmpDev,      N * N * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&rhoHost, N * N * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&tmpHost, N * N * sizeof(double)));

    // RHS with three non-zero elements at (x, y)=(0.3, 0.1), (0.4, 0.1) and (0.5, 0.6).
    for (int i = 0; i < N * N; ++i)
        rhoHost[i] = 0.0;
    rhoHost[(1 * N / 10) * N + (3 * N / 10)] += 2.0;
    rhoHost[(1 * N / 10) * N + (4 * N / 10)] += 1.0;
    rhoHost[(6 * N / 10) * N + (5 * N / 10)] += -2.0;
    CUDA_CHECK(cudaMemcpy(rhoDev, rhoHost, N * N * sizeof(double), cudaMemcpyHostToDevice));

    // Initial guess x^(0)_i = 0.
    CUDA_CHECK(cudaMemset(phikDev, 0, N * N * sizeof(double)));
    CUDA_CHECK(cudaMemset(phik1Dev, 0, N * N * sizeof(double)));

    for (int iter = 0; iter < numIterations; ++iter) {
        if (iter % 1000 == 0) {
            computeAphi(N, h, phikDev, tmpDev);

            CUDA_CHECK(cudaMemcpy(tmpHost, tmpDev, N * N * sizeof(double), cudaMemcpyDeviceToHost));
            printL1L2(iter, N, tmpHost, rhoHost);

            CUDA_CHECK(cudaMemcpy(tmpHost, phikDev, N * N * sizeof(double), cudaMemcpyDeviceToHost));
            for (int i = 0; i < N * N; ++i)
                tmpHost[i] -= -rhoHost[i];
            dumpCSV(N, tmpHost, iter);
        }

        jacobiStep(N, h, rhoDev, phikDev, phik1Dev);

        std::swap(phikDev, phik1Dev);
    }


    CUDA_CHECK(cudaFreeHost(tmpHost));
    CUDA_CHECK(cudaFreeHost(rhoHost));
    CUDA_CHECK(cudaFree(tmpDev));
    CUDA_CHECK(cudaFree(phik1Dev));
    CUDA_CHECK(cudaFree(phikDev));
    CUDA_CHECK(cudaFree(rhoDev));
}
