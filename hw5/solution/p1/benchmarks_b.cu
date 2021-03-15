#include "utils.h"
#include <algorithm>
#include <random>

/// a_i <-- b_{p_i}
__global__ void permutedCopyKernelABP(int K, double *a, const double *b, const int *p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K)
        a[idx] = b[p[idx]];
}
/// a_{p_i} <-- b_i
__global__ void permutedCopyKernelAPB(int K, double *a, const double *b, const int *p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K)
        a[p[idx]] = b[idx];
}

/// a_i <-- a_i + b_i
__global__ void additionKernel(int K, double *a, const double *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K)
        a[idx] += b[idx];
}
/// a_i <-- a_i + b_i (100x)
__global__ void repeatedAdditionKernel(int K, double *a, const double *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K) {
        for (int i = 0; i < 100; ++i)
            a[idx] += b[idx];
    }
}
/// a_i <-- a_i + b_i (100x) (optimized with restrict)
__global__ void optimizedRepeatedAdditionKernel1(
        int K, double * __restrict__ a, const double * __restrict__ b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K) {
        for (int i = 0; i < 100; ++i)
            a[idx] += b[idx];
    }
}
/// a_i <-- a_i + b_i (100x) (optimized with a temporary variable)
__global__ void optimizedRepeatedAdditionKernel2(int K, double *a, const double *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K) {
        double aFinal = a[idx];
        for (int i = 0; i < 100; ++i)
            aFinal += b[idx];
        a[idx] = aFinal;
    }
}

/// Buffer sizes we consider. The numbers are odd such that p[i]=(2*i)%K are all different.
static constexpr int kBufferSizes[] = {
    17, 65, 251, 1001, 2001, 5001,
    10'001, 25'001, 50'001, 100'001, 250'001, 500'001, 1'000'001,
    5'000'001, 20'000'001, 50'000'001,
};

void subtask_b() {
    constexpr int threadsPerBlock = 1024;
    int maxK = kBufferSizes[sizeof(kBufferSizes) / sizeof(kBufferSizes[0]) - 1];

    /// Pick a N with respect to K such that total running time is more or less uniform.
    auto pickN = [](int K) {
        return 100'000 / (int)std::sqrt(K) + 5;  // Some heuristics.
    };

    // Allocate all neccessary device and host buffers.
    // CUDA_CHECK(cudaCmd) check whether `cudaCmd` completed successfully.
    double *aDev;
    double *bDev;
    int *pDev;
    double *aHost;
    int *pHost;
    CUDA_CHECK(cudaMalloc(&aDev, maxK * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&bDev, maxK * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&pDev, maxK * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&aHost, maxK * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&pHost, maxK * sizeof(int)));
    // Set aDev, bDev and aHost to 0.0 (not really that important).
    CUDA_CHECK(cudaMemset(aDev, 0, maxK * sizeof(double)));
    CUDA_CHECK(cudaMemset(bDev, 0, maxK * sizeof(double)));
    memset(aHost, 0, maxK * sizeof(double));

    // Task 1b.1)
    for (int K : kBufferSizes) {
        double dt = benchmark(pickN(K), [=]() {
            // Synchronous upload.
            CUDA_CHECK(cudaMemcpy(aDev, aHost, K * sizeof(double), cudaMemcpyHostToDevice));
        });
        printf("upload K=%8d --> %5.2f GB/s\n", K, 1e-9 * K * sizeof(double) / dt);
    }


    // Task 1b.2)
    /// Benchmark copying for a given access pattern (permutation).
    auto benchmarkPermutedCopy = [=](const char *description, auto permutationFunc) {
        for (int K : kBufferSizes) {
            // Compute the permutation p[i].
            permutationFunc(K);
            CUDA_CHECK(cudaMemcpy(pDev, pHost, K * sizeof(int), cudaMemcpyHostToDevice));
            int numBlocks = (K + threadsPerBlock - 1) / threadsPerBlock;
            // Test both variants of permuted copying.
            double dtABP = benchmark(pickN(K), [=]() {
                // permutedCopyKernelABP<<<numBlocks, threadsPerBlock>>>(K, aDev, bDev, pDev);
                CUDA_LAUNCH(permutedCopyKernelABP, numBlocks, threadsPerBlock, K, aDev, bDev, pDev);
            });
            double dtAPB = benchmark(pickN(K), [=]() {
                // permutedCopyKernelAPB<<<numBlocks, threadsPerBlock>>>(K, aDev, bDev, pDev);
                CUDA_LAUNCH(permutedCopyKernelAPB, numBlocks, threadsPerBlock, K, aDev, bDev, pDev);
            });
            // Report how many bytes per second was written.
            printf("Case %s  -->  K=%8d  [a=b_p] %6.2f GB/s  [a_p=b] %6.2f GB/s written\n",
                   description, K,
                   1e-9 * K * sizeof(double) / dtABP,
                   1e-9 * K * sizeof(double) / dtAPB);
        }
    };

    // The patterns are already implemented, do not modify!
    std::mt19937 gen;
    benchmarkPermutedCopy("p[i]=i", [pHost](int K) {
        for (int i = 0; i < K; ++i)
            pHost[i] = i;
    });
    benchmarkPermutedCopy("p[i]=(2*i)%K", [pHost](int K) {
        for (int i = 0; i < K; ++i)
            pHost[i] = (2 * i) % K;
    });
    benchmarkPermutedCopy("p[i]=(4*i)%K", [pHost](int K) {
        for (int i = 0; i < K; ++i)
            pHost[i] = (4 * i) % K;
    });
    benchmarkPermutedCopy("p[i]=i, 32-shuffled", [pHost, &gen](int K) {
        for (int i = 0; i < K; ++i)
            pHost[i] = i;
        for (int i = 0; i < K; i += 32)
            std::shuffle(pHost + i, pHost + std::min(i + 32, K), gen);
    });
    benchmarkPermutedCopy("fully shuffled", [pHost, &gen](int K) {
        for (int i = 0; i < K; ++i)
            pHost[i] = i;
        std::shuffle(pHost, pHost + K, gen);
    });


    // Task 1b.3) and 1b.4)
    for (int K : kBufferSizes) {
        int numBlocks = (K + threadsPerBlock - 1) / threadsPerBlock;
        double dt1 = benchmark(pickN(K), [=]() {
            // additionKernel<<<numBlocks, threadsPerBlock>>>(K, aDev, bDev);
            CUDA_LAUNCH(additionKernel, numBlocks, threadsPerBlock, K, aDev, bDev);
        });
        double dt2 = benchmark(pickN(K), [=]() {
            // repeatedAdditionKernel<<<numBlocks, threadsPerBlock>>>(K, aDev, bDev);
            CUDA_LAUNCH(repeatedAdditionKernel, numBlocks, threadsPerBlock, K, aDev, bDev);
        });
        double dt3 = benchmark(pickN(K), [=]() {
            // optimizedRepeatedAdditionKernel1<<<numBlocks, threadsPerBlock>>>(K, aDev, bDev);
            CUDA_LAUNCH(optimizedRepeatedAdditionKernel1, numBlocks, threadsPerBlock, K, aDev, bDev);
        });
        double dt4 = benchmark(pickN(K), [=]() {
            // optimizedRepeatedAdditionKernel2<<<numBlocks, threadsPerBlock>>>(K, aDev, bDev);
            CUDA_LAUNCH(optimizedRepeatedAdditionKernel2, numBlocks, threadsPerBlock, K, aDev, bDev);
        });

        printf("a+b  1x -> %4.1f GFLOP/s  100x -> %5.1f GFLOP/s  restrict %6.1f GFLOP/s  tmp var %6.1f GFLOP/s\n",
               1e-9 * K / dt1,
               1e-9 * 100 * K / dt2,
               1e-9 * 100 * K / dt3,
               1e-9 * 100 * K / dt4);
    }

    CUDA_CHECK(cudaFreeHost(aHost));
    CUDA_CHECK(cudaFreeHost(pHost));
    CUDA_CHECK(cudaFree(pDev));
    CUDA_CHECK(cudaFree(bDev));
    CUDA_CHECK(cudaFree(aDev));
}

int main() {
    subtask_b();
}
