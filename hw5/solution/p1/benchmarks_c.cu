#include "utils.h"
#include <numeric>
#include <omp.h>
#include <vector>

using ll = long long;

/// Uniformly split the range [0..K-1] to numThreads and evaluate
/// what belongs to the current thread.
/// This function is used by both CPU and GPU codes.
__device__ __host__ double leibnizSeries(ll K, int threadId, int numThreads) {
    // [0..K) --> [begin..end)
    ll begin = K * threadId / numThreads;
    ll end = K * (threadId + 1) / numThreads;

    // We go from end to begin, because in principle summing from smaller to
    // larger numbers results in a better accuracy. Not really effective here
    // though.

    // Here we try to avoid having a special variable for (-1)^k (not to
    // mention std::pow...). Note that this optimization might be
    // counterproductive for CPU codes since it may prevent autovectorization.
    // However, it seems that neither clang nor gcc vectorize this loop anyway
    // (not even with --fast-math).
    double sum = 0.0;
    for (ll k = end - 1; k >= begin; --k)
        sum = 1.0 / (2 * k + 1) - sum;
    if (begin % 2 == 1)
        sum = -sum;  // Correct the sign.

    // This is the same code, but going from start to to end.
    // for (ll k = begin; k < end; ++k)
    //     sum = 1.0 / (2 * k + 1) - sum;
    // if (end % 2 == 0)
    //     sum = -sum;  // Correct the sign.

    return sum;
}

// Compute the sum of the Leibniz series. Each thread takes care of a subset of terms.
__global__ void leibnizKernel(ll K, double *partialSums) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = gridDim.x * blockDim.x;

    partialSums[idx] = leibnizSeries(K, idx, numThreads);
}

/// Run the CUDA code for the given number of blocks and threads/block.
void runCUDA(ll K, int numBlocks, int threadsPerBlock) {
    int numThreads = numBlocks * threadsPerBlock;

    // Allocate the device and host buffers.
    double *sumsDev;
    double *sumsHost;
    CUDA_CHECK(cudaMalloc(&sumsDev, numThreads * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&sumsHost, numThreads * sizeof(double)));
    CUDA_CHECK(cudaMemset(sumsDev, 0, numThreads * sizeof(double)));
    double dt = benchmark(100, [numBlocks, threadsPerBlock, K, sumsDev]() {
        // leibnizKernel<<<numBlocks, threadsPerBlock>>>(K, sumsDev);
        CUDA_LAUNCH(leibnizKernel, numBlocks, threadsPerBlock, K, sumsDev);
    });
    CUDA_CHECK(cudaMemcpy(sumsHost, sumsDev, numThreads * sizeof(double), cudaMemcpyDeviceToHost));
    double sum = std::accumulate(sumsHost, sumsHost + numThreads, 0.0);
    double pi = 4 * sum;
    printf("CUDA blocks=%5d  threads/block=%4d  iter/thread=%5lld  pi=%.12f  rel error=%.2g  Gterms/s=%.1f\n",
           numBlocks, threadsPerBlock, K / numThreads, pi, (pi - M_PI) / M_PI,
           1e-9 * K / dt);
    CUDA_CHECK(cudaFreeHost(sumsHost));
    CUDA_CHECK(cudaFree(sumsDev));
}

/// Run the OpenMP variant of the code.
void runOpenMP(ll K, int numThreads) {
    double sum;
    double dt = benchmark(10, [K, numThreads, &sum]() {
        sum = 0.0;
        #pragma omp parallel num_threads(numThreads)
        {
            int threadId = omp_get_thread_num();
            int numThreads = omp_get_num_threads();

            double local = leibnizSeries(K, threadId, numThreads);
            #pragma omp atomic
            sum += local;
        }
    });
    double pi = 4 * sum;
    printf("OpenMP threads=%d  pi=%.16g  rel error=%.2g  Gterms/s=%.1f\n",
           numThreads, pi, (pi - M_PI) / M_PI, 1e-9 * K / dt);
};


void subtask_c() {
    constexpr ll K = 2LL << 30;

    runCUDA(K, 512, 512);
    runCUDA(K, 512, 1024);
    runCUDA(K, 1024, 128);
    runCUDA(K, 1024, 256);
    runCUDA(K, 1024, 512);
    runCUDA(K, 1024, 1024);
    runCUDA(K, 32768, 64);
    runCUDA(K, 32768, 128);
    runCUDA(K, 32768, 1024);
    runOpenMP(K, 12);
}

int main() {
    subtask_c();
}
