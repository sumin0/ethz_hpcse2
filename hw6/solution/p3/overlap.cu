#include "utils.h"
#include <cuda_profiler_api.h>
#include <algorithm>
#include <vector>

/// Memory-bound dummy kernel. Do not edit.
__global__ void fastKernel(const double *a, double *b, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M)
        return;

    b[idx] = 10.0 * a[idx];
}

/// Compute-bound dummy kernel. Do not edit.
__global__ void slowKernel(const double *a, double *b, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M)
        return;

    double x = a[idx];
    for (int i = 0; i < 10000; ++i)
        x *= 1.01;

    b[idx] = (x != 0.1231 ? 10.0 : -1.0) * a[idx];
}

/// Check whether `bHost` contains the correct result. Do not edit.
void checkResults(const double *bHost, int N) {
    for (int i = 0; i < N; ++i) {
        if (bHost[i] != 100.0 * i) {
            printf("Incorrect value for i=%d:  value before kernel=%.1f  "
                   "expected after=%.1f  now=%.1f\n",
                   i, 10.0 * i, 100. * i, bHost[i]);
            exit(1);
        }
    }
}




/// Asynchronously, and in chunks, copy the array to the device, execute the
/// kernel and copy the result back.
template <typename Kernel>
void runAsync(const char *kernelName, Kernel kernel, int N, int chunkSize, int numStreams) {
    double *aHost;
    double *bHost;

    CUDA_CHECK(cudaMallocHost(&aHost, N * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&bHost, N * sizeof(double)));
    for (int i = 0; i < N; ++i)
        aHost[i] = 10.0 * i;

    // Allocate chunks and create streams.
    std::vector<double *> aDev(numStreams);
    std::vector<double *> bDev(numStreams);
    std::vector<cudaStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        CUDA_CHECK(cudaMalloc(&aDev[i], chunkSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&bDev[i], chunkSize * sizeof(double)));
    }
    for (int i = 0; i < numStreams; ++i)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    // Instead of benchmark() we use a simplified measure() which invokes the
    // function only once (to get a cleaner profiling information).
    double dt = measure([&]() {
        // Handle chunk by chunk.
        for (int chunk = 0; chunk * chunkSize < N; ++chunk) {
            int offset = chunk * chunkSize;
            int size = std::min(chunkSize, N - offset);
            int stream = chunk % numStreams;
            // Host -> device.
            CUDA_CHECK(cudaMemcpyAsync(aDev[stream], aHost + offset, size * sizeof(double),
                                       cudaMemcpyHostToDevice, streams[stream]));

            // Kernel.
            int threads = 1024;
            int blocks = (size + threads - 1) / threads;
            CUDA_LAUNCH_EX(kernel, blocks, threads, 0, streams[stream],
                           aDev[stream], bDev[stream], size);

            // Device -> host.
            CUDA_CHECK(cudaMemcpyAsync(bHost + offset, bDev[stream], size * sizeof(double),
                                       cudaMemcpyDeviceToHost, streams[stream]));
        }

        // Synchronize. This MUST be within the lambda for the time measurement to work.
        for (int i = 0; i < numStreams; ++i)
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    });

    checkResults(bHost, N);

    printf("async %s  N=%9d  chunkSize=%9d  numStreams=%d  time=%fs\n",
           kernelName, N, chunkSize, numStreams, dt);

    // Destroy streams and deallocate the chunks.
    for (int i = numStreams - 1; i >= 0; --i)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    for (int i = numStreams - 1; i >= 0; --i) {
        CUDA_CHECK(cudaFree(bDev[i]));
        CUDA_CHECK(cudaFree(aDev[i]));
    }

    CUDA_CHECK(cudaFreeHost(bHost));
    CUDA_CHECK(cudaFreeHost(aHost));
}





/// Synchronously copy the whole array to the device, execute the kernel and
/// copy the result back. Do not edit.
template <typename Kernel>
void runSync(const char *kernelName, Kernel kernel, int N) {
    double *aHost;
    double *bHost;
    double *aDev;
    double *bDev;

    CUDA_CHECK(cudaMallocHost(&aHost, N * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&bHost, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&aDev, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&bDev, N * sizeof(double)));
    for (int i = 0; i < N; ++i)
        aHost[i] = 10.0 * i;

    // Host -> device.
    double dt1 = measure([&]() {
        CUDA_CHECK(cudaMemcpy(aDev, aHost, N * sizeof(double), cudaMemcpyHostToDevice));
    });
    // Kernel.
    double dt2 = measure([&]() {
        // We cannot execute more than maxBlocks blocks, so we split the work
        // into multiple launches. That's another reason for using chunks.
        int threads = 1024;
        int maxBlocks = 65'536;
        int blocks = (N + threads - 1) / threads;
        for (int i = 0; i < blocks; i += maxBlocks) {
            CUDA_LAUNCH(kernel, std::min(maxBlocks, blocks - i), threads,
                        aDev + i * threads, bDev + i * threads, N);
        }
    });
    // Device -> host.
    double dt3 = measure([&]() {
        CUDA_CHECK(cudaMemcpy(bHost, bDev, N * sizeof(double), cudaMemcpyDeviceToHost));
    });

    checkResults(bHost, N);

    printf("sync  %s  N=%9d  upload=%fs  kernel=%fs  download=%fs  total=%fs\n",
           kernelName, N, dt1, dt2, dt3, dt1 + dt2 + dt3);

    CUDA_CHECK(cudaFree(bDev));
    CUDA_CHECK(cudaFree(aDev));
    CUDA_CHECK(cudaFreeHost(bHost));
    CUDA_CHECK(cudaFreeHost(aHost));
}

/// Selection of runs to use for profiling.
void profile() {
    runSync("fastKernel", fastKernel, 100'000'000);
    runAsync("fastKernel", fastKernel, 100'000'000, 10'000'000, 4);
    runSync("slowKernel", slowKernel, 100'000'000);
    runAsync("slowKernel", slowKernel, 100'000'000, 10'000'000, 4);
    cudaProfilerStop();
}

/// Selection of runs to use for benchmarking.
void runBenchmarks() {
    runSync("fastKernel", fastKernel, 1'000'000);
    runSync("fastKernel", fastKernel, 100'000'000);
    runAsync("fastKernel", fastKernel, 100'000'000, 100'000'000, 1);
    runAsync("fastKernel", fastKernel, 100'000'000, 10'000'000, 4);
    runAsync("fastKernel", fastKernel, 100'000'000, 10'000'000, 8);
    runAsync("fastKernel", fastKernel, 100'000'000, 1'000'000, 4);
    runAsync("fastKernel", fastKernel, 100'000'000, 1'000'000, 8);
    printf("\n");

    runSync("slowKernel", slowKernel, 1'000'000);
    runSync("slowKernel", slowKernel, 100'000'000);
    runAsync("slowKernel", slowKernel, 100'000'000, 100'000'000, 1);
    runAsync("slowKernel", slowKernel, 100'000'000, 10'000'000, 4);
    runAsync("slowKernel", slowKernel, 100'000'000, 10'000'000, 8);
    runAsync("slowKernel", slowKernel, 100'000'000, 1'000'000, 4);
    runAsync("slowKernel", slowKernel, 100'000'000, 1'000'000, 8);
}

int main() {
    // profile();
    runBenchmarks();
}
