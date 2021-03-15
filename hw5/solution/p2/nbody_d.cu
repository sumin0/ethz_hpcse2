#include <cuda_runtime.h>

__global__ void computeForcesKernel(int N, const double3 *p, double3 *f) {
    extern __shared__ double3 pCache[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double3 myP = idx < N ? p[idx] : double3{0.0, 0.0, 0.0};
    double3 ftot{0.0, 0.0, 0.0};
    for (int offset = 0; offset < N; offset += blockDim.x) {
        // Copy to shared memory. Be careful not to exceed the total number of bodies.
        int blockSize = min((int)blockDim.x, N - offset);
        if (threadIdx.x < blockSize)
            pCache[threadIdx.x] = p[offset + threadIdx.x];

        // Wait till all threads are done preparing pCache. Even though warp
        // are synchronized (at least on the architecture that Piz Daint has),
        // different warps are not.
        __syncthreads();

        // Compute. again, be careful not to exceed to total number of bodies N.
        // (i goes from 0 to blockSize-1, not to blockDim.x-1).
        for (int i = 0; i < blockSize; ++i) {
            double dx = pCache[i].x - myP.x;
            double dy = pCache[i].y - myP.y;
            double dz = pCache[i].z - myP.z;
            double inv_r = rsqrt(1e-150 + dx * dx + dy * dy + dz * dz);
            double inv_rrr = inv_r * inv_r * inv_r;
            ftot.x += dx * inv_rrr;
            ftot.y += dy * inv_rrr;
            ftot.z += dz * inv_rrr;
        }

        // Synchronize again, otherwise one warp may start overwriting pCache
        // in the next step too early.
        __syncthreads();
    }
    f[idx] = ftot;
}

void computeForces(int N, const double3 *p, double3 *f) {
    constexpr int numThreads = 1024;
    int numBlocks = (N + numThreads - 1) / numThreads;
    size_t sharedMemorySize = numThreads * sizeof(double3);
    computeForcesKernel<<<numBlocks, numThreads, sharedMemorySize>>>(N, p, f);
}
