#include <cuda_runtime.h>

__global__ void computeForcesKernel(int N, const double3 *p, double3 *f) {
    // TODO: Copy the code from `nbody_c.cu` and utilize shared memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    extern __shared__ double3 s_p[];

    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;

    double3 ref = (idx < N) ? p[idx] : double3{0.0, 0.0, 0.0};
    
    for (int b = 0; b < N; b += blockDim.x){
        if(b+tid < N) s_p[tid] = p[b + tid];
        __syncthreads();

        for (int i = 0; i < min(blockDim.x, N-b); ++i) {
            double dx = s_p[i].x - ref.x;
            double dy = s_p[i].y - ref.y;
            double dz = s_p[i].z - ref.z;
            // Instead of skipping the i == idx case, add 1e-150 to avoid division
            // by zero. (dx * inv_r will be exactly 0.0)
            double inv_r = rsqrt(1e-150 + dx * dx + dy * dy + dz * dz);
            double inv_r_3 = inv_r * inv_r * inv_r;
            fx += dx * inv_r_3;
            fy += dy * inv_r_3;
            fz += dz * inv_r_3;
        }
    }
    if(idx < N) f[idx] = double3{fx, fy, fz};
}

void computeForces(int N, const double3 *p, double3 *f) {
    constexpr int numThreads = 1024;
    int numBlocks = (N + numThreads - 1) / numThreads;

    // TODO: Set the required shared memory size.
    //       Don't bother with checking errors here.
    size_t shmSize = numThreads * sizeof(double3);
    computeForcesKernel<<<numBlocks, numThreads, shmSize>>>(N, p, f);
}
