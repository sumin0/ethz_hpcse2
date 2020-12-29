#include <cuda_runtime.h>

__global__ void computeForcesKernel(int N, const double3 *p, double3 *f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    // TODO: Copy the code from `nbody_a.cu` and fix the reduntant arithmetic operations.
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    
    for (int i = 0; i < N; ++i) {
        double dx = p[i].x - p[idx].x;
        double dy = p[i].y - p[idx].y;
        double dz = p[i].z - p[idx].z;
        // Instead of skipping the i == idx case, add 1e-150 to avoid division
        // by zero. (dx * inv_r will be exactly 0.0)
        double r = sqrt(1e-150 + dx * dx + dy * dy + dz * dz);
        double inv_r = 1 / r;
        double inv_r_3 = inv_r * inv_r * inv_r;
        fx += dx * inv_r_3;
        fy += dy * inv_r_3;
        fz += dz * inv_r_3;
    }
    f[idx] = double3{fx, fy, fz};
}

void computeForces(int N, const double3 *p, double3 *f) {
    constexpr int numThreads = 1024;
    int numBlocks = (N + numThreads - 1) / numThreads;
    computeForcesKernel<<<numBlocks, numThreads>>>(N, p, f);
}
