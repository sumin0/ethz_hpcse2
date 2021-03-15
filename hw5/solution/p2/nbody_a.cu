#include <cuda_runtime.h>

__global__ void computeForcesKernel(int N, const double3 *p, double3 *f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    double3 ftot{0.0, 0.0, 0.0};
    for (int i = 0; i < N; ++i) {
        double dx = p[i].x - p[idx].x;
        double dy = p[i].y - p[idx].y;
        double dz = p[i].z - p[idx].z;
        double r = sqrt(1e-150 + dx * dx + dy * dy + dz * dz);
        double inv_r = 1 / r;
        ftot.x += dx * inv_r * inv_r * inv_r;
        ftot.y += dy * inv_r * inv_r * inv_r;
        ftot.z += dz * inv_r * inv_r * inv_r;
    }
    f[idx] = ftot;
}

void computeForces(int N, const double3 *p, double3 *f) {
    constexpr int numThreads = 1024;
    int numBlocks = (N + numThreads - 1) / numThreads;
    computeForcesKernel<<<numBlocks, numThreads>>>(N, p, f);
}
