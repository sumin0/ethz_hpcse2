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
        double inv_r = rsqrt(1e-150 + dx * dx + dy * dy + dz * dz);
        double inv_rrr = inv_r * inv_r * inv_r;
        ftot.x += dx * inv_rrr;
        ftot.y += dy * inv_rrr;
        ftot.z += dz * inv_rrr;
    }
    f[idx] = ftot;
}

void computeForces(int N, const double3 *p, double3 *f) {
    constexpr int numThreads = 1024;
    int numBlocks = (N + numThreads - 1) / numThreads;
    computeForcesKernel<<<numBlocks, numThreads>>>(N, p, f);
}
