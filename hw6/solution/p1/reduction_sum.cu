#include "utils.h"
#include <cassert>
#include <algorithm>

/// Returns the sum of all values `a` within a warp,
/// with the correct answer returned only by the 0th thread of a warp.
__device__ double sumWarp(double a) {
    a += __shfl_down_sync(0xFFFFFFFF, a, 1);
    a += __shfl_down_sync(0xFFFFFFFF, a, 2);
    a += __shfl_down_sync(0xFFFFFFFF, a, 4);
    a += __shfl_down_sync(0xFFFFFFFF, a, 8);
    a += __shfl_down_sync(0xFFFFFFFF, a, 16);
    return a;
}

/// Returns the sum of all values `a` within a warp,
/// with the correct answer returned by all threads of a warp.
__device__ double sumWarpAll(double a) {
    a += __shfl_xor_sync(0xFFFFFFFF, a, 1);
    a += __shfl_xor_sync(0xFFFFFFFF, a, 2);
    a += __shfl_xor_sync(0xFFFFFFFF, a, 4);
    a += __shfl_xor_sync(0xFFFFFFFF, a, 8);
    a += __shfl_xor_sync(0xFFFFFFFF, a, 16);
    return a;
}


/// Returns the sum of all values `a` within a block,
/// with the correct answer returned only by the 0th thread of a block.
__device__ double sumBlock(double a) {
    __shared__ double warpSums[32];
    int warpIdx = threadIdx.x / warpSize;
    double warpSum = sumWarp(a);
    if (threadIdx.x % warpSize == 0)
        warpSums[warpIdx] = warpSum;
    __syncthreads();
    double blockSum = 0;
    if (warpIdx == 0)
        blockSum = sumWarp(warpSums[threadIdx.x]);
    return blockSum;
}

__global__ void sum1MKernel(const double *a, double *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double value = idx < N ? a[idx] : 0.0;
    double sum = sumBlock(value);
    if (threadIdx.x == 0)
        b[blockIdx.x] = sum;
}

/// Compute the sum of all values aDev[0]..aDev[N-1] for N <= 1024^2 and store the result to bDev[0].
void sum1M(const double *aDev, double *bDev, int N) {
    assert(N <= 1024 * 1024);
    int blocks = (N + 1023) / 1024;
    double *tmpDev;
    CUDA_CHECK(cudaMalloc(&tmpDev, blocks * sizeof(double)));

    sum1MKernel<<<blocks, 1024>>>(aDev, tmpDev, N);
    sum1MKernel<<<1, 1024>>>(tmpDev, bDev, blocks);

    CUDA_CHECK(cudaFree(tmpDev));
}


__global__ void sumVeryLargeKernel(const double *a, double *b, int N) {
    double value = 0.0;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x)
        value += a[idx];

    double sum = sumBlock(value);
    if (threadIdx.x == 0)
        b[blockIdx.x] = sum;
}

// Implementation of (1e) (note: it was not required to implement it).
void sumVeryLarge(const double *aDev, double *bDev, int N) {
    int blocks = std::min(1024, (N + 1023) / 1024);
    double *tmpDev;
    CUDA_CHECK(cudaMalloc(&tmpDev, blocks * sizeof(double)));

    sumVeryLargeKernel<<<blocks, 1024>>>(aDev, tmpDev, N);
    sumVeryLargeKernel<<<1, 1024>>>(aDev, bDev, N);

    CUDA_CHECK(cudaFree(tmpDev));
}


#include "reduction_sum.h"


int main() {
    testSmallSum(sumWarpTestKernel, sumWarpCheck, 32, 3);
    testSmallSum(sumWarpTestKernel, sumWarpCheck, 32, 32);
    testSmallSum(sumWarpTestKernel, sumWarpCheck, 32, 320);
    testSmallSum(sumWarpTestKernel, sumWarpCheck, 32, 1023123);
    printf("sumWarp OK.\n");

    testSmallSum(sumWarpAllTestKernel, sumWarpAllCheck, 1, 3);
    testSmallSum(sumWarpAllTestKernel, sumWarpAllCheck, 1, 32);
    testSmallSum(sumWarpAllTestKernel, sumWarpAllCheck, 1, 320);
    testSmallSum(sumWarpAllTestKernel, sumWarpAllCheck, 1, 1023123);
    printf("sumWarpAll OK.\n");

    testSmallSum(sumBlockTestKernel, sumBlockCheck, 1024, 32);
    testSmallSum(sumBlockTestKernel, sumBlockCheck, 1024, 1024);
    testSmallSum(sumBlockTestKernel, sumBlockCheck, 1024, 12341);
    testSmallSum(sumBlockTestKernel, sumBlockCheck, 1024, 1012311);
    printf("sumBlock OK.\n");

    testLargeSum("sum1M", sum1M, 32);
    testLargeSum("sum1M", sum1M, 1024);
    testLargeSum("sum1M", sum1M, 12341);
    testLargeSum("sum1M", sum1M, 1012311);
    printf("sum1M OK.\n");

    testLargeSum("sumVeryLarge", sumVeryLarge, 32);
    testLargeSum("sumVeryLarge", sumVeryLarge, 1024);
    testLargeSum("sumVeryLarge", sumVeryLarge, 12341);
    testLargeSum("sumVeryLarge", sumVeryLarge, 1012311);
    testLargeSum("sumVeryLarge", sumVeryLarge, 1001002003);
    printf("sumVeryLarge OK.\n");
}
