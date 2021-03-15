#include "utils.h"
#include <cassert>
#include <limits>

struct Pair {
    double max;
    int idx;
};

/*
__device__ Pair shfl_xor_sync(Pair value, unsigned delta) {
    return Pair{
        __shfl_xor_sync(0xFFFFFFFF, value.max, delta),
        __shfl_xor_sync(0xFFFFFFFF, value.idx, delta),
    };
}

__device__ Pair argMaxOp(Pair a, Pair b) {
    return a.max > b.max ? a : b;
}

__device__ Pair argMaxWarp(double a) {
    Pair t{a, (int)threadIdx.x & 31};
    t = argMaxOp(t, shfl_xor_sync(t, 1));
    t = argMaxOp(t, shfl_xor_sync(t, 2));
    t = argMaxOp(t, shfl_xor_sync(t, 4));
    t = argMaxOp(t, shfl_xor_sync(t, 8));
    t = argMaxOp(t, shfl_xor_sync(t, 16));
    return t;
}
*/

/// Find the maximum value `a` among all warps and return {max value, index of
/// the max}. The result must be correct on at least the 0th thread of each warp.
__device__ Pair argMaxWarp(double a) {
    double t = a;  // max;
    t = max(t, __shfl_xor_sync(0xFFFFFFFF, t, 1));
    t = max(t, __shfl_xor_sync(0xFFFFFFFF, t, 2));
    t = max(t, __shfl_xor_sync(0xFFFFFFFF, t, 4));
    t = max(t, __shfl_xor_sync(0xFFFFFFFF, t, 8));
    t = max(t, __shfl_xor_sync(0xFFFFFFFF, t, 16));
    unsigned ballot = __ballot_sync(0xFFFFFFFF, a == t);
    int idx = __ffs(ballot) - 1;
    return {t, idx};
}



/// Second stage of argMaxBlock.
/// Returns {max value of a.value, thread index 0..1023 with the max value}.
__device__ Pair argMaxWarp(Pair a) {
    Pair partial = argMaxWarp(a.max);
    double maxBlock = partial.max;
    int idxBlock = __shfl_sync(0xFFFFFFFF, a.idx, partial.idx);
    return {maxBlock, idxBlock};
}

/// Returns the argmax of all values `a` within a block,
/// with the correct answer returned at least by the 0th thread of a block.
__device__ Pair argMaxBlock(double a) {
    __shared__ Pair partials[32];
    int warpIdx = threadIdx.x / warpSize;
    Pair partial = argMaxWarp(a);
    if (threadIdx.x % warpSize == 0) {
        partials[warpIdx].max = partial.max;
        partials[warpIdx].idx = partial.idx + threadIdx.x;
    }
    __syncthreads();
    if (warpIdx == 0)
        return argMaxWarp(partials[threadIdx.x]);
    return Pair{0, 0};
}

/// Returns the argmax of all values `a` within a block,
/// with the correct answer returned only by the 0th thread of a block.
__device__ Pair argMaxBlock(Pair a) {
    __shared__ Pair partials[32];
    int warpIdx = threadIdx.x / warpSize;
    Pair partial = argMaxWarp(a);
    if (threadIdx.x % warpSize == 0)
        partials[warpIdx] = partial;
    __syncthreads();
    if (warpIdx == 0)
        return argMaxWarp(partials[threadIdx.x]);
    return Pair{-1, -1};
}

__global__ void argMax1MKernel1(const double *a, Pair *tmp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double value = idx < N ? a[idx] : 0.0;
    Pair result = argMaxBlock(value);
    if (threadIdx.x == 0) {
        tmp[blockIdx.x].max = result.max;
        tmp[blockIdx.x].idx = result.idx + blockIdx.x * blockDim.x;
    }
}
__global__ void argMax1MKernel2(const Pair *tmp, Pair *b, int numBlocks) {
    int idx = threadIdx.x;
    Pair partial = idx < numBlocks ? tmp[idx] : Pair{-1e100, -1};  // -infty should be here.
    Pair result = argMaxBlock(partial);
    if (threadIdx.x == 0)
        *b = result;
}

void argMax1M(const double *aDev, Pair *bDev, int N) {
    assert(N <= 1024 * 1024);
    int blocks = (N + 1023) / 1024;
    Pair *tmpDev;
    CUDA_CHECK(cudaMalloc(&tmpDev, blocks * sizeof(double)));

    argMax1MKernel1<<<blocks, 1024>>>(aDev, tmpDev, N);
    argMax1MKernel2<<<1, 1024>>>(tmpDev, bDev, blocks);

    CUDA_CHECK(cudaFree(tmpDev));
}

#include "reduction_argmax.h"

int main() {
    testSmallArgMax(argMaxWarpTestKernel, argMaxWarpCheck, 32, 3);
    testSmallArgMax(argMaxWarpTestKernel, argMaxWarpCheck, 32, 32);
    testSmallArgMax(argMaxWarpTestKernel, argMaxWarpCheck, 32, 320);
    testSmallArgMax(argMaxWarpTestKernel, argMaxWarpCheck, 32, 1023123);
    printf("argMaxWarp OK.\n");

    testSmallArgMax(argMaxBlockTestKernel, argMaxBlockCheck, 1024, 32);
    testSmallArgMax(argMaxBlockTestKernel, argMaxBlockCheck, 1024, 1024);
    testSmallArgMax(argMaxBlockTestKernel, argMaxBlockCheck, 1024, 12341);
    testSmallArgMax(argMaxBlockTestKernel, argMaxBlockCheck, 1024, 1012311);
    printf("argMaxBlock OK.\n");

    testLargeArgMax("argMax1M", argMax1M, 32);
    testLargeArgMax("argMax1M", argMax1M, 1024);
    testLargeArgMax("argMax1M", argMax1M, 12341);
    testLargeArgMax("argMax1M", argMax1M, 1012311);
    printf("argMax1M OK.\n");
}

