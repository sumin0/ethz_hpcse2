#include "utils.h"
#include <cassert>
#include <algorithm>

__device__ double sumWarpAll(double a) {
    // TODO: 1.a) Compute sum of all values within a warp.
    //            Only the threads with threadIdx.x % warpSize == 0 have to
    //            return the correct result.
    //            (although this function operates only on a single warp, it
    //            will be called with many threads for testing)

    // unsigned int laneId = threadIdx.x & 0x1f;
    // unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = a;

    for(unsigned int s = warpSize/2; s>0; s/=2){
        sum += __shfl_xor_sync(0xffffffff, sum, s, warpSize); // optional: warp-level reduce-all operation
        // sum += __shfl_down_sync(0xffffffff, sum, s, warpSize);
    }

    return sum;
}

/// Returns the sum of all values `a` within a warp,
/// with the correct answer returned only by the 0th thread of a warp.
__device__ double sumWarp(double a) {
    // TODO: 1.a) Compute sum of all values within a warp.
    //            Only the threads with threadIdx.x % warpSize == 0 have to
    //            return the correct result.
    //            (although this function operates only on a single warp, it
    //            will be called with many threads for testing)

    // unsigned int laneId = threadIdx.x & 0x1f;
    // unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = a;

    for(unsigned int s = warpSize/2; s>0; s/=2){
        // sum += __shfl_xor_sync(0xffffffff, sum, s, warpSize); // optional: warp-level reduce-all operation
        sum += __shfl_down_sync(0xffffffff, sum, s, warpSize);
    }

    return sum;
}

/// Returns the sum of all values `a` within a block,
/// with the correct answer returned only by the 0th thread of a block.
__device__ double sumBlock(double a) {
    // TODO: 1.c) Compute the sum of values `a` for all threads within a block.
    //            Only threadIdx.x == 0 has to return the correct result.
    // NOTE: For 1.c) implement either this or `argMaxBlock`!
    __shared__ double warp_sums[32];
    int tid = threadIdx.x;
    
    double warp_sum = sumWarp(a);

    if (tid % 32 == 0){
        unsigned int wid = tid / 32;
        warp_sums[wid] = warp_sum;
    }
    __syncthreads();

    if (tid < 32){
        warp_sum = sumWarp(warp_sums[tid]);
    }
    __syncthreads();

    return warp_sum;

//    for (unsigned int s=16; s>0; s/=2)
//    {
//        if (tid < s) warp_sums[tid] += warp_sums[tid + s];
//
//        __syncthreads();
//    }
//
//    return warp_sums[0];
}

__global__ void sum1M_helper(const double *a, double *b, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double block_sum = sumBlock(idx < N ? a[idx] : 0.0);
    if (threadIdx.x == 0){
        b[blockIdx.x] = block_sum;
    }
    /*
    cudaDeviceSynchronize();

    if (blockIdx.x == 0){
        block_sum = sumBlock(threadIdx.x < numBlocks ? b[threadIdx.x] : 0.0);
        __syncthreads();

        if (threadIdx.x == 0) b[0] = block_sum;
    }
    */

}

/// Compute the sum of all values aDev[0]..aDev[N-1] for N <= 1024^2 and store the result to bDev[0].
void sum1M(const double *aDev, double *bDev, int N) {
    assert(N <= 1024 * 1024);
    // TODO: 1.d) Implement either this or `argMax1M`.
    //            Avoid copying any data back to the host.
    //            Hint: The solution requires more CUDA operations than just
    //            calling a single kernel. Feel free to use whatever you find
    //            necessary.
    
    int numBlocks = (N + 1024 -1 ) / 1024; // < 1024
    sum1M_helper<<<numBlocks, 1024>>>(aDev, bDev, N);

    cudaDeviceSynchronize();

    sum1M_helper<<<1, 1024>>>(bDev, bDev, 1024);

}


#include "reduction_sum.h"


int main() {
    testSmallSum(sumWarpTestKernel, sumWarpCheck, 32, 3);
    testSmallSum(sumWarpTestKernel, sumWarpCheck, 32, 32);
    testSmallSum(sumWarpTestKernel, sumWarpCheck, 32, 320);
    testSmallSum(sumWarpTestKernel, sumWarpCheck, 32, 1023123);
    printf("sumWarp OK.\n");

    
    // OPTIONAL: 1a reduce-all. In case you want to try to implement it,
    //           implement a global function `__device__ double sumWarpAll(double x)`,
    //           and comment out sumWarpAll* functions in utils.h.
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
}
