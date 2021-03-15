#include "kernels.h"

/// Initialize reaction states.
__global__ void initializationKernel(
        short Sa, short Sb, short *x,
        float *t, int numSamples, int numIters)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numSamples)
        return;

    // Every sample starts with (Sa, Sb) at t=0.
    t[idx] = 0.f;
    x[0 * numIters * numSamples + idx] = Sa;
    x[1 * numIters * numSamples + idx] = Sb;
}


/// Reaction simulation. This kernel uses precomputed random uniform samples
/// (from 0 to 1) to compute up to `numIters` steps of the SSA algorithm. The
/// values of Sa and Sb are stored in `x`, the time values in `t`. Buffer
/// `iters` stores the number of performed iterations, and `isSampleDone`
/// whether or not the sample has reach the final state (t >= endTime).
__global__ void dimerizationKernel(
        int pass, const float *u,
        short *x, float *t, int *iters, char *isSampleDone,
        float endTime, int omega, int numIters, int numSamples)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rngOffset = blockIdx.x * blockDim.x * 2 * numIters + threadIdx.x;

    if (idx >= numSamples)
        return;

    // Reaction rates.
    const float k1 = 1;
    const float k2 = 1;
    const float k3 = 0.2f / omega;
    const float k4 = 20.f * omega;

    // State variables.
    float time;
    float Sa, Sb;

    // Load state.
    const bool continuation = pass > 0 && !isSampleDone[idx];
    if (continuation) {
        Sa   = x[0 * numIters * numSamples + (numIters - 1) * numSamples + idx];
        Sb   = x[1 * numIters * numSamples + (numIters - 1) * numSamples + idx];
        time = t[(numIters - 1) * numSamples + idx];
    } else {
        Sa   = x[0 * numIters * numSamples + idx];
        Sb   = x[1 * numIters * numSamples + idx];
        time = t[idx];
    }

    // Simulation loop.
    int iter;
    for (iter = 0; time < endTime && iter < numIters && (pass == 0 || !isSampleDone[idx]); ++iter) {
        // Accumulated propensities.
        const float a1 = k1*Sa;
        const float a2 = a1 + k2*Sb;
        const float a3 = a2 + k3*Sa*Sb;
        const float a4 = a3 + k4;
        const float a0 = a4;

        time -= 1 / a0 * log(u[rngOffset]);
        rngOffset += blockDim.x;

        const float beta = a0 * u[rngOffset];
        rngOffset += blockDim.x;

        const int d1 = (beta < a1);
        const int d2 = (beta >= a1 && beta < a2);
        const int d3 = (beta >= a2 && beta < a3);
        const int d4 = (beta >= a3);

        Sa += -d1 + d3;
        Sb += -d2 - d3 + d4;

        t[iter * numSamples + idx] = time;
        x[0 * numIters * numSamples + iter * numSamples + idx] = Sa;
        x[1 * numIters * numSamples + iter * numSamples + idx] = Sb;
    }

    // Termination markers.
    iters[idx]        = iter;
    isSampleDone[idx] = time >= endTime || isSampleDone[idx];
}

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

/// Overload for int -> int reduction.
__device__ int sumWarp(int a) {
    a += __shfl_down_sync(0xFFFFFFFF, a, 1);
    a += __shfl_down_sync(0xFFFFFFFF, a, 2);
    a += __shfl_down_sync(0xFFFFFFFF, a, 4);
    a += __shfl_down_sync(0xFFFFFFFF, a, 8);
    a += __shfl_down_sync(0xFFFFFFFF, a, 16);
    return a;
}

/// Overload for bool -> int reduction.
__device__ int sumBlockBool(int condition) {
    __shared__ int warpSums[32];
    int warpSum = __popc(__ballot_sync(0xFFFFFFFF, condition));
    if ((threadIdx.x & 31) == 0)
        warpSums[threadIdx.x >> 5] = warpSum;
    __syncthreads();
    int blockSum = 0;
    if (threadIdx.x < 32)
        blockSum = sumWarp(warpSums[threadIdx.x]);
    return blockSum;
}

/// Store the sum of the subarray isSampleDone[1024*b : 1024*b+1023] in blocksDoneCount[b].
__global__ void reduceIsDoneKernel(const char *isSampleDone, int *blocksDoneCount, int numSamples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int doneCount = sumBlockBool(idx < numSamples ? isSampleDone[idx] : 0);

    if (threadIdx.x == 0)
        blocksDoneCount[blockIdx.x] = doneCount;
}

/// Binary search index of target. Helper function for `binningKernel`.
__device__ int binarySearch(float target, int numIters, const float *tShifted, int numSamples)
{
    if (numIters == 0)
        return 0;

    int min = 0;
    int max = numIters - 1;

    while (true) {
        const int mid = (min + max) / 2;
        const float tmid = tShifted[mid * numSamples];

        if (target > tmid)
            min = mid + 1;
        else
            max = mid - 1;

        if (min > max)
            return min;
    }
}

/// Reduce the samples (x, t, iters) to bins.
/// In this implementation, each thread takes care of one pair (sampleIdx, binIdx).
__global__ void binningKernel(
        const short *x, const float *t, const int *iters, float *bins,
        float dtBin, int numBins, int numSamples, int numIters)
{
    const int sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;  // dim3(threads, 1, 1)
    const int binIdx    = blockIdx.y;                             // dim3(blocks, numBins, 1)

    int count = 0;
    float Sa = 0;
    float Sb = 0;

    if (sampleIdx < numSamples) {
        // We binary search through the time array to get the subtrajectory
        // that belongs to the current bin.
        const int N = iters[sampleIdx];
        int start = binarySearch( binIdx      * dtBin, N, t + sampleIdx, numSamples);
        int end   = binarySearch((binIdx + 1) * dtBin, N, t + sampleIdx, numSamples);

        // Sum up the values of Sa and Sb.
        count = end - start;
        for (int i = start; i < end; ++i) {
            Sa += x[0 * numSamples * numIters + i * numSamples + sampleIdx];
            Sb += x[1 * numSamples * numIters + i * numSamples + sampleIdx];
        }
    }

    // Block-wide reduction of partial sums.
    float countBlock = sumBlock((float)count);
    float SaBlock    = sumBlock(Sa);
    float SbBlock    = sumBlock(Sb);

    if (threadIdx.x == 0) {
        // Grid-wide reduction with atomicAdds.
        atomicAdd(&bins[0 * numBins + binIdx], SaBlock);
        atomicAdd(&bins[1 * numBins + binIdx], SbBlock);
        atomicAdd(&bins[2 * numBins + binIdx], countBlock);
    }
}

/// At the end, after counting all Sa, Sb and counts, we compute average <Sa>
/// and <Sb> for each bin.
__global__ void binNormalizationKernel(float *bins, int numBins)
{
    const int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx >= numBins)
        return;

    const int count = bins[2 * numBins + idx];
    const float invcount = count == 0 ? 0.f : 1.f / count;

    bins[0 * numBins + idx] *= invcount;
    bins[1 * numBins + idx] *= invcount;
}
