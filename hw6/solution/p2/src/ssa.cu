#include "ssa.h"
#include "kernels.h"
#include "utils.h"

#include <curand.h>

#include <algorithm>
#include <fstream>
#include <numeric>

static constexpr int NUM_SPECIES = 2;

void SSA_GPU::run()
{
    // Problem size.
    const int numIters = numItersPerPass;

    const int threads = 1024;  // Do not change this.
    const int blocks = (numSamples + threads - 1) / threads;
    if (blocks > 65536) {
        fprintf(stderr, "Number of samples larger than 64M not supported (block limit reached).\n");
        exit(1);
    }
    const long long memoryEstimate = 2ULL * numIters * threads * blocks * sizeof(float);
    printf("SSA_GPU  numItersPerPass: %d  numSamples: %d  approx required memory: ~%.1fMB\n",
           numIters, numSamples, memoryEstimate / 1024. / 1024.);

    float *uDev;            // Uniform random values vector.
    short *xDev;            // Species vector.
    float *tDev;            // Time vector.
    int *itersDev;          // Num iterations in simulation loop.
    char *isSampleDoneDev;  // isSampleDoneDev[sampleIdx] = 0 or 1.
    int *perBlockDoneDev;   // perBlockDoneDev[blockIdx] = number of samples done in the block blockIdx.
    int *perBlockDoneHost;  // A host copy.
    float *binsDev;         // Binned results (total s0..., total s1..., total count...).
    float *binsHost;        // A host copy.
    CUDA_CHECK(cudaMalloc(&uDev, 2 * numIters * threads * blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&xDev, NUM_SPECIES * numSamples * numIters * sizeof(short)));
    CUDA_CHECK(cudaMalloc(&tDev, numSamples * numIters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&itersDev,      numSamples * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&isSampleDoneDev,      numSamples * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&perBlockDoneDev,      blocks * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&perBlockDoneHost, blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&binsDev,      3 * numBins * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&binsHost, 3 * numBins * sizeof(float)));

    CUDA_CHECK(cudaMemset(itersDev,        0, numSamples * sizeof(int)));
    CUDA_CHECK(cudaMemset(isSampleDoneDev, 0, numSamples * sizeof(char)));
    CUDA_CHECK(cudaMemset(binsDev,         0, 3 * numBins * sizeof(float)));

    curandGenerator_t generator;

    // Setup RNG.
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, seed));

    printf("===== DIMERIZATION =====\n");
    const short Sa = 4 * omega;
    const short Sb = 0;

    // Set initial values of Sa, Sb and sample time (t=0 initially).
    initializationKernel<<<blocks, threads>>>(Sa, Sb, xDev, tDev, numSamples, numIters);

    // Evaluate samples in passes of `numIters` iterations.
    // (We cannot predict the total number of iterations that a sample might
    // need, so we allocate in advance buffers sufficient for `numIters` iterations.)
    for (int pass = 0; pass < 1000; ++pass) {
        // Generate random numbers needed by all threads for `numIters` iterations.
        CURAND_CHECK(curandGenerateUniform(generator, uDev, 2 * numIters * threads * blocks));

        // Evaluate up to `numIters` iterations.
        dimerizationKernel<<<blocks, threads>>>(
                pass, uDev, xDev, tDev, itersDev, isSampleDoneDev,
                endTime, omega, numIters, numSamples);

        // Bin subtrajectories.
        binningKernel<<<dim3(blocks, numBins, 1), dim3(threads, 1, 1)>>>(
                xDev, tDev, itersDev, binsDev, dtBin, numBins, numSamples, numIters);

        // Check how many samples have finished.
        reduceIsDoneKernel<<<blocks, threads>>>(isSampleDoneDev, perBlockDoneDev, numSamples);
        CUDA_CHECK(cudaMemcpy(perBlockDoneHost, perBlockDoneDev, blocks * sizeof(int), cudaMemcpyDeviceToHost));
        int remaining = numSamples - std::accumulate(perBlockDoneHost, perBlockDoneHost + blocks, (int)0);
        printf("Execution Loop %d. Remaining samples: %d/%d\n",
               pass, remaining, numSamples);

        if (remaining == 0)
            break;
    }

    // Final step of binning. Compute averages and copy from device to host.
    binNormalizationKernel<<<(numBins + threads - 1) / threads, threads>>>(binsDev, numBins);
    CUDA_CHECK(cudaMemcpy(binsHost, binsDev, 3 * numBins * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < numBins; ++i) {
        trajSa[i] = binsHost[0 * numBins + i];
        trajSb[i] = binsHost[1 * numBins + i];
        trajNumSteps[i] = static_cast<int>(binsHost[2 * numBins + i]);
    }

    CURAND_CHECK(curandDestroyGenerator(generator));

    CUDA_CHECK(cudaFreeHost(perBlockDoneHost));
    CUDA_CHECK(cudaFree(perBlockDoneDev));
    CUDA_CHECK(cudaFree(isSampleDoneDev));
    CUDA_CHECK(cudaFreeHost(binsHost));
    CUDA_CHECK(cudaFree(binsDev));
    CUDA_CHECK(cudaFree(itersDev));
    CUDA_CHECK(cudaFree(tDev));
    CUDA_CHECK(cudaFree(xDev));
    CUDA_CHECK(cudaFree(uDev));
}

void SSA_GPU::dumpTrajectoryToFile(const char *filename) {
    std::ofstream outfile(filename);

    int totalevals = 0;
    for (int i = 0; i < (int)trajSa.size(); ++i) {
        // Must rescale wrt omega.
        outfile << i*dtBin+dtBin/2 << ' '
                << (trajSa[i] / omega) << ' '
                << (trajSb[i] / omega) << '\n';
        totalevals += trajNumSteps[i];
    }
    printf("Average number of time steps per sample: %f\n", double(totalevals) / numSamples);
}
