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
    CUDA_CHECK(cudaMalloc(&uDev, 2 * numIters * threads * blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&xDev, NUM_SPECIES * numSamples * numIters * sizeof(short)));
    CUDA_CHECK(cudaMalloc(&tDev, numSamples * numIters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&itersDev,      numSamples * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&isSampleDoneDev,      numSamples * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&perBlockDoneDev,      blocks * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&perBlockDoneHost, blocks * sizeof(int)));

    CUDA_CHECK(cudaMemset(itersDev,        0, numSamples * sizeof(int)));
    CUDA_CHECK(cudaMemset(isSampleDoneDev, 0, numSamples * sizeof(char)));

    //-------------------------------------------------------------
    float *trajSaDev;
    float *trajSbDev;
    int *trajNumStepsDev;
    CUDA_CHECK(cudaMalloc(&trajSaDev, numBins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&trajSbDev, numBins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&trajNumStepsDev, numBins * sizeof(int)));

    float *trajSaHost;
    float *trajSbHost;
    int *trajNumStepsHost;
    CUDA_CHECK(cudaMallocHost(&trajSaHost, numBins * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&trajSbHost, numBins * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&trajNumStepsHost, numBins * sizeof(int)));
    
    trajSa.reserve(numBins);
    trajSb.reserve(numBins);
    trajNumSteps.reserve(numBins);
    //-------------------------------------------------------------

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

        // TODO: Implement the binning mechanism.
        //       Use the sample trajectories xDev (which store Sa and Sb), tDev
        //       (trajectory time instances), itersDev (number of iterations
        //       for each sample, in this pass).
        //
        //       Allocate whatever memory you need, and implement the binning kernel (in kernels.cu) however you like.
        //       Make sure that the result is correct:
        //              a) data from all subtrajectories from all passes should be used,
        //              b) there must be no race condition when aggregating results.
        //       You can compare your results with HW4 solutions. See README
        //       for instructions on visualization.
        //
        //       The final result has to be stored in trajSa, trajSb and
        //       trajNumSteps (average Sa, average Sb, total number of samples,
        //       respectively).
        //
        //       Regarding the performance, take advantage of the GPU
        //       parallelism (i.e. do have multiple threads and multiple
        //       blocks).
        //
        
        binningKernel<<<blocks, threads>>>(trajSaDev, trajSbDev, trajNumStepsDev,
            xDev, tDev, itersDev,
            dtBin, numIters, numSamples);
        cudaDeviceSynchronize();
            
        // Check how many samples have finished.
        reduceIsDoneKernel<<<blocks, threads>>>(isSampleDoneDev, perBlockDoneDev, numSamples);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaMemcpy(perBlockDoneHost, perBlockDoneDev, blocks * sizeof(int), cudaMemcpyDeviceToHost));
        int remaining = numSamples - std::accumulate(perBlockDoneHost, perBlockDoneHost + blocks, (int)0);
        printf("Execution Loop %d. Remaining samples: %d/%d\n",
               pass, remaining, numSamples);

        if (remaining == 0)
            break;
    }


    // TODO: Averaging. Store the result in trajSa, trajSb and trajNumSteps.
    //          trajSa[k] = average Sa in the time bin k
    //          trajSb[k] = average Sb in the time bin k
    //          trajNumSteps[k] = number of steps (Sa, Sb, t) in the time bin k
    averageKernel<<<blocks, threads>>>(trajSaDev, trajSbDev, trajNumStepsDev, numBins);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(trajSaHost, trajSaDev, numBins * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(trajSbHost, trajSbDev, numBins * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(trajNumStepsHost, trajNumStepsDev, numBins * sizeof(int), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaDeviceSynchronize());

    for(int i=0; i<numBins; i++){
        trajSa.push_back(trajSaHost[i]);
        trajSb.push_back(trajSbHost[i]);
        trajNumSteps.push_back(trajNumStepsHost[i]);
    }


    CURAND_CHECK(curandDestroyGenerator(generator));

    // TODO: Deallocate all extra buffers you allocated.

    CUDA_CHECK(cudaFreeHost(perBlockDoneHost));
    CUDA_CHECK(cudaFree(perBlockDoneDev));
    CUDA_CHECK(cudaFree(isSampleDoneDev));
    CUDA_CHECK(cudaFree(itersDev));
    CUDA_CHECK(cudaFree(tDev));
    CUDA_CHECK(cudaFree(xDev));
    CUDA_CHECK(cudaFree(uDev));

    CUDA_CHECK(cudaFree(trajSaDev));
    CUDA_CHECK(cudaFree(trajSbDev));
    CUDA_CHECK(cudaFree(trajNumStepsDev));

    CUDA_CHECK(cudaFreeHost(trajSaHost));
    CUDA_CHECK(cudaFreeHost(trajSbHost));
    CUDA_CHECK(cudaFreeHost(trajNumStepsHost));
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
