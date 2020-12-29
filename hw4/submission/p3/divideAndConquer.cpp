/**********************************************************************
 * Code: UPC++ - Homework 3
 * Author: Vlachas Pantelis (pvlachas@ethz.ch)
 * ETH Zuerich - HPCSE II (Spring 2020)
 **********************************************************************/

// Loading necessary libraries
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include <numeric>

#include <upcxx/upcxx.hpp>
#include "factor.h"

// using ofstream constructors.
#include <iostream>
#include <fstream>  

#define NUM_FACTORS 240000

int main(int argc, char* argv[])
{
    // Measuring the total time needed.
    auto start = std::chrono::system_clock::now();

    // Intializing UPCXX
    upcxx::init();
    int rankId    = upcxx::rank_me();
    int rankCount = upcxx::rank_n();

    if (rankId == 0){
        printf("Approximating the value of PI with %d series coefficients.\n", NUM_FACTORS);
    }
    int nFactors = NUM_FACTORS;

    // TODO: Specify the variables factorsPerRank, initFactor and endFactor
    int factorsPerRank = nFactors / rankCount;
    int initFactor = rankId * factorsPerRank + 1;
    int endFactor  = (rankId+1) * factorsPerRank;

    // TODO: Initialize a global pointer factorArray.
    // Rank zero has to initialize the array.
    // Do not forget to finally broadcast the global pointer to all ranks from rank 0
    // so that all ranks have access to the same global adress space
    upcxx::global_ptr<double> factorArray;
    if (rankId == 0) factorArray = upcxx::new_array<double>(nFactors);

    auto future = upcxx::broadcast(&factorArray, 1, 0);
    future.wait();

    // TODO: After broadcasting the array, each rank needs to compute the portion of the factors it is assigned
    // and then \textbf{place} the result back to the \texttt{factorArray}.
    // Do not use RPCs in this question, use the \texttt{upcxx::rput} command.
    upcxx::future<> fut_all = upcxx::make_future();

    for (int k = initFactor; k <= endFactor; k++){
        double factor = FACTOR(k);
        auto fut = upcxx::rput(&factor, factorArray + k-1, 1);
        fut_all = upcxx::when_all(fut_all, fut);
    }

    auto end = std::chrono::system_clock::now();
    double rankTime = std::chrono::duration<double>(end-start).count();

    // Saving to a separate file for each rank
    std::stringstream filename;
    filename << "./Results/divide_and_conquer_time_rank_" << rankId << ".txt";
    std::string filenameStr = filename.str();
    std::ofstream outfile(filenameStr);
    outfile << rankTime << "\n" << std::endl;
    outfile.close();

    // TODO ?:
    fut_all.wait();
    upcxx::barrier();

    // TODO: Finally, rank zero needs to compute the approximate value $\tilde{\pi}$
    // and save it to the results file along with the total time.
    // \textbf{Downcast} the global pointer to a local one and use it to compute the final approximation.
    if (rankId == 0)
    {
        double pi_approx = 0.0;
        double *lptr = factorArray.local();
        pi_approx = 4*std::accumulate(lptr, lptr+nFactors, 0.0);

        // Reporting the result
        printf("PI approximate: %.17g\n", pi_approx);
        printf("PI: %.10f\n", M_PI);
        double error = abs(pi_approx - M_PI);
        printf("Absolute error: %.17g\n", error);

        // Computing the total time
        auto end = std::chrono::system_clock::now();
        double totalTime = std::chrono::duration<double>(end-start).count();
        printf("Total Running Time: %.17gs\n", totalTime);

        // Saving the result and the total time
        std::ofstream outfile ("./Results/divide_and_conquer.txt");
        outfile << pi_approx << "," << totalTime << "\n" << std::endl;
        outfile.close();
    }

    // Finalize UPCXX
    upcxx::finalize();

    return 0;
}
