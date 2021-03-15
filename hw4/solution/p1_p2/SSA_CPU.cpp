#include "SSA_CPU.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>

void SSA_CPU::operator()()
{
	// number of reactions
    const int m = 4;
    // number of species
    const int n = 2;
    // initial conditions
    const int S0[n] = {4*omega,0};
	
	const int niters = static_cast<int>(tend*1000);
    
    double * const r48  = new double[2*niters*numSamples];
    double * const curT = new double[numSamples];
    double * const x0 = new double[numSamples];
    double * const x1 = new double[numSamples];

    // NUMA aware initialization (first touch)
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int s=0; s<numSamples; s++)
    {
        curT[s] = 0.0;
        x0[s] = 0.0;
        x1[s] = 0.0;
		for (int iter=0; iter<niters; iter++)
		{
			r48[2*s*niters + iter*2    ] = 0.;
			r48[2*s*niters + iter*2 + 1] = 0.;
		}
    }
	
	bool bNotDone = true;
	pass = 0;
	
	while (bNotDone)
	{
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i=0; i<niters*2*numSamples; i++)
			r48[i] = drand48();
		
		startTiming();
#ifdef _OPENMP
        int num_threads;
#pragma omp parallel
#pragma omp single
        {
            num_threads = omp_get_num_threads();
        }
#else
        const int num_threads = 1;
#endif 

        const int nbins = trajS1.size();
        double * const trajS1L = new double[nbins*num_threads];
        double * const trajS2L = new double[nbins*num_threads];
        int    * const ntrajL  = new int[nbins*num_threads];

        // NUMA aware initialization (first touch)
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int t=0; t <num_threads; ++t)
        {
            for(int b=0; b <nbins; ++b)
            {
                trajS1L[t*nbins+b] = 0.0;
                trajS2L[t*nbins+b] = 0.0;
                ntrajL[t*nbins+b] = 0;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for(int s = 0; s < numSamples; ++s)
		{
#ifdef _OPENMP
            const int thread_no = omp_get_thread_num();
#else
            const int thread_no = 0;
#endif
			// local version of trajectory bins
			const int nbins = trajS1.size();
			
			// init
			double time;
			double Sa;
            double Sb;
			if (pass>0 && bNotDone)
			{
				time = curT[s];
				Sa = x0[s];
				Sb = x1[s];
			}
			else
			{
				time = 0.0;
				Sa = S0[0];
				Sb = S0[1];
			}
			// propensities
			double a[m];
			
			// time stepping
			int iter = 0;
			while (time <= tend && iter<niters)
			{
			
                const double rnd0 = r48[2*s*niters + iter*2];
                const double rnd1 = r48[2*s*niters + iter*2+1];

				// store trajectory
				const int ib = static_cast<int>(time / bin_dt);     // 1 FLOP
				trajS1L[ib+thread_no*nbins] += Sa;
				trajS2L[ib+thread_no*nbins] += Sb;                  // 2 FLOP
				++ntrajL[ib+thread_no*nbins];
				
				// compute propensities
				a[0] = k[0]*Sa;
				a[1] = k[1]*Sb;                       
				a[2] = k[2]*Sa*Sb;                                  // 4 FLOP
				a[3] = k[3];
				
				// cumsum
				for (int i = 1; i < m; ++i) a[i] += a[i-1];         // 3 FLOP
				const double a0 = a[m-1];
				
				// time step
				// using the inverse sampling method, i.e. solving for tau in
				// r = \int_0^{\tau} a_0 e^{-t a_0} dt for r in ]0,1]
				time -= log1p(-rnd0) / a0;                          // 4 FLOP + RAND
				
				// choose reaction
				const double beta = a0*rnd1;                        // 1 FLOP + RAND
				
                const double d1 = ( beta < a[0]);
                const double d2 = ( beta >= a[0] && beta < a[1]);
                const double d3 = ( beta >= a[1] && beta < a[2]);
                const double d4 = ( beta >= a[2]);
				
                Sa += -d1 + d3; 
                Sb += -d2 - d3 + d4;                                // 5 FLOP
				iter++;
			}
			
			curT[s] = time;
			x0[s] = Sa;
			x1[s] = Sb;
			
			bNotDone = time <= tend && Sa!=0 && Sb!=0;
		}
        
        for(int t = 0; t < num_threads; ++t)
        {
            for (int i = 0; i < nbins; ++i) {
                trajS1[i] += trajS1L[i+t*nbins];
                trajS2[i] += trajS2L[i+t*nbins];
                ntraj[i] += ntrajL[i+t*nbins];                          // bins * (3 FLOP, 3 READ, 3 WRITE)         (assuming trajS1L, trajS2L, ntrajL) in cache
            }
        }

        delete[] ntrajL;
        delete[] trajS2L;
        delete[] trajS1L;    
		stopTiming();
		
		pass++;
	}
    
    delete[] x1;
    delete[] x0;
    delete[] curT;
    delete[] r48;    
    
    normalize_bins();
}

void SSA_CPU::normalize_bins()
{
    assert( trajS2.size() == trajS1.size() );
    assert( ntraj.size() == trajS1.size() );
    const int nbins = trajS1.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i < nbins; ++i)
    {
        trajS1[i]/=ntraj[i];
        trajS2[i]/=ntraj[i];        // 2 FLOP, 3 READ, 2 WRITE
    }
}

double SSA_CPU::getTransfers() const
{
    // We assume all local variables (including trajS1L, trajS2L, ntrajL) fit into cache.
    const int nbins = trajS1.size();
    const double time_step_iterations = std::accumulate(ntraj.begin(),ntraj.end(), 0.0); 
    return numSamples*nbins*4*sizeof(double)        // operator()   trajS1,2    1 READ, 1 WRITE each
        + numSamples*nbins*2*sizeof(int)            // operator()   ntraj       1 READ, 1 WRITE
        + 2 * time_step_iterations * sizeof(double) // operator()   read rnd
//            + nbins*4*sizeof(double)                  // normalize()  trajS1,2    1 READ, 1 WRITE 
//            + nbins*sizeof(int)                       //              ntraj       1 READ
        ;
}

double SSA_CPU::getFlops() const
{
    const unsigned int nbins = trajS1.size();
    assert(trajS2.size() == nbins);
    assert(ntraj.size() == nbins);
    const int flops_per_iter = 20;
    const double time_step_iterations = std::accumulate(ntraj.begin(),ntraj.end(), 0.0); 
    return flops_per_iter * time_step_iterations + nbins*3*numSamples*pass;
}
