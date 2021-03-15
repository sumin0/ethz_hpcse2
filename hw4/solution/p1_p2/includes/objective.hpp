#ifndef _DIRECT_HPP_
#define _DIRECT_HPP_

#include "korali.hpp"
#include "SSA_CPU.hpp"

void direct(korali::Sample& k)
{
  double k1 = k["Parameters"][0];
  double k2 = k["Parameters"][1];
  double k3 = k["Parameters"][2];
  double k4 = k["Parameters"][3];
  
  int rnd_seed = 12;
  srand48(rnd_seed);
  
  int omega = 10;
  int num_samples = 2000;
  double T = 5.0;
  double dt = 0.1;

  SSA_CPU ssa(omega, num_samples, T, dt);
  ssa.setRates(k1, k2, k3, k4);
  ssa();
  
  double s1 = ssa.getS1();
  double s2 = ssa.getS2();

  double error2 = (s1-15.)*(s1-15.)+(s2-5.)*(s2-5.);
  k["F(x)"] = -error2;
}


#endif
