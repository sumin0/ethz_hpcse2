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

  // TODO: TASK 2b)
  //    - Initialize SSA_CPU class
  //    - set the rates k1, k2, k3, and k4
  //    - run
  //    - get S1 and S2
  //    - calculate objective function

  SSA_CPU ssa(10, 2000, 5.0, 0.1);
  ssa.setRates(k1, k2, k3, k4);
  ssa();

  double S1 = ssa.getS1();
  double S2 = ssa.getS2();

  double sse = pow((S1 - 15.0), 2) + pow((S2 - 5.0), 2); // TODO

  k["F(x)"] = -sse;
}

#endif
