// Euler-Maruyama solver

#ifndef INCLUDED_EulerMaruyama_hpp_
#define INCLUDED_EulerMaruyama_hpp_

#define _USE_MATH_DEFINES

#include <cmath>
#include <functional>
#include "vec_op.hpp"
#include "MT.h"

using namespace std;

template <class T>
void EM(function<T(double, const T&)> hI // coeff. of dt
	,function<vector<T>(double, const T&)> gaI // coeff. of dW
	,double &t, T &x, double dt);

double Uniform(); // uniform random number (0,1)
double rand_normal(double mu, double sigma); // normal distribution


template <class T>
void EM(function<T(double, const T&)> hI, function<vector<T>(double, const T&)> gaI, double &t, T &x, double dt)
{
  T hIi = hI(t,x);
  vector<T> gaIi = gaI(t,x);
  
  size_t Wdim = gaIi.size();
  vector<double> dW(Wdim);

  for (double &dWe : dW) {
    dWe = rand_normal(0.,sqrt(dt));
  }

  t += dt;
  x += hIi * dt + vec_op::paralleldot(dW, gaIi);
}


double Uniform()
{
  return genrand_real3();
}

double rand_normal(double mu, double sigma)
{
  double z = sqrt(-2.*log(Uniform())) * sin(2.*M_PI*Uniform());
  return mu + sigma*z;
}

#endif


