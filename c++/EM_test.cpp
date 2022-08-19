#include <iostream>
#include <fstream>
#include "source/EulerMaruyama.hpp"

using namespace std;

vector<vector<double>> hI(double t, const vector<vector<double>> &x); // coeff. of dt
vector<vector<vector<double>>> gaI(double t, const vector<vector<double>> &x); // coeff. of dW

int main()
{
  init_genrand((unsigned)time(NULL));
  
  double dt = 0.01;
  double t = 0;
  vector<vector<double>> x(10,vector<double>{1,0});
  double tf = 10;

  ofstream ofs("EM_test.dat");
  
  while (t<tf) {
    ofs << t << ' ';
    for (vector<double> &xi : x) {
      ofs << xi[0] << ' ';
    }
    ofs << endl;

    EM<vector<vector<double>>>(hI,gaI,t,x,dt);
  }
}

vector<vector<double>> hI(double t, const vector<vector<double>> &x)
{
  vector<vector<double>> hI = x;

  for (size_t i = 0, size = hI.size(); i < size; i++) {
    hI[i][0] = x[i][1];
    hI[i][1] = -x[i][0];
  }

  return hI;
}

vector<vector<vector<double>>> gaI(double t, const vector<vector<double>> &x)
{
  vector<vector<vector<double>>> gaI(10,x);

  for (size_t a = 0, asize = gaI.size(); a < asize; a++) {
    for (size_t i = 0, isize = gaI[a].size(); i < isize; i++) {
      if (a == i) {
	gaI[a][i][0] = 0.1;
      } else {
	gaI[a][i][0] = 0;
      }
      gaI[a][i][1] = 0;
    }
  }
  
  return gaI;
}
