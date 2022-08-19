#include <iostream>
#include "source/vec_op.hpp"

using namespace std;

int main()
{
  vector<double> v{1,2,3};
  vector<vector<double>> m{{3,4},{5,6},{7,8}};

  vector<double> vm = vec_op::dot(v,m);

  for (double &e : vm) {
    cout << e << ' ';
  }
  cout << endl;
}
