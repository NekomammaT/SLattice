#include <iostream>
#include <fstream>
#include <sys/time.h>
#include "source/EulerMaruyama.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

const int LSIZE = 8+1;
const double KMAX = 3.*2.*M_PI;
const double KMIN = 2.*M_PI/3./LSIZE;
const double KS = KMAX;
vector<vector<double>> XVEC;
int XVOL;

vector<double> hI(double t, const vector<double> &Wx);
vector<vector<double>> gaI(double t, const vector<double> &Wx);

int main()
{
  init_genrand((unsigned)time(NULL));
  
  // ---------- start stop watch ----------
  struct timeval tv;
  struct timezone tz;
  double before, after;
  
  gettimeofday(&tv, &tz);
  before = (double)tv.tv_sec + (double)tv.tv_usec * 1.e-6;
  // --------------------------------------

#ifdef _OPENMP
  cout << "OpenMP : Enabled (Max # of threads = " << omp_get_max_threads() << ")" << endl;
#endif
  
  
  for (int i=0; i<LSIZE; i++) {
    for (int j=0; j<LSIZE; j++) {
      XVEC.push_back({i-(LSIZE-1)/2.,j-(LSIZE-1)/2.,0.});
    }
  }
  XVOL = XVEC.size();

  vector<vector<double>> Cmatrix(XVOL,vector<double>(XVOL,0));
  int Nsample = 1000, counter = 0;

  //#ifdef _OPENMP
  //#pragma omp parallel for
  //#endif
  for (int sample=0; sample<Nsample; sample++) {
    double t = 0;
    vector<double> Wx(XVOL,0);
    
    EM<vector<double>>(hI,gaI,t,Wx,1);
    
    for (int i=0; i<XVOL; i++) {
      for (int j=0; j<XVOL; j++) {
	//#ifdef _OPENMP
	//#pragma omp atomic
	//#endif
	Cmatrix[i][j] += Wx[i]*Wx[j];
      }
    }

    //#ifdef _OPENMP
    //#pragma omp critical
    //#endif
    {
      counter++;
      if (counter >= Nsample/100) {
	cout << ">" << flush;
	counter = 0;
      }
    }
  }
  cout << endl;

  Cmatrix *= 1./Nsample;

  /*
  for (int i=0; i<XVOL; i++) {
    for (int j=0; j<XVOL; j++) {
      cout << Cmatrix[i][j] << ' ';
    }
    cout << endl;
  }

  cout << endl;
  */

  double DeltaC=0;
  double Cnorm = 0;
  
  for (int i=0; i<XVOL; i++) {
    for (int j=0; j<XVOL; j++) {
      double Cij;
      if (i==j) {
	Cij = 1;
      } else {
	vector<double> rvec = XVEC[i] - XVEC[j];
	double r = sqrt(vec_op::dot(rvec,rvec));
	Cij = sin(KS*r)/KS/r;
      }
      
      //cout << Cij << ' ';
      
      DeltaC += (Cmatrix[i][j]-Cij)*(Cmatrix[i][j]-Cij);
      Cnorm += Cij*Cij;
    }
    //cout << endl;
  }

  //cout << endl;

  cout << "DeltaC/C = " << sqrt(DeltaC/Cnorm) << endl;

  


  // ---------- stop stop watch ----------
  gettimeofday(&tv, &tz);
  after = (double)tv.tv_sec + (double)tv.tv_usec * 1.e-6;
  cout << after - before << " sec." << endl;
  // -------------------------------------
}


vector<double> hI(double t, const vector<double> &Wx)
{
  vector<double> hI = Wx;

  for (double &hIe : hI) {
    hIe = 0;
  }

  return hI;
}

vector<vector<double>> gaI(double t, const vector<double> &Wx)
{
  int Ntheta = ceil(KS*LSIZE/2.); //ceil(M_PI*KS/KMIN);
  double Dtheta = M_PI/Ntheta;
  vector<double> thetai(Ntheta);
  for (size_t i = 0; i < Ntheta; i++) {
    thetai[i] = (i+1./2)*Dtheta;
  }

  function<int(double)> Nphi = [](double theta){
    return ceil(sin(theta)*KS*LSIZE); //ceil(2.*M_PI*sin(theta)*KS/KMIN);
  };
  function<double(double)> Dphi = [Nphi](double theta){
    return 2.*M_PI/Nphi(theta);
  };
  function<vector<double>(double)> phii = [Nphi,Dphi](double theta){
    vector<double> phii(Nphi(theta));
    for (size_t i = 0; i < Nphi(theta); i++) {
      phii[i] = (i+1./2)*Dphi(theta);
    }
    return phii;
  };

  vector<vector<double>> OmegaList;
  for (size_t i = 0; i < Ntheta; i++) {
    double theta_tmp = thetai[i];
    
    for (size_t j = 0; j < Nphi(theta_tmp); j++) {
      double phi_tmp = phii(theta_tmp)[j];
      OmegaList.push_back(vector<double>{theta_tmp,phi_tmp});
    }
  }
  int Omegavol = OmegaList.size();

  vector<vector<double>> Omegavec;
  for (vector<double> &Omegai : OmegaList) {
    double theta_tmp = Omegai[0];
    double phi_tmp = Omegai[1];
    Omegavec.push_back(vector<double>{sin(theta_tmp)*cos(phi_tmp),
				      sin(theta_tmp)*sin(phi_tmp),
				      cos(phi_tmp)});
  }

  vector<vector<double>> gaI(Omegavol,vector<double>(XVOL));
  for (int Onum = 0; Onum<Omegavol; Onum++) {
    for (int xnum = 0; xnum<XVOL; xnum++) {
      double ksrmu = KS * vec_op::dot(Omegavec[Onum],XVEC[xnum]);
      double theta_tmp = OmegaList[Onum][0];
      double DOmega = sin(theta_tmp)*Dtheta*Dphi(theta_tmp);
      gaI[Onum][xnum] = (cos(ksrmu)-sin(ksrmu))*sqrt(DOmega)/2./sqrt(M_PI);
    }
  }
  
  return gaI;
}
