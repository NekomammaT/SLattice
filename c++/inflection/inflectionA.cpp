#include <iostream>
#include <iomanip>
#include <fstream>
#include <sys/time.h>
#include "../source/EulerMaruyama.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

const int LSIZE = 32+1;
const double KMAX = 3.*2.*M_PI;
const double KMIN = 2.*M_PI/3./LSIZE;
vector<vector<double>> XVEC;
int XVOL;

const string MODEL("inflectionA");
const double AW = 0.02;
const double BW = 1;
const double CW = 0.04;
const double DW = 0;
const double GW = 3.076278e-2;
const double RW = 0.7071067;
const double CALV = 1000;
const double W0 = 12.35;
const double CUP = 0.0382;

const double DT = 1e-2;
const double QI = 9;
//const double PI = -2.37409e-7;

vector<vector<double>> hI(double t, const vector<vector<double>> &QPfield);
vector<vector<vector<double>>> gaI(double t, const vector<vector<double>> &QPfield);
double V(double Q);
double Vp(double Q);
double rho(vector<double> QP);
double H(vector<double> QP);


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

  ofstream Lfile(MODEL + string("_Lattice.dat"));
  for (int i=0; i<XVOL; i++) {
    for (double &xyz : XVEC[i]) {
      Lfile << xyz << ' ';
    }
    Lfile << endl;
  }
  
  double t = 0;
  double tf = log(KMAX/KMIN);

  double Pi = -Vp(QI)/sqrt(3*V(QI));
  vector<vector<double>> QPfield(XVOL,vector<double>{QI,Pi});

  ofstream Qfile(MODEL + string("_Q.dat")), Pfile(MODEL + string("_P.dat")),
    Hfile(MODEL + string("_H.dat")), zfile(MODEL + "_zeta.dat");

  double counter = 0;
  while (t<tf) {
    double rhobar = 0;
    
    Qfile << setprecision(15) << t << ' ';
    Pfile << setprecision(15) << t << ' ';
    Hfile << setprecision(15) << t << ' ';
    for (vector<double> &QP : QPfield) {
      Qfile << QP[0] << ' ';
      Pfile << QP[1] << ' ';
      Hfile << H(QP) << ' ';

      rhobar += rho(QP);
    }
    Qfile << endl;
    Pfile << endl;
    Hfile << endl;

    rhobar /= XVOL;
    zfile << setprecision(15) << t << ' ';
    for (vector<double> &QP : QPfield) {
      zfile << (rho(QP)-rhobar)/3./QP[1]/QP[1] << ' ';
    }
    zfile << endl;

    EM<vector<vector<double>>>(hI,gaI,t,QPfield,DT);

    if (t-counter >= tf/100) {
      cout << ">" << flush;
      counter += tf/100;
    }
  }
  cout << endl;

  

  // ---------- stop stop watch ----------
  gettimeofday(&tv, &tz);
  after = (double)tv.tv_sec + (double)tv.tv_usec * 1.e-6;
  cout << after - before << " sec." << endl;
  // -------------------------------------
}


double V(double Q) {
  return W0*W0/CALV/CALV/CALV * ( CUP/pow(CALV,1./3) + AW/(exp(Q/sqrt(3))-BW) - CW/exp(Q/sqrt(3))
				  + exp(2*Q/sqrt(3))/CALV*(DW-GW/(RW*exp(sqrt(3)*Q)/CALV+1)) );
}

double Vp(double Q) {
  return exp(-Q/sqrt(3)) * ( CW - AW*exp(2*Q/sqrt(3))/(BW-exp(Q/sqrt(3)))/(BW-exp(Q/sqrt(3)))
			     + 3*exp(2*sqrt(3)*Q)*GW*RW/(CALV+exp(sqrt(3)*Q)*RW)/(CALV+exp(sqrt(3)*Q)*RW)
			     + 2*exp(sqrt(3)*Q)*(DW-CALV*GW/(CALV+exp(sqrt(3)*Q)*RW))/CALV )
    * W0*W0/sqrt(3)/CALV/CALV/CALV;
}

double rho(vector<double> QP) {
  double Q = QP[0];
  double P = QP[1];
  return P*P/2. + V(Q);
}

double H(vector<double> QP) {
  return sqrt(rho(QP)/3.);
}

vector<vector<double>> hI(double t, const vector<vector<double>> &QPfield)
{
  vector<vector<double>> hI = QPfield;

  for (int xnum = 0; xnum < hI.size(); xnum++) {
    vector<double> QP = QPfield[xnum];
    hI[xnum][0] = QP[1]/H(QP);
    hI[xnum][1] = -3.*QP[1] - Vp(QP[0])/H(QP);
  }

  return hI;
}

vector<vector<vector<double>>> gaI(double t, const vector<vector<double>> &QPfield)
{
  double ks = KMIN*exp(t);

  int Ntheta = ceil(ks*LSIZE/2.); //ceil(M_PI*ks/KMIN);
  double Dtheta = M_PI/Ntheta;
  vector<double> thetai(Ntheta);
  for (size_t i = 0; i < Ntheta; i++) {
    thetai[i] = (i+1./2)*Dtheta;
  }

  function<int(double)> Nphi = [ks](double theta){
    return ceil(sin(theta)*ks*LSIZE); //ceil(2.*M_PI*sin(theta)*ks/KMIN);
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


  vector<vector<vector<double>>> gaI(Omegavol,vector<vector<double>>(XVOL,vector<double>(2,0)));
  for (int Onum = 0; Onum<Omegavol; Onum++) {
    for (int xnum = 0; xnum<XVOL; xnum++) {
      double ksrmu = ks * vec_op::dot(Omegavec[Onum],XVEC[xnum]);
      double theta_tmp = OmegaList[Onum][0];
      double DOmega = sin(theta_tmp)*Dtheta*Dphi(theta_tmp);

      vector<double> QP = QPfield[xnum];
      gaI[Onum][xnum][0] = H(QP)/2./M_PI * (cos(ksrmu)-sin(ksrmu))*sqrt(DOmega)/2./sqrt(M_PI);
    }
  }
  
  return gaI;
}


