#include <iostream>
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

const double MM = 1e-5;
const double DT = 1e-2;
const double QI = 15;

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

  ofstream Lfile("chaotic_Lattice.dat");
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

  ofstream Qfile("chaotic_Q.dat"), Pfile("chaotic_P.dat"), Hfile("chaotic_H.dat"), zfile("chaotic_zeta.dat");

  double counter = 0;
  while (t<tf) {
    double rhobar = 0;
    
    Qfile << t << ' ';
    Pfile << t << ' ';
    Hfile << t << ' ';
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
    zfile << t << ' ';
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
  return MM*MM*Q*Q/2.;
}

double Vp(double Q) {
  return MM*MM*Q;
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

  int Ntheta = ceil(M_PI*ks/KMIN);
  double Dtheta = M_PI/Ntheta;
  vector<double> thetai(Ntheta);
  for (size_t i = 0; i < Ntheta; i++) {
    thetai[i] = (i+1./2)*Dtheta;
  }

  function<int(double)> Nphi = [ks](double theta){
    return ceil(2.*M_PI*sin(theta)*ks/KMIN);
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


