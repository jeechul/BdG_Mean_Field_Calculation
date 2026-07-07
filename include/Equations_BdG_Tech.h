#ifndef EQUATION_BDG_TECH_H
#define EQUATION_BDG_TECH_H

#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <utility>
#include <map>
#include <algorithm>
#include <fstream>
#include <omp.h>
#include "Eigen_system.h"
#include <boost/math/special_functions/erf.hpp>

namespace BdG_Tech_RealPair
{

#pragma omp declare reduction(vec_double_plus : std::vector<double> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
                    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

template <typename EIGEN>
class Equations
{
    EIGEN eigen;
    const int L,L2;
public:
    Equations(const EIGEN &Eigen, const int size):
    eigen(Eigen),
    L(size), L2(size*2)
    {}

    inline int get_L() const { return L; }
    inline Equations& operator=(const Equations &Copy) { return *this; }

    void Gap_Num_eq(const std::vector<double> m, const std::vector<double> d, double * gaps, double * nums)
    {
        const bool IsNotDev = true;
        MatrixD BdGT(L2); double E[L2];
#ifdef __GPU__
        eigen.Quasi0_gpu(m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi0(m,d,E,BdGT,'V',IsNotDev);
#endif
        for (int i = 0; i < L; ++i)
        {
            double temp1 = 0.0;
            double temp2 = 0.0;

            for (int j=0;j<L;++j)
            {
                temp1 += BdGT(j,i)*BdGT(j,i)-BdGT(j,i+L)*BdGT(j,i+L);
                temp2 += BdGT(j,i)*BdGT(j,i+L)+BdGT(j,i+L)*BdGT(j,i);
            }

            nums[i] = temp1+1.0;
            gaps[i] = temp2;
        }
    }

    void Num_eq(const std::vector<double> m, const std::vector<double> d, double * nums)
    {
        const bool IsNotDev = true;
        MatrixD BdGT(L2); double E[L2];
#ifdef __GPU__ 
        eigen.Quasi0_gpu(m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi0(m,d,E,BdGT,'V',IsNotDev);
#endif
        for (int i = 0; i < L; i++)
        {
            double temp1 = 0.0;

            for (int j=0;j<L;++j)
            {
                temp1 += BdGT(j,i)*BdGT(j,i)-BdGT(j,i+L)*BdGT(j,i+L);
            }

            nums[i] = temp1+1.0;
        }
    }

    void Gap_eq(const std::vector<double> m, const std::vector<double> d, double * gaps)
    {	
        const bool IsNotDev = true;
        MatrixD BdGT(L2); double E[L2];
#ifdef __GPU__
        eigen.Quasi0_gpu(m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi0(m,d,E,BdGT,'V',IsNotDev);
#endif        
        for(int i = 0; i < L; i++)
        {
            double temp2 = 0.0;

            for (int j=0;j<L;++j)
            {
                temp2 += BdGT(j,i)*BdGT(j,i+L)+BdGT(j,i+L)*BdGT(j,i);
            }

            gaps[i] = temp2;
        }
    }

    double W_eq(const std::vector<double> m, const std::vector<double> d, const std::vector<double> &dilute, const double m0)
    {
        const bool IsNotDev = true;
        double result = 0;
        MatrixD BdGT(L2); double E[L2]; 
#ifdef __GPU__
        eigen.Quasi0_gpu(m,d,E,BdGT,'N',IsNotDev);
#else
        eigen.Quasi0(m,d,E,BdGT,'N',IsNotDev);
#endif 
        for(int i = 0; i < L; i++){ 
            result += E[i]-m[i]+d[i]*d[i]/dilute[i]+(m[i]-m0)*(m[i]-m0)/dilute[i];
        }

        return result;
    }

    double W_eq(const std::vector<double> m, const std::vector<double> d, const double U, const double m0)
    {
        const bool IsNotDev = true;
        double result = 0;
        MatrixD BdGT(L2); double E[L2]; 
#ifdef __GPU__
        eigen.Quasi0_gpu(m,d,E,BdGT,'N',IsNotDev);
#else
        eigen.Quasi0(m,d,E,BdGT,'N',IsNotDev);
#endif 
        for(int i = 0; i < L; i++){ 
            result += E[i]-m[i]+d[i]*d[i]/U+(m[i]-m0)*(m[i]-m0)/U;
        }

        return result;
    }

    void Compressibility_eq(const std::vector<double> m, const std::vector<double> d, double * dn_dm)
    {
        const bool IsNotDev = true;
        MatrixD BdGT(L2); double E[L2];
#ifdef __GPU__ 
        eigen.Quasi0_gpu(m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi0(m,d,E,BdGT,'V',IsNotDev);
#endif
        for (int i = 0; i < L; i++)
        {
            double temp = 0.0;
            for (int n=0; n < L; ++n)
            {
                for (int m=0; m < L2; ++m)
                {
                    double dH_dm = -BdGT(n,i)*BdGT(m,i)+BdGT(n,i+L)*BdGT(m,i+L);
                    temp += dH_dm*dH_dm*2.0/(E[n]-E[m]);
                }
            }
            dn_dm[i] = -temp;
        }
    }

    // A. Ghosal et. al. PRB 65, 014501 (2001)
    double SFW_xx_eq(const std::vector<double> m, const std::vector<double> d, const double V, const std::vector<std::pair<int,int>> pairs, const std::vector<double> T)
    {
        const bool IsNotDev = true;
        MatrixD BdGT(L2); double E[L2];
#ifdef __GPU__ 
        eigen.Quasi0_gpu(m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi0(m,d,E,BdGT,'V',IsNotDev);
#endif
        MatrixD BdG = BdGT.transpose(); 

        double K = 0.0;
        clock_t start = clock();
        //#pragma omp parallel for reduction(+:K)
        for (int idx=0; idx<pairs.size(); ++idx){
            for (int n=L; n<L2; ++n){
                int ix = pairs[idx].first, i = pairs[idx].second;
                //K += 4.0*T[idx]*BdGT(n,i+L)*BdGT(n,ix+L);
                K += 4.0*T[idx]*BdG(i+L,n)*BdG(ix+L,n);
            }
        }
        std::cout << "# Duration time of K : " << (double)(clock()-start)/CLOCKS_PER_SEC << " s" << std::endl;

        double Gamma = 0.0;
        //start = clock();
        start = omp_get_wtime();
        #pragma omp parallel for reduction(+:Gamma)
        for (int idx=0; idx<pairs.size()*pairs.size(); ++idx){
            int pidx = idx/pairs.size();
            int qidx = idx%pairs.size(); 
            int ix = pairs[pidx].first, i = pairs[pidx].second;
            int jx = pairs[qidx].first, j = pairs[qidx].second;
            for (int n=L; n<L2; ++n){
                for (int m=L; m<L2; ++m){
                    //double tempi1 = BdGT(n,ix)*BdGT(m,i+L)+BdGT(n,i+L)*BdGT(m,ix)-BdGT(n,i)*BdGT(m,ix+L)-BdGT(n,ix+L)*BdGT(m,i),
                    //    tempj1 = BdGT(m,jx+L)*BdGT(n,j)+BdGT(n,jx+L)*BdGT(m,j),
                    //    tempi2 = BdGT(n,ix+L)*BdGT(m,i)+BdGT(n,i)*BdGT(m,ix+L)-BdGT(n,i+L)*BdGT(m,ix)-BdGT(n,ix)*BdGT(m,i+L),
                    //    tempj2 = BdGT(m,jx)*BdGT(n,j+L)+BdGT(n,jx)*BdGT(m,j+L);
                    double tempi1 = BdG(ix,n)*BdG(i+L,m)+BdG(i+L,n)*BdG(ix,m)-BdG(i,n)*BdG(ix+L,m)-BdG(ix+L,n)*BdG(i,m),
                        tempj1 = BdG(jx+L,m)*BdG(j,n)+BdG(jx+L,n)*BdG(j,m),
                        tempi2 = BdG(ix+L,n)*BdG(i,m)+BdG(i,n)*BdG(ix+L,m)-BdG(i+L,n)*BdG(ix,m)-BdG(ix,n)*BdG(i+L,m),
                        tempj2 = BdG(jx,m)*BdG(j+L,n)+BdG(jx,n)*BdG(j+L,m);
                    Gamma += -2.0*T[pidx]*T[qidx]/(E[n]+E[m])*(tempi1*tempj1+tempi2*tempj2);
                }
            }
        }
        //std::cout << "# Duration time of Gamma : " << (double)(clock()-start)/CLOCKS_PER_SEC << " s" << std::endl;
        std::cout << "# Duration time of Gamma : " << (double)(omp_get_wtime()-start) << " s" << std::endl;

        //std::cout<< "# K, Gamma : " << K << "\t" << Gamma << std::endl;

        return (K-Gamma)/V;
    }

    void SFW_xx_eq(double * SFW, const std::vector<double> m, const std::vector<double> d, const double V, const std::vector<std::pair<int,int>> pairs, const std::vector<double> T)
    {
        const bool IsNotDev = true;
        MatrixD BdGT(L2); double E[L2];
#ifdef __GPU__ 
        eigen.Quasi0_gpu(m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi0(m,d,E,BdGT,'V',IsNotDev);
#endif
        MatrixD BdG = BdGT.transpose();

        clock_t start = clock();
        //#pragma omp parallel for reduction(+:K)
        for (int idx=0; idx<pairs.size(); ++idx){
            for (int n=L; n<L2; ++n){
                int ix = pairs[idx].first, i = pairs[idx].second;
                SFW[i] += 4.0*T[idx]*BdG(i+L,n)*BdG(ix+L,n)/V;
            }
        }
        //std::cout << "# Duration time of K : " << (double)(clock()-start)/CLOCKS_PER_SEC << " s" << std::endl;

        start = clock();
        //start = omp_get_wtime();
        //#pragma omp parallel for reduction(+:Gamma)
        for (int idx=0; idx<pairs.size()*pairs.size(); ++idx){
            int pidx = idx/pairs.size();
            int qidx = idx%pairs.size(); 
            int ix = pairs[pidx].first, i = pairs[pidx].second;
            int jx = pairs[qidx].first, j = pairs[qidx].second;
            for (int n=L; n<L2; ++n){
                for (int m=L; m<L2; ++m){
                    double tempi1 = BdG(ix,n)*BdG(i+L,m)+BdG(i+L,n)*BdG(ix,m)-BdG(i,n)*BdG(ix+L,m)-BdG(ix+L,n)*BdG(i,m),
                        tempj1 = BdG(jx+L,m)*BdG(j,n)+BdG(jx+L,n)*BdG(j,m),
                        tempi2 = BdG(ix+L,n)*BdG(i,m)+BdG(i,n)*BdG(ix+L,m)-BdG(i+L,n)*BdG(ix,m)-BdG(ix,n)*BdG(i+L,m),
                        tempj2 = BdG(jx,m)*BdG(j+L,n)+BdG(jx,n)*BdG(j+L,m);
                    SFW[i] -= -2.0*T[pidx]*T[qidx]/(E[n]+E[m])*(tempi1*tempj1+tempi2*tempj2)/V;
                }
            }
        }
        std::cout << "# Duration time of Gamma : " << (double)(clock()-start)/CLOCKS_PER_SEC << " s" << std::endl;
        //std::cout << "# Duration time of Gamma : " << (double)(omp_get_wtime()-start) << " s" << std::endl;
    }

    // R. Kiran et. al. PRB 110, 184506 (2024)
    double SFW_xx_eq_v2(const std::vector<double> m, const std::vector<double> d, const double V, const std::vector<std::pair<int,int>> pairs, const std::vector<double> T, const std::vector<double> P)
    {
        const bool IsNotDev = true;
        MatrixD BdGT(L2); double E[L2];
#ifdef __GPU__ 
        eigen.Quasi0_gpu(m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi0(m,d,E,BdGT,'V',IsNotDev);
#endif
        MatrixD BdG = BdGT.transpose(); 

        double K = 0.0;
        clock_t start = clock();
        //#pragma omp parallel for reduction(+:K)
        for (int idx=0; idx<pairs.size(); ++idx){
            for (int n=L; n<L2; ++n){
                int ix = pairs[idx].first, i = pairs[idx].second;
                //K += 4.0*T[idx]*BdGT(n,i+L)*BdGT(n,ix+L);
                K += 4.0*T[idx]*P[idx]*P[idx]*BdG(i+L,n)*BdG(ix+L,n);
            }
        }
        std::cout << "# Duration time of K : " << (double)(clock()-start)/CLOCKS_PER_SEC << " s" << std::endl;

        double Gamma = 0.0;
        start = clock();
        //start = omp_get_wtime();
        //#pragma omp parallel for reduction(+:Gamma)
        /*for (int idx=0; idx<pairs.size()*pairs.size(); ++idx){
            int pidx = idx/pairs.size();
            int qidx = idx%pairs.size(); 
            int ix = pairs[pidx].first, i = pairs[pidx].second;
            int jx = pairs[qidx].first, j = pairs[qidx].second;
            for (int n=L; n<L2; ++n){
                for (int m=L; m<L2; ++m){
                    //double tempi1 = BdGT(n,ix)*BdGT(m,i+L)+BdGT(n,i+L)*BdGT(m,ix)-BdGT(n,i)*BdGT(m,ix+L)-BdGT(n,ix+L)*BdGT(m,i),
                    //    tempj1 = BdGT(m,jx+L)*BdGT(n,j)+BdGT(n,jx+L)*BdGT(m,j),
                    //    tempi2 = BdGT(n,ix+L)*BdGT(m,i)+BdGT(n,i)*BdGT(m,ix+L)-BdGT(n,i+L)*BdGT(m,ix)-BdGT(n,ix)*BdGT(m,i+L),
                    //    tempj2 = BdGT(m,jx)*BdGT(n,j+L)+BdGT(n,jx)*BdGT(m,j+L);
                    double tempi1 = BdG(ix,n)*BdG(i+L,m)+BdG(i+L,n)*BdG(ix,m)-BdG(i,n)*BdG(ix+L,m)-BdG(ix+L,n)*BdG(i,m),
                        tempj1 = BdG(jx+L,m)*BdG(j,n)+BdG(jx+L,n)*BdG(j,m),
                        tempi2 = BdG(ix+L,n)*BdG(i,m)+BdG(i,n)*BdG(ix+L,m)-BdG(i+L,n)*BdG(ix,m)-BdG(ix,n)*BdG(i+L,m),
                        tempj2 = BdG(jx,m)*BdG(j+L,n)+BdG(jx,n)*BdG(j+L,m);
                    Gamma += -2.0*T[pidx]*T[qidx]*P[pidx]*P[qidx]/(E[n]+E[m])*(tempi1*tempj1+tempi2*tempj2);
                }
            }
        }*/
        for (int n=L; n<L2; ++n){
            for (int m=L; m<L2; ++m){
                double tempi1 = 0, tempi2 = 0;
                double tempj1 = 0, tempj2 = 0;
                for (int idx=0;idx<pairs.size();++idx) {
                    int ix = pairs[idx].first, i = pairs[idx].second;
                    tempi1 += T[idx]*P[idx]*(BdG(ix,n)*BdG(i+L,m)+BdG(i+L,n)*BdG(ix,m)-BdG(i,n)*BdG(ix+L,m)-BdG(ix+L,n)*BdG(i,m));
                    tempi2 += T[idx]*P[idx]*(BdG(ix+L,n)*BdG(i,m)+BdG(i,n)*BdG(ix+L,m)-BdG(i+L,n)*BdG(ix,m)-BdG(ix,n)*BdG(i+L,m));

                    tempj1 += T[idx]*P[idx]*(BdG(ix+L,m)*BdG(i,n)+BdG(ix+L,n)*BdG(i,m));
                    tempj2 += T[idx]*P[idx]*(BdG(ix,m)*BdG(i+L,n)+BdG(ix,n)*BdG(i+L,m));
                }

                Gamma += -2.0/(E[n]+E[m])*(tempi1*tempj1+tempi2*tempj2);
            }
        }

        std::cout << "# Duration time of Gamma : " << (double)(clock()-start)/CLOCKS_PER_SEC << " s" << std::endl;
        //std::cout << "# Duration time of Gamma : " << (double)(omp_get_wtime()-start) << " s" << std::endl;

        //std::cout<< "# K, Gamma : " << K << "\t" << Gamma << std::endl;

        return (K-Gamma)/V;
    }

    // R. Kiran et. al. PRB 110, 184506 (2024), sum rule : Eq. (46) in D. J. Scalapino et. al. PRB 47, 7995 (1993)
    double SFW_xx_eq_v2_with_sum_rule(const std::vector<double> m, const std::vector<double> d, const double V, const std::vector<std::pair<int,int>> pairs, const std::vector<double> T, const std::vector<double> P, const bool sigmaoutOn=true, const double omega_cutoff=0.01, const double omega_step=0.01, const int Nomega=100, const double eta=0.01)
    {
        const bool IsNotDev = true;
        MatrixD BdGT(L2); double E[L2];
#ifdef __GPU__ 
        eigen.Quasi0_gpu(m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi0(m,d,E,BdGT,'V',IsNotDev);
#endif
        MatrixD BdG = BdGT.transpose(); 

        double K = 0.0;
        clock_t start = clock();
        //#pragma omp parallel for reduction(+:K)
        for (int idx=0; idx<pairs.size(); ++idx){
            for (int n=L; n<L2; ++n){
                int ix = pairs[idx].first, i = pairs[idx].second;
                //K += 4.0*T[idx]*BdGT(n,i+L)*BdGT(n,ix+L);
                K += 4.0*T[idx]*P[idx]*P[idx]*BdG(i+L,n)*BdG(ix+L,n);
            }
        }
        std::cout << "# Duration time of K : " << (double)(clock()-start)/CLOCKS_PER_SEC << " s" << std::endl;

        std::vector<double> omega(Nomega), IMGamma(Nomega);
        for (int i=0;i<Nomega;++i) omega[i] = omega_step*(double)(i+1);

        //start = clock();
        start = omp_get_wtime();
        #pragma omp parallel for reduction(vec_double_plus:IMGamma)
        for (int wid=0; wid<Nomega; ++wid) {
            for (int idx=0; idx<pairs.size()*pairs.size(); ++idx){
                int pidx = idx/pairs.size();
                int qidx = idx%pairs.size(); 
                int ix = pairs[pidx].first, i = pairs[pidx].second;
                int jx = pairs[qidx].first, j = pairs[qidx].second;
                for (int n=L; n<L2; ++n){
                    for (int m=L; m<L2; ++m){
                        //double tempi1 = BdGT(n,ix)*BdGT(m,i+L)+BdGT(n,i+L)*BdGT(m,ix)-BdGT(n,i)*BdGT(m,ix+L)-BdGT(n,ix+L)*BdGT(m,i),
                        //    tempj1 = BdGT(m,jx+L)*BdGT(n,j)+BdGT(n,jx+L)*BdGT(m,j),
                        //    tempi2 = BdGT(n,ix+L)*BdGT(m,i)+BdGT(n,i)*BdGT(m,ix+L)-BdGT(n,i+L)*BdGT(m,ix)-BdGT(n,ix)*BdGT(m,i+L),
                        //    tempj2 = BdGT(m,jx)*BdGT(n,j+L)+BdGT(n,jx)*BdGT(m,j+L);
                        double tempi1 = BdG(ix,n)*BdG(i+L,m)+BdG(i+L,n)*BdG(ix,m)-BdG(i,n)*BdG(ix+L,m)-BdG(ix+L,n)*BdG(i,m),
                            tempj1 = BdG(jx+L,m)*BdG(j,n)+BdG(jx+L,n)*BdG(j,m),
                            tempi2 = BdG(ix+L,n)*BdG(i,m)+BdG(i,n)*BdG(ix+L,m)-BdG(i+L,n)*BdG(ix,m)-BdG(ix,n)*BdG(i+L,m),
                            tempj2 = BdG(jx,m)*BdG(j+L,n)+BdG(jx,n)*BdG(j+L,m);
                        double wEE = omega[wid]-E[n]-E[m];
                        IMGamma[wid] += -2.0*T[pidx]*T[qidx]*P[pidx]*P[qidx]*eta/(wEE*wEE+eta*eta)*(tempi1*tempj1+tempi2*tempj2);
                    }
                }
            }
            std::cout << "\r# wid, IM[Gamma[wid]] : " << wid << " , " << IMGamma[wid] << std::flush;
        }

        std::ofstream sigmaout;
        std::string filename = "sigma_xx.dat";
        if (sigmaoutOn) sigmaout.open(filename);

        double REGamma = 0;
        for (int wid=0; wid<Nomega; ++wid){
            double sigma = 1.0/(M_PI*omega[wid])*IMGamma[wid];
            if ( abs(omega[wid]) > omega_cutoff ) REGamma += sigma*omega_step;
            if (sigmaoutOn) {
                sigmaout << omega[wid] << "\t" << sigma/V << std::endl;
            }
        }
        if (sigmaoutOn) sigmaout.close();

        //std::cout << "# Duration time of Gamma : " << (double)(clock()-start)/CLOCKS_PER_SEC << " s" << std::endl;
        std::cout << "# Duration time of Gamma : " << (double)(omp_get_wtime()-start) << " s" << std::endl;

        //std::cout<< "# K, Gamma : " << K << "\t" << Gamma << std::endl;

        return (K-REGamma)/V;
    }

    // R. Kiran et. al. PRB 110, 184506 (2024)
    double SFW_xx_eq_v2_with_qy(std::vector<double> &Gamma, const std::vector<double> m, const std::vector<double> d, const double V, const std::vector<std::pair<int,int>> pairs, const std::vector<double> T, const std::vector<double> P, const std::vector<double> Ry, const int l)
    {
        const bool IsNotDev = true;
        MatrixD BdGT(L2); double E[L2];
    #ifdef __GPU__ 
        eigen.Quasi0_gpu(m,d,E,BdGT,'V',IsNotDev);
    #else
        eigen.Quasi0(m,d,E,BdGT,'V',IsNotDev);
    #endif
        MatrixD BdG = BdGT.transpose(); 

        double K = 0.0;
        clock_t start = clock();
        //#pragma omp parallel for reduction(vec_double_plus:SFW)
        for (int idx=0; idx<pairs.size(); ++idx){
            for (int n=L; n<L2; ++n){
                int ix = pairs[idx].first, i = pairs[idx].second;
                //K += 4.0*T[idx]*BdGT(n,i+L)*BdGT(n,ix+L);
                K += 4.0*T[idx]*P[idx]*P[idx]*BdG(i+L,n)*BdG(ix+L,n)/V;
            }
        }
        std::cout << "# Duration time of K : " << (double)(clock()-start)/CLOCKS_PER_SEC << " s" << std::endl;
        std::cout << "# K : " << K << std::endl;

        start = clock();
        //start = omp_get_wtime();
        //#pragma omp parallel for reduction(vec_double_plus:Gamma)
        for (int qid = 0; qid < Gamma.size(); ++qid) {
            const double qy = qid*2*M_PI/l;
            for (int idx=0; idx<pairs.size()*pairs.size(); ++idx){
                int pidx = idx/pairs.size();
                int qidx = idx%pairs.size(); 
                int ix = pairs[pidx].first, i = pairs[pidx].second;
                int jx = pairs[qidx].first, j = pairs[qidx].second;
                double ryi = Ry[i], ryj = Ry[j];
                double fourier_coeff = cos(qy*(ryj-ryi))/V;
                for (int n=L; n<L2; ++n){
                    for (int m=L; m<L2; ++m){
                        //double tempi1 = BdGT(n,ix)*BdGT(m,i+L)+BdGT(n,i+L)*BdGT(m,ix)-BdGT(n,i)*BdGT(m,ix+L)-BdGT(n,ix+L)*BdGT(m,i),
                        //    tempj1 = BdGT(m,jx+L)*BdGT(n,j)+BdGT(n,jx+L)*BdGT(m,j),
                        //    tempi2 = BdGT(n,ix+L)*BdGT(m,i)+BdGT(n,i)*BdGT(m,ix+L)-BdGT(n,i+L)*BdGT(m,ix)-BdGT(n,ix)*BdGT(m,i+L),
                        //    tempj2 = BdGT(m,jx)*BdGT(n,j+L)+BdGT(n,jx)*BdGT(m,j+L);
                        double tempi1 = BdG(ix,n)*BdG(i+L,m)+BdG(i+L,n)*BdG(ix,m)-BdG(i,n)*BdG(ix+L,m)-BdG(ix+L,n)*BdG(i,m),
                            tempj1 = BdG(jx+L,m)*BdG(j,n)+BdG(jx+L,n)*BdG(j,m),
                            tempi2 = BdG(ix+L,n)*BdG(i,m)+BdG(i,n)*BdG(ix+L,m)-BdG(i+L,n)*BdG(ix,m)-BdG(ix,n)*BdG(i+L,m),
                            tempj2 = BdG(jx,m)*BdG(j+L,n)+BdG(jx,n)*BdG(j+L,m);
                        Gamma[qid] += -2.0*T[pidx]*T[qidx]*P[pidx]*P[qidx]/(E[n]+E[m])*(tempi1*tempj1+tempi2*tempj2)*fourier_coeff;
                    }
                }
            }
            std::cout << "# qid, Gamma[qid] : " << qid << " , " << Gamma[qid] << std::endl;
        }
        std::cout << "# Duration time of Gamma : " << (double)(clock()-start)/CLOCKS_PER_SEC << " s" << std::endl;
        //std::cout << "# Duration time of Gamma : " << (double)(omp_get_wtime()-start) << " s" << std::endl;

        return K;
    }

};

template <typename EIGEN>
class GreenFunction // Local density of state calculation LDOS(w,r_i)
{
    EIGEN eigen;
    const int L,L2;
public:
    GreenFunction(const EIGEN &Eigen, const int size):
    eigen(Eigen),
    L(size), L2(size*2)
    {}

    void single_green_function(double * ldos, const std::vector<double> Ws, const int site)
    {	
        const bool IsNotDev = true;
        MatrixD PsiT(L); double E[L];
#ifdef __GPU__
        eigen.Single0_gpu(E,&PsiT[0],'V',IsNotDev);
#else
        eigen.Single0(E,&PsiT[0],'V',IsNotDev);
#endif 
        for (int i=0;i<Ws.size();++i)
        {
            for (int j=0;j<L;++j)
            {
                double prob = std::norm(PsiT[L*j+site]);
                ldos[i] -= std::imag(prob/(dcomplex(Ws[i]-E[j],0.001)*M_PI));
            }
        }
    }

    void manybody_green_function(double * ldos, const std::vector<double> Ws, const int site, const std::vector<double> m, const std::vector<double> d, double eta=0.01)
    {	
        const bool IsNotDev = true;
        MatrixD BdGT(L2); double E[L2];
#ifdef __GPU__
        eigen.Quasi0_gpu(m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi0(m,d,E,BdGT,'V',IsNotDev);
#endif       
        for (int i=0;i<Ws.size();++i)
        {
            for (int j=0;j<L2;++j)
            {
                double prob = std::norm(BdGT[L2*j+site]);
                ldos[i] -= std::imag(prob/(dcomplex(Ws[i]-E[j],eta)*M_PI));
            }
        }
    }

    void manybody_green_function_for_many_sites(double * ldos, const std::vector<double> Ws, const int Nsite, const std::vector<double> m, const std::vector<double> d, double eta=0.01)
    {
        const bool IsNotDev = true;
        MatrixD BdGT(L2); double E[L2];
#ifdef __GPU__
        eigen.Quasi0_gpu(m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi0(m,d,E,BdGT,'V',IsNotDev);
#endif
        for (int s=0; s<Nsite; ++s)
        {
            for (int i=0;i<Ws.size();++i)
            {
                for (int j=0;j<L2;++j)
                {
                    double prob = std::norm(BdGT[L2*j+s]);
                    ldos[s*Ws.size()+i] -= std::imag(prob/(dcomplex(Ws[i]-E[j],eta)*M_PI));
                }
            }
        }
    }

    void manybody_green_function_for_many_sites_with_GaussianDeltaFunction(double * ldos, const std::vector<double> Ws, const int Nsite, const std::vector<double> m, const std::vector<double> d, double eta=0.01)
    {
        const bool IsNotDev = true;
        MatrixD BdGT(L2); double E[L2];
#ifdef __GPU__
        eigen.Quasi0_gpu(m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi0(m,d,E,BdGT,'V',IsNotDev);
#endif
        for (int s=0; s<Nsite; ++s)
        {
            for (int i=0;i<Ws.size();++i)
            {
                for (int j=0;j<L2;++j)
                {
                    double prob = std::norm(BdGT[L2*j+s]);
                    ldos[s*Ws.size()+i] += prob*exp(-(Ws[i]-E[j])*(Ws[i]-E[j])/(4.0*eta))/sqrt(4.0*M_PI*eta);
                }
            }
        }
    }
};


template <typename EIGEN>
class LDOS_smearing // Local density of state calculation LDOS(w,r_i) [Ref. : Two-dimensional Photonic Crystal: A square grid of dielectric veins, F. Weissenbacher (2018)]
{
    EIGEN eigen;
    const int L,L2;
public:
    LDOS_smearing(const EIGEN &Eigen, const int size):
    eigen(Eigen),
    L(size), L2(size*2)
    {}

    void Gaussian(double * ldos, const std::vector<double> Ws, const int Nsite, const std::vector<double> m, const std::vector<double> d, const double sf=2.0, const double xi=1.0)
    {
        const bool IsNotDev = true;
        MatrixD BdGT(L2); double E[L2];
#ifdef __GPU__
        eigen.Quasi0_gpu(m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi0(m,d,E,BdGT,'V',IsNotDev);
#endif
        const double dw = Ws[1]-Ws[0], sigma = dw*sf, cutoff = xi*sigma;
        for (int s=0; s<Nsite; ++s)
        {
            for (int i=0;i<Ws.size();++i)
            {
                for (int j=0;j<L2;++j)
                {
                    double prob = std::norm(BdGT[L2*j+s]);
                    double nu = Ws[i]-E[j];
                    ldos[s*Ws.size()+i] += (abs(nu)<=cutoff)? prob*exp(-nu*nu/(2.0*sigma))/(sqrt(M_PI)*sigma*boost::math::erf(xi)) : 0.0;
                }
            }
        }
    }

    void Lorentzian(double * ldos, const std::vector<double> Ws, const int Nsite, const std::vector<double> m, const std::vector<double> d, const double sf=2.0, const double xi=1.0)
    {
        const bool IsNotDev = true;
        MatrixD BdGT(L2); double E[L2];
#ifdef __GPU__
        eigen.Quasi0_gpu(m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi0(m,d,E,BdGT,'V',IsNotDev);
#endif
        const double dw = Ws[1]-Ws[0], alpha = 0.5*dw*sf, cutoff = xi*alpha;
        for (int s=0; s<Nsite; ++s)
        {
            for (int i=0;i<Ws.size();++i)
            {
                for (int j=0;j<L2;++j)
                {
                    double prob = std::norm(BdGT[L2*j+s]);
                    double nu = Ws[i]-E[j];
                    ldos[s*Ws.size()+i] += (abs(nu)<=cutoff)? prob*alpha/(2.0*(nu*nu+alpha*alpha)*atan(xi)) : 0.0;
                }
            }
        }
    }

    void Uniform(double * ldos, const std::vector<double> Ws, const int Nsite, const std::vector<double> m, const std::vector<double> d, const double sf=2.0)
    {
        const bool IsNotDev = true;
        MatrixD BdGT(L2); double E[L2];
#ifdef __GPU__
        eigen.Quasi0_gpu(m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi0(m,d,E,BdGT,'V',IsNotDev);
#endif
        const double dw = Ws[1]-Ws[0], omega = dw*sf, cutoff = omega*0.5;
        for (int s=0; s<Nsite; ++s)
        {
            for (int i=0;i<Ws.size();++i)
            {
                for (int j=0;j<L2;++j)
                {
                    double prob = std::norm(BdGT[L2*j+s]);
                    double nu = Ws[i]-E[j];
                    ldos[s*Ws.size()+i] += (abs(nu)<=cutoff)? prob/omega : 0.0;
                }
            }
        }
    }
};


template <typename EIGEN>
class SingleinReal
{
    EIGEN eigen;
    const int L;
public:
    SingleinReal(const EIGEN &Eigen, const int size):
    eigen(Eigen),
    L(size)
    {}

    void onsiteDOS(double * ldos, const std::vector<double> Ws, const int site)
    {
        const bool IsNotDev = true;
        double * PsiT = new double[L*L]; double E[L];
#ifdef __GPU__
        eigen.Single0_gpu(E,PsiT,'V',IsNotDev);
#else 
        eigen.Single0(E,PsiT,'V',IsNotDev);
#endif
        for (int i=0;i<Ws.size();++i)
        {
            for (int j=0;j<L;++j)
            {
                double prob = std::norm(PsiT[L*j+site]);
                ldos[i] -= std::imag(prob/(dcomplex(Ws[i]-E[j],0.001)*M_PI*(double)L));
            }
        }
        delete[] PsiT;
    }

    void totalDOS(double * ldos, const std::vector<double> Ws)
    {
        const bool IsNotDev = true;
        double * PsiT = new double[L*L]; double E[L];
#ifdef __GPU__
        eigen.Single0_gpu(E,PsiT,'N',IsNotDev);
#else
        eigen.Single0(E,PsiT,'N',IsNotDev);
#endif
        for (int i=0;i<Ws.size();++i)
        {
            for (int j=0;j<L;++j)
            {
                ldos[i] -= std::imag(1.0/(dcomplex(Ws[i]-E[j],0.01)*M_PI*(double)L));
            }
        }
        delete[] PsiT;
    }

    void totalDOS_global_mem(double * PsiT, double * ldos, const std::vector<double> Ws)
    {
        const bool IsNotDev = true;
        double E[L];
#ifdef __GPU__
        eigen.Single0_gpu(E,PsiT,'N',IsNotDev);
#else
        eigen.Single0(E,PsiT,'N',IsNotDev);
#endif
        for (int i=0;i<Ws.size();++i)
        {
            for (int j=0;j<L;++j)
            {
                ldos[i] -= std::imag(1.0/(dcomplex(Ws[i]-E[j],0.01)*M_PI*(double)L));
            }
        }
    }

    double occupation(double * ns, int Nt)
    {
        const bool IsNotDev = true;
        double * PsiT = new double[L*L]; double E[L];
#ifdef __GPU__
        eigen.Single0_gpu(E,PsiT,'V',IsNotDev);
#else
        eigen.Single0(E,PsiT,'V',IsNotDev);
#endif
        std::vector<double> Er;
        std::map<double,std::vector<int>> Eid;

        for (int i=0;i<L;++i)
        {
            double e = std::round(E[i]*1e7)/1e7;
            if (std::find(Er.begin(),Er.end(),e) == Er.end()) {
                Er.push_back(e);
                Eid.insert({e,std::vector<int>(1,i)});
            }
            else {
                Eid[e].push_back(i);
            }
        }

        double sumE = 0.0, Nacc = 0.0;
        for (const auto e : Er)
        {
            int Nd = Eid[e].size();

            double fac = 1.0;
            if ((Nt-Nacc-Nd)<0) fac = (Nt-Nacc)/(double)Nd;

            for (int site=0; site<L; ++site)
            {
                for (const auto id : Eid[e])
                {
                    double prob = std::norm(PsiT[L*id+site])*fac;
                    ns[site] += prob;
                    Nacc += prob;
                }
            }
            sumE += e*Nd*fac;
        }

        delete[] PsiT;
        for (const auto e : Er){ 
            Eid[e].clear();
            std::vector<int>().swap(Eid[e]);
        }
        Eid.clear();

        return sumE;
    }
};

} // namespace BdG_Tech_RealPair


namespace BdG_Tech_ComplexPair
{

template <typename EIGEN>
class Equations
{
    EIGEN eigen;
    const int L,L2;
public:
    Equations(const EIGEN &Eigen, const int size):
    eigen(Eigen),
    L(size), L2(size*2)
    {}

    inline int get_L() const { return L; }
    inline Equations& operator=(const Equations &Copy) { return *this; }

    void Gap_Num_eq(const std::vector<double> m, const std::vector<dcomplex> d, dcomplex * gaps, double * nums)
    {
        const bool IsNotDev = true;
        MatrixC BdGT(L2); double E[L2];
#ifdef __GPU__
        eigen.Quasi_gpu(0.0,0.0,m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi(0.0,0.0,m,d,E,BdGT,'V',IsNotDev);
#endif
        for (int i = 0; i < L; ++i)
        {
            dcomplex temp1 = 0.0;
            dcomplex temp2 = 0.0;

            for (int j=0;j<L;++j)
            {
                temp1 += std::conj(BdGT(j,i))*BdGT(j,i)-std::conj(BdGT(j,i+L))*BdGT(j,i+L);
                temp2 += std::conj(BdGT(j,i+L))*BdGT(j,i);
            }

            nums[i] = temp1.real()+1.0;
            gaps[i] = temp2;
        }
    }

    void Num_eq(const std::vector<double> m, const std::vector<dcomplex> d, double * nums)
    {
        const bool IsNotDev = true;
        MatrixC BdGT(L2); double E[L2];
#ifdef __GPU__ 
        eigen.Quasi_gpu(0.0,0.0,m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi(0.0,0.0,m,d,E,BdGT,'V',IsNotDev);
#endif
        for (int i = 0; i < L; i++)
        {
            dcomplex temp1 = 0.0;

            for (int j=0;j<L;++j)
            {
                temp1 += std::conj(BdGT(j,i))*BdGT(j,i)-std::conj(BdGT(j,i+L))*BdGT(j,i+L);
            }

            nums[i] = temp1.real()+1.0;
        }
    }

    void Gap_eq(const std::vector<double> m, const std::vector<dcomplex> d, dcomplex * gaps)
    {	
        const bool IsNotDev = true;
        MatrixC BdGT(L2); double E[L2];
#ifdef __GPU__
        eigen.Quasi_gpu(0.0,0.0,m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi(0.0,0.0,m,d,E,BdGT,'V',IsNotDev);
#endif        
        for(int i = 0; i < L; i++)
        {
            dcomplex temp2 = 0.0;

            for (int j=0;j<L;++j)
            {
                temp2 += std::conj(BdGT(j,i+L))*BdGT(j,i);
            }

            gaps[i] = temp2;
        }
    }

    double W_eq(const std::vector<double> m, const std::vector<dcomplex> d, const std::vector<double> &dilute, const double m0)
    {
        const bool IsNotDev = true;
        double result = 0;
        MatrixC BdGT(L2); double E[L2]; 
#ifdef __GPU__
        eigen.Quasi_gpu(0.0,0.0,m,d,E,BdGT,'N',IsNotDev);
#else
        eigen.Quasi(0.0,0.0,m,d,E,BdGT,'N',IsNotDev);
#endif 
        for(int i = 0; i < L; i++){ 
            result += E[i]-m[i]+std::norm(d[i])/dilute[i]+(m[i]-m0)*(m[i]-m0)/dilute[i];
        }

        return result;
    }

    double W_eq(const std::vector<double> m, const std::vector<dcomplex> d, const double U, const double m0)
    {
        const bool IsNotDev = true;
        double result = 0;
        MatrixC BdGT(L2); double E[L2]; 
#ifdef __GPU__
        eigen.Quasi_gpu(0.0,0.0,m,d,E,BdGT,'N',IsNotDev);
#else
        eigen.Quasi(0.0,0.0,m,d,E,BdGT,'N',IsNotDev);
#endif 
        for(int i = 0; i < L; i++){ 
            result += E[i]-m[i]+std::norm(d[i])/U+(m[i]-m0)*(m[i]-m0)/U;
        }

        return result;
    }

};

} // namespace BdG_Tech_ComplexPair

#endif
