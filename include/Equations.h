#ifndef EQUATION_H
#define EQUATION_H

#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <utility>
#include <map>
#include <algorithm>
#include <mpi.h>
#include "Eigen_system.h"

#define MPIERR(X) { if(X != MPI_SUCCESS) { std::cout<<"ERROR!"<<__LINE__<<std::endl; exit(1); } }

using thrDoubleComplex=thrust::complex<double>;

namespace Kernel
{

__global__ void _CurrentOpCal_(const int L, const int L2, 
    thrDoubleComplex * M_conv, thrDoubleComplex * M_geom,
    const thrDoubleComplex * BdG, const thrDoubleComplex * g_dh_g)
{
    int gid = blockIdx.x*blockDim.x+threadIdx.x;

    int i=gid/L2, j=gid%L2;
    thrDoubleComplex M_conv_p=thrDoubleComplex(0,0), M_conv_m=thrDoubleComplex(0,0);
    thrDoubleComplex M_geom_p=thrDoubleComplex(0,0), M_geom_m=thrDoubleComplex(0,0);
    
    for(int m=0;m<L;m++)
    for(int n=0;n<L;n++){
        if(m==n)
        { 
            M_conv_p += thrust::conj(BdG[m*L2+i])*BdG[n*L2+j]*g_dh_g[m*L+n]; 
            M_conv_m += thrust::conj(BdG[(m+L)*L2+j])*BdG[(n+L)*L2+i]*g_dh_g[m*L+n]; 
        }
        else
        { 
            M_geom_p += thrust::conj(BdG[m*L2+i])*BdG[n*L2+j]*g_dh_g[m*L+n]; 
            M_geom_m += thrust::conj(BdG[(m+L)*L2+j])*BdG[(n+L)*L2+i]*g_dh_g[m*L+n]; 
        }
    }

    M_conv[i*L2+j] = M_conv_p*M_conv_m; 
    M_geom[i*L2+j] = M_geom_p*M_geom_m+M_conv_p*M_geom_m+M_geom_p*M_conv_m; 
}

} // namespace Kernel

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

    void Gap_Num_eq(const double k_x, const double k_y, double * gaps, double * nums, double c, const std::vector<double> m, const std::vector<double> d)
    {
        const bool IsNotDev = true;
        MatrixC BdGT(L2); double E[L2]; 
#ifdef __GPU__
        eigen.Quasi_gpu(k_x,k_y,m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi(k_x,k_y,m,d,E,BdGT,'V',IsNotDev);
#endif
        for (int i = 0; i < L; ++i)
        {
            dcomplex temp1 = 0.0;
            dcomplex temp2 = 0.0;

            for (int j=0;j<L;++j)
            {
                temp1 += std::conj(BdGT(j,i))*BdGT(j,i)-std::conj(BdGT(j,i+L))*BdGT(j,i+L);
                temp2 += std::conj(BdGT(j,i))*BdGT(j,i+L)+std::conj(BdGT(j,i+L))*BdGT(j,i);
            }
            
            nums[i] += (temp1.real()+1.0)*c;
            gaps[i] += temp2.real()*c; 		
        }
    }

    void Num_eq(const double k_x, const double k_y, double * nums, double c, const std::vector<double> m, const std::vector<double> d)
    {
        const bool IsNotDev = true;
        MatrixC BdGT(L2); double E[L2]; 
#ifdef __GPU__
        eigen.Quasi_gpu(k_x,k_y,m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi(k_x,k_y,m,d,E,BdGT,'V',IsNotDev);
#endif
        for (int i = 0; i < L; ++i)
        {
            dcomplex temp1 = 0.0;

            for (int j=0;j<L;++j)
            {
                temp1 += std::conj(BdGT(j,i))*BdGT(j,i)-std::conj(BdGT(j,i+L))*BdGT(j,i+L);
            }
            
            nums[i] += (temp1.real()+1.0)*c;
        };
    }

    void Gap_eq(const double k_x, const double k_y, double * gaps, double c, const std::vector<double> m, const std::vector<double> d)
    {	
        const bool IsNotDev = true;
        MatrixC BdGT(L2); double E[L2];
#ifdef __GPU__
        eigen.Quasi_gpu(k_x,k_y,m,d,E,BdGT,'V',IsNotDev);
#else
        eigen.Quasi(k_x,k_y,m,d,E,BdGT,'V',IsNotDev);
#endif
        for(int i = 0; i < L; ++i)
        {
            dcomplex temp2 = 0.0;

            for (int j=0;j<L;++j)
            {
                temp2 += std::conj(BdGT(j,i))*BdGT(j,i+L)+std::conj(BdGT(j,i+L))*BdGT(j,i);
            }
            
            gaps[i] += temp2.real()*c; 		
        }
    }

    double W_eq(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, const std::vector<double> &dilute, const double m0)
    {
        const bool IsNotDev = true;
        double result = 0;
        MatrixC BdGT(L2); double E[L2]; 
#ifdef __GPU__
        eigen.Quasi_gpu(k_x,k_y,m,d,E,BdGT,'N',IsNotDev);
#else
        eigen.Quasi(k_x,k_y,m,d,E,BdGT,'N',IsNotDev);
#endif        
        for(int i = 0; i < L; i++){ 
            result += E[i]-m[i]+d[i]*d[i]/dilute[i]+(m[i]-m0)*(m[i]-m0)/dilute[i];
        }

        return result;
    }

    double W_eq(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, const double U, const double m0)
    {
        const bool IsNotDev = true;
        double result = 0;
        MatrixC BdGT(L2); double E[L2];
#ifdef __GPU__
        eigen.Quasi_gpu(k_x,k_y,m,d,E,BdGT,'N',IsNotDev);
#else        
        eigen.Quasi(k_x,k_y,m,d,E,BdGT,'N',IsNotDev);
#endif        
        for(int i = 0; i < L; i++){ 
            result += E[i]-m[i]+d[i]*d[i]/U+(m[i]-m0)*(m[i]-m0)/U;
        }

        return result;
    }

    std::pair<double,double> SFW_eq(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, const double V, const double beta)
    {
        MatrixC g_T(L), g(L), BdG(L2), PsiT(L2), Psi(L2), G_T(L2);
        double e[L], E[L2];
#ifdef __GPU__
        eigen.Single_gpu(k_x,k_y,e,g_T,'V',true);
        eigen.Quasi_gpu(k_x,k_y,m,d,E,PsiT,'V',true);
#else
        eigen.Single(k_x,k_y,e,g_T,'V',true);
        eigen.Quasi(k_x,k_y,m,d,E,PsiT,'V',true);
#endif
        g = g_T.Hermitian_transpose();
        Psi = PsiT.Hermitian_transpose();
        G_T.put_Block(g_T,0,L,0,L);
        G_T.put_Block(g_T,L,L2,L,L2);
        BdG = G_T.zgemm(1.0,Psi);

        MatrixC dh(L);
        eigen.Make_dH0(k_x,k_y,dh);
        MatrixC g_dh_g = (g_T.zgemm(1.0,dh)).zgemm(1.0,g);
  
        MatrixC M_conv(L2), M_geom(L2);
        dcomplex SFW_conv=0, SFW_geom=0;
#ifdef __GPU__
        thrDoubleComplex *M_conv_ptr, *M_geom_ptr, *BdG_ptr, *g_dh_g_ptr;
        cudaMalloc((void**)&M_conv_ptr,L2*L2*sizeof(thrDoubleComplex));
        cudaMalloc((void**)&M_geom_ptr,L2*L2*sizeof(thrDoubleComplex));
        cudaMalloc((void**)&BdG_ptr,L2*L2*sizeof(thrDoubleComplex));
        cudaMalloc((void**)&g_dh_g_ptr,L*L*sizeof(thrDoubleComplex));

        cudaMemcpy(BdG_ptr,&BdG[0],L2*L2*sizeof(thrDoubleComplex),cudaMemcpyHostToDevice);
        cudaMemcpy(g_dh_g_ptr,&g_dh_g[0],L*L*sizeof(thrDoubleComplex),cudaMemcpyHostToDevice);

        int numBlocks = L2, threadsPerBlock = L2; 
        Kernel::_CurrentOpCal_<<<numBlocks,threadsPerBlock>>>(L,L2,M_conv_ptr,M_geom_ptr,BdG_ptr,g_dh_g_ptr);
        cudaDeviceSynchronize();

        cudaMemcpy(&M_conv[0],M_conv_ptr,L2*L2*sizeof(thrDoubleComplex),cudaMemcpyDeviceToHost);
        cudaMemcpy(&M_geom[0],M_geom_ptr,L2*L2*sizeof(thrDoubleComplex),cudaMemcpyDeviceToHost);

        cudaFree(M_conv_ptr);
        cudaFree(M_geom_ptr);
        cudaFree(BdG_ptr);
        cudaFree(g_dh_g_ptr);
#else
        for (int idx=0;idx<L2*L2;++idx){
            int i=idx/L2, j=idx%L2;
            dcomplex M_conv_p=0, M_conv_m=0;
            dcomplex M_geom_p=0, M_geom_m=0;
            
            for(int m=0;m<L;m++)
            for(int n=0;n<L;n++){
                if(m==n){ M_conv_p += std::conj(BdG(m,i))*BdG(n,j)*g_dh_g(m,n); }
                else{ M_geom_p += std::conj(BdG(m,i))*BdG(n,j)*g_dh_g(m,n); }
            }

            for(int q=L;q<L2;q++)
            for(int p=L;p<L2;p++){
                if(p==q){ M_conv_m += std::conj(BdG(q,j))*BdG(p,i)*g_dh_g(q-L,p-L); }
                else{ M_geom_m += std::conj(BdG(q,j))*BdG(p,i)*g_dh_g(q-L,p-L); }
            }

            M_conv(i,j) = M_conv_p*M_conv_m; 
            M_geom(i,j) = M_geom_p*M_geom_m+M_conv_p*M_geom_m+M_geom_p*M_conv_m; 
        }
#endif
    	for(int i=0;i<L;i++)
        {
            double coef1 = beta/(2.0*std::pow(std::cosh(beta*E[L2-1-i]/2.0),2))*-4.0,
                coef2 = std::tanh(beta*E[L2-1-i]/2.)/E[L2-1-i]*-2.0;
            SFW_conv += coef1*M_conv(i,i)+coef2*(M_conv(L2-1-i,i)+M_conv(i,L2-1-i));	
            SFW_geom += coef1*M_geom(i,i)+coef2*(M_geom(L2-1-i,i)+M_geom(i,L2-1-i));
            for(int j=i+1;j<L;j++)
            {
                double coef3 = (std::tanh(beta*E[L2-1-i]/2.)+std::tanh(beta*E[L2-1-j]/2.))/(E[L2-1-i]+E[L2-1-j])*-2.0,
                    coef4 = (std::tanh(beta*E[L+i]/2.)+std::tanh(beta*E[L+j]/2.))/(E[L+i]+E[L+j])*-2.0;
                SFW_conv += coef3*(M_conv(i,L2-1-j)+M_conv(L2-1-j,i))+coef4*(M_conv(L-1-i,L+j)+M_conv(L+j,L-1-i));
                SFW_geom += coef3*(M_geom(i,L2-1-j)+M_geom(L2-1-j,i))+coef4*(M_geom(L-1-i,L+j)+M_geom(L+j,L-1-i));
            }
        }

        return std::pair<double,double>(std::real(SFW_conv)/V,std::real(SFW_geom)/V);
    }
};


template <typename EIGEN>
class GreenFunction
{
    EIGEN eigen;
    const int L,L2;
public:
    GreenFunction(const EIGEN &Eigen, const int size):
    eigen(Eigen),
    L(size), L2(size*2)
    {}

    void manybody_green_function(const double k_x, const double k_y, double * ldos, const std::vector<double> Ws, const int site, const std::vector<double> m, const std::vector<double> d)
    {	
        const bool IsNotDev = true;
        MatrixD BdGT(L2); double E[L2];
#ifdef __GPU__
        eigen.Quasi_gpu(k_x,k_y,m,d,E,BdGT,'V',IsNotDev);
#else 
        eigen.Quasi(k_x,k_y,m,d,E,BdGT,'V',IsNotDev);
#endif       
        for (int i=0;i<Ws.size();++i)
        {
            for (int j=0;j<L2;++j)
            {
                double prob = std::norm(BdGT[L2*j+site]);
                ldos[i] -= std::imag(prob/(dcomplex(Ws[i]-E[j],0.001)*M_PI));
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


template <typename EIGEN, typename PROFILE>
class Gauss : public Equations<EIGEN>
{
    const int Ngrid, L, L2;
    std::vector<double> k_x, k_y, xg, wg;
    const double kx_intv[2], ky_intv[2], RBZarea;
public:
    Gauss(const EIGEN &eigen, const int size, const int Ngrid):
    Equations<EIGEN>(eigen, size),
    Ngrid(Ngrid),
    L(size), L2(size*2),
    RBZarea(M_PI*M_PI*PROFILE::RBZ_x*PROFILE::RBZ_y),
    kx_intv{-M_PI/2.*PROFILE::RBZ_x,M_PI/2.*PROFILE::RBZ_x},
    ky_intv{-M_PI/2.*PROFILE::RBZ_y,M_PI/2.*PROFILE::RBZ_y}
    {
        k_x.resize(Ngrid);
        k_y.resize(Ngrid);
        GaussLobattoPoints(Ngrid,xg,wg);

        for(int i=0;i<Ngrid;i++){
            k_x[i] = (kx_intv[0]+kx_intv[1])/2.+(kx_intv[1]-kx_intv[0])*xg[i]/2.;
            k_y[i] = (ky_intv[0]+ky_intv[1])/2.+(ky_intv[1]-ky_intv[0])*xg[i]/2.;
        }
    }

    inline int get_L() const { return L; }
    inline Gauss& operator=(const Gauss &Copy) { return *this; }

    void Gap_Num (const std::vector<double> m, const std::vector<double> d, double * gaps, double * nums, const int ranks, const int nprocs)
    {
        std::vector<double> Pgaps(L,0), Pnums(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Gap_Num_eq(k_x[x],k_y[y],&Pgaps[0],&Pnums[0],(ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.,m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pgaps[0],&gaps[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));
        MPIERR(MPI_Reduce(&Pnums[0],&nums[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                gaps[i] = gaps[i]/RBZarea;
                nums[i] = nums[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&gaps[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
        MPIERR(MPI_Bcast(&nums[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void Num (const std::vector<double> m, const std::vector<double> d, double * nums, const int ranks, const int nprocs)
    {
        std::vector<double> Pnums(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Num_eq(k_x[x],k_y[y],&Pnums[0],(ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.,m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pnums[0],&nums[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                nums[i] = nums[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&nums[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void Gap (const std::vector<double> m, const std::vector<double> d, double * gaps, const int ranks, const int nprocs) 
    {
        std::vector<double> Pgaps(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Gap_eq(k_x[x],k_y[y],&Pgaps[0],(ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.,m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pgaps[0],&gaps[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                gaps[i] = gaps[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&gaps[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    double W (const std::vector<double> m, const std::vector<double> d, const std::vector<double> &dilute, const double m0, const int ranks, const int nprocs)
    {
        double Presult=0,result=0;

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = (ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.;
            Presult += this->W_eq(k_x[x],k_y[y],m,d,dilute,m0)*weight;
        }

        MPIERR(MPI_Reduce(&Presult,&result,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result = result/RBZarea;
        }

        MPIERR(MPI_Bcast(&result,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return result;
    }

    double W (const std::vector<double> m, const std::vector<double> d, const double U, const double m0, const int ranks, const int nprocs)
    {
        double Presult=0,result=0;

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = (ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.;
            Presult += this->W_eq(k_x[x],k_y[y],m,d,U,m0)*weight;
        }

        MPIERR(MPI_Reduce(&Presult,&result,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result = result/RBZarea;
        }

        MPIERR(MPI_Bcast(&result,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return result;
    }

    std::pair<double,double> SFW (const std::vector<double> m, const std::vector<double> d, const double V, const double beta, const int ranks, const int nprocs)
    {
        std::vector<double> Presult(2,0), result(2,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = (ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.;
            std::pair<double,double> sfw = this->SFW_eq(k_x[x],k_y[y],m,d,V,beta);
            Presult[0] += sfw.first*weight;
            Presult[1] += sfw.second*weight;
        }

        MPIERR(MPI_Reduce(&Presult[0],&result[0],2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result[0] = result[0]/RBZarea;
            result[1] = result[1]/RBZarea;
        }

        MPIERR(MPI_Bcast(&result[0],2,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return std::pair<double,double>(result[0],result[1]);
    }
};


template <typename EIGEN, typename PROFILE>
class Trapz : public Equations<EIGEN>
{
    const int Ngrid, L, L2;
    std::vector<double> k_x, k_y, wg;
    const double dk_x, dk_y, RBZarea;
public:
    Trapz(const EIGEN &eigen, const int size, const int Ngrid):
    Equations<EIGEN>(eigen, size),
    Ngrid(Ngrid),
    L(size), L2(size*2),
    RBZarea(M_PI*M_PI*PROFILE::RBZ_x*PROFILE::RBZ_y),
    dk_x(M_PI/((double)Ngrid)*PROFILE::RBZ_x),
    dk_y(M_PI/((double)Ngrid)*PROFILE::RBZ_y)
    {
        k_x.resize(Ngrid);
        k_y.resize(Ngrid);
        wg.resize(Ngrid);

        for(int i=0;i<Ngrid;i++){
            k_x[i] = (2.*i - (Ngrid-1) - 1.)/(2.*(Ngrid-1)) * M_PI*PROFILE::RBZ_x;	// (something)*(a length of the range)
            k_y[i] = (2.*i - (Ngrid-1) - 1.)/(2.*(Ngrid-1)) * M_PI*PROFILE::RBZ_y;
                
            if (i > 0 && i < Ngrid-1)
                wg[i] = 2.0;
            else 
                wg[i] = 1.0;
        }
    }

    inline int get_L() const { return L; }
    inline Trapz& operator=(const Trapz &Copy) { return *this; }

    void Gap_Num (const std::vector<double> m, const std::vector<double> d, double * gaps, double * nums, const int ranks, const int nprocs)
    {
        std::vector<double> Pgaps(L,0), Pnums(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Gap_Num_eq(k_x[x],k_y[y],&Pgaps[0],&Pnums[0],dk_x*dk_y*0.25*wg[x]*wg[y],m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pgaps[0],&gaps[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));
        MPIERR(MPI_Reduce(&Pnums[0],&nums[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                gaps[i] = gaps[i]/RBZarea;
                nums[i] = nums[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&gaps[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
        MPIERR(MPI_Bcast(&nums[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void Num (const std::vector<double> m, const std::vector<double> d, double * nums, const int ranks, const int nprocs)
    {
        std::vector<double> Pnums(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Num_eq(k_x[x],k_y[y],&Pnums[0],dk_x*dk_y*0.25*wg[x]*wg[y],m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pnums[0],&nums[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                nums[i] = nums[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&nums[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void Gap (const std::vector<double> m, const std::vector<double> d, double * gaps, const int ranks, const int nprocs)
    {
        std::vector<double> Pgaps(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Gap_eq(k_x[x],k_y[y],&Pgaps[0],dk_x*dk_y*0.25*wg[x]*wg[y],m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pgaps[0],&gaps[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                gaps[i] = gaps[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&gaps[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    double W (const std::vector<double> m, const std::vector<double> d, const std::vector<double> &dilute, const double m0, const int ranks, const int nprocs)
    {
        double Presult=0,result=0;

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = dk_x*dk_y*0.25*wg[x]*wg[y];
            Presult += this->W_eq(k_x[x],k_y[y],m,d,dilute,m0)*weight;
        }

        MPIERR(MPI_Reduce(&Presult,&result,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result = result/RBZarea;
        }

        MPIERR(MPI_Bcast(&result,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return result;
    }

    double W (const std::vector<double> m, const std::vector<double> d, const double U, const double m0, const int ranks, const int nprocs)
    {
        double Presult=0,result=0;

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = dk_x*dk_y*0.25*wg[x]*wg[y];
            Presult += this->W_eq(k_x[x],k_y[y],m,d,U,m0)*weight;
        }

        MPIERR(MPI_Reduce(&Presult,&result,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result = result/RBZarea;
        }

        MPIERR(MPI_Bcast(&result,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return result;
    }

    std::pair<double,double> SFW (const std::vector<double> m, const std::vector<double> d, const double V, const double beta, const int ranks, const int nprocs)
    {
        std::vector<double> Presult(2,0), result(2,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = dk_x*dk_y*0.25*wg[x]*wg[y];
            std::pair<double,double> sfw = this->SFW_eq(k_x[x],k_y[y],m,d,V,beta);
            Presult[0] += sfw.first*weight;
            Presult[1] += sfw.second*weight;
        }

        MPIERR(MPI_Reduce(&Presult[0],&result[0],2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result[0] = result[0]/RBZarea;
            result[1] = result[1]/RBZarea;
        }

        MPIERR(MPI_Bcast(&result[0],2,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return std::pair<double,double>(result[0],result[1]);
    }
};


template <typename EIGEN, typename PROFILE>
class Simp : public Equations<EIGEN>
{
    const int Ngrid, L, L2;
    std::vector<double> k_x, k_y, wg;
    const double dk_x, dk_y, RBZarea;
public:
    Simp(const EIGEN &eigen, const int size, const int Ngrid):
    Equations<EIGEN>(eigen, size),
    Ngrid(Ngrid),
    L(size), L2(size*2),
    RBZarea(M_PI*M_PI*PROFILE::RBZ_x*PROFILE::RBZ_y),
    dk_x(M_PI/((double)Ngrid)*PROFILE::RBZ_x),
    dk_y(M_PI/((double)Ngrid)*PROFILE::RBZ_y)
    {
        k_x.resize(Ngrid);
        k_y.resize(Ngrid);
        wg.resize(Ngrid);

        for(int i=0;i<Ngrid;i++){
            k_x[i] = (2.*i - (Ngrid-1) - 1.)/(2.*(Ngrid-1)) * M_PI*PROFILE::RBZ_x;	// (something)*(a length of the range)
            k_y[i] = (2.*i - (Ngrid-1) - 1.)/(2.*(Ngrid-1)) * M_PI*PROFILE::RBZ_y;
                
            if (i > 0 && i < Ngrid-1)
            {
                if (i%2==1)
                    wg[i] = 4.0/3.0;
                else
                    wg[i] = 2.0/3.0;
            }
            else
            {
                wg[i] = 1.0/3.0;
            }
        }
    }

    inline int get_L() const { return L; }
    inline Simp& operator=(const Simp &Copy) { return *this; }

    void Gap_Num (const std::vector<double> m, const std::vector<double> d, double * gaps, double * nums, const int ranks, const int nprocs)
    {
        std::vector<double> Pgaps(L,0), Pnums(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Gap_Num_eq(k_x[x],k_y[y],&Pgaps[0],&Pnums[0],dk_x*dk_y*0.25*wg[x]*wg[y],m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pgaps[0],&gaps[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));
        MPIERR(MPI_Reduce(&Pnums[0],&nums[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                gaps[i] = gaps[i]/RBZarea;
                nums[i] = nums[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&gaps[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
        MPIERR(MPI_Bcast(&nums[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void Num (const std::vector<double> m, const std::vector<double> d, double * nums, const int ranks, const int nprocs)
    {
        std::vector<double> Pnums(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Num_eq(k_x[x],k_y[y],&Pnums[0],dk_x*dk_y*0.25*wg[x]*wg[y],m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pnums[0],&nums[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                nums[i] = nums[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&nums[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void Gap (const std::vector<double> m, const std::vector<double> d, double * gaps, const int ranks, const int nprocs)
    {
        std::vector<double> Pgaps(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Gap_eq(k_x[x],k_y[y],&Pgaps[0],dk_x*dk_y*0.25*wg[x]*wg[y],m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pgaps[0],&gaps[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                gaps[i] = gaps[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&gaps[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    double W (const std::vector<double> m, const std::vector<double> d, const std::vector<double> &dilute, const double m0, const int ranks, const int nprocs)
    {
        double Presult=0,result=0;

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = dk_x*dk_y*0.25*wg[x]*wg[y];
            Presult += this->W_eq(k_x[x],k_y[y],m,d,dilute,m0)*weight;
        }

        MPIERR(MPI_Reduce(&Presult,&result,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result = result/RBZarea;
        }

        MPIERR(MPI_Bcast(&result,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return result;
    }

    double W (const std::vector<double> m, const std::vector<double> d, const double U, const double m0, const int ranks, const int nprocs)
    {
        double Presult=0,result=0;

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = dk_x*dk_y*0.25*wg[x]*wg[y];
            Presult += this->W_eq(k_x[x],k_y[y],m,d,U,m0)*weight;
        }

        MPIERR(MPI_Reduce(&Presult,&result,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result = result/RBZarea;
        }

        MPIERR(MPI_Bcast(&result,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return result;
    }

    std::pair<double,double> SFW (const std::vector<double> m, const std::vector<double> d, const double V, const double beta, const int ranks, const int nprocs)
    {
        std::vector<double> Presult(2,0), result(2,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = dk_x*dk_y*0.25*wg[x]*wg[y];
            std::pair<double,double> sfw = this->SFW_eq(k_x[x],k_y[y],m,d,V,beta);
            Presult[0] += sfw.first*weight;
            Presult[1] += sfw.second*weight;
        }

        MPIERR(MPI_Reduce(&Presult[0],&result[0],2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result[0] = result[0]/RBZarea;
            result[1] = result[1]/RBZarea;
        }

        MPIERR(MPI_Bcast(&result[0],2,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return std::pair<double,double>(result[0],result[1]);
    }
};


template <typename EIGEN, typename PROFILE>
class Stack : public GreenFunction<EIGEN>
{
    const int Ngrid, L, L2;
    std::vector<double> k_x, k_y;
    const double dk_x, dk_y, RBZarea;
public:
    Stack(const EIGEN &eigen, const int size, const int Ngrid):
    GreenFunction<EIGEN>(eigen, size),
    Ngrid(Ngrid),
    L(size), L2(size*2),
    RBZarea(M_PI*M_PI*PROFILE::RBZ_x*PROFILE::RBZ_y),
    dk_x(M_PI/((double)Ngrid)*PROFILE::RBZ_x),
    dk_y(M_PI/((double)Ngrid)*PROFILE::RBZ_y)
    {
        k_x.resize(Ngrid);
        k_y.resize(Ngrid);

        for(int i=0;i<Ngrid;i++){
            k_x[i] = (2.*i - (Ngrid-1) - 1.)/(2.*(Ngrid-1)) * M_PI*PROFILE::RBZ_x;	// (something)*(a length of the range)
            k_y[i] = (2.*i - (Ngrid-1) - 1.)/(2.*(Ngrid-1)) * M_PI*PROFILE::RBZ_y;
        }
    }

    inline int get_L() const { return L; }
    inline Stack& operator=(const Stack &Copy) { return *this; }

    void ManybodyLDOS (const std::vector<double> m, const std::vector<double> d, double * ldos, const int site, const std::vector<double> Ws, const int ranks, const int nprocs)
    {
        std::vector<double> Pldos(Ws.size(),0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->manybody_green_function(k_x[x],k_y[y],&Pldos[0],Ws,site,m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pldos[0],&ldos[0],Ws.size(),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<Ws.size();i++){ 
                ldos[i] = ldos[i]/(Ngrid*Ngrid);
		    }
        }

        MPIERR(MPI_Bcast(&ldos[0],Ws.size(),MPI_DOUBLE,0,MPI_COMM_WORLD));
    }
};


#endif
