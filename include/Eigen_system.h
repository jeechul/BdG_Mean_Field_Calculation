#ifndef EIGEN_SYSTEM_H
#define EIGEN_SYSTEM_H

#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include "Matrix_operator.h"

#ifdef __GPU__
#include "EIGEN_cusolver.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#define lapack_complex_double std::complex<double>
#include <mkl/mkl_lapacke.h>

// Default : col-major
namespace EIGENSOLVER 
{

int zheev(MatrixC &H0, double * E, const int N, const char JOBZ)
{
	const char UPLO = 'L'; // if JOBZ is 'V', val and vec. if JOBZ is 'N', val only.
	const int LDA = N;
    int INFO = 0;

	INFO = LAPACKE_zheev(LAPACK_COL_MAJOR, JOBZ, UPLO, N, &H0[0], LDA, E);
    return INFO;
};

int dsyev(double * H0, double * E, const int N, const char JOBZ)
{
	const char UPLO = 'L'; // if JOBZ is 'V', val and vec. if JOBZ is 'N', val only.
	const int LDA = N;
	int INFO = 0;

    int Major = LAPACK_COL_MAJOR;

	INFO = LAPACKE_dsyev(Major, JOBZ, UPLO, N, H0, LDA, E);
    return INFO;
};

int dsyev(MatrixD &H0, double * E, const int N, const char JOBZ)
{
	const char UPLO = 'L'; // if JOBZ is 'V', val and vec. if JOBZ is 'N', val only.
	const int LDA = N;
	int INFO = 0;

	INFO = LAPACKE_dsyev(LAPACK_COL_MAJOR, JOBZ, UPLO, N, &H0[0], LDA, E);
    return INFO;
};

#ifdef __GPU__
void cuDsyev(double * H0, double * E, const int N, const char JOBZ)
{
	thrust::device_vector<double> V(N*N);
	thrust::device_vector<double> W(N);
    thrust::copy(H0,H0+N*N,V.begin());

	double *ptr_V = thrust::raw_pointer_cast(V.data());
	double *ptr_W = thrust::raw_pointer_cast(W.data()); 

	cusolverDsyevd(N,ptr_V,ptr_W,JOBZ);
	
	thrust::copy(V.begin(),V.end(),H0);
	thrust::copy(W.begin(),W.end(),E);
    cudaDeviceSynchronize();
};

void cuDsyev(MatrixD &H0, double * E, const int N, const char JOBZ)
{
	thrust::device_vector<double> V(H0.data());
	thrust::device_vector<double> W(N);

	double *ptr_V = thrust::raw_pointer_cast(V.data());
	double *ptr_W = thrust::raw_pointer_cast(W.data()); 

	cusolverDsyevd(N,ptr_V,ptr_W,JOBZ);
	
	thrust::copy(V.begin(),V.end(),&H0[0]);
	thrust::copy(W.begin(),W.end(),E);
    cudaDeviceSynchronize();
};

void cuZheev(dcomplex * H0, double * E, const int N, const char JOBZ)
{
	thrust::device_vector<dcomplex> V(N*N);
	thrust::device_vector<double> W(N);
    thrust::copy(H0,H0+N*N,V.begin());

	cuDoubleComplex *ptr_V = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(V.data()));
	double *ptr_W = thrust::raw_pointer_cast(W.data()); 

	cusolverZheevd(N,ptr_V,ptr_W,JOBZ);
	
	thrust::copy(V.begin(),V.end(),H0);
	thrust::copy(W.begin(),W.end(),E);
    cudaDeviceSynchronize();
};

void cuZheev(MatrixC &H0, double * E, const int N, const char JOBZ)
{
	thrust::device_vector<dcomplex> V(H0.data());
	thrust::device_vector<double> W(N);

	cuDoubleComplex *ptr_V = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(V.data()));
	double *ptr_W = thrust::raw_pointer_cast(W.data()); 

	cusolverZheevd(N,ptr_V,ptr_W,JOBZ);
	
	thrust::copy(V.begin(),V.end(),&H0[0]);
	thrust::copy(W.begin(),W.end(),E);
    cudaDeviceSynchronize();
};
#endif

} // namespace EIGENSOLVER


namespace BdG_Tech_RealPair
{

template <typename LATTICE, typename PROFILE>
class EigenSet
{
    const int L, L2;
public:
    EigenSet(const int size):
    L(size), L2(size*2)
    {}

    void Make_dH0(const double k_x, const double k_y, MatrixC &dH0)
    {
        LATTICE geometry(k_x,k_y,L);
        dH0 = geometry.make_mat(PROFILE::lattices,false);
    }

	void Single(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		Evec = geometry.make_mat(PROFILE::lattices,IsNotDev);
		EIGENSOLVER::zheev(Evec,Eval,L,JOBZ);
	}
#ifdef __GPU__
    void Single_gpu(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		Evec = geometry.make_mat(PROFILE::lattices,IsNotDev);
		EIGENSOLVER::cuZheev(Evec,Eval,L,JOBZ);
	}
#endif
    void Single0(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
		geometry.make_mat_real(Evec,PROFILE::lattices,IsNotDev);
		EIGENSOLVER::dsyev(Evec,Eval,L,JOBZ);
	}
#ifdef __GPU__
    void Single0_gpu(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
		geometry.make_mat_real(Evec,PROFILE::lattices,IsNotDev);
		EIGENSOLVER::cuDsyev(Evec,Eval,L,JOBZ);
	}
#endif
	void Quasi(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat(PROFILE::lattices,IsNotDev),L);

		for (int i=0;i<L;++i) h(i,i) = -m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0);

		EIGENSOLVER::zheev(eig_vec,eig_val,L2,JOBZ);
	}
#ifdef __GPU__
    void Quasi_gpu(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat(PROFILE::lattices,IsNotDev),L);

		for (int i=0;i<L;++i) h(i,i) = -m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::cuZheev(eig_vec,eig_val,L2,JOBZ);
	}
#endif
    void Quasi0(const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixD &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
        MatrixD h(L);
		geometry.make_mat_real(&h[0],PROFILE::lattices,IsNotDev);

		for (int i=0;i<L;++i) h(i,i) = -m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::dsyev(eig_vec,eig_val,L2,JOBZ);
	}
#ifdef __GPU__
    void Quasi0_gpu(const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixD &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
        MatrixD h(L);
		geometry.make_mat_real(&h[0],PROFILE::lattices,IsNotDev);

		for (int i=0;i<L;++i) h(i,i) = -m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::cuDsyev(eig_vec,eig_val,L2,JOBZ);
	}
#endif
};

template <typename LATTICE, typename PROFILE>
class EigenSetMCM // g^{up triangle} = g^{down triangle} : with inversion symmetry
{
    double *gamma;
    const int L, L2;
    const bool IsRev;
public:
    EigenSetMCM(const int size, double *g, bool IsRev_=false):
    gamma(g),
    IsRev(IsRev_),
    L(size), L2(size*2)
    {}

    inline void set_xconf(double *gamma_){ gamma = gamma_; }

    void Make_dH0(const double k_x, const double k_y, MatrixC &dH0)
    {
        LATTICE geometry(k_x,k_y,L);
        dH0 = geometry.make_mat(PROFILE::lattices,gamma,false);
        if (IsRev) dH0.minus();
    }

	void Single(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		Evec = geometry.make_mat(PROFILE::lattices,gamma,IsNotDev);
        for(int i=0;i<L;++i) Evec(i,i) = -2.0*gamma[i]*gamma[i];
        if (IsRev){
            Evec.minus();
		    EIGENSOLVER::zheev(Evec,Eval,L,JOBZ);
        }
        else {
		    EIGENSOLVER::zheev(Evec,Eval,L,JOBZ);
        }
	}
#ifdef __GPU__
    void Single_gpu(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		Evec = geometry.make_mat(PROFILE::lattices,gamma,IsNotDev);
        for(int i=0;i<L;++i) Evec(i,i) = -2.0*gamma[i]*gamma[i];
        if (IsRev){
            Evec.minus();
		    EIGENSOLVER::cuZheev(Evec,Eval,L,JOBZ);
        }
        else {
		    EIGENSOLVER::cuZheev(Evec,Eval,L,JOBZ);
        }
	}
#endif
    void Single0(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
		geometry.make_mat_real(Evec,PROFILE::lattices,gamma,IsNotDev);
        for(int i=0;i<L;++i) Evec[i*L+i] = -2.0*gamma[i]*gamma[i];
		EIGENSOLVER::dsyev(Evec,Eval,L,JOBZ); // it is transposed and do not support reversed band
	}
#ifdef __GPU__
    void Single0_gpu(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
		geometry.make_mat_real(Evec,PROFILE::lattices,gamma,IsNotDev);
        for(int i=0;i<L;++i) Evec[i*L+i] = -2.0*gamma[i]*gamma[i];
		EIGENSOLVER::cuDsyev(Evec,Eval,L,JOBZ);
	}
#endif
	void Quasi(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat(PROFILE::lattices,gamma,IsNotDev),L);

        if (IsRev){
            for(int i=0;i<L;++i) h(i,i) = -2.0*gamma[i]*gamma[i]+m[i];
            h.minus();
        }
        else {
            for(int i=0;i<L;++i) h(i,i) = -2.0*gamma[i]*gamma[i]-m[i];
        }

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::zheev(eig_vec,eig_val,L2,JOBZ);
	}
#ifdef __GPU__
    void Quasi_gpu(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat(PROFILE::lattices,gamma,IsNotDev),L);

        if (IsRev){
		    for (int i=0;i<L;++i) h(i,i) = -2.0*gamma[i]*gamma[i]+m[i];
            h.minus();
        }
        else {
		    for (int i=0;i<L;++i) h(i,i) = -2.0*gamma[i]*gamma[i]-m[i];
        }

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::cuZheev(eig_vec,eig_val,L2,JOBZ);
	}
#endif
    void Quasi0(const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixD &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
        MatrixD h(L);
		geometry.make_mat_real(&h[0],PROFILE::lattices,gamma,IsNotDev);

		for (int i=0;i<L;++i) h(i,i) = -2.0*gamma[i]*gamma[i]-m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::dsyev(eig_vec,eig_val,L2,JOBZ);
	}
#ifdef __GPU__
    void Quasi0_gpu(const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixD &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
        MatrixD h(L);
		geometry.make_mat_real(&h[0],PROFILE::lattices,gamma,IsNotDev);

		for (int i=0;i<L;++i) h(i,i) = -2.0*gamma[i]*gamma[i]-m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::cuDsyev(eig_vec,eig_val,L2,JOBZ);
	}
#endif
};

template <typename LATTICE, typename PROFILE>
class EigenSetRandomPot
{
    double *X;
    const int L, L2;
public:
    EigenSetRandomPot(const int size, double *X):
    X(X),
    L(size), L2(size*2)
    {}

    inline void set_xconf(double *X_){ X = X_; }

    void Make_dH0(const double k_x, const double k_y, MatrixC &dH0)
    {
        LATTICE geometry(k_x,k_y,L);
        dH0 = geometry.make_mat(PROFILE::lattices,false);
    }

	void Single(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		Evec = geometry.make_mat(PROFILE::lattices,IsNotDev);
        for (int i=0; i<L; ++i) Evec(i,i) = -X[i];
		EIGENSOLVER::zheev(Evec,Eval,L,JOBZ);
	}
#ifdef __GPU__
    void Single_gpu(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		Evec = geometry.make_mat(PROFILE::lattices,IsNotDev);
        for (int i=0; i<L; ++i) Evec(i,i) = -X[i];
		EIGENSOLVER::cuZheev(Evec,Eval,L,JOBZ);
	}
#endif
    void Single0(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
		geometry.make_mat_real(Evec,PROFILE::lattices,IsNotDev);
        for (int i=0; i<L; ++i) Evec[i*L+i] = -X[i];
		EIGENSOLVER::dsyev(Evec,Eval,L,JOBZ);
	}
#ifdef __GPU__
    void Single0_gpu(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
		geometry.make_mat_real(Evec,PROFILE::lattices,IsNotDev);
        for (int i=0; i<L; ++i) Evec[i*L+i] = -X[i];
		EIGENSOLVER::cuDsyev(Evec,Eval,L,JOBZ);
	}
#endif
	void Quasi(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat(PROFILE::lattices,IsNotDev),L);
        for (int i=0; i<L; ++i) h(i,i) = -X[i]-m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::zheev(eig_vec,eig_val,L2,JOBZ);
	}
#ifdef __GPU__
    void Quasi_gpu(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat(PROFILE::lattices,IsNotDev),L);

		for (int i=0;i<L;++i) h(i,i) = -X[i]-m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::cuZheev(eig_vec,eig_val,L2,JOBZ);
	}
#endif
    void Quasi0(const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixD &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
        MatrixD h(L);
		geometry.make_mat_real(&h[0],PROFILE::lattices,IsNotDev);

		for (int i=0;i<L;++i) h(i,i) = -X[i]-m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::dsyev(eig_vec,eig_val,L2,JOBZ);
	}
#ifdef __GPU__
    void Quasi0_gpu(const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixD &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
        MatrixD h(L);
		geometry.make_mat_real(&h[0],PROFILE::lattices,IsNotDev);

		for (int i=0;i<L;++i) h(i,i) = -X[i]-m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::cuDsyev(eig_vec,eig_val,L2,JOBZ);
	}
#endif
};

template <typename LATTICE, typename PROFILE>
class EigenSetMCMonlyHop
{
    double *gamma;
    const int L, L2;
    const bool IsRev;
public:
    EigenSetMCMonlyHop(const int size, double *g, bool IsRev_=false):
    gamma(g),
    IsRev(IsRev_),
    L(size), L2(size*2)
    {}

    inline void set_xconf(double *gamma_){ gamma = gamma_; }

    void Make_dH0(const double k_x, const double k_y, MatrixC &dH0)
    {
        LATTICE geometry(k_x,k_y,L);
        dH0 = geometry.make_mat(PROFILE::lattices,gamma,false);
        if (IsRev) dH0.minus();
    }

	void Single(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		Evec = geometry.make_mat(PROFILE::lattices,gamma,IsNotDev);
        if (IsRev){
            Evec.minus();
		    EIGENSOLVER::zheev(Evec,Eval,L,JOBZ);
        }
        else {
		    EIGENSOLVER::zheev(Evec,Eval,L,JOBZ);
        }
	}
#ifdef __GPU__
    void Single_gpu(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		Evec = geometry.make_mat(PROFILE::lattices,gamma,IsNotDev);
        if (IsRev){
            Evec.minus();
		    EIGENSOLVER::cuZheev(Evec,Eval,L,JOBZ);
        }
        else {
		    EIGENSOLVER::cuZheev(Evec,Eval,L,JOBZ);
        }
	}
#endif
    void Single0(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
		geometry.make_mat_real(Evec,PROFILE::lattices,gamma,IsNotDev);
		EIGENSOLVER::dsyev(Evec,Eval,L,JOBZ); // it is transposed and do not support reversed band
	}
#ifdef __GPU__
    void Single0_gpu(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
		geometry.make_mat_real(Evec,PROFILE::lattices,gamma,IsNotDev);
		EIGENSOLVER::cuDsyev(Evec,Eval,L,JOBZ);
	}
#endif
	void Quasi(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat(PROFILE::lattices,gamma,IsNotDev),L);

        if (IsRev){
            for(int i=0;i<L;++i) h(i,i) = m[i];
            h.minus();
        }
        else {
            for(int i=0;i<L;++i) h(i,i) = -m[i];
        }

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::zheev(eig_vec,eig_val,L2,JOBZ);
	}
#ifdef __GPU__
    void Quasi_gpu(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat(PROFILE::lattices,gamma,IsNotDev),L);

        if (IsRev){
		    for (int i=0;i<L;++i) h(i,i) = m[i];
            h.minus();
        }
        else {
		    for (int i=0;i<L;++i) h(i,i) = -m[i];
        }

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::cuZheev(eig_vec,eig_val,L2,JOBZ);
	}
#endif
    void Quasi0(const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixD &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
        MatrixD h(L);
		geometry.make_mat_real(&h[0],PROFILE::lattices,gamma,IsNotDev);

		for (int i=0;i<L;++i) h(i,i) = -m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::dsyev(eig_vec,eig_val,L2,JOBZ);
	}
#ifdef __GPU__
    void Quasi0_gpu(const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixD &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
        MatrixD h(L);
		geometry.make_mat_real(&h[0],PROFILE::lattices,gamma,IsNotDev);

		for (int i=0;i<L;++i) h(i,i) = -m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::cuDsyev(eig_vec,eig_val,L2,JOBZ);
	}
#endif
};

template <typename LATTICE, typename PROFILE>
class EigenSetBDM // g^{up triangle} = 1/g^{down triangle} : without inversion symmetry
{
    double *gamma;
    const int L, L2;
    const bool IsRev;
public:
    EigenSetBDM(const int size, double *g, bool IsRev_=false):
    gamma(g),
    IsRev(IsRev_),
    L(size), L2(size*2)
    {}

    inline void set_xconf(double *gamma_){ gamma = gamma_; }

    void Make_dH0(const double k_x, const double k_y, MatrixC &dH0)
    {
        LATTICE geometry(k_x,k_y,L);
        dH0 = geometry.make_mat_BDM(PROFILE::lattices,gamma,false);
        if (IsRev) dH0.minus();
    }

    void Single(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev, bool Switch=true)
    {
        LATTICE geometry(k_x,k_y,L);
        Evec = geometry.make_mat_BDM(PROFILE::lattices,gamma,IsNotDev);
        for(int i=0;i<L;++i){
            Evec(i,i) = -gamma[i]*gamma[i]-1.0/(gamma[i]*gamma[i]);
        }
        if (IsRev){
            Evec.minus(); 
            EIGENSOLVER::zheev(Evec,Eval,L,JOBZ);
        }
        else
            EIGENSOLVER::zheev(Evec,Eval,L,JOBZ);
    }
#ifdef __GPU__
    void Single_gpu(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		Evec = geometry.make_mat_BDM(PROFILE::lattices,gamma,IsNotDev);
        for(int i=0;i<L;++i) Evec(i,i) = -gamma[i]*gamma[i]-1.0/(gamma[i]*gamma[i]);
        if (IsRev){
            Evec.minus();
		    EIGENSOLVER::cuZheev(Evec,Eval,L,JOBZ);
        }
        else {
		    EIGENSOLVER::cuZheev(Evec,Eval,L,JOBZ);
        }
	}
#endif
    void Single0(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
    {
        LATTICE geometry(0.0,0.0,L);
        geometry.make_mat_BDM_real(Evec,PROFILE::lattices,gamma,IsNotDev);
        for(int i=0;i<L;++i) Evec[i*L+i] = -gamma[i]*gamma[i]-1.0/(gamma[i]*gamma[i]);
        EIGENSOLVER::dsyev(Evec,Eval,L,JOBZ); // it is transposed and do not support reversed band
    }
#ifdef __GPU__
    void Single0_gpu(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
		geometry.make_mat_BDM_real(Evec,PROFILE::lattices,gamma,IsNotDev);
        for(int i=0;i<L;++i) Evec[i*L+i] = -gamma[i]*gamma[i]-1.0/(gamma[i]*gamma[i]);
		EIGENSOLVER::cuDsyev(Evec,Eval,L,JOBZ);
	}
#endif
    void Quasi(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev, bool Switch=true)
    {
        LATTICE geometry(k_x,k_y,L);
        MatrixC h(geometry.make_mat_BDM(PROFILE::lattices,gamma,IsNotDev),L);

        if (IsRev){
            for(int i=0;i<L;++i) h(i,i) = -gamma[i]*gamma[i]-1.0/(gamma[i]*gamma[i])+m[i];
            h.minus();
        }
        else {
            for(int i=0;i<L;++i) h(i,i) = -gamma[i]*gamma[i]-1.0/(gamma[i]*gamma[i])-m[i];
        }

        eig_vec.put_Block(h,0,L,0,L);
        eig_vec.put_Block(h.minus(),L,L2,L,L2);
        eig_vec.fill_diagonal(d,0,L);
        eig_vec.fill_diagonal(d,L,0);

        EIGENSOLVER::zheev(eig_vec,eig_val,L2,JOBZ);
    }
#ifdef __GPU__
    void Quasi_gpu(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat_BDM(PROFILE::lattices,gamma,IsNotDev),L);

        if (IsRev){
		    for (int i=0;i<L;++i) h(i,i) = -gamma[i]*gamma[i]-1.0/(gamma[i]*gamma[i])+m[i];
            h.minus();
        }
        else {
		    for (int i=0;i<L;++i) h(i,i) = -gamma[i]*gamma[i]-1.0/(gamma[i]*gamma[i])-m[i];
        }

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::cuZheev(eig_vec,eig_val,L2,JOBZ);
	}
#endif
    void Quasi0(const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixD &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
        MatrixD h(L);
		geometry.make_mat_BDM_real(&h[0],PROFILE::lattices,gamma,IsNotDev);

		for (int i=0;i<L;++i) h(i,i) = -gamma[i]*gamma[i]-1.0/(gamma[i]*gamma[i])-m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::dsyev(eig_vec,eig_val,L2,JOBZ);
	}
#ifdef __GPU__
    void Quasi0_gpu(const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixD &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
        MatrixD h(L);
		geometry.make_mat_BDM_real(&h[0],PROFILE::lattices,gamma,IsNotDev);

		for (int i=0;i<L;++i) h(i,i) = -gamma[i]*gamma[i]-1.0/(gamma[i]*gamma[i])-m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::cuDsyev(eig_vec,eig_val,L2,JOBZ);
	}
#endif
};

template <typename LATTICE, typename PROFILE>
class EigenSetTilt
{
    double eta;
    const int L, L2;
public:
    EigenSetTilt(const int size, double eta):
    eta(eta),
    L(size), L2(size*2)
    {}

    void Make_dH0(const double k_x, const double k_y, MatrixC &dH0)
    {
        LATTICE geometry(k_x,k_y,L);
        dH0 = geometry.make_mat_tilt(PROFILE::lattices,eta,false);
    }

	void Single(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		Evec = geometry.make_mat_tilt(PROFILE::lattices,eta,IsNotDev);
		EIGENSOLVER::zheev(Evec,Eval,L,JOBZ);
	}
#ifdef __GPU__
    void Single_gpu(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		Evec = geometry.make_mat_tilt(PROFILE::lattices,eta,IsNotDev);
		EIGENSOLVER::cuZheev(Evec,Eval,L,JOBZ);
	}
#endif
    void Single0(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
		geometry.make_mat_tilt_real(Evec,PROFILE::lattices,eta,IsNotDev);
		EIGENSOLVER::dsyev(Evec,Eval,L,JOBZ);
	}
#ifdef __GPU__
    void Single0_gpu(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
		geometry.make_mat_tilt_real(Evec,PROFILE::lattices,eta,IsNotDev);
		EIGENSOLVER::cuDsyev(Evec,Eval,L,JOBZ);
	}
#endif
	void Quasi(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat_tilt(PROFILE::lattices,eta,IsNotDev),L);

		for (int i=0;i<L;++i) h(i,i) = -m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0);

		EIGENSOLVER::zheev(eig_vec,eig_val,L2,JOBZ);
	}
#ifdef __GPU__
    void Quasi_gpu(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat_tilt(PROFILE::lattices,eta,IsNotDev),L);

		for (int i=0;i<L;++i) h(i,i) = -m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::cuZheev(eig_vec,eig_val,L2,JOBZ);
	}
#endif
    void Quasi0(const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixD &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
        MatrixD h(L);
		geometry.make_mat_tilt_real(&h[0],PROFILE::lattices,eta,IsNotDev);

		for (int i=0;i<L;++i) h(i,i) = -m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::dsyev(eig_vec,eig_val,L2,JOBZ);
	}
#ifdef __GPU__
    void Quasi0_gpu(const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixD &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
        MatrixD h(L);
		geometry.make_mat_tilt_real(&h[0],PROFILE::lattices,eta,IsNotDev);

		for (int i=0;i<L;++i) h(i,i) = -m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);
		eig_vec.fill_diagonal(d,0,L);
		eig_vec.fill_diagonal(d,L,0); 

		EIGENSOLVER::cuDsyev(eig_vec,eig_val,L2,JOBZ);
	}
#endif
};


} // namespace BdG_Tech_RealPair


namespace BdG_Tech_ComplexPair
{

template <typename LATTICE, typename PROFILE>
class EigenSet
{
    const int L, L2;
public:
    EigenSet(const int size):
    L(size), L2(size*2)
    {}

    void Make_dH0(const double k_x, const double k_y, MatrixC &dH0)
    {
        LATTICE geometry(k_x,k_y,L);
        dH0 = geometry.make_mat(PROFILE::lattices,gamma,false);
    }

	void Single(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		Evec = geometry.make_mat(PROFILE::lattices,IsNotDev);
		EIGENSOLVER::zheev(Evec,Eval,L,JOBZ);
	}

    void Single0(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
		geometry.make_mat_real(Evec,PROFILE::lattices,IsNotDev);
		EIGENSOLVER::dsyev(Evec,Eval,L,JOBZ);
	}
#ifdef __GPU__
    void Single0_gpu(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
		geometry.make_mat_real(Evec,PROFILE::lattices,IsNotDev);
		EIGENSOLVER::cuDsyev(Evec,Eval,L,JOBZ);
	}
#endif
	void Quasi(const double k_x, const double k_y, const std::vector<double> m, const std::vector<dcomplex> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat(PROFILE::lattices,IsNotDev),L);

		for (int i=0;i<L;++i) h(i,i) = -m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);

        for (int i=0;i<L;++i)
        {
            eig_vec(i,i+L) = d[i];
            eig_vec(i+L,i) = std::conj(d[i]); 
        }

		EIGENSOLVER::zheev(eig_vec,eig_val,L2,JOBZ);
	}
#ifdef __GPU__
    void Quasi_gpu(const double k_x, const double k_y, const std::vector<double> m, const std::vector<dcomplex> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat(PROFILE::lattices,IsNotDev),L);

		for (int i=0;i<L;++i) h(i,i) = -m[i];

		eig_vec.put_Block(h,0,L,0,L);
		eig_vec.put_Block(h.minus(),L,L2,L,L2);

        for (int i=0;i<L;++i)
        {
            eig_vec(i,i+L) = d[i];
            eig_vec(i+L,i) = std::conj(d[i]); 
        }

		EIGENSOLVER::cuZheev(eig_vec,eig_val,L2,JOBZ);
	}
#endif
};

} // namespace BdG_Tech_ComplexPair

#endif
