#ifndef __LUSOLVER__
#define __LUSOLVER__

#include <iostream>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <cuComplex.h>
#include <cusolverDn.h>
#include <assert.h>

void cusolverLUsolve(const int m, const int nrhs, double *d_A, double *d_B)
{
	cusolverDnHandle_t cusolverH = NULL;
	cudaStream_t stream = NULL;
	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;

	const int lda = m;
	const int ldb = m;
	int *d_Ipiv = NULL;
	int *d_info = NULL;
	int info = 0;
	int lwork = 0;
	double *d_work = NULL;

	/* step 1: create cusolver handle, bind a stream */
	status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == status);
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	status = cusolverDnSetStream(cusolverH, stream);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* step 2: copy A to device */
	cudaMalloc ((void**)&d_Ipiv, sizeof(int) * m);
	cudaMalloc ((void**)&d_info, sizeof(int));

	/* step 3: query working space of getrf */
	status = cusolverDnDgetrf_bufferSize(cusolverH,m,m,d_A,lda,&lwork);
	assert(CUSOLVER_STATUS_SUCCESS == status);
	cudaMalloc((void**)&d_work, sizeof(double)*lwork);

	/* step 4: LU factorization */
	status = cusolverDnDgetrf(cusolverH,m,m,d_A,lda,d_work,d_Ipiv,d_info);
	cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
	if ( 0 > info ){
		std::cout<<(-info)<<"-th parameter is wrong \n"<<std::endl;
		exit(1);
	}

	/* step 5: solve A*X = B */
	status = cusolverDnDgetrs(cusolverH,CUBLAS_OP_N,m,nrhs,d_A,lda,d_Ipiv,d_B,ldb,d_info);
	cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* free resources */
	if (d_Ipiv ) cudaFree(d_Ipiv);
	if (d_info ) cudaFree(d_info);
	if (d_work ) cudaFree(d_work);
	if (cusolverH ) cusolverDnDestroy(cusolverH);
	if (stream ) cudaStreamDestroy(stream);
}

#endif








