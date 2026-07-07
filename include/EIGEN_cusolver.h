#ifndef __EIGEN__
#define __EIGEN__

#include <iostream>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <cuComplex.h>
#include <cusolverDn.h>
#include <cusolverMg.h>
#include <assert.h>

void cusolverDsyevd(const int m, double *d_A, double *d_W, const char JOBZ)
{
	cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	
	const int lda = m;
	//double *d_A = NULL;
    //double *d_W = NULL;
    int *devInfo = NULL;
    double *d_work = NULL;
    int  lwork = 0;
    int info_gpu = 0;

	// step 1: create cusolver/cublas handle
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	// step 2: copy A and B to device
	//cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
    //cudaMalloc ((void**)&d_W, sizeof(double) * m);
    cudaMalloc ((void**)&devInfo, sizeof(int));

	//cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
	
	// step 3: query working space of syevd
	cusolverEigMode_t jobz; 
    if (JOBZ == 'V') 
        jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and vectors.
    else if (JOBZ == 'N')
        jobz = CUSOLVER_EIG_MODE_NOVECTOR; // compute eigenvalues only.
    else 
        assert(0);
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolver_status = cusolverDnDsyevd_bufferSize(cusolverH,jobz,uplo,m,d_A,lda,d_W,&lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cudaMalloc((void**)&d_work, sizeof(double)*lwork);

	// step 4: compute spectrum
	cusolver_status = cusolverDnDsyevd(cusolverH,jobz,uplo,m,d_A,lda,d_W,d_work,lwork,devInfo);
    cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	//cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
	//cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
    cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(0 == info_gpu);

	//cudaFree(d_W);
	//cudaFree(d_A);
    cudaFree(devInfo);
    cudaFree(d_work);
    cusolverDnDestroy(cusolverH);
}

void cusolverZheevd(const int m, cuDoubleComplex *d_A, double *d_W, const char JOBZ)
{
	cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	
	const int lda = m;
	//double *d_A = NULL;
    //double *d_W = NULL;
    int *devInfo = NULL;
    cuDoubleComplex *d_work = NULL;
    int lwork = 0;
    int info_gpu = 0;

	// step 1: create cusolver/cublas handle
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	// step 2: copy A and B to device
	//cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
    //cudaMalloc ((void**)&d_W, sizeof(double) * m);
    cudaMalloc ((void**)&devInfo, sizeof(int));

	//cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
	
	// step 3: query working space of zheevd
    cusolverEigMode_t jobz; 
    if (JOBZ == 'V') 
        jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and vectors.
    else if (JOBZ == 'N')
        jobz = CUSOLVER_EIG_MODE_NOVECTOR; // compute eigenvalues only.
    else 
        assert(0);
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolver_status = cusolverDnZheevd_bufferSize(cusolverH,jobz,uplo,m,d_A,lda,d_W,&lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex)*lwork);

	// step 4: compute spectrum
	cusolver_status = cusolverDnZheevd(cusolverH,jobz,uplo,m,d_A,lda,d_W,d_work,lwork,devInfo);
    cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	//cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
	//cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
    cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(0 == info_gpu);

	//cudaFree(d_W);
	//cudaFree(d_A);
    cudaFree(devInfo);
    cudaFree(d_work);
    cusolverDnDestroy(cusolverH);
}

void cusolverDsyevdj(const int m, double *d_A, double *d_W)
{
	cusolverDnHandle_t cusolverH = NULL;
 	cudaStream_t stream = NULL;
 	syevjInfo_t syevj_params = NULL;
 	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;

	const int lda = m;
	//double *d_A = NULL;
    //double *d_W = NULL;
    int *devInfo = NULL;
    double *d_work = NULL;
    int lwork = 0;

    int info_gpu = 0;

	/* configuration of syevj */
 	const double tol = 1.e-7;
 	const int max_sweeps = 15;
 	const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
 	const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	/* step 1: create cusolver handle, bind a stream */
 	status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == status);
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	status = cusolverDnSetStream(cusolverH, stream);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* step 2: configuration of syevj */
	status = cusolverDnCreateSyevjInfo(&syevj_params);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* default value of tolerance is machine zero */
	status = cusolverDnXsyevjSetTolerance(
	syevj_params,
	tol);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* default value of max. sweeps is 100 */
	status = cusolverDnXsyevjSetMaxSweeps(
	syevj_params,
	max_sweeps);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaMalloc ((void**)&devInfo, sizeof(int));

	/* step 4: query working space of syevj */
	status = cusolverDnDsyevj_bufferSize(
	cusolverH,
	jobz,
	uplo,
	m,
	d_A,
	lda,
	d_W,
	&lwork,
	syevj_params);
	assert(CUSOLVER_STATUS_SUCCESS == status);
	cudaMalloc((void**)&d_work, sizeof(double)*lwork);
	
	/* step 5: compute eigen-pair */
	status = cusolverDnDsyevj(
	cusolverH,
	jobz,
	uplo,
	m,
	d_A,
	lda,
	d_W,
	d_work,
	lwork,
	devInfo,
	syevj_params);
	cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(0 == info_gpu);

    cudaFree(devInfo);
    cudaFree(d_work);
    cusolverDnDestroy(cusolverH);
	cudaStreamDestroy(stream);
	cusolverDnDestroySyevjInfo(syevj_params);
}

void cusolverZheevdj(const int m, cuDoubleComplex *d_A, double *d_W)
{
	cusolverDnHandle_t cusolverH = NULL;
 	cudaStream_t stream = NULL;
 	syevjInfo_t syevj_params = NULL;
 	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;

	const int lda = m;
	//double *d_A = NULL;
    //double *d_W = NULL;
    int *devInfo = NULL;
    cuDoubleComplex *d_work = NULL;
    int lwork = 0;

    int info_gpu = 0;

	/* configuration of syevj */
 	const double tol = 1.e-7;
 	const int max_sweeps = 15;
 	const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
 	const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	/* step 1: create cusolver handle, bind a stream */
 	status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == status);
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	status = cusolverDnSetStream(cusolverH, stream);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* step 2: configuration of syevj */
	status = cusolverDnCreateSyevjInfo(&syevj_params);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* default value of tolerance is machine zero */
	status = cusolverDnXsyevjSetTolerance(
	syevj_params,
	tol);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* default value of max. sweeps is 100 */
	status = cusolverDnXsyevjSetMaxSweeps(
	syevj_params,
	max_sweeps);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaMalloc ((void**)&devInfo, sizeof(int));

	/* step 4: query working space of syevj */
	status = cusolverDnZheevj_bufferSize(
	cusolverH,
	jobz,
	uplo,
	m,
	d_A,
	lda,
	d_W,
	&lwork,
	syevj_params);
	assert(CUSOLVER_STATUS_SUCCESS == status);
	cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex)*lwork);
	
	/* step 5: compute eigen-pair */
	status = cusolverDnZheevj(
	cusolverH,
	jobz,
	uplo,
	m,
	d_A,
	lda,
	d_W,
	d_work,
	lwork,
	devInfo,
	syevj_params);
	cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(0 == info_gpu);

    cudaFree(devInfo);
    cudaFree(d_work);
    cusolverDnDestroy(cusolverH);
	cudaStreamDestroy(stream);
	cusolverDnDestroySyevjInfo(syevj_params);
}

#endif
