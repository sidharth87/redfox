/*! \file Sort.cu
	\author Gregory Diamos <gregory.diamos>
	\date Wednesday December 1, 2010
	\brief The source file for the C interface to CUDA sorting routines.
*/

// Redfox Includes
#include <redfox/nvcc/interface/RelationalAlgebraKernel.h>
#include <redfox/ra/interface/ModernGPUJoin.h>
#include <redfox/ra/interface/Tuple.h>

#include <redfox/ra/interface/moderngpu/include/kernels/join.cuh>

#include <stdio.h>
#include <iostream>
#include <string.h>

class gpu128
{
public:
	typedef long long unsigned int type;

public:
	type a[2];
};

struct compare_sort_gpu128
{
  typedef gpu128 type;

  __host__ __device__
  bool operator()(type i, type j)
  {
   if (i.a[1] != j.a[1])
	return (i.a[1] < j.a[1]);

    return (i.a[0] < j.a[0]);
  }
};

namespace redfox
{

struct compare_string
{
  __host__ __device__
  bool operator()(unsigned char * i, unsigned char *j)
  {
     const char *string1 = (char *) i;
     const char *string2 = (char *) j;

//     return(strcmp(string1, string2) < 0);     
     int ii = 0;
     
     while(string1[ii] != '\0' && string2[ii] != '\0')
     {
     	if(string1[ii] < string2[ii])
     		return true;
    	else if(string1[ii] > string2[ii]) 
		return false;

     	ii++;
     }
     
     if(string1[ii] == '\0' && string2[ii] != '\0')
     	return true;
     else
	return false;
  }
};

struct compare_string2
{
  __host__ __device__
  bool operator()(unsigned long long int i, unsigned long long int j)
  {
     const char *string1 = (char *) i;
     const char *string2 = (char *) j;

//     return(strcmp(string1, string2) < 0);     
     int ii = 0;
     
     while(string1[ii] != '\0' && string2[ii] != '\0')
     {
     	if(string1[ii] < string2[ii])
     		return true;
    	else if(string1[ii] > string2[ii]) 
		return false;

     	ii++;
     }
     
     if(string1[ii] == '\0' && string2[ii] != '\0')
     	return true;
     else
	return false;
  }
};
void check(cudaError_t status)
{
	if(status != cudaSuccess)
	{
		std::cerr << cudaGetErrorString(status) << "\n";
	
		std::abort();
	}
}

void find_bounds_128(void* lower_bound, void* left_count, unsigned long long int *result_size, 
	void* left_key, unsigned long long int left_size, 
	void* right_key, unsigned long long int right_size)
{
//	unsigned short left[100];
//	unsigned short right[100];

//	unsigned long long int *left_host = (unsigned long long int*)malloc(16);
//	unsigned long long int *right_host = (unsigned long long int*)malloc(6001215 * 8);
//	check(cudaMemcpy(left_host, (unsigned long long int *)left_key, 16,
//		cudaMemcpyDeviceToHost));
//
//	check(cudaMemcpy(right_host, (unsigned long long int *)right_key, 6001215 * 8,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 6001214; i > 6001204; --i)
//		printf("%u %x %x\n", i, left[i], right[i]);
//	int *left_host = (int *)malloc(30142*sizeof(int));
//	check(cudaMemcpy(left_host, (int *)left_key, 30142*sizeof(int),
//		cudaMemcpyDeviceToHost));
//	int *right_host = (int *)malloc(1500000*sizeof(int));
//	check(cudaMemcpy(right_host, (int *)right_key, 1500000*sizeof(int),
//		cudaMemcpyDeviceToHost));

//	for(unsigned int i = 0; i < 2; ++i)
//		printf("left %u %llx\n", i, left_host[i]);
//	for(unsigned int i = 0; i < 100; ++i)
//		printf("right %u %llx\n", i, right_host[i]);
//	for(unsigned int i = 6001214; i > 6001204; --i)
//		printf("right %u %llx\n", i, right_host[i]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

//	size_t freeMem, totalMem;
//	cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";
//
//	std::cout << left_size << "  " << right_size << "\n";

//	mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);
//	context->Start();

		
	typedef gpu128 type;

	const mgpu::MgpuSearchType LeftType = mgpu::MgpuSearchTypeIndex;
	const mgpu::MgpuBounds Bounds = mgpu::MgpuBoundsLower;
	const mgpu::MgpuSearchType RightType = mgpu::MgpuSearchTypeNone;

	mgpu::SortedSearch<Bounds, LeftType, RightType, type*, type*, int *, int *>((type*)left_key,
		(int)left_size, (type*)right_key, (int)right_size, (int *)lower_bound, (int*)0, compare_sort_gpu128(),
		/**context,*/ (int *)0, (int *)0);

//int *lower_bound_host = (int *)malloc(2*sizeof(int));
//check(cudaMemcpy(lower_bound_host, (int *)lower_bound, 2*sizeof(int),
//	cudaMemcpyDeviceToHost));

//std::cout << "lower bound 0: " << lower_bound_host[0] << " lower bound 1: " << lower_bound_host[1] << "\n";

	mgpu::SortedEqualityCount<type*, type *, int *, int *, struct compare_sort_gpu128, struct mgpu::SortedEqualityOp>
		((type *)left_key, (int)left_size, 
		(type *)right_key, (int)right_size, (int *)lower_bound, (int *)left_count, 
		compare_sort_gpu128(), mgpu::SortedEqualityOp()/*, 
		*context*/);

//int *left_count_host = (int *)malloc(2*sizeof(int));
//check(cudaMemcpy(left_count_host, (int *)left_count, 2*sizeof(int),
//	cudaMemcpyDeviceToHost));

//std::cout << "left count 0: " << left_count_host[0] << " left count 1: " << left_count_host[1] << "\n";

	// Scan the product counts. This is part of the load-balancing search.
	unsigned long long int total = mgpu::Scan((int *)left_count, (int)left_size/*, *context*/);
//printf("total %llu\n", total);

	check(cudaMemcpy(result_size, &total, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

//	exe_time += context->Split();

//        cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("join_find_bounds %lf\n", exe_time);
	printf("**********************join size %llu\n", total);

//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 10,
//		cudaMemcpyDeviceToHost));
//
//	check(cudaMemcpy(data_value, (double *)value_begin, 80,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 10; ++i)
//		printf("%u %x %lf\n", i, data_key[i], data_value[i]);
}

void find_bounds_string(void* lower_bound, void* left_count, unsigned long long int *result_size, 
	void* left_key, unsigned long long int left_size, 
	void* right_key, unsigned long long int right_size)
{
//	unsigned short left[100];
//	unsigned short right[100];

//	unsigned long long int *left_host = (unsigned long long int*)malloc(16);
//	unsigned long long int *right_host = (unsigned long long int*)malloc(6001215 * 8);
//	check(cudaMemcpy(left_host, (unsigned long long int *)left_key, 16,
//		cudaMemcpyDeviceToHost));
//
//	check(cudaMemcpy(right_host, (unsigned long long int *)right_key, 6001215 * 8,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 6001214; i > 6001204; --i)
//		printf("%u %x %x\n", i, left[i], right[i]);
//	int *left_host = (int *)malloc(30142*sizeof(int));
//	check(cudaMemcpy(left_host, (int *)left_key, 30142*sizeof(int),
//		cudaMemcpyDeviceToHost));
//	int *right_host = (int *)malloc(1500000*sizeof(int));
//	check(cudaMemcpy(right_host, (int *)right_key, 1500000*sizeof(int),
//		cudaMemcpyDeviceToHost));

//	for(unsigned int i = 0; i < 2; ++i)
//		printf("left %u %llx\n", i, left_host[i]);
//	for(unsigned int i = 0; i < 100; ++i)
//		printf("right %u %llx\n", i, right_host[i]);
//	for(unsigned int i = 6001214; i > 6001204; --i)
//		printf("right %u %llx\n", i, right_host[i]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

//	size_t freeMem, totalMem;
//	cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";
//
//	std::cout << left_size << "  " << right_size << "\n";

//	mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);
//	context->Start();

	const mgpu::MgpuSearchType LeftType = mgpu::MgpuSearchTypeIndex;
	const mgpu::MgpuBounds Bounds = mgpu::MgpuBoundsLower;
	const mgpu::MgpuSearchType RightType = mgpu::MgpuSearchTypeNone;

	mgpu::SortedSearch<Bounds, LeftType, RightType, unsigned long long int*, unsigned long long int*, int *, int *>((unsigned long long int*)left_key,
		(int)left_size, (unsigned long long int*)right_key, (int)right_size, (int *)lower_bound, (int*)0, compare_string2(),/*mgpu::less<T>(),
		*context,*/ (int *)0, (int *)0);

//int *lower_bound_host = (int *)malloc(2*sizeof(int));
//check(cudaMemcpy(lower_bound_host, (int *)lower_bound, 2*sizeof(int),
//	cudaMemcpyDeviceToHost));

//std::cout << "lower bound 0: " << lower_bound_host[0] << " lower bound 1: " << lower_bound_host[1] << "\n";

	mgpu::SortedEqualityCount<unsigned long long int*, unsigned long long int *, int *, int *, struct compare_string2, struct mgpu::SortedEqualityOp>
		((unsigned long long int *)left_key, (int)left_size, 
		(unsigned long long int *)right_key, (int)right_size, (int *)lower_bound, (int *)left_count, 
		compare_string2(), mgpu::SortedEqualityOp()/*, 
		*context*/);

//int *left_count_host = (int *)malloc(2*sizeof(int));
//check(cudaMemcpy(left_count_host, (int *)left_count, 2*sizeof(int),
//	cudaMemcpyDeviceToHost));

//std::cout << "left count 0: " << left_count_host[0] << " left count 1: " << left_count_host[1] << "\n";

	// Scan the product counts. This is part of the load-balancing search.
	unsigned long long int total = mgpu::Scan((int *)left_count, (int)left_size/*, *context*/);
//printf("total %llu\n", total);

	check(cudaMemcpy(result_size, &total, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

//	exe_time += context->Split();

//        cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("join_find_bounds %lf\n", exe_time);
	printf("**********************join size %llu\n", total);

//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 10,
//		cudaMemcpyDeviceToHost));
//
//	check(cudaMemcpy(data_value, (double *)value_begin, 80,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 10; ++i)
//		printf("%u %x %lf\n", i, data_key[i], data_value[i]);
}

void find_bounds_64(void* lower_bound, void* left_count, unsigned long long int *result_size, 
	void* left_key, unsigned long long int left_size, 
	void* right_key, unsigned long long int right_size)
{
//	unsigned short left[100];
//	unsigned short right[100];

//	unsigned long long int *left_host = (unsigned long long int*)malloc(16);
//	unsigned long long int *right_host = (unsigned long long int*)malloc(6001215 * 8);
//	check(cudaMemcpy(left_host, (unsigned long long int *)left_key, 16,
//		cudaMemcpyDeviceToHost));
//
//	check(cudaMemcpy(right_host, (unsigned long long int *)right_key, 6001215 * 8,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 6001214; i > 6001204; --i)
//		printf("%u %x %x\n", i, left[i], right[i]);
//	int *left_host = (int *)malloc(30142*sizeof(int));
//	check(cudaMemcpy(left_host, (int *)left_key, 30142*sizeof(int),
//		cudaMemcpyDeviceToHost));
//	int *right_host = (int *)malloc(1500000*sizeof(int));
//	check(cudaMemcpy(right_host, (int *)right_key, 1500000*sizeof(int),
//		cudaMemcpyDeviceToHost));

//	for(unsigned int i = 0; i < 2; ++i)
//		printf("left %u %llx\n", i, left_host[i]);
//	for(unsigned int i = 0; i < 100; ++i)
//		printf("right %u %llx\n", i, right_host[i]);
//	for(unsigned int i = 6001214; i > 6001204; --i)
//		printf("right %u %llx\n", i, right_host[i]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

//	size_t freeMem, totalMem;
//	cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";
//
//	std::cout << left_size << "  " << right_size << "\n";

//	mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);
//	context->Start();

	const mgpu::MgpuSearchType LeftType = mgpu::MgpuSearchTypeIndex;
	const mgpu::MgpuBounds Bounds = mgpu::MgpuBoundsLower;
	const mgpu::MgpuSearchType RightType = mgpu::MgpuSearchTypeNone;

	mgpu::SortedSearch<Bounds, LeftType, RightType, unsigned long long int*, unsigned long long int*, int *, int *>((unsigned long long int*)left_key,
		(int)left_size, (unsigned long long int*)right_key, (int)right_size, (int *)lower_bound, (int*)0, /*mgpu::less<T>(),
		*context,*/ (int *)0, (int *)0);

//int *lower_bound_host = (int *)malloc(2*sizeof(int));
//check(cudaMemcpy(lower_bound_host, (int *)lower_bound, 2*sizeof(int),
//	cudaMemcpyDeviceToHost));

//std::cout << "lower bound 0: " << lower_bound_host[0] << " lower bound 1: " << lower_bound_host[1] << "\n";

	mgpu::SortedEqualityCount<unsigned long long int*, unsigned long long int *, int *, int *, struct mgpu::SortedEqualityOp>
		((unsigned long long int *)left_key, (int)left_size, 
		(unsigned long long int *)right_key, (int)right_size, (int *)lower_bound, (int *)left_count, 
		mgpu::SortedEqualityOp()/*, 
		*context*/);

//int *left_count_host = (int *)malloc(2*sizeof(int));
//check(cudaMemcpy(left_count_host, (int *)left_count, 2*sizeof(int),
//	cudaMemcpyDeviceToHost));

//std::cout << "left count 0: " << left_count_host[0] << " left count 1: " << left_count_host[1] << "\n";

	// Scan the product counts. This is part of the load-balancing search.
	unsigned long long int total = mgpu::Scan((int *)left_count, (int)left_size/*, *context*/);
//printf("total %llu\n", total);

	check(cudaMemcpy(result_size, &total, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

//	exe_time += context->Split();

//        cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("join_find_bounds %lf\n", exe_time);
	printf("**********************join size %llu\n", total);

//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 10,
//		cudaMemcpyDeviceToHost));
//
//	check(cudaMemcpy(data_value, (double *)value_begin, 80,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 10; ++i)
//		printf("%u %x %lf\n", i, data_key[i], data_value[i]);
}

void find_bounds_16(void* lower_bound, void* left_count, unsigned long long int *result_size, 
	void* left_key, unsigned long long int left_size, 
	void* right_key, unsigned long long int right_size)
{
//	unsigned short left[100];
//	unsigned short right[100];

//	unsigned short *left_host = (unsigned short*)malloc(6001215*2);
//	unsigned short *right = (unsigned short*)malloc(100*2);
//	check(cudaMemcpy(left_host, (unsigned short *)left_key, 6001215*2,
//		cudaMemcpyDeviceToHost));

//	check(cudaMemcpy(right, (unsigned short *)right_key, 200,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 6001214; i > 6001204; --i)
//		printf("%u %x %x\n", i, left[i], right[i]);
//	int *left_host = (int *)malloc(30142*sizeof(int));
//	check(cudaMemcpy(left_host, (int *)left_key, 30142*sizeof(int),
//		cudaMemcpyDeviceToHost));
//	int *right_host = (int *)malloc(1500000*sizeof(int));
//	check(cudaMemcpy(right_host, (int *)right_key, 1500000*sizeof(int),
//		cudaMemcpyDeviceToHost));
//	for(unsigned int i = 0; i < 6001215; ++i)
//		if(left_host[i] < left_host[i - 1]) printf("error %d %d %d\n", i, left_host[i], left_host[i - 1]);
//		printf("left %u %u\n", i, left_host[i]);
//	for(unsigned int i = 0; i < 100; ++i)
//		if(right_host[i] < right_host[i - 1]) printf("error %d %d %d\n", i, right_host[i], right_host[i - 1]);
//		printf("right %u %u\n", i, right[i]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

//	size_t freeMem, totalMem;
//	cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";
//
//	std::cout << left_size << "  " << right_size << "\n";

//	mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);
//	context->Start();

	typedef unsigned short* T;

	const mgpu::MgpuSearchType LeftType = mgpu::MgpuSearchTypeIndex;
	const mgpu::MgpuBounds Bounds = mgpu::MgpuBoundsLower;
	const mgpu::MgpuSearchType RightType = mgpu::MgpuSearchTypeNone;

	mgpu::SortedSearch<Bounds, LeftType, RightType, unsigned short *, unsigned short *, int *, int *>((unsigned short*)left_key,
		(int)left_size, (unsigned short *)right_key, (int)right_size, (int *)lower_bound, (int*)0, /*mgpu::less<T>(),
		*context,*/ (int *)0, (int *)0);

	mgpu::SortedEqualityCount<unsigned short*, unsigned short*, int *, int *, struct mgpu::SortedEqualityOp>
		((unsigned short*)left_key, (int)left_size, 
		(unsigned short*)right_key, (int)right_size, (int *)lower_bound, (int *)left_count, mgpu::SortedEqualityOp()/*, 
		*context*/);

	// Scan the product counts. This is part of the load-balancing search.
	unsigned long long int total = mgpu::Scan((int *)left_count, (int)left_size/*, *context*/);
//printf("total %llu\n", total);

	check(cudaMemcpy(result_size, &total, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

//	exe_time += context->Split();

//        cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("join_find_bounds %lf\n", exe_time);
	printf("**********************join size %llu\n", total);

//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 10,
//		cudaMemcpyDeviceToHost));
//
//	check(cudaMemcpy(data_value, (double *)value_begin, 80,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 10; ++i)
//		printf("%u %x %lf\n", i, data_key[i], data_value[i]);
}

void find_bounds_32(void* lower_bound, void* left_count, unsigned long long int *result_size, 
	void* left_key, unsigned long long int left_size, 
	void* right_key, unsigned long long int right_size)
{
//	unsigned int left[6001215];
//	unsigned int right[6001215];

//	unsigned int *left = (unsigned int*)malloc(6001215*4);
//	unsigned int *right = (unsigned int*)malloc(6001215*4);
//	check(cudaMemcpy(left, (unsigned int *)left_key, 4*6001215,
//		cudaMemcpyDeviceToHost));
//
//	check(cudaMemcpy(right, (unsigned int *)right_key, 4*6001215,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 6001214; i > 6001204; --i)
//		printf("%u %x %x\n", i, left[i], right[i]);
//	int *left_host = (int *)malloc(30142*sizeof(int));
//	check(cudaMemcpy(left_host, (int *)left_key, 30142*sizeof(int),
//		cudaMemcpyDeviceToHost));
//	int *right_host = (int *)malloc(1500000*sizeof(int));
//	check(cudaMemcpy(right_host, (int *)right_key, 1500000*sizeof(int),
//		cudaMemcpyDeviceToHost));
//	for(unsigned int i = 1; i < 100; ++i)
//		if(left_host[i] < left_host[i - 1]) printf("error %d %d %d\n", i, left_host[i], left_host[i - 1]);
//		printf("left %d %d\n", i, left_host[i]);
//	for(unsigned int i = 1; i < 100; ++i)
//		if(right_host[i] < right_host[i - 1]) printf("error %d %d %d\n", i, right_host[i], right_host[i - 1]);
//		printf("right %d %d\n", i, right_host[i]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

//	size_t freeMem, totalMem;
//	cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";
//
//	std::cout << left_size << "  " << right_size << "\n";

//	mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);
//	context->Start();

	typedef unsigned int* T;

	const mgpu::MgpuSearchType LeftType = mgpu::MgpuSearchTypeIndex;
	const mgpu::MgpuBounds Bounds = mgpu::MgpuBoundsLower;
	const mgpu::MgpuSearchType RightType = mgpu::MgpuSearchTypeNone;

	mgpu::SortedSearch<Bounds, LeftType, RightType, unsigned int *, unsigned int *, int *, int *>((unsigned int*)left_key,
		(int)left_size, (unsigned int *)right_key, (int)right_size, (int *)lower_bound, (int*)0, /*mgpu::less<T>(),
		*context,*/ (int *)0, (int *)0);

	mgpu::SortedEqualityCount<unsigned int*, unsigned int*, int *, int *, struct mgpu::SortedEqualityOp>((unsigned int*)left_key, (int)left_size, 
		(unsigned int*)right_key, (int)right_size, (int *)lower_bound, (int *)left_count, mgpu::SortedEqualityOp()/*, 
		*context*/);

	// Scan the product counts. This is part of the load-balancing search.
	unsigned long long int total = mgpu::Scan((int *)left_count, (int)left_size/*, *context*/);
//printf("total %llu\n", total);

	check(cudaMemcpy(result_size, &total, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

//	exe_time += context->Split();

//        cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("join_find_bounds %lf\n", exe_time);
	printf("**********************join size %llu\n", total);

//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 10,
//		cudaMemcpyDeviceToHost));
//
//	check(cudaMemcpy(data_value, (double *)value_begin, 80,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 10; ++i)
//		printf("%u %x %lf\n", i, data_key[i], data_value[i]);
}

void find_bounds_8(void* lower_bound, void* left_count, unsigned long long int *result_size, 
	void* left_key, unsigned long long int left_size, 
	void* right_key, unsigned long long int right_size)
{
//	unsigned char data_key[10];
//	double data_value[10];
//
//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 10,
//		cudaMemcpyDeviceToHost));
//
//	check(cudaMemcpy(data_value, (double *)value_begin, 80,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 10; ++i)
//		printf("%u %x %lf\n", i, data_key[i], data_value[i]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

//	size_t freeMem, totalMem;
//	cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";

//	mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);
//	context->Start();

	typedef unsigned char* T;

	const mgpu::MgpuSearchType LeftType = mgpu::MgpuSearchTypeIndex;
	const mgpu::MgpuBounds Bounds = mgpu::MgpuBoundsLower;
	const mgpu::MgpuSearchType RightType = mgpu::MgpuSearchTypeNone;

	mgpu::SortedSearch<Bounds, LeftType, RightType, unsigned char *, unsigned char *, int *, int *>((unsigned char*)left_key,
		(int)left_size, (unsigned char *)right_key, (int)right_size, (int *)lower_bound, (int*)0,/* mgpu::less<T>(),
		*context,*/ (int *)0, (int *)0);

	mgpu::SortedEqualityCount<unsigned char*, unsigned char*, int *, int *, struct mgpu::SortedEqualityOp>
		((unsigned char*)left_key, (int)left_size, 
		(unsigned char*)right_key, (int)right_size, (int *)lower_bound, (int *)left_count, mgpu::SortedEqualityOp()/*, 
		*context*/);

	// Scan the product counts. This is part of the load-balancing search.
	unsigned long long int total = mgpu::Scan((int *)left_count, (int)left_size/*, *context*/);
//printf("total %llu\n", total);

	check(cudaMemcpy(result_size, &total, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

//	exe_time += context->Split();

//        cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("join_find_bounds %lf\n", exe_time);
	printf("**********************join size %llu\n", total);

//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 10,
//		cudaMemcpyDeviceToHost));
//
//	check(cudaMemcpy(data_value, (double *)value_begin, 80,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 10; ++i)
//		printf("%u %x %lf\n", i, data_key[i], data_value[i]);
}

void join(int* left_indices, int* right_indices, unsigned long long int result_size, 
	int* lowerBound, int* leftCount, 
	unsigned long long int input_size)
{
//	unsigned char data_key[10];
//	double data_value[10];
//
//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 10,
//		cudaMemcpyDeviceToHost));
//
//	check(cudaMemcpy(data_value, (double *)value_begin, 80,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 10; ++i)
//		printf("%u %x %lf\n", i, data_key[i], data_value[i]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

//	size_t freeMem, totalMem
//	cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";

//	mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);
//	context->Start();

	const int NT = 128;
	const int VT = 7;
	typedef mgpu::LaunchBoxVT<NT, VT> Tuning;
	int2 launch = Tuning::GetLaunchParams(/**context*/);
	int NV = launch.x * launch.y;
	
	const mgpu::MgpuBounds Bounds = mgpu::MgpuBoundsUpper;
//	MGPU_MEM(int) partitionsDevice = mgpu::MergePathPartitions<Bounds>(
	int* partitionsDevice = mgpu::MergePathPartitions<Bounds>(
		mgpu::counting_iterator<int>(0), result_size, leftCount,
		input_size, NV, 0, mgpu::less<int>()/*, *context*/);

	int numBlocks = MGPU_DIV_UP(result_size + input_size, NV);
	mgpu::KernelLeftJoin<Tuning, false>
		<<<numBlocks, launch.x/*, 0, context->Stream()*/>>>(result_size, 
//		lowerBound, leftCount, input_size, partitionsDevice->get(),
		lowerBound, leftCount, input_size, partitionsDevice,
		left_indices, right_indices);

	check(cudaFree(partitionsDevice));
//	exe_time += context->Split();

//      cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("join_main %lf\n", exe_time);

//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 10,
//		cudaMemcpyDeviceToHost));
//
//	check(cudaMemcpy(data_value, (double *)value_begin, 80,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 10; ++i)
//		printf("%u %x %lf\n", i, data_key[i], data_value[i]);
}
}


