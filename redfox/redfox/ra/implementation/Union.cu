/*! \file Merge.cu
	\author Gregory Diamos <gregory.diamos>	\date Wednesday December 1, 2010
	\brief The source file for the C interface to CUDA sorting routines.
*/
#ifndef SORT_CU_INCLUDED
#define SORT_CU_INCLUDED

// Redfox Includes
#include <redfox/nvcc/interface/RelationalAlgebraKernel.h>
#include <redfox/ra/interface/Union.h>
#include <redfox/ra/interface/Tuple.h>

// Thrust Includes
#include <thrust/set_operations.h>
#include <thrust/device_ptr.h>

// Hydrazine Includes
//#include <hydrazine/implementation/debug.h>

#include <stdio.h>

struct compare_sort_gpu128
{
  __host__ __device__
  bool operator()(ra::tuple::PackedNBytes<2> i, ra::tuple::PackedNBytes<2> j)
  {
   if (i.a[1] != j.a[1])
	return (i.a[1] < j.a[1]);

    return (i.a[0] < j.a[0]);
  }
};

namespace redfox
{

void check(cudaError_t status)
{
	if(status != cudaSuccess)
	{
		std::cerr << cudaGetErrorString(status) << "\n";
	
		std::abort();
	}
}

void set_union(void *result, unsigned long long int *size, void* lbegin, void* lend, 
	void* rbegin, void* rend, unsigned int type)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	unsigned long long int size_host;

	switch(type)
	{
	default: printf("Invalid Type.\n");
	case nvcc::RelationalAlgebraKernel::I8:
	{
		thrust::device_ptr<unsigned char> result_end = thrust::set_union(
			thrust::device_ptr<unsigned char>((unsigned char *)lbegin),
			thrust::device_ptr<unsigned char>((unsigned char *)lend),
			thrust::device_ptr<unsigned char>((unsigned char *)rbegin),
			thrust::device_ptr<unsigned char>((unsigned char *)rend),
			thrust::device_ptr<unsigned char>((unsigned char *)result));

		size_host = (result_end - thrust::device_ptr<unsigned char>((unsigned char *)result)) * sizeof(unsigned char);
		break;
	}
	case nvcc::RelationalAlgebraKernel::I16:
	{
		thrust::device_ptr<unsigned short> result_end = thrust::set_union(
			thrust::device_ptr<unsigned short>((unsigned short*)lbegin),
			thrust::device_ptr<unsigned short>((unsigned short*)lend),
			thrust::device_ptr<unsigned short>((unsigned short*)rbegin),
			thrust::device_ptr<unsigned short>((unsigned short*)rend),
			thrust::device_ptr<unsigned short>((unsigned short*)result));

		size_host = (result_end - thrust::device_ptr<unsigned short>((unsigned short *)result)) * sizeof(unsigned short);
		break;
	}
	case nvcc::RelationalAlgebraKernel::I32:
	{
		thrust::device_ptr<unsigned int> result_end = thrust::set_union(
			thrust::device_ptr<unsigned int>((unsigned int*)lbegin),
			thrust::device_ptr<unsigned int>((unsigned int*)lend),
			thrust::device_ptr<unsigned int>((unsigned int*)rbegin),
			thrust::device_ptr<unsigned int>((unsigned int*)rend),
			thrust::device_ptr<unsigned int>((unsigned int*)result));

		size_host = (result_end - thrust::device_ptr<unsigned int>((unsigned int *)result)) * sizeof(unsigned int);

//		unsigned int merge_result[10];
//		check(cudaMemcpy(merge_result, (unsigned int *)result, 4 * 10,
//			cudaMemcpyDeviceToHost));
//	
//		for(int i = 0; i < 10; ++i)
//		{
//			printf("%u %llx\n", i, merge_result[i]);
//		}

		break;
	}
	case nvcc::RelationalAlgebraKernel::I64:
	{
		typedef thrust::device_ptr<long long unsigned int> ptr;
	
		thrust::device_ptr<long long unsigned int> result_end = thrust::set_union(
			ptr((long long unsigned int*)lbegin),
			ptr((long long unsigned int*)lend),
			ptr((long long unsigned int*)rbegin),
			ptr((long long unsigned int*)rend),
			ptr((long long unsigned int*)result));

		size_host = (result_end - thrust::device_ptr<unsigned long long int>((unsigned long long int *)result)) * sizeof(unsigned long long int);
		break;
	}
	case nvcc::RelationalAlgebraKernel::I128:
	{
		typedef ra::tuple::PackedNBytes<2> type;
		typedef thrust::device_ptr<type> ptr;
	
		ptr result_end = thrust::set_union(
			ptr((type*)lbegin),
			ptr((type*)lend),
			ptr((type*)rbegin),
			ptr((type*)rend),
			ptr((type*)result), compare_sort_gpu128());

		size_host = (result_end - ptr((type *)result)) * sizeof(type);
		break;
	}
	}

	check(cudaMemcpy(size, &size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("union %f\n", exe_time);
	printf("after union size %llu \n", size_host);
}

}

#endif

