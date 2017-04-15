/*! \file Merge.cu
	\author Gregory Diamos <gregory.diamos>	\date Wednesday December 1, 2010
	\brief The source file for the C interface to CUDA sorting routines.
*/
#ifndef SORT_CU_INCLUDED
#define SORT_CU_INCLUDED

// Redfox Includes
#include <redfox/nvcc/interface/RelationalAlgebraKernel.h>
#include <redfox/ra/interface/Difference.h>
#include <redfox/ra/interface/Tuple.h>

// Thrust Includes
#include <thrust/set_operations.h>
#include <thrust/device_ptr.h>

// Hydrazine Includes
//#include <hydrazine/implementation/debug.h>

#include <stdio.h>

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

void difference(void *result, unsigned long long int *size, void* lbegin, void* lend, 
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
		thrust::device_ptr<unsigned char> result_end = thrust::set_difference(
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
		thrust::device_ptr<unsigned short> result_end = thrust::set_difference(
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
//		unsigned int lsize = (unsigned int*)lend-(unsigned int*)lbegin;
//		unsigned int rsize = (unsigned int*)rend-(unsigned int*)rbegin;
//
//unsigned int* left = (unsigned int *)malloc(lsize*sizeof(unsigned int));
//check(cudaMemcpy(left, (unsigned int *)lbegin, lsize * sizeof(unsigned int),
//		cudaMemcpyDeviceToHost));
//unsigned int* right = (unsigned int *)malloc(rsize*sizeof(unsigned int));
//check(cudaMemcpy(right, (unsigned int *)rbegin, rsize * sizeof(unsigned int),
//		cudaMemcpyDeviceToHost));
//
//
//	printf("lsize %u rsize %u\n", lsize, rsize);
//for(int i = 0; i < lsize; ++i)
//	printf("left %u %u\n", i, left[i]);
//for(int i = 0; i < rsize; ++i)
//	printf("right %u %u\n", i, right[i]);

		thrust::device_ptr<unsigned int> result_end = thrust::set_difference(
			thrust::device_ptr<unsigned int>((unsigned int*)lbegin),
			thrust::device_ptr<unsigned int>((unsigned int*)lend),
			thrust::device_ptr<unsigned int>((unsigned int*)rbegin),
			thrust::device_ptr<unsigned int>((unsigned int*)rend),
			thrust::device_ptr<unsigned int>((unsigned int*)result));

		size_host = (result_end - thrust::device_ptr<unsigned int>((unsigned int *)result)) * sizeof(unsigned int);
		break;
	}
	case nvcc::RelationalAlgebraKernel::I64:
	{
		typedef thrust::device_ptr<long long unsigned int> ptr;
	
		thrust::device_ptr<long long unsigned int> result_end = thrust::set_difference(
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
	
		ptr result_end = thrust::set_difference(
			ptr((type*)lbegin),
			ptr((type*)lend),
			ptr((type*)rbegin),
			ptr((type*)rend),
			ptr((type*)result));

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

	printf("differenece %f\n", exe_time);
	printf("after difference size %llu \n", size_host);
}

}

#endif

