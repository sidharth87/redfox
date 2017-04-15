/*! \file Unique.cu
	\author Gregory Diamos <gregory.diamos>
	\date Wednesday December 8, 2010
	\brief The source file for the C interface to CUDA unique routines.
*/

#ifndef UNIQUE_CU_INCLUDED
#define UNIQUE_CU_INCLUDED

// Redfox Includes
#include <redfox/nvcc/interface/RelationalAlgebraKernel.h>
#include <redfox/ra/interface/Unique.h>
#include <redfox/ra/interface/Tuple.h>

// Thrust Includes
#include <thrust/device_ptr.h>
#include <thrust/unique.h>

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

void unique(void* begin, unsigned long long int *size, unsigned long long int type)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	unsigned long long int ret = 0;

	unsigned long long int size_host;
	check(cudaMemcpy(&size_host, size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

	switch(type)
	{
	case nvcc::RelationalAlgebraKernel::I8:
	{
		if(size_host == 1) return;
		ret = sizeof(unsigned char) * (thrust::unique(
			thrust::device_ptr<unsigned char>((unsigned char*)begin),
			thrust::device_ptr<unsigned char>((unsigned char*)begin + (size_host / sizeof(unsigned char)))) 
			- thrust::device_ptr<unsigned char>((unsigned char*)begin));

		break;
	}
	case nvcc::RelationalAlgebraKernel::I16:
	{
		if(size_host == 2) return;
		ret = sizeof(unsigned short) * (thrust::unique(
			thrust::device_ptr<unsigned short>((unsigned short*)begin),
			thrust::device_ptr<unsigned short>((unsigned short*)begin + (size_host / sizeof(unsigned short))))
			- thrust::device_ptr<unsigned short>((unsigned short*)begin));

		break;
	}
	case nvcc::RelationalAlgebraKernel::I32:
	{
		if(size_host == 4) return;
		ret = sizeof(unsigned int) * (thrust::unique(
			thrust::device_ptr<unsigned int>((unsigned int*)begin),
			thrust::device_ptr<unsigned int>((unsigned int*)begin + (size_host / sizeof(unsigned int))))
			- thrust::device_ptr<unsigned int>((unsigned int*)begin));
	
		break;
	}
	case nvcc::RelationalAlgebraKernel::I64:
	{
		if(size_host == 8) return;
		typedef thrust::device_ptr<unsigned long long int> ptr;
	
		ret = sizeof(unsigned long long int) * (thrust::unique(
			ptr((unsigned long long int*)begin),
			ptr((unsigned long long int*)begin + (size_host / sizeof(unsigned long long int))))
			- ptr((unsigned long long int*)begin));

		break;
	}
	default: 
	{
		printf("Invalid Type.\n");
	}
	}

	check(cudaMemcpy(size, &ret, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
 	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("unique %f\n", exe_time);

	printf("before unique %llu\n", size_host);
	printf("after unique size %llu\n", ret);

	return;
}

}

#endif

