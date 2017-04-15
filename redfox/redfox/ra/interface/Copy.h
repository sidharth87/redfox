/*! \file   Copy.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday November 5, 2010
	\brief The header file for the copy series of utility functions
*/

#ifndef COPY_H_INCLUDED
#define COPY_H_INCLUDED

#include<stdio.h>

namespace ra
{

typedef long long unsigned int uint;

namespace cuda
{

template<typename Type>
__device__ void copy(Type* _output, const Type* _begin, const uint* bytes)
{
	uint step = gridDim.x * blockDim.x;
	uint id   = blockIdx.x * blockDim.x + threadIdx.x;
	
	uint elements  = *bytes;
	uint chunks    = elements / (sizeof(uint) * step);

	uint* begin  = (uint*) _begin;
	uint* output = (uint*) _output;
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("copy %llu\n", elements);
	uint index = id;
	for(uint i = 0; i < chunks; ++i, index += step)
	{
		output[index] = begin[index];
	}
	
	index = id + chunks * step * sizeof(uint);
	
	for( ; index < elements; index += step)
	{
		((char*)output)[index] = ((char*)begin)[index];
	}
//if(threadIdx.x == 0 && blockIdx.x == 0)
//{
//for(unsigned int i = 0; i < elements/4; ++i)
//	printf("%u %llx\n", i, output[i]);
//}
}

}

}

#endif

