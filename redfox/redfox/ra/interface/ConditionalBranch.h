/*! \file   ConditionalBranch.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday December 10, 2010
	\brief The header file for the branch series of utility functions
*/

#ifndef CONDITIONAL_BRANCH_H_INCLUDED
#define CONDITIONAL_BRANCH_H_INCLUDED

namespace ra
{

typedef long long unsigned int uint;

namespace cuda
{

template<typename Type>
__device__ void conditionalBranch(uint* target, 
	const Type* _beginA, const Type* _endA,
	const Type* _beginB, const Type* _endB)
{
	uint elements  = _endA - _beginA;
	uint bElements = _endB - _beginB;

	if(elements != bElements)
	{
		if(threadIdx.x == 0 && blockIdx.x == 0)
		{
			*target = 1;
		}
		return;
	}

	uint step   = gridDim.x * blockDim.x;
	uint chunks = elements / (sizeof(uint) * step);

	uint* beginA = (uint*) _beginA;
	uint* beginB = (uint*) _beginB;

	uint id   = blockIdx.x * blockDim.x + threadIdx.x;
	
	uint index = id;
	for(uint i = 0; i < chunks; ++i, index += step)
	{
		if(beginA[index] != beginB[index])
		{
			*target = 1;
			return;
		}
	}
	
	index = id + chunks * step * sizeof(uint);
	
	for( ; index < elements; index += step)
	{
		if(((char*)beginA)[index] != ((char*)beginB)[index])
		{
			*target = 1;
			return;
		}
	}
}

}

}

#endif

