/*! \file Product.h
	\date Saturday November 6, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the product family of RA functions.
*/

/*  x = {(3,a),(4,a)}, y = {(True,2),(False,9)}
  product       x y  -->  {(3,a,True,2),(4,a,True,2),
                          (3,a,False,9),(4,a,False,9)} */

#ifndef PRODUCT_H_INCLUDED
#define PRODUCT_H_INCLUDED

// RedFox Includes
#include <stdio.h>
#include <redfox/ra/interface/Tuple.h>

namespace ra
{

namespace cuda
{

typedef long long unsigned int uint;

template<typename ResultTuple, typename LeftTuple, typename RightTuple>
__device__ void resultSize(uint* size, const uint leftSize,
	const uint rightSize)
{
	*size = ((leftSize) * (rightSize)) *
		sizeof(typename ResultTuple::BasicType);

//	printf("**********Product left tuple size %u\n", sizeof(LeftTuple::BasicType));
//	printf("**********Product leftsize %u\n", leftSize);
//	printf("**********Product rightsize %u\n", rightSize);
//	printf("**********Product resultsize %u\n", *size);
}

template<typename ResultTuple, typename LeftTuple, typename RightTuple>
__device__ void product(typename ResultTuple::BasicType* result,
	const typename LeftTuple::BasicType* leftBegin,
	const typename LeftTuple::BasicType* leftEnd,
	const typename RightTuple::BasicType* rightBegin,
	const typename RightTuple::BasicType* rightEnd)
{
	uint threads = blockDim.x * gridDim.x;
	uint id      = threadIdx.x + blockDim.x * blockIdx.x;
	
	uint leftSize   = leftEnd  - leftBegin;
	uint rightSize  = rightEnd - rightBegin;
	uint resultSize = leftSize * rightSize;
	
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("******before product %llx\n", leftBegin[0].a[0]);
	for(uint i = id; i < resultSize; i += threads)
	{
		if(i < resultSize)
		{
			uint leftIndex  = i / rightSize;
			uint rightIndex = i % rightSize;
			
			result[i] = tuple::combine<LeftTuple, RightTuple, ResultTuple, 0>(
				leftBegin[leftIndex], rightBegin[rightIndex]);
		}
	}
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llx %llx %llx\n", result[0].a[0], result[0].a[1], result[0].a[2]);
//	if(threadIdx.x == 0 && blockIdx.x == 0) printf("product %u\n", leftSize);
//	if(threadIdx.x == 0 && blockIdx.x == 0) printf("product %u\n", rightSize);
//	if(threadIdx.x == 0 && blockIdx.x == 0) printf("product %u\n", resultSize);
}

//template<typename ResultTuple, typename LeftTuple, typename RightTuple>
//__device__ void product2(typename ResultTuple::BasicType* result,
//	const typename LeftTuple::BasicType* leftBegin,
//	const typename LeftTuple::BasicType* leftEnd,
//	const typename RightTuple::BasicType* rightBegin,
//	const typename RightTuple::BasicType* rightEnd)
//{
//	uint threads = blockDim.x * gridDim.x;
//	uint id      = threadIdx.x + blockDim.x * blockIdx.x;
//	
//	uint leftSize   = leftEnd  - leftBegin;
//	uint rightSize  = rightEnd - rightBegin;
//	uint resultSize = leftSize * rightSize;
//	
////if(threadIdx.x == 0 && blockIdx.x == 0) 
////for(unsigned int i = (resultSize - 1); i > (resultSize - 100); --i)
////printf("******before product %u %llx\n", i, leftBegin[i].a[7]);
//	for(uint i = id; i < resultSize; i += threads)
//	{
//		if(i < resultSize)
//		{
//			uint leftIndex  = i / rightSize;
//			uint rightIndex = i % rightSize;
//			
//			result[i] = tuple::combine<LeftTuple, RightTuple, ResultTuple, 0>(
//				leftBegin[leftIndex], rightBegin[rightIndex]);
//		}
//	}
//
////if(threadIdx.x == 0 && blockIdx.x == 0) 
////for(unsigned int i = (resultSize - 1); i > (resultSize - 100); --i)
////printf("******after product %u %llx\n", i, result[i].a[8]);
//
////if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llx %llx %llx\n", result[0].a[0], result[0].a[1], result[0].a[2]);
////	if(threadIdx.x == 0 && blockIdx.x == 0) printf("product %u\n", leftSize);
////	if(threadIdx.x == 0 && blockIdx.x == 0) printf("product %u\n", rightSize);
////	if(threadIdx.x == 0 && blockIdx.x == 0) printf("product %u\n", resultSize);
//}
}

}

#endif

