/*! \file Project.h
	\date Wednesday August 8, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the Project family of cuda functions.
*/

#ifndef PROJECT_H_INCLUDED
#define PROJECT_H_INCLUDED

#include <stdio.h>

#include <redfox/ra/interface/Tuple.h>

//#include <hydrazine/cuda/Cuda.h>
#include <hydrazine/interface/macros.h>
#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace ra
{

namespace cuda
{

template<unsigned int d0, unsigned int d1, unsigned int d2,
	unsigned int d3, unsigned int d4,unsigned int d5,
	unsigned int d6, unsigned int d7,unsigned int d8,
	unsigned int d9, unsigned int d10,unsigned int d11,
	unsigned int d12, unsigned int d13,unsigned int d14,
	unsigned int d15>
class PermuteMap
{
public:
	static const unsigned int dimension0 = d0;
	static const unsigned int dimension1 = d1;
	static const unsigned int dimension2 = d2;
	static const unsigned int dimension3 = d3;
	static const unsigned int dimension4 = d4;
	static const unsigned int dimension5 = d5;
	static const unsigned int dimension6 = d6;
	static const unsigned int dimension7 = d7;
	static const unsigned int dimension8 = d8;
	static const unsigned int dimension9 = d9;
	static const unsigned int dimension10 = d10;
	static const unsigned int dimension11 = d11;
	static const unsigned int dimension12 = d12;
	static const unsigned int dimension13 = d13;
	static const unsigned int dimension14 = d14;
	static const unsigned int dimension15 = d15;
};

template<typename Tuple, typename ResultTuple, typename PermuteMap>
__device__ void permuteElement(typename ResultTuple::BasicType& out,
	const typename Tuple::BasicType& in)
{
//	typedef typename ResultTuple::BasicType OutType;
//	typename ResultTuple::BasicType temp = 0;


	typedef unsigned long long int OutType;
	typename ResultTuple::BasicType temp = 0;

	temp = tuple::insert<OutType, 0, ResultTuple>(temp, 
		tuple::extract<OutType, PermuteMap::dimension0, Tuple>(in));
	temp = tuple::insert<OutType, 1, ResultTuple>(temp, 
		tuple::extract<OutType, PermuteMap::dimension1, Tuple>(in));
	temp = tuple::insert<OutType, 2, ResultTuple>(temp, 
		tuple::extract<OutType, PermuteMap::dimension2, Tuple>(in));
	temp = tuple::insert<OutType, 3, ResultTuple>(temp, 
		tuple::extract<OutType, PermuteMap::dimension3, Tuple>(in));
	temp = tuple::insert<OutType, 4, ResultTuple>(temp, 
		tuple::extract<OutType, PermuteMap::dimension4, Tuple>(in));
	temp = tuple::insert<OutType, 5, ResultTuple>(temp, 
		tuple::extract<OutType, PermuteMap::dimension5, Tuple>(in));
	temp = tuple::insert<OutType, 6, ResultTuple>(temp, 
		tuple::extract<OutType, PermuteMap::dimension6, Tuple>(in));
	temp = tuple::insert<OutType, 7, ResultTuple>(temp, 
		tuple::extract<OutType, PermuteMap::dimension7, Tuple>(in));
	temp = tuple::insert<OutType, 8, ResultTuple>(temp, 
		tuple::extract<OutType, PermuteMap::dimension8, Tuple>(in));
	temp = tuple::insert<OutType, 9, ResultTuple>(temp, 
		tuple::extract<OutType, PermuteMap::dimension9, Tuple>(in));
	temp = tuple::insert<OutType, 10, ResultTuple>(temp, 
		tuple::extract<OutType, PermuteMap::dimension10, Tuple>(in));
	temp = tuple::insert<OutType, 11, ResultTuple>(temp, 
		tuple::extract<OutType, PermuteMap::dimension11, Tuple>(in));
	temp = tuple::insert<OutType, 12, ResultTuple>(temp, 
		tuple::extract<OutType, PermuteMap::dimension12, Tuple>(in));
	temp = tuple::insert<OutType, 13, ResultTuple>(temp, 
		tuple::extract<OutType, PermuteMap::dimension13, Tuple>(in));
	temp = tuple::insert<OutType, 14, ResultTuple>(temp, 
		tuple::extract<OutType, PermuteMap::dimension14, Tuple>(in));
	temp = tuple::insert<OutType, 15, ResultTuple>(temp, 
		tuple::extract<OutType, PermuteMap::dimension15, Tuple>(in));

	out = temp;
}

template<typename ResultTuple, typename Tuple, typename PermuteMap>
__device__ void permute(typename ResultTuple::BasicType* result,
	const typename Tuple::BasicType* begin,
	const typename Tuple::BasicType* end)
{
	unsigned int step     = gridDim.x * blockDim.x;
	unsigned int start    = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int elements = end - begin;
	
	for(unsigned int i = start; i < elements; i += step)
	{
		if(i < elements)
		{
			permuteElement<Tuple, ResultTuple, PermuteMap>(result[i], begin[i]);
		}
	}

//	if(threadIdx.x == 0 && blockIdx.x == 0 && elements > 1)
//		for(unsigned int i = 0; i < 300; ++i) 
//			printf("after project %u %llu\n", i, begin[i].a[1]);
}

template<typename ResultType, typename SourceType>
__device__ void getResultSize(long long unsigned int* size)
{
//	printf("*****project result size %llu\n", *size);
//	printf("*****project result size %llu\n", sizeof(ResultType));
//	printf("*****project result size %llu\n", sizeof(SourceType));
	*size = (*size * sizeof(ResultType))/ sizeof(SourceType);
//	printf("*****project result size %llu, %p\n", *size, size);
}

//template<typename Tuple, typename ResultTuple, typename PermuteMap>
//__device__ void permuteElement2(typename ResultTuple::BasicType& out,
//	const typename Tuple::BasicType& in)
//{
//	typedef unsigned long long int OutType;
////	typename ResultTuple::BasicType temp = in;
//	typename ResultTuple::BasicType temp = 0;
//	typename ResultTuple::BasicType temp_out = 0;
////if(threadIdx.x == 2 && blockIdx.x == 0){
////	unsigned long long int  tmp = tuple::extract<unsigned long long int, PermuteMap::dimension0, Tuple>(in);
//// printf("***permute element %llx\n", tmp);}
//
//	if(blockIdx.x == 0 && threadIdx.x == 0)
//	{
//		printf("***************in[0] %llx %llx\n", in.a[5], in.a[6]);
//
//		OutType temp0 = tuple::extract<OutType, PermuteMap::dimension0, Tuple>(in);
//		printf("***************temp0 %llx\n", temp0);
//		temp_out = tuple::insert2<OutType, 0, ResultTuple>(temp_out, temp0);
//		printf("after insert temp0 %llx %llx\n", temp_out.a[0], temp_out.a[1]);
//
//		OutType temp1 = tuple::extract<OutType, PermuteMap::dimension1, Tuple>(in);
//		printf("***************temp1 %llx\n", temp1);
//		temp_out = tuple::insert<OutType, 1, ResultTuple>(temp_out, temp1);
//		printf("after insert temp1 %llx %llx\n", temp_out.a[0], temp_out.a[1]);
//
////		OutType temp2 = tuple::extract<OutType, PermuteMap::dimension2, Tuple>(in);
////		printf("***************temp2 %llu\n", temp2);
////		temp_out = tuple::insert<OutType, 2, ResultTuple>(temp_out, temp2);
////		printf("after insert temp2 %llx %llx\n", temp_out.a[0], temp_out.a[1]);
////
////		OutType temp3 = tuple::extract<OutType, PermuteMap::dimension3, Tuple>(in);
////		printf("***************temp3 %llu\n", temp3);
////		temp_out = tuple::insert<OutType, 3, ResultTuple>(temp_out, temp3);
////		printf("after insert temp3 %llx %llx\n", temp_out.a[0], temp_out.a[1]);
////
////		OutType temp4 = tuple::extract<OutType, PermuteMap::dimension4, Tuple>(in);
////		printf("***************temp4 %llu\n", temp4);
////		temp_out = tuple::insert<OutType, 4, ResultTuple>(temp_out, temp4);
////		printf("after insert temp4 %llx %llx\n", temp_out.a[0], temp_out.a[1]);
////
////		OutType temp5 = tuple::extract<OutType, PermuteMap::dimension5, Tuple>(in);
////		printf("***************temp5 %llu\n", temp5);
////		temp_out = tuple::insert2<OutType, 5, ResultTuple>(temp_out, temp5);
////		printf("after insert temp5 %llx %llx\n", temp_out.a[0], temp_out.a[1]);
//	}
//
//	temp = tuple::insert<OutType, 0, ResultTuple>(temp, 
//		tuple::extract<OutType, PermuteMap::dimension0, Tuple>(in));
//
//	temp = tuple::insert<OutType, 1, ResultTuple>(temp, 
//		tuple::extract<OutType, PermuteMap::dimension1, Tuple>(in));
//
////	if(temp.a[1] != 0x0 && temp.a[1] != 0x1 && temp.a[1] != 0x3 && temp.a[1] != 0x5)
////	{
////		printf("*****during projection %u %u\n", blockIdx.x, threadIdx.x);
////		printf("*****during projection %llx %llx\n", temp.a[1], temp.a[0]);
////		printf("*****during projection %llx %llx %llx %llx %llx %llx %llx %llx %llx %llx\n", 
////							       in.a[10], in.a[9], in.a[8],
////							       in.a[7], in.a[6], in.a[5], in.a[4], in.a[3], in.a[2], in.a[1], in.a[0]);
////	}
//
//	temp = tuple::insert<OutType, 2, ResultTuple>(temp, 
//		tuple::extract<OutType, PermuteMap::dimension2, Tuple>(in));
//	temp = tuple::insert<OutType, 3, ResultTuple>(temp, 
//		tuple::extract<OutType, PermuteMap::dimension3, Tuple>(in));
//	temp = tuple::insert<OutType, 4, ResultTuple>(temp, 
//		tuple::extract<OutType, PermuteMap::dimension4, Tuple>(in));
//	temp = tuple::insert<OutType, 5, ResultTuple>(temp, 
//		tuple::extract<OutType, PermuteMap::dimension5, Tuple>(in));
//	temp = tuple::insert<OutType, 6, ResultTuple>(temp, 
//		tuple::extract<OutType, PermuteMap::dimension6, Tuple>(in));
//	temp = tuple::insert<OutType, 7, ResultTuple>(temp, 
//		tuple::extract<OutType, PermuteMap::dimension7, Tuple>(in));
//	temp = tuple::insert<OutType, 8, ResultTuple>(temp, 
//		tuple::extract<OutType, PermuteMap::dimension8, Tuple>(in));
//	temp = tuple::insert<OutType, 9, ResultTuple>(temp, 
//		tuple::extract<OutType, PermuteMap::dimension9, Tuple>(in));
//	temp = tuple::insert<OutType, 10, ResultTuple>(temp, 
//		tuple::extract<OutType, PermuteMap::dimension10, Tuple>(in));
//	temp = tuple::insert<OutType, 11, ResultTuple>(temp, 
//		tuple::extract<OutType, PermuteMap::dimension11, Tuple>(in));
//	temp = tuple::insert<OutType, 12, ResultTuple>(temp, 
//		tuple::extract<OutType, PermuteMap::dimension12, Tuple>(in));
//	temp = tuple::insert<OutType, 13, ResultTuple>(temp, 
//		tuple::extract<OutType, PermuteMap::dimension13, Tuple>(in));
//	temp = tuple::insert<OutType, 14, ResultTuple>(temp, 
//		tuple::extract<OutType, PermuteMap::dimension14, Tuple>(in));
//	temp = tuple::insert<OutType, 15, ResultTuple>(temp, 
//		tuple::extract<OutType, PermuteMap::dimension15, Tuple>(in));
//
//	out = temp;
//}

//template<typename ResultTuple, typename Tuple, typename PermuteMap>
//__device__ void permute2(typename ResultTuple::BasicType* result,
//	const typename Tuple::BasicType* begin,
//	const typename Tuple::BasicType* end)
//{
//	unsigned int step     = gridDim.x * blockDim.x;
//	unsigned int start    = threadIdx.x + blockIdx.x * blockDim.x;
//	unsigned int elements = end - begin;
//
////	if(blockIdx.x == 0 && threadIdx.x == 0)
////		printf("project elements %u\n", elements);
//	
//	if(threadIdx.x == 0 && blockIdx.x == 0)
//	{
//		for(unsigned int i = 0; i < 10; i++)
//			printf("before projection %u, %llu, %llu\n", i, begin[i].a[1], begin[i].a[0]);
//	}
//
//	for(unsigned int i = start; i < elements; i += step)
//	{
//		if(i < elements)
//		{
//			permuteElement<Tuple, ResultTuple, PermuteMap>(result[i], begin[i]);
//		}
//	}
//
////	if(threadIdx.x == 0 && blockIdx.x == 0)
////	{
////		for(unsigned int i = 0; i < 6384; i++)
////		 printf("after projection %s\n", (char *)result[i]);
////	}
//}
}

}

#endif

