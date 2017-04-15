/*! \file Join.h
	\date Thursday December 2, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the join family of functions.

	This set of functions is intended to be simple and correct, rather
		than complicated and efficient.
*/

#ifndef JOIN_H_INCLUDED
#define JOIN_H_INCLUDED

#ifndef PARTITIONS
#define PARTITIONS 30
#endif

#include <stdio.h>

// RedFox Includes
#include <redfox/ra/interface/Tuple.h>
#include <redfox/ra/interface/Scan.h>
#include <redfox/ra/interface/Comparisons.h>

#define cache_size 128 

namespace ra
{

namespace cuda
{

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

template<typename Left, typename Right, unsigned int keyFields>
__device__ unsigned int lowerBound(const typename Left::BasicType key,
	const typename Right::BasicType* begin,
	const typename Right::BasicType* end)
{
	unsigned int low = 0;
	unsigned int high = end - begin;
	
	while(low < high)
	{
		unsigned int median = (low + ((high - low) >> 1));
		if(tuple::stripValues<Right, keyFields>(begin[median]) < key)
		{
			low = median + 1;
		}
		else
		{
			high = median;
		}
	}
	
	return low;
}

template<typename Left, typename Right, unsigned int keyFields>
__device__ unsigned int upperBound(const typename Left::BasicType key,
	const typename Right::BasicType* begin,
	const typename Right::BasicType* end)
{
	unsigned int low = 0;
	unsigned int high = end - begin;
	
	while(low < high)
	{
		unsigned int median = (low + ((high - low) >> 1));
		if(key < tuple::stripValues<Right, keyFields>(begin[median]))
		{
			high = median;
		}
		else
		{
			low = median + 1;
		}
	}
	
	return low;
}

//template<typename Left, typename Right, unsigned int keyFields>
//__device__ unsigned int upperBound2(const typename Left::BasicType key,
//	const typename Right::BasicType* begin,
//	const typename Right::BasicType* end)
//{
//	unsigned int low = 0;
//	unsigned int high = end - begin;
//
//	while(low < high)
//	{
//		unsigned int median = (low + (high - low) / 2);
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("***median %u\n", median);
//
//typename Right::BasicType rkey = tuple::stripValues<Right, keyFields>(begin[median]);
//
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("***rkey %llx %llx %llx %llx\n", rkey.a[0], rkey.a[1], rkey.a[2], rkey.a[3]);
//		if(key < rkey)
//		{
//			high = median;
//		}
//		else
//		{
//			low = median + 1;
//		}
//	}
//	
//	return low;
//}

__device__ unsigned int log2(unsigned int value)
{
	unsigned int targetlevel = 0;
	while (value >>= 1) ++targetlevel;
	
	return targetlevel;
}

//template<typename Left, typename Right, unsigned int keyFields>
//__device__ void findBounds2(
//	const typename Left::BasicType*  leftBegin,
//	const typename Left::BasicType*  leftEnd,
//	const typename Right::BasicType* rightBegin,
//	const typename Right::BasicType* rightEnd,
//	unsigned int* lowerBounds,
//	unsigned int* upperBounds,
//	unsigned int* outBounds)
//{
//	typedef typename Left::BasicType  LeftType;
//	typedef typename Right::BasicType RightType;
//
//	const unsigned int leftElements  = leftEnd  - leftBegin;
//	const unsigned int rightElements = rightEnd - rightBegin;
//	
//	unsigned int start = blockIdx.x * blockDim.x + threadIdx.x;
//	unsigned int step  = blockDim.x * gridDim.x;
//	
//	const unsigned int partitions    = gridDim.x;
//	const unsigned int partitionSize = (leftElements / partitions) + 1;
//
//	if(threadIdx.x == 0 && blockIdx.x == 0)
//	{
//		printf("%u\n", leftElements);
//		printf("%u\n", rightElements);
//		printf("left  %llx %llx\n", leftBegin[leftElements - 1].a[0], leftBegin[leftElements - 1].a[1]);
//		printf("right %llx\n", rightBegin[0]);
//	}
//	for(unsigned int i = start; i < partitions; i += step)
//	{
//		unsigned int leftIndex = partitionSize * i;
//		unsigned int leftMax   = MIN(partitionSize * (i + 1), leftElements);
//
//		if(leftIndex < leftElements)
//		{
//			LeftType key = tuple::stripValues<Left, keyFields>(
//				leftBegin[leftIndex]);
//
//			if(i == 126) 
//				printf("*********lower key %llx\n", key.a[0]);
//
////			if(blockIdx.x == 0 && threadIdx.x == 0) 
////				printf("*********right %llx %llx\n", rightBegin[leftIndex].a[0], rightBegin[leftIndex].a[1]);
//
////			RightType rkey = tuple::stripValues<Right, keyFields>(
////				rightBegin[leftIndex]);
////
////			if(blockIdx.x == 0 && threadIdx.x == 0 && i == 0) 
////				printf("*********right key %llx", rkey);
//
//			unsigned int lBound = lowerBound<Left, Right, keyFields>(key,
//				rightBegin, rightEnd);
//
//			if(i == 126) 
//				printf("*********lBound %u\n", lBound);
//
//			lowerBounds[i] = lBound;
//
//			unsigned int uBound = 0;
//			if(leftMax <= leftElements)
//			{
//				LeftType key = tuple::stripValues<Left, keyFields>(
//					leftBegin[leftMax-1]);
//
//				if(i == 126) 
//					printf("*********upper key %llx\n", key.a[0]);
//
//				uBound = upperBound<Left, Right, keyFields>(key,
//					rightBegin, rightEnd);
//
//				if(i == 126) 
//					printf("*********uBound %u\n", uBound);
//			}
//			else
//			{
//				uBound = rightElements;
//			}
//
//			upperBounds[i] = uBound;
////			outBounds[i]   = (uBound - lBound) * (leftMax - leftIndex);
//			outBounds[i]   = MAX((uBound - lBound), (leftMax - leftIndex));
//		}
//		else
//		{
//			lowerBounds[i] = rightElements;
//			upperBounds[i] = rightElements;
//			outBounds[i]   = 0;
//		}
//	}
//
//	if(blockIdx.x == 0 && threadIdx.x ==0)
//		for(int i = 0; i < 128; i++) 
//			printf("*********findbound result %u, %u, %u\n", lowerBounds[i], upperBounds[i], outBounds[i]);
//}

template<typename Left, typename Right, unsigned int keyFields>
__device__ void findBounds(
	const typename Left::BasicType*  leftBegin,
	const typename Left::BasicType*  leftEnd,
	const typename Right::BasicType* rightBegin,
	const typename Right::BasicType* rightEnd,
	unsigned int* lowerBounds,
	unsigned int* upperBounds,
	unsigned int* outBounds)
{
	typedef typename Left::BasicType  LeftType;
	typedef typename Right::BasicType RightType;

	const unsigned int leftElements  = leftEnd  - leftBegin;
	const unsigned int rightElements = rightEnd - rightBegin;
	
	unsigned int start = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int step  = blockDim.x * gridDim.x;
	
	const unsigned int partitions    = gridDim.x;
	const unsigned int partitionSize = (leftElements / partitions) + 1;

//	if (threadIdx.x == 0 && blockIdx.x == 0)
//		printf("%x %llx %llx\n", leftBegin[0], rightBegin[0].a[0], rightBegin[0].a[1]);

	for(unsigned int i = start; i < partitions; i += step)
	{
		unsigned int leftIndex = partitionSize * i;
		unsigned int leftMax   = MIN(partitionSize * (i + 1), leftElements);

		if(leftIndex < leftElements)
		{
			LeftType key = tuple::stripValues<Left, keyFields>(
				leftBegin[leftIndex]);

//			if(i == 0) 
//				printf("*********low key %x\n", key);

			unsigned int lBound = lowerBound<Left, Right, keyFields>(key,
				rightBegin, rightEnd);

//			if(i == 0) 
//				printf("*********lBound %u\n", lBound);

			lowerBounds[i] = lBound;

			unsigned int uBound = 0;
			if(leftMax <= leftElements)
			{
				LeftType key = tuple::stripValues<Left, keyFields>(
					leftBegin[leftMax-1]);

//				if(i == 0) 
//					printf("*********high key %x\n", leftBegin[leftMax - 1]);

				uBound = upperBound<Left, Right, keyFields>(key,
					rightBegin, rightEnd);
			}
			else
			{
				uBound = rightElements;
			}

			upperBounds[i] = uBound;
//			outBounds[i]   = (uBound - lBound) * (leftMax - leftIndex);
//			outBounds[i]   = MIN((uBound - lBound), (leftMax - leftIndex));
			outBounds[i]   = MAX((uBound - lBound), (leftMax - leftIndex));
		}
		else
		{
			lowerBounds[i] = rightElements;
			upperBounds[i] = rightElements;
			outBounds[i]   = 0;
		}
	}

//	if(threadIdx.x ==0) 
//		printf("*********findbound result %u: %u %u %u\n", blockIdx.x, lowerBounds[blockIdx.x], upperBounds[blockIdx.x], outBounds[blockIdx.x]);
}

template<typename ResultType, typename LeftType, typename RightType>
__device__ void getTempSize(long long unsigned int* size,
	const long long unsigned int* leftSize,
	const long long unsigned int* rightSize)
{
//	*size = sizeof(ResultType)
//		* (*leftSize  / sizeof(LeftType))
//		* (*rightSize / sizeof(RightType));

//	*size = sizeof(ResultType)
//		* MIN((*leftSize  / sizeof(LeftType)),
//		 (*rightSize / sizeof(RightType)));

	*size = 2 * sizeof(ResultType)
		* MAX((*leftSize  / sizeof(LeftType)),
		 (*rightSize / sizeof(RightType)));

//printf("*******************************get temp size %llu\n", *size);
//printf("*******************************result type size %d\n", sizeof(ResultType));
//printf("*******************************left type size %d\n", sizeof(LeftType));
//printf("*******************************right type size %d\n", sizeof(RightType));
//printf("*******************************left size %llu\n", *leftSize);
//printf("*******************************right size %llu\n", *rightSize);
}

template<typename type>
__device__ type* dma(type* out, const type* in, const type* inEnd)
{
	unsigned int elements = inEnd - in;
	
	for(unsigned int i = threadIdx.x; i < elements; i+=blockDim.x)
	{
		out[i] = in[i];
	}
	
	return out + elements;
}

template<typename Left, typename Right, typename Out,
        unsigned int keyFields, unsigned int threads>
__device__ unsigned int joinBlock(typename Out::BasicType* out,
	const typename Left::BasicType* left,  const typename Left::BasicType* leftEnd,
	const typename Right::BasicType* right, const typename Right::BasicType* rightEnd)
{
	typedef typename Left::BasicType  LeftType;
	typedef typename Right::BasicType RightType;
	typedef typename Out::BasicType   OutType;

	__shared__ OutType cache[cache_size];
	__syncthreads();

	const RightType* r = right + threadIdx.x;
	
	RightType rValue = 0;
	RightType rKey = 0;
	unsigned int rightElements = rightEnd - right;
	unsigned int foundCount = 0;	
	
        unsigned int lower = 0;
	unsigned int higher = 0;

	if(threadIdx.x < rightElements)
	{
		rKey = tuple::stripValues<Right, keyFields>(*r);
                rValue = *r;

		lower  = lowerBound<Right, Left, keyFields>(rKey, left, leftEnd);
		higher = upperBound<Right, Left, keyFields>(rKey, left, leftEnd);
		
		foundCount = higher - lower;
	}
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("*****foundCount %u\n", foundCount);	
	__syncthreads();

	unsigned int total = 0;
	unsigned int index = exclusiveScan<threads, unsigned int>(foundCount, total);

	__syncthreads();
	
	if(total <= cache_size)
	{
		for(unsigned int c = 0; c < foundCount; ++c)
		{
			LeftType lValue = left[lower+c];
			cache[index + c] = tuple::combine<Left, Right, Out, keyFields>(lValue, rValue);
		}
		
		__syncthreads();
		
		dma<OutType>(out, cache, cache + total);

	}
	else
	{
		printf("something is wrong.\n");
//		__shared__ unsigned int sharedCopiedThisTime;
//	
//		unsigned int copiedSoFar = 0;
//		bool done = false;
//	
//		while(copiedSoFar < total)
//		{
//			if(index + foundCount <= cache_size && !done)
//			{
//				for(unsigned int c = 0; c < foundCount; ++c)
//				{
//					LeftType lValue = left[lower + c];
//					cache[index + c] = tuple::combine<Left, Right, Out, keyFields>(lValue, rValue);
//				}
//			
//				done = true;
//			}
//			
//			if(index <= cache_size && index + foundCount > cache_size) //overbounded thread
//			{
//				sharedCopiedThisTime = index;
//			}
//			else if (threadIdx.x == threads - 1 && done)
//			    sharedCopiedThisTime = index + foundCount;
//		
//			__syncthreads();
//		
//			unsigned int copiedThisTime = sharedCopiedThisTime;
//		
//			index -= copiedThisTime;
//			copiedSoFar += copiedThisTime;
//		
//			out = dma<OutType>(out, cache, cache + copiedThisTime);
//		}
	}

	return total;
}

template<typename Left, typename Right, typename Out,
        unsigned int keyFields, unsigned int threads>
__device__ unsigned int joinBlock_string(typename Out::BasicType* out,
	const typename Left::BasicType* left,  const typename Left::BasicType* leftEnd,
	const typename Right::BasicType* right, const typename Right::BasicType* rightEnd)
{
	typedef typename Left::BasicType  LeftType;
	typedef typename Right::BasicType RightType;
	typedef typename Out::BasicType   OutType;

	__shared__ OutType cache[cache_size];
	__syncthreads();

	const RightType* r = right + threadIdx.x;
	
	RightType rValue = 0;
	RightType rKey = 0;
	unsigned int rightElements = rightEnd - right;
	unsigned int foundCount = 0;	
	
        unsigned int lower = 0;
	unsigned int higher = 0;

	if(threadIdx.x < rightElements)
	{
		rKey = tuple::stripValues<Right, keyFields - 1>(*r);
                rValue = *r;

		lower  = lowerBound<Right, Left, keyFields - 1>(rKey, left, leftEnd);
		higher = upperBound<Right, Left, keyFields - 1>(rKey, left, leftEnd);
	
		char *rString = (char *)(tuple::extract<unsigned long long int, keyFields - 1, Right>(*r));
		typedef ra::comparisons::eqstring<char *> Comparison;
		Comparison comp;
	
		for(; lower < higher; lower++)
		{
			char *lString = (char *)(tuple::extract<unsigned long long int, keyFields - 1, Left>(left[lower]));

			if(comp(rString, lString))
				break;
		}

		for(; higher > lower; higher--)
		{
			char *lString = (char *)(tuple::extract<unsigned long long int, keyFields - 1, Left>(left[higher - 1]));

			if(comp(rString, lString))
				break;
		}

		foundCount = higher - lower;
	}
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("*****foundCount %u\n", foundCount);	
	__syncthreads();

	unsigned int total = 0;
	unsigned int index = exclusiveScan<threads, unsigned int>(foundCount, total);

	__syncthreads();
	
	if(total <= cache_size)
	{
		for(unsigned int c = 0; c < foundCount; ++c)
		{
			LeftType lValue = left[lower+c];
			cache[index + c] = tuple::combine<Left, Right, Out, keyFields>(lValue, rValue);
		}
		
		__syncthreads();
		
		dma<OutType>(out, cache, cache + total);

	}
	else
	{
		printf("something is wrong.\n");
//		__shared__ unsigned int sharedCopiedThisTime;
//	
//		unsigned int copiedSoFar = 0;
//		bool done = false;
//	
//		while(copiedSoFar < total)
//		{
//			if(index + foundCount <= cache_size && !done)
//			{
//				for(unsigned int c = 0; c < foundCount; ++c)
//				{
//					LeftType lValue = left[lower + c];
//					cache[index + c] = tuple::combine<Left, Right, Out, keyFields>(lValue, rValue);
//				}
//			
//				done = true;
//			}
//			
//			if(index <= cache_size && index + foundCount > cache_size) //overbounded thread
//			{
//				sharedCopiedThisTime = index;
//			}
//			else if (threadIdx.x == threads - 1 && done)
//			    sharedCopiedThisTime = index + foundCount;
//		
//			__syncthreads();
//		
//			unsigned int copiedThisTime = sharedCopiedThisTime;
//		
//			index -= copiedThisTime;
//			copiedSoFar += copiedThisTime;
//		
//			out = dma<OutType>(out, cache, cache + copiedThisTime);
//		}
	}

	return total;
}

template<typename Left, typename Right, typename Out,
	unsigned int keyFields, unsigned int threads>
__device__ void join(
	const typename Left::BasicType*  leftBegin,
	const typename Left::BasicType*  leftEnd,
	const typename Right::BasicType* rightBegin,
	const typename Right::BasicType* rightEnd,
	typename Out::BasicType*         output,
	unsigned int*                    histogram,
	unsigned int*                    lowerBounds,
	unsigned int*                    upperBounds,
	unsigned int*                    outBounds)
{
	typedef typename Left::BasicType  LeftType;
	typedef typename Right::BasicType RightType;
	typedef typename Out::BasicType   OutType;

	__shared__ LeftType leftCache[threads];
//	__syncthreads();
	__shared__ RightType rightCache[threads];
	__syncthreads();

	const unsigned int leftElements  = leftEnd - leftBegin;
	
	unsigned int id = blockIdx.x;
	
	const unsigned int partitions    = gridDim.x;
	const unsigned int partitionSize = (leftElements / partitions) + 1;
	
	const LeftType* l    = leftBegin
		+ MIN(partitionSize * id,       leftElements);
	const LeftType* lend = leftBegin
		+ MIN(partitionSize * (id + 1), leftElements);

	const RightType* r    = rightBegin + lowerBounds[id];
	const RightType* rend = rightBegin + upperBounds[id];
	
	OutType* oBegin = output + outBounds[id] - outBounds[0];
	OutType* o      = oBegin;
	
	while(l != lend && r != rend)
	{
		unsigned int leftBlockSize  = MIN(lend - l, threads);
		unsigned int rightBlockSize = MIN(rend - r, threads);

		const LeftType* leftBlockEnd  = l + leftBlockSize;
		const RightType* rightBlockEnd = r + rightBlockSize;

		dma<LeftType>(leftCache,  l, leftBlockEnd );
		dma<RightType>(rightCache, r, rightBlockEnd);

		__syncthreads();

		LeftType lMaxValue = tuple::stripValues<Left, keyFields>(*(leftCache + leftBlockSize - 1));
		RightType rMinValue = tuple::stripValues<Right, keyFields>(*rightCache);
	
		if(lMaxValue < rMinValue)
		{
			l = leftBlockEnd;
		}
		else
		{
			LeftType lMinValue = tuple::stripValues<Left, keyFields>(*leftCache);
			RightType rMaxValue = tuple::stripValues<Right, keyFields>(*(rightCache + rightBlockSize - 1));
						
			if(rMaxValue < lMinValue)
			{
				r = rightBlockEnd;
			}
			else
			{
				unsigned int joined = joinBlock<Left, Right, Out, keyFields, threads>(o,
					leftCache,  leftCache  + leftBlockSize,
					rightCache, rightCache + rightBlockSize);
//				if(threadIdx.x == 0 && blockIdx.x == 0) printf("***joined %u\n", joined);
				o += joined;
				const RightType* ri = rightBlockEnd;

				for(; ri != rend;)
				{
					rightBlockSize = MIN(threads, rend - rightBlockEnd);
					rightBlockEnd  = rightBlockEnd + rightBlockSize;
					dma<RightType>(rightCache, ri, rightBlockEnd);
				
					__syncthreads();
			
					rMinValue = tuple::stripValues<Right, keyFields>(*rightCache);
   			
					if(lMaxValue < rMinValue) break;
			
					joined = joinBlock<Left, Right, Out, keyFields, threads>(o,
						leftCache,  leftCache  + leftBlockSize,
						rightCache, rightCache + rightBlockSize);
	                   
					o += joined;
					ri = rightBlockEnd;
				}
        	
				l = leftBlockEnd;
			}
			__syncthreads();
		}
	}	

	if(threadIdx.x == 0) 
	{
		histogram[id] = o - oBegin;
//		printf("%u %u\n", id, histogram[id]);
	}
//	if(blockIdx.x == 0 && threadIdx.x == 0) printf("join result %x\n", oBegin);
}

template<typename Left, typename Right, typename Out,
	unsigned int keyFields, unsigned int threads>
__device__ void join_string(
	const typename Left::BasicType*  leftBegin,
	const typename Left::BasicType*  leftEnd,
	const typename Right::BasicType* rightBegin,
	const typename Right::BasicType* rightEnd,
	typename Out::BasicType*         output,
	unsigned int*                    histogram,
	unsigned int*                    lowerBounds,
	unsigned int*                    upperBounds,
	unsigned int*                    outBounds)
{
	typedef typename Left::BasicType  LeftType;
	typedef typename Right::BasicType RightType;
	typedef typename Out::BasicType   OutType;

	__shared__ LeftType leftCache[threads];
//	__syncthreads();
	__shared__ RightType rightCache[threads];
	__syncthreads();

	const unsigned int leftElements  = leftEnd - leftBegin;
	
	unsigned int id = blockIdx.x;
	
	const unsigned int partitions    = gridDim.x;
	const unsigned int partitionSize = (leftElements / partitions) + 1;
	
	const LeftType* l    = leftBegin
		+ MIN(partitionSize * id,       leftElements);
	const LeftType* lend = leftBegin
		+ MIN(partitionSize * (id + 1), leftElements);

	const RightType* r    = rightBegin + lowerBounds[id];
	const RightType* rend = rightBegin + upperBounds[id];
	
	OutType* oBegin = output + outBounds[id] - outBounds[0];
	OutType* o      = oBegin;
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("before join left %llx %llx\n", leftBegin[0].a[5], leftBegin[0].a[6]);
	while(l != lend && r != rend)
	{
		unsigned int leftBlockSize  = MIN(lend - l, threads);
		unsigned int rightBlockSize = MIN(rend - r, threads);

		const LeftType* leftBlockEnd  = l + leftBlockSize;
		const RightType* rightBlockEnd = r + rightBlockSize;

		dma<LeftType>(leftCache,  l, leftBlockEnd );
		dma<RightType>(rightCache, r, rightBlockEnd);

		__syncthreads();

		LeftType lMaxValue = tuple::stripValues<Left, keyFields - 1>(*(leftCache + leftBlockSize - 1));
		RightType rMinValue = tuple::stripValues<Right, keyFields - 1>(*rightCache);
	
		if(lMaxValue < rMinValue)
		{
			l = leftBlockEnd;
		}
		else
		{
			LeftType lMinValue = tuple::stripValues<Left, keyFields - 1>(*leftCache);
			RightType rMaxValue = tuple::stripValues<Right, keyFields - 1>(*(rightCache + rightBlockSize - 1));
						
			if(rMaxValue < lMinValue)
			{
				r = rightBlockEnd;
			}
			else
			{
				unsigned int joined = joinBlock_string<Left, Right, Out, keyFields, threads>(o,
					leftCache,  leftCache  + leftBlockSize,
					rightCache, rightCache + rightBlockSize);
				o += joined;
				const RightType* ri = rightBlockEnd;

				for(; ri != rend;)
				{
					rightBlockSize = MIN(threads, rend - rightBlockEnd);
					rightBlockEnd  = rightBlockEnd + rightBlockSize;
					dma<RightType>(rightCache, ri, rightBlockEnd);
				
					__syncthreads();
			
					rMinValue = tuple::stripValues<Right, keyFields - 1>(*rightCache);
   			
					if(lMaxValue < rMinValue) break;
			
					joined = joinBlock_string<Left, Right, Out, keyFields, threads>(o,
						leftCache,  leftCache  + leftBlockSize,
						rightCache, rightCache + rightBlockSize);
	                   
					o += joined;
					ri = rightBlockEnd;
				}
        	
				l = leftBlockEnd;
			}
			__syncthreads();
		}
	}	

	if(threadIdx.x == 0) 
	{
		histogram[id] = o - oBegin;
//		printf("%u %u\n", id, histogram[id]);
	}
//	if(blockIdx.x == 0 && threadIdx.x == 0) printf("join result %x\n", oBegin);


//if(threadIdx.x == 0 && blockIdx.x == 0) printf("aftre join result %llx %llx\n", output[0].a[5], output[0].a[6]);
}

//template<typename Left, typename Right, typename Out,
//        unsigned int keyFields, unsigned int threads>
//__device__ unsigned int joinBlock2(typename Out::BasicType* out,
//	const typename Left::BasicType* left,  const typename Left::BasicType* leftEnd,
//	const typename Right::BasicType* right, const typename Right::BasicType* rightEnd)
//{
//	typedef typename Left::BasicType  LeftType;
//	typedef typename Right::BasicType RightType;
//	typedef typename Out::BasicType   OutType;
//
//	__shared__ OutType cache[cache_size];
//	__syncthreads();
//
//	const RightType* r = right + threadIdx.x;
//	
//	RightType rValue = 0;
//	RightType rKey = 0;
//	unsigned int rightElements = rightEnd - right;
//	unsigned int foundCount = 0;	
//	
//	unsigned int lower = 0;
//	unsigned int higher = 0;
//
//	if(threadIdx.x < rightElements)
//	{
//		rKey = tuple::stripValues<Right, keyFields>(*r);
//		rValue = *r;
////if(threadIdx.x == 0 && blockIdx.x == 0) 
////{
////LeftType lKey = tuple::stripValues<Left, keyFields>(left[64]);
////
////printf("*****%llx %llx %llx %llx\n", lKey.a[0], lKey.a[1], lKey.a[2], lKey.a[3]);	
////printf("*****%llx %llx\n", rKey.a[0], rKey.a[1]);	
////}
//		lower  = lowerBound<Right, Left, keyFields>(rKey, left, leftEnd);
//		higher = upperBound<Right, Left, keyFields>(rKey, left, leftEnd);
//		
//		foundCount = higher - lower;
//	}
////if(threadIdx.x < 65 && (threadIdx.x + blockIdx.x * 65) < 8192 && foundCount != 1) printf("*****foundCount %u %u %u\n", blockIdx.x, threadIdx.x, foundCount);	
//	__syncthreads();
//
//	unsigned int total = 0;
//	unsigned int index = 0;
////	if(blockIdx.x == 0) {
////		index = exclusiveScan2<threads, unsigned int>(foundCount, total);
////	}
////	else
//		index = exclusiveScan<threads, unsigned int>(foundCount, total);
//
//	__syncthreads();
//	
////if(blockIdx.x < 126 && total != 65 ) 
////{
////	if(threadIdx.x == 0)	
////		printf("*****blk %u total %u\n", blockIdx.x, total);
////
////	printf("*****thr %u foundCount %u, index %u\n", threadIdx.x, foundCount, index);
////}
//
//	if(total <= cache_size)
//	{
//		for(unsigned int c = 0; c < foundCount; ++c)
//		{
//			LeftType lValue = left[lower+c];
////if(blockIdx.x == 0 && threadIdx.x == 0)
////{
////	cache[index + c] = tuple::combine2<Left, Right, Out, keyFields>(lValue, rValue);
////	printf("cta join %llx %llx\n", lValue.a[0], cache[0].a[2]);
////
////}
////else
//	cache[index + c] = tuple::combine<Left, Right, Out, keyFields>(lValue, rValue);
//		}
//		
//		__syncthreads();
//		
//		dma<OutType>(out, cache, cache + total);
//
//	}
/////	else
////	{
////		__shared__ unsigned int sharedCopiedThisTime;
////	
////		unsigned int copiedSoFar = 0;
////		bool done = false;
////	
////		while(copiedSoFar < total)
////		{
////			if(index + foundCount <= cache_size && !done)
////			{
////				for(unsigned int c = 0; c < foundCount; ++c)
////				{
////					LeftType lValue = left[lower + c];
////					cache[index + c] = tuple::combine<Left, Right, Out, keyFields>(lValue, rValue);
////				}
////			
////				done = true;
////			}
////			
////			if(index <= cache_size && index + foundCount > cache_size) //overbounded thread
////			{
////				sharedCopiedThisTime = index;
////			}
////			else if (threadIdx.x == threads - 1 && done)
////			    sharedCopiedThisTime = index + foundCount;
////		
////			__syncthreads();
////		
////			unsigned int copiedThisTime = sharedCopiedThisTime;
////		
////			index -= copiedThisTime;
////			copiedSoFar += copiedThisTime;
////		
////			out = dma<OutType>(out, cache, cache + copiedThisTime);
////		}
////	}
//
//	return total;
//}

//template<typename Left, typename Right, typename Out,
//	unsigned int keyFields, unsigned int threads>
//__device__ void join2(
//	const typename Left::BasicType*  leftBegin,
//	const typename Left::BasicType*  leftEnd,
//	const typename Right::BasicType* rightBegin,
//	const typename Right::BasicType* rightEnd,
//	typename Out::BasicType*         output,
//	unsigned int*                    histogram,
//	unsigned int*                    lowerBounds,
//	unsigned int*                    upperBounds,
//	unsigned int*                    outBounds)
//{
//
//	typedef typename Left::BasicType  LeftType;
//	typedef typename Right::BasicType RightType;
//	typedef typename Out::BasicType   OutType;
//
//	__shared__ LeftType leftCache[threads];
//	__shared__ RightType rightCache[threads];
//	__syncthreads();
//
//	const unsigned int leftElements  = leftEnd - leftBegin;
//	const unsigned int rightElements  = rightEnd - rightBegin;
//	
////	if(blockIdx.x == 0 && threadIdx.x == 0)
////		printf("*************************************join2 %llx\n", leftBegin[0].a[0]);
//
//	unsigned int id = blockIdx.x;
//	
//	const unsigned int partitions    = gridDim.x;
//	const unsigned int partitionSize = (leftElements / partitions) + 1;
//
//	//if(threadIdx.x == 0 && blockIdx.x == 0)
//	//{
//	//	for(int i = 0; i < 8192; ++i)
//	//	{
//	//		unsigned long long int left = (leftBegin[i].a[2]) >> 3;
//	//		unsigned long long int right = rightBegin[i].a[1];
//	//		if(left != right)
//	//		{
//	//			printf("*****before join %u %llx %llx\n", i, left, right);
//	////			return;
//	//		}
//	//	}
//	//}
////if(threadIdx.x == 0 && blockIdx.x == 0) printf("*****%llx %llx %llx %llx\n", leftBegin[64].a[0], leftBegin[64].a[1], leftBegin[64].a[2], leftBegin[64].a[3]);	
//	const LeftType* l    = leftBegin
//		+ MIN(partitionSize * id,       leftElements);
//	const LeftType* lend = leftBegin
//		+ MIN(partitionSize * (id + 1), leftElements);
//
//	const RightType* r    = rightBegin + lowerBounds[id];
//	const RightType* rend = rightBegin + upperBounds[id];
//	
//	OutType* oBegin = output + outBounds[id] - outBounds[0];
//	OutType* o      = oBegin;
//	
//	while(l != lend && r != rend)
//	{
//		unsigned int leftBlockSize  = MIN(lend - l, threads);
//		unsigned int rightBlockSize = MIN(rend - r, threads);
//
//		const LeftType* leftBlockEnd  = l + leftBlockSize;
//		const RightType* rightBlockEnd = r + rightBlockSize;
//
//		dma<LeftType>(leftCache,  l, leftBlockEnd );
//		dma<RightType>(rightCache, r, rightBlockEnd);
//
//		__syncthreads();
//
//		LeftType lMaxValue = tuple::stripValues<Left, keyFields>(*(leftCache + leftBlockSize - 1));
//		RightType rMinValue = tuple::stripValues<Right, keyFields>(*rightCache);
//	
//		if(lMaxValue < rMinValue)
//		{
//			l = leftBlockEnd;
//		}
//		else
//		{
//			LeftType lMinValue = tuple::stripValues<Left, keyFields>(*leftCache);
//			RightType rMaxValue = tuple::stripValues<Right, keyFields>(*(rightCache + rightBlockSize - 1));
//
//			//if(lMinValue.a[0] != rMinValue.a[0] && threadIdx.x == 0) printf("between join1 %u %llx %llx", blockIdx.x, lMinValue.a[0], rMinValue.a[0]);
//			//if(lMaxValue.a[0] != rMaxValue.a[0] && threadIdx.x == 0) printf("between join2 %u %llx %llx", blockIdx.x, lMaxValue.a[0], rMaxValue.a[0]);
//						
//			if(rMaxValue < lMinValue)
//			{
//				r = rightBlockEnd;
//			}
//			else
//			{
//				unsigned int joined = joinBlock<Left, Right, Out, keyFields, threads>(o,
//					leftCache,  leftCache  + leftBlockSize,
//					rightCache, rightCache + rightBlockSize);
////				if(threadIdx.x == 0 && blockIdx.x == 0) printf("*** join block %llx\n", o[0].a[2]);
//				o += joined;
//				const RightType* ri = rightBlockEnd;
//
//				for(; ri != rend;)
//				{
//					rightBlockSize = MIN(threads, rend - rightBlockEnd);
//					rightBlockEnd  = rightBlockEnd + rightBlockSize;
//					dma<RightType>(rightCache, ri, rightBlockEnd);
//				
//					__syncthreads();
//			
//					rMinValue = tuple::stripValues<Right, keyFields>(*rightCache);
//   			
//					if(lMaxValue < rMinValue) break;
//			
//					joined = joinBlock<Left, Right, Out, keyFields, threads>(o,
//						leftCache,  leftCache  + leftBlockSize,
//						rightCache, rightCache + rightBlockSize);
//	                   
//					o += joined;
//					ri = rightBlockEnd;
//				}
//        	
//				l = leftBlockEnd;
//			}
//			__syncthreads();
//		}
//	}	
//
//	if(threadIdx.x == 0) 
//	{
//		histogram[id] = o - oBegin;
////		if(blockIdx.x == 0) printf("%u %u\n", id, histogram[id]);
//	}
//
////	if(blockIdx.x == 0 && threadIdx.x == 0)
////		printf("*************************************join2 %llx\n", output[0].a[2]);
//
//
////	if(blockIdx.x == 0 && threadIdx.x == 0) 
////		for(unsigned int i = 0; i < gridDim.x; ++i)
////			printf("***** %u %u\n", i, histogram[i]);
//}

template<typename ResultType, unsigned int index>
__device__ void getResultSize(long long unsigned int* size,
	const unsigned int* histogram)
{
	*size = histogram[index] * sizeof(ResultType);
//	printf("****************join get size %u\n", histogram[index]);
}

template<typename Tuple>
__device__ void gather(
	typename Tuple::BasicType* begin,
	typename Tuple::BasicType* end,
	const typename Tuple::BasicType* inBegin,
	const typename Tuple::BasicType* inEnd,
	const unsigned int* outBounds,
	const unsigned int* histogram)
{
	typedef typename Tuple::BasicType type;
	typedef typename Tuple::BasicType* pointer;
	typedef const typename Tuple::BasicType* const_pointer;
	
	const_pointer inWindowBegin = inBegin + outBounds[blockIdx.x];
	
	unsigned int beginIndex = histogram[blockIdx.x];
	unsigned int endIndex   = histogram[blockIdx.x + 1];
	
	unsigned int elements = endIndex - beginIndex;
	
	unsigned int start = threadIdx.x;
	unsigned int step  = blockDim.x;
	
	for(unsigned int i = start; i < elements; i += step)
	{
		begin[beginIndex + i] = inWindowBegin[i];
	}

//	if(threadIdx.x == 0 && blockIdx.x == 0)
//	{
//		printf("join gather 0 %x\n", begin[0]);
//		printf("join gather 8192 %x\n", begin[8191]);
//	}
}

template<typename Tuple>
__device__ void gather2(
	typename Tuple::BasicType* begin,
	typename Tuple::BasicType* end,
	const typename Tuple::BasicType* inBegin,
	const typename Tuple::BasicType* inEnd,
	const unsigned int* outBounds,
	const unsigned int* histogram)
{
	typedef typename Tuple::BasicType type;
	typedef typename Tuple::BasicType* pointer;
	typedef const typename Tuple::BasicType* const_pointer;
	
	const_pointer inWindowBegin = inBegin + outBounds[blockIdx.x];
	
	unsigned int beginIndex = histogram[blockIdx.x];
	unsigned int endIndex   = histogram[blockIdx.x + 1];
	
	unsigned int elements = endIndex - beginIndex;
	
	unsigned int start = threadIdx.x;
	unsigned int step  = blockDim.x;
	
//	if(threadIdx.x == 0 && blockIdx.x == 0)
//	{
//		printf("join gather %llx %llx %llx %llx %llx %llx\n", inWindowBegin[0].a[0], inWindowBegin[0].a[1], inWindowBegin[0].a[2], inWindowBegin[0].a[3], inWindowBegin[0].a[4], inWindowBegin[0].a[5]);
//	}

	for(unsigned int i = start; i < elements; i += step)
	{
		begin[beginIndex + i] = inWindowBegin[i];
	}
__syncthreads();
//	if(threadIdx.x == 0 && blockIdx.x == 0)
//	{
//		printf("join gather %llx %llx %llx %llx %llx %llx %llx\n", begin[0].a[0], begin[0].a[1], begin[0].a[2], begin[0].a[3], begin[0].a[4], begin[0].a[5], begin[0].a[6]);
//	}
}
}

}

#endif

