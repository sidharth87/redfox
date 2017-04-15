/*! \file   Select.h
	\date   Thursday January 6, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  Select a set of elements that match a predicate.
*/

#ifndef SELECT_H_INCLUDED
#define SELECT_H_INCLUDED

#ifndef PARTITIONS
#define PARTITIONS 30
#endif

// RedFox Includes
#include <redfox/ra/interface/Tuple.h>
#include <redfox/ra/interface/Comparisons.h>
#include <redfox/ra/interface/Scan.h>
#include <redfox/ra/interface/Date.h>

#include <stdio.h>

namespace ra
{

namespace cuda
{

#define MIN(x, y) (((x) < (y)) ? (x) : (y));

template<typename Tuple, typename Comparison,
	typename DataType, unsigned int threads, unsigned int field>
__device__ void select_field_constant(typename Tuple::BasicType* output, 
	unsigned int* histogram, const typename Tuple::BasicType* input,
	const typename Tuple::BasicType* inputEnd, DataType constant)
{
	typedef typename Tuple::BasicType Element;

	__shared__ Element outCache[threads];

	__syncthreads();

	const unsigned int id    = threadIdx.x;
	const unsigned int block  = blockIdx.x;

	const unsigned int elements = inputEnd - input;
	const unsigned int numTiles = (elements + threads - 1) / threads;

	uint2 task;
	unsigned int begin, end;
	task.x = numTiles / PARTITIONS;
	task.y = numTiles - task.x * PARTITIONS;
	begin = task.x * block;
	begin += MIN(block, task.y);
	end = begin + task.x + (block < task.y);
	begin *= threads;
	end = MIN(elements, end * threads);

	const unsigned int step  = blockDim.x;
	
	unsigned int total = 0;

//if(threadIdx.x == 0) printf("select block %u %u %u\n", blockIdx.x, blockDim.x, begin);	
	for(unsigned int index = begin; index < end; index += step)
	{
		unsigned int myIndex = index + id;
		unsigned int match   = 0;

		Element value;

		if(myIndex < end)
		{
			value = input[myIndex];

			Comparison comp;

			DataType key = tuple::extract<DataType, field, Tuple>(value);
//if(blockIdx.x == 127) printf("***********************key %u %s\n", myIndex, key);
			if(comp(key, constant))
			{
				match = 1;
			}
		}
				
		__syncthreads();
		
		unsigned int max = 0;
		unsigned int localIndex = exclusiveScan<threads>(match, max);
		
		if(match != 0)
		{
			outCache[localIndex] = value;
		}
		
		__syncthreads();
		
		if(id < max)
		{
			output[begin + total + id] = outCache[id];
		}
		
		total += max;
	}
	
	if(threadIdx.x == 0)
	{
		histogram[blockIdx.x] = total;
	}

//	if(threadIdx.x == 0)
//		printf("%u %u\n", blockIdx.x, total);
//	if(threadIdx.x == 0 && blockIdx.x == 1)
//		for(unsigned int i = 0; i < total; ++i)
//			printf("%u %llu\n", begin + i, output[begin + i].a[1]);
}

template<typename Tuple, typename Comparison,
	typename DataType, unsigned int threads, unsigned int field>
__device__ void select_field_floatconstant(typename Tuple::BasicType* output, 
	unsigned int* histogram, const typename Tuple::BasicType* input,
	const typename Tuple::BasicType* inputEnd, DataType constant)
{
	typedef typename Tuple::BasicType Element;

	__shared__ Element outCache[threads];

	__syncthreads();

	const unsigned int id    = threadIdx.x;
	const unsigned int block  = blockIdx.x;

	const unsigned int elements = inputEnd - input;
	const unsigned int numTiles = (elements + threads - 1) / threads;

	uint2 task;
	unsigned int begin, end;
	task.x = numTiles / PARTITIONS;
	task.y = numTiles - task.x * PARTITIONS;
	begin = task.x * block;
	begin += MIN(block, task.y);
	end = begin + task.x + (block < task.y);
	begin *= threads;
	end = MIN(elements, end * threads);

	const unsigned int step  = blockDim.x;
	
	unsigned int total = 0;

	for(unsigned int index = begin; index < end; index += step)
	{
		unsigned int myIndex = index + id;
		unsigned int match   = 0;

		Element value;

		if(myIndex < end)
		{
			value = input[myIndex];

			Comparison comp;

			DataType key = tuple::extract_double<field, Tuple>(value);
//if(blockIdx.x == 127) printf("***********************key %u %s\n", myIndex, key);
			if(comp(key, constant))
			{
				match = 1;
			}
		}
				
		__syncthreads();
		
		unsigned int max = 0;
		unsigned int localIndex = exclusiveScan<threads>(match, max);
		
		if(match != 0)
		{
			outCache[localIndex] = value;
		}
		
		__syncthreads();
		
		if(id < max)
		{
			output[begin + total + id] = outCache[id];
		}
		
		total += max;
	}
	
	if(threadIdx.x == 0)
	{
		histogram[blockIdx.x] = total;
	}

//	if(threadIdx.x == 0 && blockIdx.x == 0)
//		for(unsigned int i = 0; i < 128; ++i)
//			printf("%u %u\n", i, histogram[i]);
}

//template<typename Tuple, typename Comparison,
//	typename DataType, unsigned int threads, unsigned int field>
//__device__ void select_field_constant2(typename Tuple::BasicType* output, 
//	unsigned int* histogram, const typename Tuple::BasicType* input,
//	const typename Tuple::BasicType* inputEnd, DataType constant)
//{
//	typedef typename Tuple::BasicType Element;
//
//	__shared__ Element outCache[threads];
//
//	__syncthreads();
//
//	const unsigned int id    = threadIdx.x;
//	const unsigned int step  = blockDim.x;
//	
//	const unsigned int elements = inputEnd - input;
//	const unsigned int partitions    = PARTITIONS;
//	const unsigned int partitionSize = (elements / partitions) + 1;
//	
//	const unsigned int begin = MIN(blockIdx.x * partitionSize,       elements);
//	const unsigned int end   = MIN((blockIdx.x + 1) * partitionSize, elements);
//	
//	unsigned int total = 0;
////if(blockIdx.x == 0 && threadIdx.x == 0) printf("************%u\n", input);
//if(blockIdx.x == 0 && threadIdx.x == 0) printf("************constant %s\n", constant);
//	for(unsigned int index = begin; index < end; index += step)
//	{
//		unsigned int myIndex = index + id;
//		unsigned int match   = 0;
//
//		Element value;
//
//		if(myIndex < end)
//		{
//			value = input[myIndex];
//
//			Comparison comp;
//
//			DataType key = tuple::extract<DataType, field, Tuple>(value);
////printf("***********************key %s\n", key);
//			if(comp(key, constant))
//			{
//				match = 1;
//			}
//		}
//				
//		__syncthreads();
//		
//		unsigned int max = 0;
//		unsigned int localIndex = exclusiveScan<threads>(match, max);
//		
//		if(match != 0)
//		{
//			outCache[localIndex] = value;
//		}
//		
//		__syncthreads();
//		
//		if(id < max)
//		{
//			output[begin + total + id] = outCache[id];
//		}
//		
//		total += max;
//	}
//	
//	if(threadIdx.x == 0)
//	{
//		histogram[blockIdx.x] = total;
//	}
//}

template<typename Tuple, typename Comparison0, typename Comparison1,
	typename DataType, unsigned int threads, unsigned int field0, unsigned int field1>
__device__ void select_field_constant_2(typename Tuple::BasicType* output, 
	unsigned int* histogram, const typename Tuple::BasicType* input,
	const typename Tuple::BasicType* inputEnd, DataType constant0, DataType constant1)
{
	typedef typename Tuple::BasicType Element;

	__shared__ Element outCache[threads];

	__syncthreads();

	const unsigned int id    = threadIdx.x;
	const unsigned int block  = blockIdx.x;

	const unsigned int elements = inputEnd - input;
	const unsigned int numTiles = (elements + threads - 1) / threads;

	uint2 task;
	unsigned int begin, end;
	task.x = numTiles / PARTITIONS;
	task.y = numTiles - task.x * PARTITIONS;
	begin = task.x * block;
	begin += MIN(block, task.y);
	end = begin + task.x + (block < task.y);
	begin *= threads;
	end = MIN(elements, end * threads);

	const unsigned int step  = blockDim.x;
	
	unsigned int total = 0;

	for(unsigned int index = begin; index < end; index += step)
	{
		unsigned int myIndex = index + id;
		unsigned int match   = 0;

		Element value;

		if(myIndex < end)
		{
			value = input[myIndex];

			Comparison0 comp0;
			Comparison1 comp1;

			DataType key0 = tuple::extract<DataType, field0, Tuple>(value);
			DataType key1 = tuple::extract<DataType, field1, Tuple>(value);

			if(comp0(key0, constant0) && comp1(key1, constant1))
			{
				match = 1;
			}
		}
				
		__syncthreads();
		
		unsigned int max = 0;
		unsigned int localIndex = exclusiveScan<threads>(match, max);
		
		if(match != 0)
		{
			outCache[localIndex] = value;
		}
		
		__syncthreads();
		
		if(id < max)
		{
			output[begin + total + id] = outCache[id];
		}
		
		total += max;
	}
	
	if(threadIdx.x == 0)
	{
		histogram[blockIdx.x] = total;
	}
}

template<typename Tuple, typename Comparison0, typename Comparison1,
	typename DataType, unsigned int threads, unsigned int field00,
	unsigned int field01, unsigned int field10, unsigned int field11>
__device__ void select_field_field_2(typename Tuple::BasicType* output, 
	unsigned int* histogram, const typename Tuple::BasicType* input,
	const typename Tuple::BasicType* inputEnd)
{
	typedef typename Tuple::BasicType Element;

	__shared__ Element outCache[threads];

	__syncthreads();

	const unsigned int id    = threadIdx.x;
	const unsigned int block  = blockIdx.x;

	const unsigned int elements = inputEnd - input;
	const unsigned int numTiles = (elements + threads - 1) / threads;

	uint2 task;
	unsigned int begin, end;
	task.x = numTiles / PARTITIONS;
	task.y = numTiles - task.x * PARTITIONS;
	begin = task.x * block;
	begin += MIN(block, task.y);
	end = begin + task.x + (block < task.y);
	begin *= threads;
	end = MIN(elements, end * threads);

	const unsigned int step  = blockDim.x;
	
	unsigned int total = 0;

	for(unsigned int index = begin; index < end; index += step)
	{
		unsigned int myIndex = index + id;
		unsigned int match   = 0;

		Element value;

		if(myIndex < end)
		{
			value = input[myIndex];

			Comparison0 comp0;
			Comparison1 comp1;

			DataType key00 = tuple::extract<DataType, field00, Tuple>(value);
			DataType key01 = tuple::extract<DataType, field01, Tuple>(value);

			DataType key10 = tuple::extract<DataType, field10, Tuple>(value);
			DataType key11 = tuple::extract<DataType, field11, Tuple>(value);

//if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llu %llu %llu %llu\n", key00, key01, key10, key11);
		
			if(comp0(key00, key01) && comp1(key10, key11))
			{
				match = 1;
			}
		}
		
//		if(blockIdx.x == 126) printf("### match %u %u\n", threadIdx.x, match);		
	
		__syncthreads();

		unsigned int max = 0;
		unsigned int localIndex = exclusiveScan<threads, unsigned int>(match, max);
		
//		if(blockIdx.x == 126) printf("### max %u %u\n", threadIdx.x, localIndex);
		
		if(match != 0)
		{
			outCache[localIndex] = value;
		}
		
		__syncthreads();
		
		if(id < max)
		{
			output[begin + total + id] = outCache[id];
		}
		
		total += max;
	}
	
	if(threadIdx.x == 0)
	{
		histogram[blockIdx.x] = total;
	}

//	if(threadIdx.x == 0 && blockIdx.x == 0)
//		for(int i = 0; i < gridDim.x; i++)
//			printf("%u\n", histogram[i]);
}

template<typename Tuple, typename Comparison0, typename Comparison1,
	typename DataType, unsigned int threads, unsigned int field00,
	unsigned int left, unsigned int right, unsigned int field10, unsigned int field11>
__device__ void select_addmonth_field_field(typename Tuple::BasicType* output, 
	unsigned int* histogram, const typename Tuple::BasicType* input,
	const typename Tuple::BasicType* inputEnd)
{
	typedef typename Tuple::BasicType Element;

	__shared__ Element outCache[threads];

	__syncthreads();

	const unsigned int id    = threadIdx.x;
	const unsigned int block  = blockIdx.x;

	const unsigned int elements = inputEnd - input;
	const unsigned int numTiles = (elements + threads - 1) / threads;

	uint2 task;
	unsigned int begin, end;
	task.x = numTiles / PARTITIONS;
	task.y = numTiles - task.x * PARTITIONS;
	begin = task.x * block;
	begin += MIN(block, task.y);
	end = begin + task.x + (block < task.y);
	begin *= threads;
	end = MIN(elements, end * threads);

	const unsigned int step  = blockDim.x;
	
	unsigned int total = 0;

	for(unsigned int index = begin; index < end; index += step)
	{
		unsigned int myIndex = index + id;
		unsigned int match   = 0;

		Element value;

		if(myIndex < end)
		{
			value = input[myIndex];

			Comparison0 comp0;
			Comparison1 comp1;

			DataType key00 = tuple::extract<DataType, field00, Tuple>(value);
			DataType date = tuple::extract<DataType, left, Tuple>(value);
			int year, month, day;
			ra::cuda::int2date(date, year, month, day);
			month += right;
			ra::cuda::wrapdate(year, month);
			date = ra::cuda::date2int(year, month, day);

			DataType key10 = tuple::extract<DataType, field10, Tuple>(value);
			DataType key11 = tuple::extract<DataType, field11, Tuple>(value);

//if(threadIdx.x == 9 && blockIdx.x == 0) printf("%d\n", key0);		
//if(threadIdx.x == 9 && blockIdx.x == 0) printf("%d\n", key1);		
			if(comp0(key00, date) && comp1(key10, key11))
			{
				match = 1;
			}
		}
		
//		if(blockIdx.x == 126) printf("### match %u %u\n", threadIdx.x, match);		
	
		__syncthreads();

		unsigned int max = 0;
		unsigned int localIndex = exclusiveScan<threads, unsigned int>(match, max);
		
//		if(blockIdx.x == 126) printf("### max %u %u\n", threadIdx.x, localIndex);
		
		if(match != 0)
		{
			outCache[localIndex] = value;
		}
		
		__syncthreads();
		
		if(id < max)
		{
			output[begin + total + id] = outCache[id];
		}
		
		total += max;
	}
	
	if(threadIdx.x == 0)
	{
		histogram[blockIdx.x] = total;
	}

//	if(threadIdx.x == 0 && blockIdx.x == 0)
//		for(int i = 0; i < gridDim.x; i++)
//			printf("%u\n", histogram[i]);
}

template<typename Tuple, typename Comparison0, typename Comparison1,
	typename DataType, unsigned int threads, unsigned int field00,
	unsigned int left, unsigned int right, unsigned int field10, unsigned int field11>
__device__ void select_add_field_field(typename Tuple::BasicType* output, 
	unsigned int* histogram, const typename Tuple::BasicType* input,
	const typename Tuple::BasicType* inputEnd)
{
	typedef typename Tuple::BasicType Element;

	__shared__ Element outCache[threads];

	__syncthreads();

	const unsigned int id    = threadIdx.x;
	const unsigned int block  = blockIdx.x;

	const unsigned int elements = inputEnd - input;
	const unsigned int numTiles = (elements + threads - 1) / threads;

	uint2 task;
	unsigned int begin, end;
	task.x = numTiles / PARTITIONS;
	task.y = numTiles - task.x * PARTITIONS;
	begin = task.x * block;
	begin += MIN(block, task.y);
	end = begin + task.x + (block < task.y);
	begin *= threads;
	end = MIN(elements, end * threads);

	const unsigned int step  = blockDim.x;
	
	unsigned int total = 0;

	for(unsigned int index = begin; index < end; index += step)
	{
		unsigned int myIndex = index + id;
		unsigned int match   = 0;

		Element value;

		if(myIndex < end)
		{
			value = input[myIndex];

			Comparison0 comp0;
			Comparison1 comp1;

			DataType key00 = tuple::extract_double<field00, Tuple>(value);
//if(threadIdx.x == 0 && blockIdx.x == 0 && index == begin) printf("%f\n", key00);		
			DataType data = tuple::extract_double<left, Tuple>(value);
//if(threadIdx.x == 0 && blockIdx.x == 0 && index == begin) printf("%f\n", data);		
//if(threadIdx.x == 0 && blockIdx.x == 0 && index == begin) printf("%u\n", right);		
			data += right;
//if(threadIdx.x == 0 && blockIdx.x == 0 && index == begin) printf("%f\n", data);		

			DataType key10 = tuple::extract_double<field10, Tuple>(value);
			DataType key11 = tuple::extract_double<field11, Tuple>(value);

//if(threadIdx.x == 0 && blockIdx.x == 0 && index == begin) printf("%f\n", key10);		
//if(threadIdx.x == 0 && blockIdx.x == 0 && index == begin) printf("%f\n", key11);		
//if(threadIdx.x == 9 && blockIdx.x == 0) printf("%d\n", key1);		
			if(comp0(key00, data) && comp1(key10, key11))
			{
				match = 1;
			}
		}
		
//		if(blockIdx.x == 126) printf("### match %u %u\n", threadIdx.x, match);		
	
		__syncthreads();

		unsigned int max = 0;
		unsigned int localIndex = exclusiveScan<threads, unsigned int>(match, max);
		
//		if(blockIdx.x == 126) printf("### max %u %u\n", threadIdx.x, localIndex);
		
		if(match != 0)
		{
			outCache[localIndex] = value;
		}
		
		__syncthreads();
		
		if(id < max)
		{
			output[begin + total + id] = outCache[id];
		}
		
		total += max;
	}
	
	if(threadIdx.x == 0)
	{
		histogram[blockIdx.x] = total;
	}

//	if(threadIdx.x == 0 && blockIdx.x == 0)
//		for(int i = 0; i < gridDim.x; i++)
//			printf("%u\n", histogram[i]);
}


#if 1 
template<typename Tuple, typename Comparison,
	typename DataType, unsigned int threads, unsigned int field0,
	unsigned int field1>
__device__ void select_field_field(typename Tuple::BasicType* output, 
	unsigned int* histogram, const typename Tuple::BasicType* input,
	const typename Tuple::BasicType* inputEnd)
{
	typedef typename Tuple::BasicType Element;

	__shared__ Element outCache[threads];

	__syncthreads();

	const unsigned int id    = threadIdx.x;
	const unsigned int block  = blockIdx.x;

	const unsigned int elements = inputEnd - input;
	const unsigned int numTiles = (elements + threads - 1) / threads;

	uint2 task;
	unsigned int begin, end;
	task.x = numTiles / PARTITIONS;
	task.y = numTiles - task.x * PARTITIONS;
	begin = task.x * block;
	begin += MIN(block, task.y);
	end = begin + task.x + (block < task.y);
	begin *= threads;
	end = MIN(elements, end * threads);

	const unsigned int step  = blockDim.x;
	
	unsigned int total = 0;
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("********** %llx %llx\n", input[0].a[1], input[0].a[2]);
	for(unsigned int index = begin; index < end; index += step)
	{
		unsigned int myIndex = index + id;
		unsigned int match   = 0;

		Element value;

		if(myIndex < end)
		{
			value = input[myIndex];

			Comparison comp;

			DataType key0 = tuple::extract<DataType, field0, Tuple>(value);
			DataType key1 = tuple::extract<DataType, field1, Tuple>(value);
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("key0 %llu\n", key0);		
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("key1 %llu\n", key1);		
			if(comp(key0, key1))
			{
				match = 1;
			}
		}
		
//		if(blockIdx.x == 126) printf("### match %u %u\n", threadIdx.x, match);		
	
		__syncthreads();

		unsigned int max = 0;
		unsigned int localIndex = exclusiveScan<threads, unsigned int>(match, max);
		
//		if(blockIdx.x == 126) printf("### max %u %u\n", threadIdx.x, localIndex);
		
		if(match != 0)
		{
			outCache[localIndex] = value;
		}
		
		__syncthreads();
		
		if(id < max)
		{
			output[begin + total + id] = outCache[id];
		}
		
		total += max;
	}
	
	if(threadIdx.x == 0)
	{
		histogram[blockIdx.x] = total;
	}

//	if(threadIdx.x == 0 && blockIdx.x == 0)
//		for(int i = 0; i < gridDim.x; i++)
//			printf("%d %u\n", i, histogram[i]);
}
#else

template<typename Tuple, typename Comparison,
	typename DataType, unsigned int NT, unsigned int field0,
	unsigned int field1>
__device__ void select_field_field(typename Tuple::BasicType* output, 
	unsigned int* histogram, const typename Tuple::BasicType* input,
	const typename Tuple::BasicType* inputEnd)
{
#define VT 1 
	typedef typename Tuple::BasicType T;

	const unsigned int tid    = threadIdx.x;
	const unsigned int block  = blockIdx.x;

	const unsigned int NV = NT * VT;
	const unsigned int count = inputEnd - input;
	const unsigned int numTiles = (count + NV - 1) / NV;

	uint2 task, range;
	task.x = numTiles / PARTITIONS;
	task.y = numTiles - task.x * PARTITIONS;
	range.x = task.x * block;
	range.x += MIN(block, task.y);
	range.y = range.x + task.x + (block < task.y);
	range.x *= NV;
	range.y = MIN(count, range.y * NV);

	typedef CTAScan<NT, unsigned int> S;
	union Shared
	{
		unsigned int indices[NV];
		typename S::Storage scan;
		unsigned int values[NV*16]; 	
	};

	__shared__ Shared shared;

	unsigned int total = 0;

	for(unsigned int gid = range.x; gid < range.y; gid += NV)
	{
		unsigned int sourceCount = MIN(NV, range.y - gid);

		// Set the flags to 1. The default is to copy a value.
		#pragma unroll
		for(unsigned int i = 0; i < VT; ++i)
		{
			unsigned int index = NT * i + tid;
			shared.indices[index] = index < sourceCount;
		}
	
		__syncthreads();

		// Load the data into register.
		T values[VT];
		unsigned int indices[VT];
		unsigned int matches[VT];
		unsigned int matches2[VT];
		unsigned int max;

		if(sourceCount == NV)
		{
			#pragma unroll
			for(unsigned int i = 0; i < VT; ++i)
				values[i] = input[gid + NT * i + tid];
		}
		else
		{
			#pragma unroll
			for(unsigned int i = 0; i < VT; ++i)
			{
				unsigned int index = NT * i + tid;
				if(index < sourceCount)
					values[i] = input[gid + index];
			}
		}

		// Set the counter to 0 for each index we've loaded.
		#pragma unroll
		for(unsigned int i = 0; i < VT; ++i)
		{
			unsigned int index = NT * i + tid;

			if(index < sourceCount)
			{
				Comparison comp;
	
				DataType key0 = tuple::extract<DataType, field0, Tuple>(values[i]);
				DataType key1 = tuple::extract<DataType, field1, Tuple>(values[i]);

				if(!comp(key0, key1))
				{
					shared.indices[index] = 0;
				}
			}
		}

		__syncthreads();

		#pragma unroll
		for(unsigned int i = 0; i < VT; ++i)
		{
			matches[i] = shared.indices[NT * i + tid];			
		}

		// Run a raking scan over the flags. We count the set flags - this is the 
		// number of elements to load in per thread.
		unsigned int x = 0;
		#pragma unroll
		for(unsigned int i = 0; i < VT; ++i)
			x += matches2[i] = shared.indices[VT * tid + i];
		__syncthreads();


		// Run a CTA scan and scatter the gather indices to shared memory.
		unsigned int scan = S::Scan(tid, x, shared.scan, &max);
//if(gid == 0 && block == 0)
//	printf("%u %u %u\n", tid, scan, max);

		#pragma unroll
		for(unsigned int i = 0; i < VT; ++i)
		{
			unsigned int index = VT * tid + i;
			unsigned int scan2 = scan + matches2[i]; 
			shared.indices[index] = scan;
			scan = scan2;
		}
		__syncthreads();

//if(gid == 0 && block == 0 && tid == 0)
//for(unsigned int i = 0; i < 128; ++i)
//{
//printf("%u, %u\n", i, shared.indices[i]);
//}
//__syncthreads();

		#pragma unroll
		for(unsigned int i = 0; i < VT; ++i)
		{
			indices[i] = shared.indices[NT * i + tid];			
		}

		__syncthreads();
		#pragma unroll
		for(unsigned int i = 0; i < VT; ++i)
			if(matches[i]) 
			{
				T* shared_values = (T*) shared.values;	
				shared_values[indices[i]] = values[i];
			}
		__syncthreads();

		#pragma unroll
		for(unsigned int i = 0; i < VT; ++i)
		{
			unsigned int index = NT * i + tid;

			if(index < max)
			{
				T* shared_values = (T*) shared.values;	
				output[range.x + total + index] = shared_values[index];
			}
		}
		
		total += max;
	}
	
	if(threadIdx.x == 0)
	{
		histogram[blockIdx.x] = total;
	}

//	if(threadIdx.x == 0 && blockIdx.x == 0)
//		for(int i = 0; i < gridDim.x; i++)
//			printf("%d %u\n", i, histogram[i]);
}
#endif

template<typename Tuple, typename Comparison,
	unsigned int threads, unsigned int field0,
	unsigned int field1>
__device__ void select_fielddouble_fielddouble(typename Tuple::BasicType* output, 
	unsigned int* histogram, const typename Tuple::BasicType* input,
	const typename Tuple::BasicType* inputEnd)
{
	typedef typename Tuple::BasicType Element;

	__shared__ Element outCache[threads];

	__syncthreads();

	const unsigned int id    = threadIdx.x;
	const unsigned int block  = blockIdx.x;

	const unsigned int elements = inputEnd - input;
	const unsigned int numTiles = (elements + threads - 1) / threads;

	uint2 task;
	unsigned int begin, end;
	task.x = numTiles / PARTITIONS;
	task.y = numTiles - task.x * PARTITIONS;
	begin = task.x * block;
	begin += MIN(block, task.y);
	end = begin + task.x + (block < task.y);
	begin *= threads;
	end = MIN(elements, end * threads);

	const unsigned int step  = blockDim.x;
	
	unsigned int total = 0;

	for(unsigned int index = begin; index < end; index += step)
	{
		unsigned int myIndex = index + id;
		unsigned int match   = 0;

		Element value;

		if(myIndex < end)
		{
			value = input[myIndex];

			Comparison comp;

			double key0 = tuple::extract_double<field0, Tuple>(value);
			double key1 = tuple::extract_double<field1, Tuple>(value);

//			unsigned long long int temp_key0 = tuple::extract<unsigned long long int, field0, Tuple>(value);
//			unsigned long long int temp_key1 = tuple::extract<unsigned long long int, field1, Tuple>(value);

//if(threadIdx.x == 0 && blockIdx.x == 0) printf("%f\n", key1);		
			if(comp(key0, key1))
			{
				match = 1;
			}
//printf("%u %lf %lf %u\n", myIndex, key0, key1, match);		
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llx %llx %u\n", temp_key0, temp_key1, (key0 < key1));		
		}
		
//		if(blockIdx.x == 126) printf("### match %u %u\n", threadIdx.x, match);		
	
		__syncthreads();

		unsigned int max = 0;
		unsigned int localIndex = exclusiveScan<threads, unsigned int>(match, max);
		
//		if(blockIdx.x == 126) printf("### max %u %u\n", threadIdx.x, localIndex);
		
		if(match != 0)
		{
			outCache[localIndex] = value;
		}
		
		__syncthreads();
		
		if(id < max)
		{
			output[begin + total + id] = outCache[id];
		}
		
		total += max;
	}
	
	if(threadIdx.x == 0)
	{
		histogram[blockIdx.x] = total;
	}

//	if(threadIdx.x == 0 && blockIdx.x == 0)
//		for(int i = 0; i < gridDim.x; i++)
//			printf("%d %u\n", i, histogram[i]);
}

template<typename ResultType, unsigned int partitions>
__device__ void getResultSize(long long unsigned int* size,
	const unsigned int* histogram)
{
	*size = histogram[partitions] * sizeof(ResultType);
	printf("**********************select result size %u\n", histogram[partitions]);
}

#if 0
template<typename Tuple>
__device__ void gather(
	typename Tuple::BasicType* begin,
	const typename Tuple::BasicType* tempBegin,
	const typename Tuple::BasicType* tempEnd,
	const unsigned int* histogram)
{
	typedef typename Tuple::BasicType type;
	typedef typename Tuple::BasicType* pointer;
	typedef const typename Tuple::BasicType* const_pointer;
	
	const unsigned int tempElements = tempEnd - tempBegin;

	const unsigned int partitions    = PARTITIONS;
	const unsigned int partitionSize = (tempElements / partitions) + 1;
	
	const_pointer inWindowBegin = tempBegin 
		+ MIN(blockIdx.x * partitionSize, tempElements);
	
	unsigned int beginIndex = histogram[blockIdx.x];
	unsigned int endIndex   = histogram[blockIdx.x + 1];
	
	unsigned int elements = endIndex - beginIndex;
	
	unsigned int start = threadIdx.x;
	unsigned int step  = blockDim.x;
	
	for(unsigned int i = start; i < elements; i += step)
	{
		begin[beginIndex + i] = inWindowBegin[i];
	}
}
#else
template<typename Tuple, unsigned int NT>
__device__ void gather(
	typename Tuple::BasicType* begin,
	const typename Tuple::BasicType* tempBegin,
	const typename Tuple::BasicType* tempEnd,
	const unsigned int* histogram)
{
#define VT 1 
	typedef typename Tuple::BasicType type;
	typedef typename Tuple::BasicType* pointer;
	typedef const typename Tuple::BasicType* const_pointer;
	
	const unsigned int tid    = threadIdx.x;
	const unsigned int block  = blockIdx.x;

	const unsigned int NV = NT * VT;
	const unsigned int count = tempEnd - tempBegin;
	const unsigned int numTiles = (count + NV - 1) / NV;

	uint2 task;
	unsigned int range;
	task.x = numTiles / PARTITIONS;
	task.y = numTiles - task.x * PARTITIONS;
	range = task.x * block;
	range += MIN(block, task.y);
	range *= NV;

	const_pointer inWindowBegin = tempBegin + range;

//if(threadIdx.x == 0 && blockIdx.x == 1)
//	for(int i = 0; i < 300; ++i)
//		printf("%d %llu\n", i, inWindowBegin[i].a[1]); 
//if(threadIdx.x == 0) printf("gather block %u %u %u %u %u %u %u %u\n", blockIdx.x, count, numTiles, task.x, task.y, NV, NT, VT);	
	unsigned int beginIndex = histogram[blockIdx.x];
	unsigned int endIndex   = histogram[blockIdx.x + 1];
	
	unsigned int elements = endIndex - beginIndex;
	
//	unsigned int start = threadIdx.x;
//	unsigned int step  = blockDim.x;

#define VT1 2 
	unsigned int start = 0;
	unsigned int step = blockDim.x * VT1;
	
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("%u %u %u\n", NT, VT1, blockDim.x);
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("%u %u %u\n", start, elements, step);

	for(unsigned int i = start; i < elements; i += step)
	{
//		if(threadIdx.x == 0 && blockIdx.x == 0) printf("%u\n", i);
		type reg[VT1];

		#pragma unroll 
		for(unsigned int j = 0; j < VT1; j++)
		{
			unsigned int index = blockDim.x * j + tid + i;

			if(index < elements)
				reg[j] = inWindowBegin[index];
		}

		#pragma unroll
		for(unsigned int j = 0; j < VT1; j++)
		{
			unsigned int index = blockDim.x * j + tid + i;

			if(index < elements)
				begin[beginIndex + index] = reg[j];
		} 
//		begin[beginIndex + i] = inWindowBegin[i];
	}
}
#endif
}
}

#endif

