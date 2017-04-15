/*! \file Sort.h
	\author Gregory Diamos <gregory.diamos>
	\date Wednesday December 1, 2010
	\brief The header file for the C interface to CUDA sorting routines.
*/

#ifndef MODERNGPU_JOIN_H_INCLUDED
#define MODERNGPU_JOIN_H_INCLUDED

#include <redfox/ra/interface/Tuple.h>

namespace redfox
{

extern void find_bounds_string(void* lower_bound, void* left_count, unsigned long long int *result_size, 
	void* left_key, unsigned long long int left_size, 
	void* right_key, unsigned long long int right_size);

extern void find_bounds_128(void* lower_bound, void* left_count, unsigned long long int *result_size, 
	void* left_key, unsigned long long int left_size, 
	void* right_key, unsigned long long int right_size);

extern void find_bounds_64(void* lower_bound, void* left_count, unsigned long long int *result_size, 
	void* left_key, unsigned long long int left_size, 
	void* right_key, unsigned long long int right_size);

extern void find_bounds_32(void* lower_bound, void* left_count, unsigned long long int *result_size, 
	void* left_key, unsigned long long int left_size, 
	void* right_key, unsigned long long int right_size);

extern void find_bounds_16(void* lower_bound, void* left_count, unsigned long long int *result_size, 
	void* left_key, unsigned long long int left_size, 
	void* right_key, unsigned long long int right_size);

extern void find_bounds_8(void* lower_bound, void* left_count, unsigned long long int *result_size, 
	void* left_key, unsigned long long int left_size, 
	void* right_key, unsigned long long int right_size);

extern void join(int* left_indices, int* right_indices, unsigned long long int result_size, 
	int* lowerBound, int* leftCount, 
	unsigned long long int input_size);

template<typename Tuple, typename LeftKey, unsigned int NT, unsigned int keys>
__device__ void gather_key_key(
	typename Tuple::BasicType* result_begin,
	typename LeftKey::BasicType* left_begin,
	int* left_index,
	unsigned long long int size)
{

#define VT 1 
#define MIN(x,y) ((x)<(y)?(x):(y))
	typedef typename Tuple::BasicType type;
	typedef typename LeftKey::BasicType ltype;
	
	const unsigned int tid    = threadIdx.x;
	const unsigned int block  = blockIdx.x;

	const unsigned int NV = NT * VT;
	const unsigned int numTiles = (size + NV - 1) / NV;

	uint2 task, range;
	task.x = numTiles / gridDim.x;
	task.y = numTiles - task.x * gridDim.x;
	range.x = task.x * block;
	range.x += MIN(block, task.y);
	range.y = range.x + task.x + (block < task.y);
	range.x *= NV;
	range.y = MIN(size, range.y * NV);

	for(unsigned int i = range.x; i < range.y; i += NV)
	{
		unsigned int index = i + tid;

		if (index < size)
			result_begin[index] = left_begin[left_index[index]];
	}
#undef VT
}

template<typename Tuple, typename LeftKey, typename RightValue, unsigned int NT, unsigned int keys>
__device__ void gather_key_value(
	typename Tuple::BasicType* result_begin,
	typename LeftKey::BasicType* left_begin,
	int* left_index,
	typename RightValue::BasicType* right_begin,
	int* right_index,
	unsigned long long int size)
{
#define VT 1 
#define MIN(x,y) ((x)<(y)?(x):(y))
	typedef typename Tuple::BasicType type;
	typedef typename LeftKey::BasicType ltype;
	typedef typename RightValue::BasicType rtype;
	
	const unsigned int tid    = threadIdx.x;
	const unsigned int block  = blockIdx.x;

	const unsigned int NV = NT * VT;
	const unsigned int numTiles = (size + NV - 1) / NV;

	uint2 task, range;
	task.x = numTiles / gridDim.x;
	task.y = numTiles - task.x * gridDim.x;
	range.x = task.x * block;
	range.x += MIN(block, task.y);
	range.y = range.x + task.x + (block < task.y);
	range.x *= NV;
	range.y = MIN(size, range.y * NV);

	for(unsigned int i = range.x; i < range.y; i += NV)
	{
		unsigned int index = i + tid;

		if (index < size)
			result_begin[index] = ra::tuple::restoreValues<Tuple, keys>((type)(left_begin[left_index[index]])) | (type)(right_begin[right_index[index]]);
	}
#undef VT
}

template<typename Tuple, typename LeftKey, typename LeftValue, unsigned int NT, unsigned int keys>
__device__ void gather_value_key(
	typename Tuple::BasicType* result_begin,
	typename LeftKey::BasicType* left_key_begin,
	typename LeftValue::BasicType* left_value_begin,
	int* left_index,
	unsigned long long int size)
{
#define VT 1 
#define MIN(x,y) ((x)<(y)?(x):(y))
	typedef typename Tuple::BasicType type;
	typedef typename LeftKey::BasicType lkey;
	typedef typename LeftValue::BasicType lvalue;
	
	const unsigned int tid    = threadIdx.x;
	const unsigned int block  = blockIdx.x;

	const unsigned int NV = NT * VT;
	const unsigned int numTiles = (size + NV - 1) / NV;

	uint2 task, range;
	task.x = numTiles / gridDim.x;
	task.y = numTiles - task.x * gridDim.x;
	range.x = task.x * block;
	range.x += MIN(block, task.y);
	range.y = range.x + task.x + (block < task.y);
	range.x *= NV;
	range.y = MIN(size, range.y * NV);

	for(unsigned int i = range.x; i < range.y; i += NV)
	{
		unsigned int index = i + tid;

		if (index < size)
			result_begin[index] = ra::tuple::restoreValues<Tuple, keys>((type)(left_key_begin[left_index[index]])) | (type)(left_value_begin[left_index[index]]);
	}
#undef VT
}

template<typename Tuple, typename LeftKey, typename LeftValue, typename RightValue, unsigned int NT, unsigned int keys>
__device__ void gather_value_value(
	typename Tuple::BasicType* result_begin,
	typename LeftKey::BasicType* left_key_begin,
	typename LeftValue::BasicType* left_value_begin,
	int* left_index,
	typename RightValue::BasicType* right_value_begin,
	int* right_index,
	unsigned long long int size)
{
#define VT 1 
#define MIN(x,y) ((x)<(y)?(x):(y))
	typedef typename Tuple::BasicType type;
	typedef typename LeftKey::BasicType lkey;
	typedef typename LeftValue::BasicType lvalue;
	typedef typename RightValue::BasicType rvalue;
	
	const unsigned int tid    = threadIdx.x;
	const unsigned int block  = blockIdx.x;

	const unsigned int NV = NT * VT;
	const unsigned int numTiles = (size + NV - 1) / NV;

	uint2 task, range;
	task.x = numTiles / gridDim.x;
	task.y = numTiles - task.x * gridDim.x;
	range.x = task.x * block;
	range.x += MIN(block, task.y);
	range.y = range.x + task.x + (block < task.y);
	range.x *= NV;
	range.y = MIN(size, range.y * NV);

	for(unsigned int i = range.x; i < range.y; i += NV)
	{
		unsigned int index = i + tid;

		if (index < size)
			result_begin[index] = ra::tuple::restoreValues<Tuple, keys>((type)(left_key_begin[left_index[index]])) | ((type)(left_value_begin[left_index[index]]) << (RightValue::bits)) | (type)(right_value_begin[right_index[index]]);
	}
#undef VT
}
}

#endif

