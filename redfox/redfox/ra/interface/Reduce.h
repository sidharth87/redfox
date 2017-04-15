/*! \file Sort.h
	\author Gregory Diamos <gregory.diamos>
	\date Wednesday December 1, 2010
	\brief The header file for the C interface to CUDA sorting routines.
*/

#ifndef REDUCE_H_INCLUDED
#define REDUCE_H_INCLUDED

#include <redfox/ra/interface/Tuple.h>

// Thrust Includes
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>

namespace redfox
{

//template<typename T>
//class WrappedType 
//{
//public:
//	typedef T type;
//};

//template<typename KeyTuple>
//extern void count(const typename KeyTuple::BasicType* key_begin, 
//	typename KeyTuple::BasicType* result_key_begin, 
//	unsigned long long int* result_value_begin, unsigned long long int *key_size, 
//	unsigned long long int *value_size);

//template<typename KeyTuple, typename ValueType, typename ReduceType, 
//	 typename binary_pred, typename binary_op>
//extern void reduce(const typename KeyTuple::BasicType* key_begin, 
//	const typename ValueType::type* value_begin,  
//	typename KeyTuple::BasicType* result_key_begin, 
//	typename ReduceType::type* result_value_begin, 
//	unsigned long long int *key_size,
//	unsigned long long int *value_size);

extern void count_8(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size);

extern void count_16(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size);

extern void count_32(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size);

extern void count_string(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size);

extern void count_128(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size);

extern void count_256(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size);

extern void total_string_double(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size);

extern void total_8_double(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size);

extern void total_16_double(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size);

extern void total_double(void* result,
	const void* begin, 
	const void* end);

extern void max_double(void* result,
	const void* begin, 
	const void* end);

extern void count(void* result,
	const void* begin, 
	const void* end);

extern void total_32_double(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size);

extern void total_64_double(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size);

extern void total_64_32(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size);

extern void total_string_32(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size);

extern void total_128_double(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size);

extern void min_32_double(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size);
}

#endif

