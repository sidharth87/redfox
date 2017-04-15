/*! \file Project.h
	\date Wednesday August 8, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the Project family of cuda functions.
*/

#ifndef SPLIT_H_INCLUDED
#define SPLIT_H_INCLUDED

#include <redfox/ra/interface/Tuple.h>
#include <stdio.h>
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
template<typename KeyTuple, typename ValueTuple, typename Tuple, unsigned int domains>
__device__ void splitElement(typename KeyTuple::BasicType& key,
	typename ValueTuple::BasicType& value,
	const typename Tuple::BasicType& in)
{
	typedef typename KeyTuple::BasicType KeyType;
	typedef typename ValueTuple::BasicType ValueType;

	switch(domains)
	{
	case 1:
		key = (KeyType)tuple::stripValues<Tuple, 1>(in);
		value = tuple::extract<ValueType, 1, Tuple>(in);
		break;
	case 2:
	{
//		key = (KeyType)tuple::stripValues<Tuple, 2>(in);
		unsigned long long int key0 = tuple::extract<unsigned long long int, 0, Tuple>(in);
		unsigned long long int key1 = tuple::extract<unsigned long long int, 1, Tuple>(in);
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llx %llx %llx\n", key0, key1, key2);	
		key = 0;	
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llx %llx\n", key.a[0], key.a[1]);	
		key = tuple::insert<unsigned long long int, 0, KeyTuple>(key, key0); 
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llx %llx\n", key.a[0], key.a[1]);	
		key = tuple::insert<unsigned long long int, 1, KeyTuple>(key, key1); 
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llx %llx\n", key.a[0], key.a[1]);	

		value = tuple::extract<ValueType, 2, Tuple>(in);
		break;
	}
	case 3:
	{	
//		key = (KeyType)tuple::stripValues<Tuple, 3>(in);
		unsigned long long int key0 = tuple::extract<unsigned long long int, 0, Tuple>(in);
		unsigned long long int key1 = tuple::extract<unsigned long long int, 1, Tuple>(in);
		unsigned long long int key2 = tuple::extract<unsigned long long int, 2, Tuple>(in);
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llx %llx %llx\n", key0, key1, key2);	
		key = 0;	
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llx %llx\n", key.a[0], key.a[1]);	
		key = tuple::insert<unsigned long long int, 0, KeyTuple>(key, key0); 
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llx %llx\n", key.a[0], key.a[1]);	
		key = tuple::insert<unsigned long long int, 1, KeyTuple>(key, key1); 
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llx %llx\n", key.a[0], key.a[1]);	
		key = tuple::insert<unsigned long long int, 2, KeyTuple>(key, key2); 
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llx %llx\n", key.a[0], key.a[1]);	

		value = tuple::extract<ValueType, 3, Tuple>(in);
		break;
	}
	case 4:
		key = (KeyType)tuple::stripValues<Tuple, 4>(in);
		value = tuple::extract<ValueType, 4, Tuple>(in);
		break;
	case 5:
		key = (KeyType)tuple::stripValues<Tuple, 5>(in);
		value = tuple::extract<ValueType, 5, Tuple>(in);
		break;
	case 6:
		key = (KeyType)tuple::stripValues<Tuple, 6>(in);
		value = tuple::extract<ValueType, 6, Tuple>(in);
		break;
	case 7:
		key = (KeyType)tuple::stripValues<Tuple, 7>(in);
		value = tuple::extract<ValueType, 7, Tuple>(in);
		break;
	case 8:
		key = (KeyType)tuple::stripValues<Tuple, 8>(in);
		value = tuple::extract<ValueType, 8, Tuple>(in);
		break;
	case 9:
		key = (KeyType)tuple::stripValues<Tuple, 9>(in);
		value = tuple::extract<ValueType, 9, Tuple>(in);
		break;
	case 10:
		key = (KeyType)tuple::stripValues<Tuple, 10>(in);
		value = tuple::extract<ValueType, 10, Tuple>(in);
		break;
	case 11:
		key = (KeyType)tuple::stripValues<Tuple, 11>(in);
		value = tuple::extract<ValueType, 11, Tuple>(in);
		break;
	case 12:
		key = (KeyType)tuple::stripValues<Tuple, 12>(in);
		value = tuple::extract<ValueType, 12, Tuple>(in);
		break;
	case 13:
		key = (KeyType)tuple::stripValues<Tuple, 13>(in);
		value = tuple::extract<ValueType, 13, Tuple>(in);
		break;
	case 14:
		key = (KeyType)tuple::stripValues<Tuple, 14>(in);
		value = tuple::extract<ValueType, 14, Tuple>(in);
		break;
	case 15:
		key = (KeyType)tuple::stripValues<Tuple, 15>(in);
		value = tuple::extract<ValueType, 15, Tuple>(in);
		break;
	}
}

template<typename KeyTuple, typename Tuple, unsigned int domains>
__device__ void splitKeyElement(typename KeyTuple::BasicType& key,
	const typename Tuple::BasicType& in)
{
	typedef typename KeyTuple::BasicType KeyType;

	key = (KeyType)tuple::stripValues<Tuple, domains>(in);
//	switch(domains)
//	{
//	case 1:
//		key = (KeyType)tuple::stripValues<Tuple, 1>(in);
//		break;
//	case 2:
//	{
//		key = (KeyType)tuple::stripValues<Tuple, 2>(in);
////		unsigned long long int key0 = tuple::extract<unsigned long long int, 0, Tuple>(in);
////		unsigned long long int key1 = tuple::extract<unsigned long long int, 1, Tuple>(in);
////		key = 0;	
////		key = tuple::insert<unsigned long long int, 0, KeyTuple>(key, key0); 
////		key = tuple::insert<unsigned long long int, 1, KeyTuple>(key, key1); 
//
//		break;
//	}
//	case 3:
//	{	
//		key = (KeyType)tuple::stripValues<Tuple, 3>(in);
////		unsigned long long int key0 = tuple::extract<unsigned long long int, 0, Tuple>(in);
////		unsigned long long int key1 = tuple::extract<unsigned long long int, 1, Tuple>(in);
////		unsigned long long int key2 = tuple::extract<unsigned long long int, 2, Tuple>(in);
////		key = 0;	
////		key = tuple::insert<unsigned long long int, 0, KeyTuple>(key, key0); 
////		key = tuple::insert<unsigned long long int, 1, KeyTuple>(key, key1); 
////		key = tuple::insert<unsigned long long int, 2, KeyTuple>(key, key2); 
//
//		break;
//	}
//	case 4:
//		key = (KeyType)tuple::stripValues<Tuple, 4>(in);
//		break;
//	case 5:
//		key = (KeyType)tuple::stripValues<Tuple, 5>(in);
//		break;
//	case 6:
//		key = (KeyType)tuple::stripValues<Tuple, 6>(in);
//		break;
//	case 7:
//		key = (KeyType)tuple::stripValues<Tuple, 7>(in);
//		break;
//	case 8:
//		key = (KeyType)tuple::stripValues<Tuple, 8>(in);
//		break;
//	case 9:
//		key = (KeyType)tuple::stripValues<Tuple, 9>(in);
//		break;
//	case 10:
//		key = (KeyType)tuple::stripValues<Tuple, 10>(in);
//		break;
//	case 11:
//		key = (KeyType)tuple::stripValues<Tuple, 11>(in);
//		break;
//	case 12:
//		key = (KeyType)tuple::stripValues<Tuple, 12>(in);
//		break;
//	case 13:
//		key = (KeyType)tuple::stripValues<Tuple, 13>(in);
//		break;
//	case 14:
//		key = (KeyType)tuple::stripValues<Tuple, 14>(in);
//		break;
//	case 15:
//		key = (KeyType)tuple::stripValues<Tuple, 15>(in);
//		break;
//	}
}

template<typename ValueTuple, typename Tuple, unsigned int domains>
__device__ void splitValueElement(typename ValueTuple::BasicType& value,
	const typename Tuple::BasicType& in)
{
	typedef typename ValueTuple::BasicType ValueType;

	value = (ValueType)tuple::stripKeys<Tuple, domains>(in);
}

//template<typename ValueTuple, typename Tuple, unsigned int domains>
//__device__ void splitValueElement2(typename ValueTuple::BasicType& value,
//	const typename Tuple::BasicType& in)
//{
//	typedef typename ValueTuple::BasicType ValueType;
//
//	value = (ValueType)tuple::stripKeys2<Tuple, domains>(in);
//}
//template<typename KeyTuple, typename Tuple, unsigned int domains>
//__device__ void splitKeyElement2(typename KeyTuple::BasicType& key,
//	const typename Tuple::BasicType& in, unsigned int i)
//{
//	typedef typename KeyTuple::BasicType KeyType;
//
//	unsigned long long int temp = 0;
//	switch(domains)
//	{
//	case 1:
//		key = (KeyType)tuple::stripValues<Tuple, 1>(in);
//		break;
//	case 2:
//	{
////		key = (KeyType)tuple::stripValues<Tuple, 2>(in);
//		unsigned long long int key0 = tuple::extract<unsigned long long int, 0, Tuple>(in);
//		unsigned long long int key1 = tuple::extract<unsigned long long int, 1, Tuple>(in);
//if(i == 6001214) printf("element2 key0 %llx key1 %llx\n", key0, key1);	
////if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llx %llx\n", key.a[0], key.a[1]);	
//		temp = tuple::insert<unsigned long long int, 0, KeyTuple>(temp, key0); 
//if(i == 6001214) printf("element2 temp %llx\n", temp);	
//		temp = tuple::insert<unsigned long long int, 1, KeyTuple>(temp, key1); 
//if(i == 6001214) printf("element2 temp %llx\n", temp);	
//		key = (KeyType)temp;
//if(i == 6001214) printf("ement2 key %x\n", key);	
//		break;
//	}
//	case 3:
//	{	
////		key = (KeyType)tuple::stripValues<Tuple, 3>(in);
//		unsigned long long int key0 = tuple::extract<unsigned long long int, 0, Tuple>(in);
//		unsigned long long int key1 = tuple::extract<unsigned long long int, 1, Tuple>(in);
//		unsigned long long int key2 = tuple::extract<unsigned long long int, 2, Tuple>(in);
////if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llx %llx %llx\n", key0, key1, key2);	
//		key = 0;	
////if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llx %llx\n", key.a[0], key.a[1]);	
//		key = tuple::insert<unsigned long long int, 0, KeyTuple>(key, key0); 
////if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llx %llx\n", key.a[0], key.a[1]);	
//		key = tuple::insert<unsigned long long int, 1, KeyTuple>(key, key1); 
////if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llx %llx\n", key.a[0], key.a[1]);	
//		key = tuple::insert<unsigned long long int, 2, KeyTuple>(key, key2); 
////if(threadIdx.x == 0 && blockIdx.x == 0) printf("%llx %llx\n", key.a[0], key.a[1]);	
//
//		break;
//	}
//	case 4:
//		key = (KeyType)tuple::stripValues<Tuple, 4>(in);
//		break;
//	case 5:
//		key = (KeyType)tuple::stripValues<Tuple, 5>(in);
//		break;
//	case 6:
//		key = (KeyType)tuple::stripValues<Tuple, 6>(in);
//		break;
//	case 7:
//		key = (KeyType)tuple::stripValues<Tuple, 7>(in);
//		break;
//	case 8:
//		key = (KeyType)tuple::stripValues<Tuple, 8>(in);
//		break;
//	case 9:
//		key = (KeyType)tuple::stripValues<Tuple, 9>(in);
//		break;
//	case 10:
//		key = (KeyType)tuple::stripValues<Tuple, 10>(in);
//		break;
//	case 11:
//		key = (KeyType)tuple::stripValues<Tuple, 11>(in);
//		break;
//	case 12:
//		key = (KeyType)tuple::stripValues<Tuple, 12>(in);
//		break;
//	case 13:
//		key = (KeyType)tuple::stripValues<Tuple, 13>(in);
//		break;
//	case 14:
//		key = (KeyType)tuple::stripValues<Tuple, 14>(in);
//		break;
//	case 15:
//		key = (KeyType)tuple::stripValues<Tuple, 15>(in);
//		break;
//	}
//}

template<typename KeyTuple, typename ValueTuple, typename Tuple, unsigned int domains>
__device__ void split(typename KeyTuple::BasicType* key,
	typename ValueTuple::BasicType* value,
	const typename Tuple::BasicType* begin,
	const typename Tuple::BasicType* end)
{
	unsigned int step     = gridDim.x * blockDim.x;
	unsigned int start    = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int elements = end - begin;

//	if(threadIdx.x == 0 && blockIdx.x == 0) 
//	{
//		for(unsigned int i = 0; i < 300; ++i)
//			printf("%u %llu %llx\n", i, begin[i].a[0], begin[i].a[1]);
//	}

	for(unsigned int i = start; i < elements; i += step)
	{
		if(i < elements)
		{
			splitKeyElement<KeyTuple, Tuple, domains>(key[i], begin[i]);
			splitValueElement<ValueTuple, Tuple, domains>(value[i], begin[i]);
		}
	}

//	if(threadIdx.x == 0 && blockIdx.x == 0) 
//	{
//		for(unsigned int i = 0; i < 300; ++i)
//			printf("%u %u\n", i, key[i]);
//	}
}

//template<typename KeyTuple, typename ValueTuple, typename Tuple, unsigned int domains>
//__device__ void split2(typename KeyTuple::BasicType* key,
//	typename ValueTuple::BasicType* value,
//	const typename Tuple::BasicType* begin,
//	const typename Tuple::BasicType* end)
//{
//	unsigned int step     = gridDim.x * blockDim.x;
//	unsigned int start    = threadIdx.x + blockIdx.x * blockDim.x;
//	unsigned int elements = end - begin;
//
//	if(threadIdx.x == 0 && blockIdx.x == 0) 
//	{
////		for(unsigned int i = 0; i < 300; ++i)
//			printf("before split %llx\n", begin[0].a[0]);
//	}
//
//	for(unsigned int i = start; i < elements; i += step)
//	{
//		if(i < elements)
//		{
//			splitKeyElement<KeyTuple, Tuple, domains>(key[i], begin[i]);
//			splitValueElement2<ValueTuple, Tuple, domains>(value[i], begin[i]);
//		}
//	}
//
//	if(threadIdx.x == 0 && blockIdx.x == 0) 
//	{
////		for(unsigned int i = 0; i < 300; ++i)
//			printf("after split  %llx\n", value[0].a[0]);
//	}
//}

template<typename KeyTuple, typename Tuple, unsigned int domains>
__device__ void split_key(typename KeyTuple::BasicType* key,
	const typename Tuple::BasicType* begin,
	const typename Tuple::BasicType* end)
{
	unsigned int step     = gridDim.x * blockDim.x;
	unsigned int start    = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int elements = end - begin;
	
	for(unsigned int i = start; i < elements; i += step)
	{
		if(i < elements)
			splitKeyElement<KeyTuple, Tuple, domains>(key[i], begin[i]);
	}

//	if(threadIdx.x == 0 && blockIdx.x == 0) 
//	{
//		printf("begin %llx %llx\n", begin[6001214].a[0], begin[6001214].a[1]);
//		printf("key %llx\n", key[6001214]);
//	}
}

//template<typename KeyTuple, typename Tuple, unsigned int domains>
//__device__ void split_key2(typename KeyTuple::BasicType* key,
//	const typename Tuple::BasicType* begin,
//	const typename Tuple::BasicType* end)
//{
//	unsigned int step     = gridDim.x * blockDim.x;
//	unsigned int start    = threadIdx.x + blockIdx.x * blockDim.x;
//	unsigned int elements = end - begin;
//
//	if(threadIdx.x == 0 && blockIdx.x == 0) 
//	{
//		printf("*** before split key begin %u\n", elements);
//	
//		for(unsigned int i  = (elements - 1); i > (elements - 100); --i)
//			printf("before key %u %llx\n", i, begin[i].a[8]);
//	}
//
//	for(unsigned int i = start; i < elements; i += step)
//	{
//		if(i < elements)
//			splitKeyElement<KeyTuple, Tuple, domains>(key[i], begin[i]);
//	}
//
//	if(threadIdx.x == 0 && blockIdx.x == 0) 
//	{
//		for(unsigned int i  = (elements - 1); i > (elements - 100); --i)
//		printf("key %u %x\n", i, key[i]);
////		splitKeyElement2<KeyTuple, Tuple, domains>(key[6001214], begin[6001214]);
////		printf("key %x\n", key[6001214]);
////		unsigned long long int key0 = tuple::extract<unsigned long long int, 0, Tuple>(begin[6001214]);
////		unsigned long long int key1 = tuple::extract<unsigned long long int, 1, Tuple>(begin[6001214]);
////		printf("key0 %llx key1 %llx\n", key0, key1);
////		unsigned long long int temp = 0;
////		temp = tuple::insert<unsigned long long int, 0, KeyTuple>(temp, key0); 
////		printf("temp %llx\n", temp);
////		temp = tuple::insert<unsigned long long int, 1, KeyTuple>(temp, key1); 
////		printf("temp %llx\n", temp);
////		unsigned int value = (unsigned int)temp;
////		printf("value %x\n", value);
//	}
//}

template<typename KeyTuple, typename ValueTuple, typename SourceTuple>
__device__ void getSplitResultSize(long long unsigned int* size, long long unsigned int* key_size,
	long long unsigned int* value_size)
{
	*key_size = (*size / sizeof(typename SourceTuple::BasicType)) * sizeof(typename KeyTuple::BasicType);
	*value_size = (*size / sizeof(typename SourceTuple::BasicType)) * sizeof(typename ValueTuple::BasicType);

//	printf("split result size %llu %llu %llu\n", *size, *key_size, *value_size);
}

template<typename KeyTuple, typename ValueTuple, typename Tuple, unsigned int domains>
__device__ void mergeElement(typename Tuple::BasicType& result,
	const typename KeyTuple::BasicType& key,
	const typename ValueTuple::BasicType& value)
{
	typedef typename ValueTuple::BasicType ValueType;
	typename Tuple::BasicType temp = 0;

	temp = tuple::restoreValues<Tuple, domains>((typename Tuple::BasicType)key);

//	temp = tuple::insert<ValueType, domains, Tuple>(temp, value);
	
	result = temp | ((typename Tuple::BasicType)value);
//	result = temp;
}

//template<typename KeyTuple, typename ValueTuple, typename Tuple, unsigned int domains>
//__device__ void mergeElement2(typename Tuple::BasicType& result,
//	const typename KeyTuple::BasicType& key,
//	const typename ValueTuple::BasicType& value)
//{
//	typedef typename ValueTuple::BasicType ValueType;
//	typename Tuple::BasicType temp = 0;
//
//	temp = tuple::restoreValues<Tuple, domains>((typename Tuple::BasicType)key);
////if(thread)printf("after key %u, %llx\n", threadIdx.x, temp);
////	temp = tuple::insert<ValueType, domains, Tuple>(temp, value);
////printf("after value %u, %llx\n", threadIdx.x, temp);
//	result = temp | ((typename Tuple::BasicType)value);
////	result = temp;
//}

template<typename KeyTuple, typename ValueTuple, typename Tuple, unsigned int domains>
__device__ void merge(typename Tuple::BasicType* result,
	const typename KeyTuple::BasicType* key_begin,
	const typename KeyTuple::BasicType* key_end,
	const typename ValueTuple::BasicType* value_begin,
	const typename ValueTuple::BasicType* value_end)
{
	unsigned int step     = gridDim.x * blockDim.x;
	unsigned int start    = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int elements = value_end - value_begin;
	
//	if(threadIdx.x == 0 && blockIdx.x == 0)
//		for(unsigned int i = 0; i < 30000; ++i) 
//			printf("%u key %s\n", i, (char *)key_begin[i]);

	for(unsigned int i = start; i < elements; i += step)
	{
		if(i < elements)
			mergeElement<KeyTuple, ValueTuple, Tuple, domains>(result[i], key_begin[i], value_begin[i]);
//			printf("%u\n", i);
	}

//	if(threadIdx.x == 0 && blockIdx.x == 0) printf("after merge %llx %llx\n", result[3].a[0], result[3].a[1]);
}

//template<typename KeyTuple, typename ValueTuple, typename Tuple, unsigned int domains>
//__device__ void merge2(typename Tuple::BasicType* result,
//	const typename KeyTuple::BasicType* key_begin,
//	const typename KeyTuple::BasicType* key_end,
//	const typename ValueTuple::BasicType* value_begin,
//	const typename ValueTuple::BasicType* value_end)
//{
//	unsigned int step     = gridDim.x * blockDim.x;
//	unsigned int start    = threadIdx.x + blockIdx.x * blockDim.x;
//	unsigned int elements = value_end - value_begin;
//	
//	if(threadIdx.x == 0 && blockIdx.x == 0) printf("before merge %llx\n", value_begin[0].a[0]);
//
//	for(unsigned int i = start; i < elements; i += step)
//	{
//		if(i < elements)
//			mergeElement<KeyTuple, ValueTuple, Tuple, domains>(result[i], key_begin[i], value_begin[i]);
//	}
//
//	if(threadIdx.x == 0 && blockIdx.x == 0) printf("after merge %llx\n", result[0].a[0]);
//}

template<typename ResultType, typename ValueType>
__device__ void getMergeResultSize(long long unsigned int* size, 
	long long unsigned int* value_size)
{
	*size = (*value_size / sizeof(ValueType)) * sizeof(ResultType);

//	printf("merge result type size %u\n", sizeof(ResultType));
//	printf("merge value type size  %u\n", sizeof(ValueType));
//	printf("merge result size %llu\n", *size);
//	printf("merge value  size %llu\n", *value_size);
}

template<typename SourceTuple>
__device__ void getGenerateResultSize(long long unsigned int* generate_size, 
	long long unsigned int* size)
{
	*generate_size = (*size / sizeof(typename SourceTuple::BasicType)) * sizeof(unsigned int);

//	printf("generate result size %llu\n",  *generate_size);
}

__device__ void generate(
	unsigned int* begin,
	unsigned int* end)
{
	unsigned int step     = gridDim.x * blockDim.x;
	unsigned int start    = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int elements = end - begin;
	
	for(unsigned int i = start; i < elements; i += step)
	{
		if(i < elements)
			begin[i] = 1;
	}

//	if(threadIdx.x == 0 && blockIdx.x == 0) 
//	{
//		printf("begin %x\n", begin[0]);
//	}
}

}

}

#endif

