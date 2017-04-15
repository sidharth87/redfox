/*! \file Sort.cu
	\author Gregory Diamos <gregory.diamos>
	\date Wednesday December 1, 2010
	\brief The source file for the C interface to CUDA sorting routines.
*/

#ifndef SORT_CU_INCLUDED
#define SORT_CU_INCLUDED

// Redfox Includes
#include <redfox/nvcc/interface/RelationalAlgebraKernel.h>
#include <redfox/ra/interface/Reduce.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

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

struct compare_string
{
  __host__ __device__
  bool operator()(unsigned long long int i, unsigned long long int j)
  {
     const char *string1 = (char *) i;
     const char *string2 = (char *) j;
     
     int ii = 0;
     
     while(string1[ii] != '\0' && string2[ii] != '\0')
     {
     	if(string1[ii] != string2[ii])
     		return false;
     
     	ii++;
     }
     
     if(string1[ii] == '\0' && string2[ii] == '\0')
     	return true;
     else
     	return false;
  }
};

//struct compare_string_32
//{
//  __host__ __device__
//  bool operator()(ra::tuple::PackedNBytes<2> i, ra::tuple::PackedNBytes<2> j)
//  {
//     printf("i %llx %llx\n", i.a[0], i.a[1]);
//     printf("j %llx %llx\n", j.a[0], j.a[1]);
//     if(i.a[0] != j.a[0]) return false;
//
//     const char *string1 = (char *)(i.a[1]);
//     const char *string2 = (char *)(j.a[1]);
//     
//     int ii = 0;
//     
//     while(string1[ii] != '\0' && string2[ii] != '\0')
//     {
//     	if(string1[ii] != string2[ii])
//     		return false;
//     
//     	ii++;
//     }
//     
//     if(string1[ii] == '\0' && string2[ii] == '\0')
//     	return true;
//     else
//     	return false;
//  }
//};


struct compare_128
{
  __host__ __device__
  bool operator()(ra::tuple::PackedNBytes<2> i, ra::tuple::PackedNBytes<2> j)
  {
     return ((i.a[0] == j.a[0]) && (i.a[1] == j.a[1]));
  }
};

struct compare_256
{
  __host__ __device__
  bool operator()(ra::tuple::PackedNBytes<4> i, ra::tuple::PackedNBytes<4> j)
  {
     return ((i.a[0] == j.a[0]) && (i.a[1] == j.a[1]) && (i.a[2] == j.a[2]) && (i.a[3] == j.a[3]));
  }
};
//template<typename KeyTuple>
//void count(const typename KeyTuple::BasicType* key_begin, 
//	const typename KeyTuple::BasicType* key_end, 
//	typename KeyTuple::BasicType* result_key_begin, 
//	unsigned long long int* result_value_begin, unsigned long long int *key_size, 
//	unsigned long long int *value_size)
//{
//	typedef typename KeyTuple::BasicType type;
//
//	thrust::pair<int *, int *> size;
//	thrust::equal_to<type> pred;
//	thrust::plus<unsigned long long int>ope;
//
//	int N = (key_end - key_begin) / (sizeof(type));
//	thrust::host_vector<unsigned long long int>value_host(N);
//	thrust::generate(value_host.begin(), value_host.end(), 1);
//	thrust::device_vector<unsigned long long int>value_device
//		 = value_host;
//	size = thrust::reduce_by_key(
//		thrust::device_ptr<type*>(key_begin),
//		thrust::device_ptr<type*>(key_end),
//		value_device.begin(),
//		thrust::device_ptr<type*>(result_key_begin),
//		thrust::device_ptr<unsigned long long int*>(result_value_begin),
//		pred, ope);
//
//	*key_size = *(size.first) * sizeof(type);
//	*value_size = *(size.second) * sizeof(unsigned long long int);
//}

//template<typename KeyTuple, typename ValueType, typename ReduceType, 
//	 typename binary_pred, typename binary_op>
//void reduce(const typename KeyTuple::BasicType* key_begin,
//	const typename ValueType::type* value_begin, 
//	typename KeyTuple::BasicType* result_key_begin, 
//	typename ReduceType::type* result_value_begin, 
//	unsigned long long int *key_size,
//	unsigned long long int *value_size)
//{
//	unsigned long long int key_size_host;
//
//	check(cudaMemcpy(&key_size_host, key_size, sizeof(unsigned long long int),
//		cudaMemcpyDeviceToHost));
//
//	unsigned long long int value_size_host;
//
//	check(cudaMemcpy(&value_size_host, value_size, sizeof(unsigned long long int),
//		cudaMemcpyDeviceToHost));
//
//	thrust::pair<int *, int *> size;
//
//	typedef typename KeyTuple::BasicType keytype;
//	typedef typename ValueType::type valuetype;
//	typedef typename ReduceType::type reducetype;
//	
//	binary_pred pred;
//	binary_op op;
//
//	size = thrust::reduce_by_key(
//		thrust::device_ptr<keytype*>(key_begin),
//		thrust::device_ptr<keytype*>(key_begin + (key_size_host / sizeof(keytype))),
//		thrust::device_ptr<valuetype*>(value_begin),
//		thrust::device_ptr<keytype*>(result_key_begin),
//		thrust::device_ptr<reducetype*>(result_value_begin),
//		pred, op);
//	key_size_host = *(size.first) * sizeof(keytype);
//	value_size_host = *(size.second) * sizeof(reducetype);
//
//	check(cudaMemcpy(key_size, &key_size_host, sizeof(unsigned long long int),
//		cudaMemcpyHostToDevice));
//	check(cudaMemcpy(value_size, &value_size_host, sizeof(unsigned long long int),
//		cudaMemcpyHostToDevice));
//}
void count_8(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	unsigned long long int key_size_host;
	check(cudaMemcpy(&key_size_host, key_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

	unsigned long long int value_size_host;
	check(cudaMemcpy(&value_size_host, value_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

//	printf("before reduce size %llu %llu\n", key_size_host, value_size_host);
//
//	unsigned char data_key[8060];
//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 8060,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce key %x, %x\n", data_key[0], data_key[8059]);
//
//	unsigned long long int data_value[8060];
//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 8060 * 8,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce value %llx, %llx\n", data_value[0], data_value[8059]);
//	double *pointer = (double *)data_value;
//	printf("before reduce value %lf, %lf\n", pointer[0], pointer[8059]);

	thrust::pair<thrust::device_ptr<unsigned char>, thrust::device_ptr<unsigned int> > size;

//	thrust::equal_to<unsigned char> binary_pred;
//	thrust::plus<double> binary_op;

	size = thrust::reduce_by_key(
		thrust::device_ptr<const unsigned char>((const unsigned char*)key_begin),
		thrust::device_ptr<const unsigned char>((const unsigned char*)key_begin + (key_size_host / sizeof(unsigned char))),
		thrust::device_ptr<const unsigned int>((const unsigned int *)value_begin),
		thrust::device_ptr<unsigned char>((unsigned char*)result_key_begin),
		thrust::device_ptr<unsigned int>((unsigned int *)result_value_begin)/*,
		binary_pred, binary_op*/);

	key_size_host = (size.first - thrust::device_ptr<unsigned char>((unsigned char *)result_key_begin)) * sizeof(unsigned char);
	value_size_host = (size.second - thrust::device_ptr<unsigned int>((unsigned int *)result_value_begin)) * sizeof(unsigned int);

	check(cudaMemcpy(key_size, &key_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));
	check(cudaMemcpy(value_size, &value_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
 	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("count %f\n", exe_time);

	printf("after count size %llu %llu\n", key_size_host, value_size_host);

	unsigned char reduce_key[4];
	check(cudaMemcpy(reduce_key, (unsigned char *)result_key_begin, 4,
		cudaMemcpyDeviceToHost));

	unsigned int reduce_value[4];
	check(cudaMemcpy(reduce_value, (unsigned int *)result_value_begin, 16,
		cudaMemcpyDeviceToHost));

	for(int i = 0; i < 4; ++i)
	{
		printf("%x %x %u\n", i, reduce_key[i], reduce_value[i]);
	}
}

void count_16(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	unsigned long long int key_size_host;
	check(cudaMemcpy(&key_size_host, key_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

	unsigned long long int value_size_host;
	check(cudaMemcpy(&value_size_host, value_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

//	printf("before reduce size %llu %llu\n", key_size_host, value_size_host);
//
//	unsigned char data_key[8060];
//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 8060,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce key %x, %x\n", data_key[0], data_key[8059]);
//
//	unsigned long long int data_value[8060];
//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 8060 * 8,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce value %llx, %llx\n", data_value[0], data_value[8059]);
//	double *pointer = (double *)data_value;
//	printf("before reduce value %lf, %lf\n", pointer[0], pointer[8059]);

	thrust::pair<thrust::device_ptr<unsigned short>, thrust::device_ptr<unsigned int> > size;

//	thrust::equal_to<unsigned char> binary_pred;
//	thrust::plus<double> binary_op;

	size = thrust::reduce_by_key(
		thrust::device_ptr<const unsigned short>((const unsigned short*)key_begin),
		thrust::device_ptr<const unsigned short>((const unsigned short*)key_begin + (key_size_host / sizeof(unsigned short))),
		thrust::device_ptr<const unsigned int>((const unsigned int *)value_begin),
		thrust::device_ptr<unsigned short>((unsigned short*)result_key_begin),
		thrust::device_ptr<unsigned int>((unsigned int *)result_value_begin)/*,
		binary_pred, binary_op*/);

	key_size_host = (size.first - thrust::device_ptr<unsigned short>((unsigned short *)result_key_begin)) * sizeof(unsigned short);
	value_size_host = (size.second - thrust::device_ptr<unsigned int>((unsigned int *)result_value_begin)) * sizeof(unsigned int);

	check(cudaMemcpy(key_size, &key_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));
	check(cudaMemcpy(value_size, &value_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
 	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("count %f\n", exe_time);

	printf("after count size %llu %llu\n", key_size_host, value_size_host);
//
//	unsigned char reduce_key[4];
//	check(cudaMemcpy(reduce_key, (unsigned char *)result_key_begin, 4,
//		cudaMemcpyDeviceToHost));
//
//	for(int i = 0; i < 4; ++i)
//	{
//		printf("%x %x\n", i, reduce_key[i]);
//	}
//
//	unsigned int reduce_value[4];
//	check(cudaMemcpy(reduce_value, (unsigned int *)result_value_begin, 16,
//		cudaMemcpyDeviceToHost));
//
//	for(int i = 0; i < 4; ++i)
//	{
//		printf("%x %u\n", i, reduce_value[i]);
//	}
}
void count_32(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	unsigned long long int key_size_host;
	check(cudaMemcpy(&key_size_host, key_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

	unsigned long long int value_size_host;
	check(cudaMemcpy(&value_size_host, value_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

//	printf("before reduce size %llu %llu\n", key_size_host, value_size_host);
//
//	unsigned char data_key[8060];
//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 8060,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce key %x, %x\n", data_key[0], data_key[8059]);
//
//	unsigned long long int data_value[8060];
//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 8060 * 8,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce value %llx, %llx\n", data_value[0], data_value[8059]);
//	double *pointer = (double *)data_value;
//	printf("before reduce value %lf, %lf\n", pointer[0], pointer[8059]);

//unsigned int* data_key = (unsigned int *)malloc(key_size_host);
//check(cudaMemcpy(data_key, (unsigned int *)key_begin, key_size_host,
//		cudaMemcpyDeviceToHost));
//
//for(int i = 0; i < key_size_host/4; ++i)
//	printf("%u\n", data_key[i]);

	thrust::pair<thrust::device_ptr<unsigned int>, thrust::device_ptr<unsigned int> > size;

//	thrust::equal_to<unsigned char> binary_pred;
//	thrust::plus<double> binary_op;

	size = thrust::reduce_by_key(
		thrust::device_ptr<const unsigned int>((const unsigned int*)key_begin),
		thrust::device_ptr<const unsigned int>((const unsigned int*)key_begin + (key_size_host / sizeof(unsigned int))),
		thrust::device_ptr<const unsigned int>((const unsigned int *)value_begin),
		thrust::device_ptr<unsigned int>((unsigned int*)result_key_begin),
		thrust::device_ptr<unsigned int>((unsigned int *)result_value_begin)/*,
		binary_pred, binary_op*/);

	key_size_host = (size.first - thrust::device_ptr<unsigned int>((unsigned int *)result_key_begin)) * sizeof(unsigned int);
	value_size_host = (size.second - thrust::device_ptr<unsigned int>((unsigned int *)result_value_begin)) * sizeof(unsigned int);

	check(cudaMemcpy(key_size, &key_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));
	check(cudaMemcpy(value_size, &value_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
 	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("count %f\n", exe_time);

	printf("after count size %llu %llu\n", key_size_host, value_size_host);

//	unsigned int reduce_key[41];
//	check(cudaMemcpy(reduce_key, (unsigned int *)result_key_begin, 4 * 41,
//		cudaMemcpyDeviceToHost));
//
//	unsigned int reduce_value[41];
//	check(cudaMemcpy(reduce_value, (unsigned int *)result_value_begin, 4 * 41,
//		cudaMemcpyDeviceToHost));
//
//
//	for(int i = 0; i < 41; ++i)
//	{
//		printf("%u %u %u\n", i, reduce_key[i], reduce_value[i]);
//	}
}

void count_string(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	unsigned long long int key_size_host;
	check(cudaMemcpy(&key_size_host, key_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

	unsigned long long int value_size_host;
	check(cudaMemcpy(&value_size_host, value_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

//	printf("before reduce size %llu %llu\n", key_size_host, value_size_host);
//
//	unsigned char data_key[8060];
//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 8060,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce key %x, %x\n", data_key[0], data_key[8059]);
//
//	unsigned long long int data_value[8060];
//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 8060 * 8,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce value %llx, %llx\n", data_value[0], data_value[8059]);
//	double *pointer = (double *)data_value;
//	printf("before reduce value %lf, %lf\n", pointer[0], pointer[8059]);

	thrust::pair<thrust::device_ptr<unsigned long long int>, thrust::device_ptr<unsigned int> > size;

//	thrust::equal_to<unsigned char> binary_pred;
	thrust::plus<unsigned int> binary_op;

	size = thrust::reduce_by_key(
		thrust::device_ptr<const unsigned long long int>((const unsigned long long int*)key_begin),
		thrust::device_ptr<const unsigned long long int>((const unsigned long long int*)key_begin + (key_size_host / sizeof(unsigned long long int))),
		thrust::device_ptr<const unsigned int>((const unsigned int *)value_begin),
		thrust::device_ptr<unsigned long long int>((unsigned long long int*)result_key_begin),
		thrust::device_ptr<unsigned int>((unsigned int *)result_value_begin),
		compare_string(), binary_op);

	key_size_host = (size.first - thrust::device_ptr<unsigned long long int>((unsigned long long int *)result_key_begin)) * sizeof(unsigned long long int);
	value_size_host = (size.second - thrust::device_ptr<unsigned int>((unsigned int *)result_value_begin)) * sizeof(unsigned int);

	check(cudaMemcpy(key_size, &key_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));
	check(cudaMemcpy(value_size, &value_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
 	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("count %f\n", exe_time);

	printf("after count size %llu %llu\n", key_size_host, value_size_host);

	unsigned int reduce_value[7];
	check(cudaMemcpy(reduce_value, (unsigned int *)result_value_begin, 28,
		cudaMemcpyDeviceToHost));

	for(int i = 0; i < 7; ++i)
	{
		printf("%x %u\n", i, reduce_value[i]);
	}
}

void count_128(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	unsigned long long int key_size_host;
	check(cudaMemcpy(&key_size_host, key_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

	unsigned long long int value_size_host;
	check(cudaMemcpy(&value_size_host, value_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

//	printf("before reduce size %llu %llu\n", key_size_host, value_size_host);

	typedef ra::tuple::PackedNBytes<2> type;
	typedef const ra::tuple::PackedNBytes<2> const_type;
	thrust::pair<thrust::device_ptr<type>, thrust::device_ptr<unsigned int> > size;

//	type data_key[10];
//	check(cudaMemcpy(data_key, (type *)key_begin, 16 * 10,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 10; ++i)
//	printf("before reduce key %llx, %llx\n", data_key[i].a[0], data_key[i].a[1]);
//
//	unsigned long long int data_value[8060];
//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 8060 * 8,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce value %llx, %llx\n", data_value[0], data_value[8059]);
//	double *pointer = (double *)data_value;
//	printf("before reduce value %lf, %lf\n", pointer[0], pointer[8059]);

//	thrust::equal_to<unsigned char> binary_pred;
	thrust::plus<unsigned int> binary_op;

	size = thrust::reduce_by_key(
		thrust::device_ptr<const_type>((const_type*)key_begin),
		thrust::device_ptr<const_type>((const_type*)key_begin + (key_size_host / sizeof(type))),
		thrust::device_ptr<const unsigned int>((const unsigned int *)value_begin),
		thrust::device_ptr<type>((type*)result_key_begin),
		thrust::device_ptr<unsigned int>((unsigned int *)result_value_begin),
		compare_128(), binary_op);

	key_size_host = (size.first - thrust::device_ptr<type>((type *)result_key_begin)) * sizeof(type);
	value_size_host = (size.second - thrust::device_ptr<unsigned int>((unsigned int *)result_value_begin)) * sizeof(unsigned int);

	check(cudaMemcpy(key_size, &key_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));
	check(cudaMemcpy(value_size, &value_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
 	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("count %f\n", exe_time);

	printf("after count size %llu %llu\n", key_size_host, value_size_host);

	type reduce_key[10];
	check(cudaMemcpy(reduce_key, (type *)key_begin, 16 * 10,
		cudaMemcpyDeviceToHost));

//	unsigned int reduce_value[10];
//	check(cudaMemcpy(reduce_value, (unsigned int *)result_value_begin, 40,
//		cudaMemcpyDeviceToHost));
//
//	for(int i = 0; i < 10; ++i)
//	{
//		printf("%d %llx %llx %u\n", i, reduce_key[i].a[0], reduce_key[i].a[1], reduce_value[i]);
//	}
}

void count_256(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	unsigned long long int key_size_host;
	check(cudaMemcpy(&key_size_host, key_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

	unsigned long long int value_size_host;
	check(cudaMemcpy(&value_size_host, value_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

//	printf("before reduce size %llu %llu\n", key_size_host, value_size_host);

	typedef ra::tuple::PackedNBytes<4> type;
	typedef const ra::tuple::PackedNBytes<4> const_type;
	thrust::pair<thrust::device_ptr<type>, thrust::device_ptr<unsigned int> > size;

//	type data_key[10];
//	check(cudaMemcpy(data_key, (type *)key_begin, 16 * 10,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 10; ++i)
//	printf("before reduce key %llx, %llx\n", data_key[i].a[0], data_key[i].a[1]);
//
//	unsigned long long int data_value[8060];
//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 8060 * 8,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce value %llx, %llx\n", data_value[0], data_value[8059]);
//	double *pointer = (double *)data_value;
//	printf("before reduce value %lf, %lf\n", pointer[0], pointer[8059]);

//	thrust::equal_to<unsigned char> binary_pred;
	thrust::plus<unsigned int> binary_op;

	size = thrust::reduce_by_key(
		thrust::device_ptr<const_type>((const_type*)key_begin),
		thrust::device_ptr<const_type>((const_type*)key_begin + (key_size_host / sizeof(type))),
		thrust::device_ptr<const unsigned int>((const unsigned int *)value_begin),
		thrust::device_ptr<type>((type*)result_key_begin),
		thrust::device_ptr<unsigned int>((unsigned int *)result_value_begin),
		compare_256(), binary_op);

	key_size_host = (size.first - thrust::device_ptr<type>((type *)result_key_begin)) * sizeof(type);
	value_size_host = (size.second - thrust::device_ptr<unsigned int>((unsigned int *)result_value_begin)) * sizeof(unsigned int);

	check(cudaMemcpy(key_size, &key_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));
	check(cudaMemcpy(value_size, &value_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
 	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("count %f\n", exe_time);

	printf("after count size %llu %llu\n", key_size_host, value_size_host);

//	type reduce_key[10];
//	check(cudaMemcpy(reduce_key, (type *)key_begin, 16 * 10,
//		cudaMemcpyDeviceToHost));
//
//	unsigned int reduce_value[10];
//	check(cudaMemcpy(reduce_value, (unsigned int *)result_value_begin, 40,
//		cudaMemcpyDeviceToHost));
//
//	for(int i = 0; i < 10; ++i)
//	{
//		printf("%d %llx %llx %u\n", i, reduce_key[i].a[0], reduce_key[i].a[1], reduce_value[i]);
//	}
}

void total_string_32(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	unsigned long long int key_size_host;
	check(cudaMemcpy(&key_size_host, key_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

	unsigned long long int value_size_host;
	check(cudaMemcpy(&value_size_host, value_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

//	printf("before reduce size %llu %llu\n", key_size_host, value_size_host);
//
//	unsigned char data_key[100];
//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 100,
//		cudaMemcpyDeviceToHost));
//
//	unsigned long long int data_value[100];
//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 800,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 100; ++i)
//		printf("before reduce %u, %d, %llx\n", i, data_key[i], data_value[i]);
	thrust::pair<thrust::device_ptr<unsigned long long int>, thrust::device_ptr<unsigned int> > size;

//	thrust::equal_to<unsigned char> binary_pred;
	thrust::plus<unsigned int> binary_op;

	size = thrust::reduce_by_key(
		thrust::device_ptr<const unsigned long long int>((const unsigned long long int*)key_begin),
		thrust::device_ptr<const unsigned long long int>((const unsigned long long int*)key_begin + (key_size_host / sizeof(unsigned long long int))),
		thrust::device_ptr<const unsigned int>((const unsigned int *)value_begin),
		thrust::device_ptr<unsigned long long int>((unsigned long long int*)result_key_begin),
		thrust::device_ptr<unsigned int>((unsigned int *)result_value_begin),
		compare_string(), binary_op);

	key_size_host = (size.first - thrust::device_ptr<unsigned long long int>((unsigned long long int *)result_key_begin)) * sizeof(unsigned long long int);
	value_size_host = (size.second - thrust::device_ptr<unsigned int>((unsigned int *)result_value_begin)) * sizeof(unsigned int);

	check(cudaMemcpy(key_size, &key_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));
	check(cudaMemcpy(value_size, &value_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("reduce %f\n", exe_time);

	printf("after reduce size %llu %llu\n", key_size_host, value_size_host);

//	unsigned char reduce_key[5];
//	check(cudaMemcpy(reduce_key, (unsigned char *)result_key_begin, 5,
//		cudaMemcpyDeviceToHost));

	unsigned int reduce_value[2];
	check(cudaMemcpy(reduce_value, (unsigned int *)result_value_begin, 8,
		cudaMemcpyDeviceToHost));

	for(int i = 0; i < 2; ++i)
	{
		printf("%d %u\n", i, reduce_value[i]);
	}
}

void total_string_double(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	unsigned long long int key_size_host;
	check(cudaMemcpy(&key_size_host, key_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

	unsigned long long int value_size_host;
	check(cudaMemcpy(&value_size_host, value_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

//	printf("before reduce size %llu %llu\n", key_size_host, value_size_host);
//
//	unsigned char data_key[100];
//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 100,
//		cudaMemcpyDeviceToHost));
//
//	unsigned long long int data_value[100];
//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 800,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 100; ++i)
//		printf("before reduce %u, %d, %llx\n", i, data_key[i], data_value[i]);
	thrust::pair<thrust::device_ptr<unsigned long long int>, thrust::device_ptr<double> > size;

//	thrust::equal_to<unsigned char> binary_pred;
	thrust::plus<double> binary_op;

	size = thrust::reduce_by_key(
		thrust::device_ptr<const unsigned long long int>((const unsigned long long int*)key_begin),
		thrust::device_ptr<const unsigned long long int>((const unsigned long long int*)key_begin + (key_size_host / sizeof(unsigned long long int))),
		thrust::device_ptr<const double>((const double *)value_begin),
		thrust::device_ptr<unsigned long long int>((unsigned long long int*)result_key_begin),
		thrust::device_ptr<double>((double *)result_value_begin),
		compare_string(), binary_op);

	key_size_host = (size.first - thrust::device_ptr<unsigned long long int>((unsigned long long int *)result_key_begin)) * sizeof(unsigned long long int);
	value_size_host = (size.second - thrust::device_ptr<double>((double *)result_value_begin)) * sizeof(double);

	check(cudaMemcpy(key_size, &key_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));
	check(cudaMemcpy(value_size, &value_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("reduce %f\n", exe_time);

	printf("after reduce size %llu %llu\n", key_size_host, value_size_host);

//	unsigned char reduce_key[5];
//	check(cudaMemcpy(reduce_key, (unsigned char *)result_key_begin, 5,
//		cudaMemcpyDeviceToHost));

	double reduce_value[7];
	check(cudaMemcpy(reduce_value, (double *)result_value_begin, 56,
		cudaMemcpyDeviceToHost));

	for(int i = 0; i < 7; ++i)
	{
		printf("%d %lf\n", i, reduce_value[i]);
	}
}

void total_8_double(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	unsigned long long int key_size_host;
	check(cudaMemcpy(&key_size_host, key_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

	unsigned long long int value_size_host;
	check(cudaMemcpy(&value_size_host, value_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

//	printf("before reduce size %llu %llu\n", key_size_host, value_size_host);
//
//	unsigned char data_key[100];
//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 100,
//		cudaMemcpyDeviceToHost));
//
//	unsigned long long int data_value[100];
//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 800,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 100; ++i)
//		printf("before reduce %u, %d, %llx\n", i, data_key[i], data_value[i]);

	thrust::pair<thrust::device_ptr<unsigned char>, thrust::device_ptr<double> > size;

//	thrust::equal_to<unsigned char> binary_pred;
//	thrust::plus<double> binary_op;

	size = thrust::reduce_by_key(
		thrust::device_ptr<const unsigned char>((const unsigned char*)key_begin),
		thrust::device_ptr<const unsigned char>((const unsigned char*)key_begin + (key_size_host / sizeof(unsigned char))),
		thrust::device_ptr<const double>((const double *)value_begin),
		thrust::device_ptr<unsigned char>((unsigned char*)result_key_begin),
		thrust::device_ptr<double>((double *)result_value_begin)/*,
		binary_pred, binary_op*/);

	key_size_host = (size.first - thrust::device_ptr<unsigned char>((unsigned char *)result_key_begin)) * sizeof(unsigned char);
	value_size_host = (size.second - thrust::device_ptr<double>((double *)result_value_begin)) * sizeof(double);

	check(cudaMemcpy(key_size, &key_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));
	check(cudaMemcpy(value_size, &value_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("reduce %f\n", exe_time);

	printf("after reduce size %llu %llu\n", key_size_host, value_size_host);

	unsigned char reduce_key[5];
	check(cudaMemcpy(reduce_key, (unsigned char *)result_key_begin, 5,
		cudaMemcpyDeviceToHost));

	double reduce_value[5];
	check(cudaMemcpy(reduce_value, (double *)result_value_begin, 40,
		cudaMemcpyDeviceToHost));

	for(int i = 0; i < 5; ++i)
	{
		printf("%x %x %lf\n", i, reduce_key[i], reduce_value[i]);
	}
}

void total_16_double(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	unsigned long long int key_size_host;
	check(cudaMemcpy(&key_size_host, key_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

	unsigned long long int value_size_host;
	check(cudaMemcpy(&value_size_host, value_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

//	printf("before reduce size %llu %llu\n", key_size_host, value_size_host);
//
//	unsigned char data_key[8060];
//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 8060,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce key %x, %x\n", data_key[0], data_key[8059]);
//
//	unsigned long long int data_value[8060];
//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 8060 * 8,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce value %llx, %llx\n", data_value[0], data_value[8059]);
//	double *pointer = (double *)data_value;
//	printf("before reduce value %lf, %lf\n", pointer[0], pointer[8059]);

	thrust::pair<thrust::device_ptr<unsigned short>, thrust::device_ptr<double> > size;

//	thrust::equal_to<unsigned char> binary_pred;
//	thrust::plus<double> binary_op;

	size = thrust::reduce_by_key(
		thrust::device_ptr<const unsigned short>((const unsigned short*)key_begin),
		thrust::device_ptr<const unsigned short>((const unsigned short*)key_begin + (key_size_host / sizeof(unsigned short))),
		thrust::device_ptr<const double>((const double *)value_begin),
		thrust::device_ptr<unsigned short>((unsigned short*)result_key_begin),
		thrust::device_ptr<double>((double *)result_value_begin)/*,
		binary_pred, binary_op*/);

	key_size_host = (size.first - thrust::device_ptr<unsigned short>((unsigned short *)result_key_begin)) * sizeof(unsigned short);
	value_size_host = (size.second - thrust::device_ptr<double>((double *)result_value_begin)) * sizeof(double);

	check(cudaMemcpy(key_size, &key_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));
	check(cudaMemcpy(value_size, &value_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("reduce %f\n", exe_time);

	printf("after reduce size %llu %llu\n", key_size_host, value_size_host);
//
//	unsigned char reduce_key[5];
//	check(cudaMemcpy(reduce_key, (unsigned char *)result_key_begin, 5,
//		cudaMemcpyDeviceToHost));
//
//	double reduce_value[5];
//	check(cudaMemcpy(reduce_value, (double *)result_value_begin, 40,
//		cudaMemcpyDeviceToHost));
//
//	for(int i = 0; i < 5; ++i)
//	{
//		printf("%x %x %lf\n", i, reduce_key[i], reduce_value[i]);
//	}
}

void total_double(void* result,
	const void* begin, 
	const void* end)
{
//	double data[10];
//	check(cudaMemcpy(data, (double *)begin, 10 * 8,
//		cudaMemcpyDeviceToHost));
//
//	for(int i = 0; i < 10; ++i)
//		printf("before reduce %d, %llx\n", i, data[i]);
//
//	for(int i = 37897; i > 37887; --i)
//		printf("before reduce %d, %f\n", i, data[i]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	double ret;
//	thrust::plus<double> binary_op;

	ret = thrust::reduce(
		thrust::device_ptr<const double>((const double*)begin),
		thrust::device_ptr<const double>((const double*)end)/*,
		binary_op*/);

	check(cudaMemcpy(result, &ret, sizeof(double),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("reduce %f\n", exe_time);

	printf("after reduce result %f\n", ret);
}

void count(void* result,
	const void* begin, 
	const void* end)
{
//	unsigned int data[10];
//	check(cudaMemcpy(data, (unsigned int *)begin, 10 * 4,
//		cudaMemcpyDeviceToHost));
//
//	for(int i = 0; i < 10; ++i)
//		printf("before reduce %d, %u\n", i, data[i]);

//	for(int i = 37897; i > 37887; --i)
//		printf("before reduce %d, %f\n", i, data[i]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	unsigned int ret;
//	thrust::plus<double> binary_op;

	ret = thrust::reduce(
		thrust::device_ptr<const unsigned int>((const unsigned int*)begin),
		thrust::device_ptr<const unsigned int>((const unsigned int*)end)/*,
		binary_op*/);

	check(cudaMemcpy(result, &ret, sizeof(unsigned int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("reduce %f\n", exe_time);

	printf("after reduce result %u\n", ret);
}

void max_double(void* result,
	const void* begin, 
	const void* end)
{
//	double data[10];
//	check(cudaMemcpy(data, (double *)begin, 10 * 8,
//		cudaMemcpyDeviceToHost));
//
//	for(int i = 0; i < 10; ++i)
//		printf("before reduce %d, %llx\n", i, data[i]);
//
//	for(int i = 37897; i > 37887; --i)
//		printf("before reduce %d, %f\n", i, data[i]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

//	double *data = (double *)malloc(sizeof(double) * 225925);
//	check(cudaMemcpy(data, (double *)begin, sizeof(double) * 225925,
//		cudaMemcpyDeviceToHost));
//
//	double maxi = 0.0f;
//	for(unsigned int i = 0; i < 225925; ++i)
//		if(data[i] > maxi)
//		{
//			maxi = data[i];
//			printf("***%u %lf\n", i, data[i]);
//		}

//	unsigned long long int data[10000];
//
//	check(cudaMemcpy(data, (unsigned long long int *)begin, 80000,
//		cudaMemcpyDeviceToHost));
//
//	unsigned long long int maxi = 0.0f;
//	for(unsigned int i = 0; i < 10000; ++i)
//		if(data[i] > maxi)
//		{
//			maxi = data[i];
//			printf("***%u %llx\n", i, data[i]);
//		}
//
//
//	unsigned long long int data_int[10000];
//	check(cudaMemcpy(data_int, (unsigned long long int *)begin, 80000,
//		cudaMemcpyDeviceToHost));
//	printf("***%llx\n", data_int[8448]);

	unsigned long long int ret;
	thrust::maximum<unsigned long long int> binary_op;

	ret = thrust::reduce(
		thrust::device_ptr<const unsigned long long int>((const unsigned long long int*)begin),
		thrust::device_ptr<const unsigned long long int>((const unsigned long long int*)end),
		0ll, binary_op);

//	ret = 1772627.208700f;

	check(cudaMemcpy(result, &ret, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("reduce %f\n", exe_time);

	printf("after reduce result %llx\n", ret);

//	unsigned long long int result_int;
//	check(cudaMemcpy(&result_int, result, sizeof(unsigned long long int),
//		cudaMemcpyDeviceToHost));
//
//	printf("after reduce result %llx\n", result_int);
}

void total_64_32(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	unsigned long long int key_size_host;
	check(cudaMemcpy(&key_size_host, key_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

	unsigned long long int value_size_host;
	check(cudaMemcpy(&value_size_host, value_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

//	printf("before reduce size %llu %llu\n", key_size_host, value_size_host);
//
//	unsigned char data_key[8060];
//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 8060,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce key %x, %x\n", data_key[0], data_key[8059]);
//
//	unsigned long long int data_value[8060];
//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 8060 * 8,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce value %llx, %llx\n", data_value[0], data_value[8059]);
//	double *pointer = (double *)data_value;
//	printf("before reduce value %lf, %lf\n", pointer[0], pointer[8059]);

	thrust::pair<thrust::device_ptr<unsigned long long int>, thrust::device_ptr<unsigned int> > size;

	thrust::equal_to<unsigned long long int> binary_pred;
	thrust::plus<unsigned int> binary_op;

	size = thrust::reduce_by_key(
		thrust::device_ptr<const unsigned long long int>((const unsigned long long int*)key_begin),
		thrust::device_ptr<const unsigned long long int>((const unsigned long long int*)key_begin + (key_size_host / sizeof(unsigned long long int))),
		thrust::device_ptr<const unsigned int>((const unsigned int *)value_begin),
		thrust::device_ptr<unsigned long long int>((unsigned long long int*)result_key_begin),
		thrust::device_ptr<unsigned int>((unsigned int *)result_value_begin),
		binary_pred, binary_op);

	key_size_host = (size.first - thrust::device_ptr<unsigned long long int>((unsigned long long int *)result_key_begin)) * sizeof(unsigned long long int);
	value_size_host = (size.second - thrust::device_ptr<unsigned int>((unsigned int *)result_value_begin)) * sizeof(unsigned int);

	check(cudaMemcpy(key_size, &key_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));
	check(cudaMemcpy(value_size, &value_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("reduce %f\n", exe_time);

	printf("after reduce size %llu %llu\n", key_size_host, value_size_host);

	unsigned long long int reduce_key[2];
	check(cudaMemcpy(reduce_key, (unsigned long long int *)result_key_begin, 16,
		cudaMemcpyDeviceToHost));

	unsigned int reduce_value[2];
	check(cudaMemcpy(reduce_value, (unsigned int *)result_value_begin, 8,
		cudaMemcpyDeviceToHost));

	for(int i = 0; i < 2; ++i)
	{
		printf("%d %llu %u\n", i, reduce_key[i], reduce_value[i]);
	}
}

void total_64_double(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	unsigned long long int key_size_host;
	check(cudaMemcpy(&key_size_host, key_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

	unsigned long long int value_size_host;
	check(cudaMemcpy(&value_size_host, value_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

//	printf("before reduce size %llu %llu\n", key_size_host, value_size_host);
//
//	unsigned char data_key[8060];
//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 8060,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce key %x, %x\n", data_key[0], data_key[8059]);
//
//	unsigned long long int data_value[8060];
//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 8060 * 8,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce value %llx, %llx\n", data_value[0], data_value[8059]);
//	double *pointer = (double *)data_value;
//	printf("before reduce value %lf, %lf\n", pointer[0], pointer[8059]);

	thrust::pair<thrust::device_ptr<unsigned long long int>, thrust::device_ptr<double> > size;

//	thrust::equal_to<unsigned char> binary_pred;
//	thrust::plus<double> binary_op;

	size = thrust::reduce_by_key(
		thrust::device_ptr<const unsigned long long int>((const unsigned long long int*)key_begin),
		thrust::device_ptr<const unsigned long long int>((const unsigned long long int*)key_begin + (key_size_host / sizeof(unsigned long long int))),
		thrust::device_ptr<const double>((const double *)value_begin),
		thrust::device_ptr<unsigned long long int>((unsigned long long int*)result_key_begin),
		thrust::device_ptr<double>((double *)result_value_begin)/*,
		binary_pred, binary_op*/);

	key_size_host = (size.first - thrust::device_ptr<unsigned long long int>((unsigned long long int *)result_key_begin)) * sizeof(unsigned long long int);
	value_size_host = (size.second - thrust::device_ptr<double>((double *)result_value_begin)) * sizeof(double);

	check(cudaMemcpy(key_size, &key_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));
	check(cudaMemcpy(value_size, &value_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("reduce %f\n", exe_time);

	printf("after reduce size %llu %llu\n", key_size_host, value_size_host);

	unsigned long long int reduce_key[2];
	check(cudaMemcpy(reduce_key, (unsigned long long int *)result_key_begin, 16,
		cudaMemcpyDeviceToHost));

	double reduce_value[2];
	check(cudaMemcpy(reduce_value, (double *)result_value_begin, 16,
		cudaMemcpyDeviceToHost));

	for(int i = 0; i < 2; ++i)
	{
		printf("%d %llu %f\n", i, reduce_key[i], reduce_value[i]);
	}
}

void total_32_double(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	unsigned long long int key_size_host;
	check(cudaMemcpy(&key_size_host, key_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

	unsigned long long int value_size_host;
	check(cudaMemcpy(&value_size_host, value_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

//	printf("before reduce size %llu %llu\n", key_size_host, value_size_host);
//
//	unsigned int data_key[458];
//	check(cudaMemcpy(data_key, (unsigned int *)key_begin, 458 * 4,
//		cudaMemcpyDeviceToHost));
//
//	double data_value[458];
//	check(cudaMemcpy(data_value, (double *)value_begin, 458 * 8,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i  = 0 ; i < 458; ++i)
//		printf("before reduce %u, %u, %u, %lf\n", i, (data_key[i] >> 14), (data_key[i] & 0x3fff), data_value[i]);

	thrust::pair<thrust::device_ptr<unsigned int>, thrust::device_ptr<double> > size;

//	thrust::equal_to<unsigned char> binary_pred;
//	thrust::plus<double> binary_op;

	size = thrust::reduce_by_key(
		thrust::device_ptr<const unsigned int>((const unsigned int*)key_begin),
		thrust::device_ptr<const unsigned int>((const unsigned int*)key_begin + (key_size_host / sizeof(unsigned int))),
		thrust::device_ptr<const double>((const double *)value_begin),
		thrust::device_ptr<unsigned int>((unsigned int*)result_key_begin),
		thrust::device_ptr<double>((double *)result_value_begin)/*,
		binary_pred, binary_op*/);

	key_size_host = (size.first - thrust::device_ptr<unsigned int>((unsigned int *)result_key_begin)) * sizeof(unsigned int);
	value_size_host = (size.second - thrust::device_ptr<double>((double *)result_value_begin)) * sizeof(double);

	check(cudaMemcpy(key_size, &key_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));
	check(cudaMemcpy(value_size, &value_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("reduce %f\n", exe_time);

	printf("after reduce size %llu %llu\n", key_size_host, value_size_host);

//	unsigned int reduce_key[277];
//	check(cudaMemcpy(reduce_key, (unsigned int *)result_key_begin, 277 * 4,
//		cudaMemcpyDeviceToHost));
//
//	double reduce_value[277];
//	check(cudaMemcpy(reduce_value, (double *)result_value_begin, 277 * 8,
//		cudaMemcpyDeviceToHost));
//
//	for(int i = 0; i < 277; ++i)
//	{
//		printf("after reduce %u %u %lf\n", (reduce_key[i] >> 14), (reduce_key[i] & 0x3fff), reduce_value[i]);
//	}
}

void total_128_double(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	unsigned long long int key_size_host;
	check(cudaMemcpy(&key_size_host, key_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

	unsigned long long int value_size_host;
	check(cudaMemcpy(&value_size_host, value_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

//	printf("before reduce size %llu %llu\n", key_size_host, value_size_host);
//
//	ra::tuple::PackedNBytes<2> data_key[10];
//	check(cudaMemcpy(&data_key, (ra::tuple::PackedNBytes<2> *)key_begin, 160,
//		cudaMemcpyDeviceToHost));
//	for(int i = 0; i < 10;  ++i)
//	printf("before reduce key %d %llx, %llx\n", i, data_key[i].a[0], data_key[i].a[1]);

//	unsigned long long int data_value[8060];
//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 8060 * 8,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce value %llx, %llx\n", data_value[0], data_value[8059]);
//	double *pointer = (double *)data_value;
//	printf("before reduce value %lf, %lf\n", pointer[0], pointer[8059]);

	typedef thrust::device_ptr<ra::tuple::PackedNBytes<2> > ptr;
	thrust::pair<ptr, thrust::device_ptr<double> > size;

//	thrust::equal_to<unsigned char> binary_pred;
	thrust::plus<double> binary_op;

	size = thrust::reduce_by_key(
		ptr((ra::tuple::PackedNBytes<2>*)key_begin),
		ptr((ra::tuple::PackedNBytes<2>*)key_begin + (key_size_host / sizeof(ra::tuple::PackedNBytes<2>))),
		thrust::device_ptr<const double>((const double *)value_begin),
		ptr((ra::tuple::PackedNBytes<2>*)result_key_begin),
		thrust::device_ptr<double>((double *)result_value_begin),
		compare_128(), binary_op);

	key_size_host = (size.first - ptr((ra::tuple::PackedNBytes<2>*)result_key_begin)) * sizeof(ra::tuple::PackedNBytes<2>);
	value_size_host = (size.second - thrust::device_ptr<double>((double *)result_value_begin)) * sizeof(double);

	check(cudaMemcpy(key_size, &key_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));
	check(cudaMemcpy(value_size, &value_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("reduce %f\n", exe_time);

	printf("after reduce size %llu %llu\n", key_size_host, value_size_host);

	ra::tuple::PackedNBytes<2> reduce_key[50];
	check(cudaMemcpy(reduce_key, (ra::tuple::PackedNBytes<2> *)result_key_begin, 16 * 50,
		cudaMemcpyDeviceToHost));

	double reduce_value[50];
	check(cudaMemcpy(reduce_value, (double *)result_value_begin, 8 * 50,
		cudaMemcpyDeviceToHost));

	for(int i = 0; i < 50; ++i)
	{
		printf("%d %llx %llx %lf\n", i, reduce_key[i].a[0], reduce_key[i].a[1], reduce_value[i]);
	}
}

void min_32_double(const void* key_begin,
	const void* value_begin, 
	void* result_key_begin, 
	void* result_value_begin, 
	unsigned long long int *key_size,
	unsigned long long int *value_size)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	unsigned long long int key_size_host;
	check(cudaMemcpy(&key_size_host, key_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

	unsigned long long int value_size_host;
	check(cudaMemcpy(&value_size_host, value_size, sizeof(unsigned long long int),
		cudaMemcpyDeviceToHost));

//	printf("before reduce size %llu %llu\n", key_size_host, value_size_host);
//
//	unsigned int data_key[642];
//	check(cudaMemcpy(data_key, (unsigned int *)key_begin, 642 * 4,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce key %u, %u\n", data_key[0], data_key[641]);
//
//	unsigned long long int data_value[642];
//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 642 * 8,
//		cudaMemcpyDeviceToHost));
//	printf("before reduce value %llx, %llx\n", data_value[0], data_value[641]);
//	double *pointer = (double *)data_value;
//	printf("before reduce value %lf, %lf\n", pointer[0], pointer[641]);

	thrust::pair<thrust::device_ptr<unsigned int>, thrust::device_ptr<double> > size;

	thrust::equal_to<unsigned int> binary_pred;
	thrust::minimum<double> binary_op;

	size = thrust::reduce_by_key(
		thrust::device_ptr<const unsigned int>((const unsigned int*)key_begin),
		thrust::device_ptr<const unsigned int>((const unsigned int*)key_begin + (key_size_host / sizeof(unsigned int))),
		thrust::device_ptr<const double>((const double *)value_begin),
		thrust::device_ptr<unsigned int>((unsigned int*)result_key_begin),
		thrust::device_ptr<double>((double *)result_value_begin),
		binary_pred, binary_op);

	key_size_host = (size.first - thrust::device_ptr<unsigned int>((unsigned int *)result_key_begin)) * sizeof(unsigned int);
	value_size_host = (size.second - thrust::device_ptr<double>((double *)result_value_begin)) * sizeof(double);

	check(cudaMemcpy(key_size, &key_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));
	check(cudaMemcpy(value_size, &value_size_host, sizeof(unsigned long long int),
		cudaMemcpyHostToDevice));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("reduce %f\n", exe_time);

	printf("after reduce size %llu %llu\n", key_size_host, value_size_host);
//
//	unsigned int reduce_key[460];
//	check(cudaMemcpy(reduce_key, (unsigned int *)result_key_begin, 460 * 4,
//		cudaMemcpyDeviceToHost));
//
//	double reduce_value[460];
//	check(cudaMemcpy(reduce_value, (double *)result_value_begin, 460 * 8,
//		cudaMemcpyDeviceToHost));
//
//	for(int i = 0; i < 460; ++i)
//	{
//		printf("%u %u %lf\n", i, reduce_key[i], reduce_value[i]);
//	}
}
}

#endif

