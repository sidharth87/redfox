/*! \file Sort.cu
	\author Gregory Diamos <gregory.diamos>
	\date Wednesday December 1, 2010
	\brief The source file for the C interface to CUDA sorting routines.
*/

// Redfox Includes
#include <redfox/nvcc/interface/RelationalAlgebraKernel.h>
#include <redfox/ra/interface/ModernGPUSort.h>
#include <redfox/ra/interface/Tuple.h>

#include <redfox/ra/interface/moderngpu/include/kernels/mergesort.cuh>

#include <stdio.h>
#include <iostream>

class gpu128
{
public:
	typedef long long unsigned int type;

public:
	type a[2];
};

class gpu256
{
public:
	typedef long long unsigned int type;

public:
	type a[4];
};

class gpu512
{
public:
	typedef long long unsigned int type;

public:
	type a[8];
};

struct compare_sort_string
{
  __host__ __device__
  bool operator()(unsigned long long int i, unsigned long long int j)
  {
    char *string1 = (char *)i;
    char *string2 = (char *)j;

    int ii = 0;
    
    while(string1[ii] != '\0' && string2[ii] != '\0')
    {
    	if(string1[ii] != string2[ii])
    		return (string1[ii] < string2[ii]);
    
    	ii++;
    }
    
    if(string1[ii] == '\0' && string2[ii] != '\0')
    	return true;
    else
    	return false;
  }
};

struct compare_sort_gpu128
{
  typedef gpu128 type;

  __host__ __device__
  bool operator()(type i, type j)
  {
   if (i.a[1] != j.a[1])
	return (i.a[1] < j.a[1]);

    return (i.a[0] < j.a[0]);
  }
};

struct compare_sort_gpu256
{
  typedef gpu256 type;

  __host__ __device__
  bool operator()(type i, type j)
  {
   if (i.a[3] != j.a[3])
	return (i.a[3] < j.a[3]);

   if (i.a[2] != j.a[2])
	return (i.a[2] < j.a[2]);

   if (i.a[1] != j.a[1])
	return (i.a[1] < j.a[1]);

   return (i.a[0] < j.a[0]);
  }
};

struct compare_sort_gpu512
{
  typedef gpu512 type;

  __host__ __device__
  bool operator()(type i, type j)
  {
   if (i.a[7] != j.a[7])
	return (i.a[7] < j.a[7]);

   if (i.a[6] != j.a[6])
	return (i.a[6] < j.a[6]);

   if (i.a[5] != j.a[5])
	return (i.a[5] < j.a[5]);

   if (i.a[4] != j.a[4])
	return (i.a[4] < j.a[4]);

   if (i.a[3] != j.a[3])
	return (i.a[3] < j.a[3]);

   if (i.a[2] != j.a[2])
	return (i.a[2] < j.a[2]);

   if (i.a[1] != j.a[1])
	return (i.a[1] < j.a[1]);

   return (i.a[0] < j.a[0]);
  }
};


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

void sort_string_pair(void* key_begin, void* value_begin, unsigned long long int size, 
	unsigned long long int key_type, unsigned long long int value_type)
{
//	unsigned int data_key[30142];
//	unsigned long long int data_value[10];

//	check(cudaMemcpy(data_key, (unsigned int *)key_begin, 30142*4,
//		cudaMemcpyDeviceToHost));

//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 80,
//		cudaMemcpyDeviceToHost));
//printf("size %llu\n", size);
//	for(unsigned int i = 0; i <300; ++i)
//		printf("%u %u \n", i, data_key[i]);
//	for(unsigned int i = 30141; i > 30131; --i)
//		printf("%u %u \n", i, data_key[i]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

//	size_t freeMem, totalMem;
//	cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";

//	mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);
//
//	context->Start();
	if (value_type == nvcc::RelationalAlgebraKernel::I8)
	{
		mgpu::MergesortPairs<unsigned long long int, unsigned char>((unsigned long long int*)key_begin, (unsigned char*)value_begin, 
				size, compare_sort_string()/*, *context*/);
	}
	else if (value_type == nvcc::RelationalAlgebraKernel::I16)
	{
		mgpu::MergesortPairs<unsigned long long int, unsigned short>((unsigned long long int*)key_begin, (unsigned short*)value_begin, 
				size, compare_sort_string()/*, *context*/);
	}
	else if (value_type == nvcc::RelationalAlgebraKernel::I32)
	{
		mgpu::MergesortPairs<unsigned long long int, unsigned int>((unsigned long long int*)key_begin, (unsigned int*)value_begin, 
				size, compare_sort_string()/*, *context*/);
	}
	else if (value_type == nvcc::RelationalAlgebraKernel::I64)
	{
		mgpu::MergesortPairs<unsigned long long int, unsigned long long int>((unsigned long long int*)key_begin, (unsigned long long int*)value_begin, 
				size, compare_sort_string()/*, *context*/);
	}
	else if (value_type == nvcc::RelationalAlgebraKernel::I128)
	{
		typedef gpu128 type;
		mgpu::MergesortPairs<unsigned long long int, type>((unsigned long long int*)key_begin, (type *)value_begin, 
				size, compare_sort_string()/*, *context*/);
	}
	else if (value_type == nvcc::RelationalAlgebraKernel::I256)
	{
		typedef gpu256 type;
		mgpu::MergesortPairs<unsigned long long int, type>((unsigned long long int*)key_begin, (type *)value_begin, 
				size, compare_sort_string()/*, *context*/);
	}
	else if (value_type == nvcc::RelationalAlgebraKernel::I512)
	{
		typedef gpu512 type;
		mgpu::MergesortPairs<unsigned long long int, type>((unsigned long long int*)key_begin, (type *)value_begin, 
				size, compare_sort_string()/*, *context*/);
	}

//	exe_time += context->Split();

//        cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("sort %lf\n", exe_time);

//	check(cudaMemcpy(data_key, (unsigned int *)key_begin, 40,
//		cudaMemcpyDeviceToHost));
//
//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 80,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 10; ++i)
//		printf("%u %u %llx\n", i, data_key[i], data_value[i]);
}

void sort_pair(void* key_begin, void* value_begin, unsigned long long int size, 
	unsigned long long int key_type, unsigned long long int value_type)
{
//	unsigned int data_key[30142];
//	unsigned long long int data_value[10];

//	check(cudaMemcpy(data_key, (unsigned int *)key_begin, 30142*4,
//		cudaMemcpyDeviceToHost));

//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 80,
//		cudaMemcpyDeviceToHost));
//printf("size %llu\n", size);
//	for(unsigned int i = 0; i <300; ++i)
//		printf("%u %u \n", i, data_key[i]);
//	for(unsigned int i = 30141; i > 30131; --i)
//		printf("%u %u \n", i, data_key[i]);

//	ra::tuple::PackedNBytes<2> data_key[10];
//	check(cudaMemcpy(&data_key, (ra::tuple::PackedNBytes<2> *)key_begin, 160,
//		cudaMemcpyDeviceToHost));
//	for(int i = 0; i < 10;  ++i)
//	printf("before reduce key %d %llx, %llx\n", i, data_key[i].a[0], data_key[i].a[1]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

//	size_t freeMem, totalMem;
//	cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";

//	mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);
//
//	context->Start();
	if (key_type == nvcc::RelationalAlgebraKernel::I8 && value_type == nvcc::RelationalAlgebraKernel::I16)
	{
		mgpu::MergesortPairs<unsigned char, unsigned short>((unsigned char*)key_begin, (unsigned short *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I8 && value_type == nvcc::RelationalAlgebraKernel::I32)
	{
		mgpu::MergesortPairs<unsigned char, unsigned int>((unsigned char*)key_begin, (unsigned int *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I8 && value_type == nvcc::RelationalAlgebraKernel::I64)
	{
		mgpu::MergesortPairs<unsigned char, unsigned long long int>((unsigned char*)key_begin, (unsigned long long int *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I8 && value_type == nvcc::RelationalAlgebraKernel::I128)
	{
		typedef gpu128 type;
		mgpu::MergesortPairs<unsigned char, type>((unsigned char*)key_begin, (type *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I8 && value_type == nvcc::RelationalAlgebraKernel::I256)
	{
		typedef gpu256 type;
		mgpu::MergesortPairs<unsigned char, type>((unsigned char*)key_begin, (type *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I8 && value_type == nvcc::RelationalAlgebraKernel::I512)
	{
		typedef gpu512 type;
		mgpu::MergesortPairs<unsigned char, type>((unsigned char*)key_begin, (type *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I16 && value_type == nvcc::RelationalAlgebraKernel::I8)
	{
		mgpu::MergesortPairs<unsigned short, unsigned char>((unsigned short*)key_begin, (unsigned char *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I16 && value_type == nvcc::RelationalAlgebraKernel::I32)
	{
		mgpu::MergesortPairs<unsigned short, unsigned int>((unsigned short*)key_begin, (unsigned int *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I16 && value_type == nvcc::RelationalAlgebraKernel::I64)
	{
		mgpu::MergesortPairs<unsigned short, unsigned long long int>((unsigned short*)key_begin, (unsigned long long int *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I16 && value_type == nvcc::RelationalAlgebraKernel::I128)
	{
		typedef gpu128 type;
		mgpu::MergesortPairs<unsigned short, type>((unsigned short*)key_begin, (type *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I16 && value_type == nvcc::RelationalAlgebraKernel::I256)
	{
		typedef gpu256 type;
		mgpu::MergesortPairs<unsigned short, type>((unsigned short*)key_begin, (type *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I16 && value_type == nvcc::RelationalAlgebraKernel::I512)
	{
		typedef gpu512 type;
		mgpu::MergesortPairs<unsigned short, type>((unsigned short*)key_begin, (type *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I32 && value_type == nvcc::RelationalAlgebraKernel::I8)
	{
		mgpu::MergesortPairs<unsigned int, unsigned char>((unsigned int*)key_begin, (unsigned char *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I32 && value_type == nvcc::RelationalAlgebraKernel::I16)
	{
		mgpu::MergesortPairs<unsigned int, unsigned short>((unsigned int*)key_begin, (unsigned short *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I32 && value_type == nvcc::RelationalAlgebraKernel::I32)
	{
		mgpu::MergesortPairs<unsigned int, unsigned int>((unsigned int*)key_begin, (unsigned int *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I32 && value_type == nvcc::RelationalAlgebraKernel::I64)
	{
		mgpu::MergesortPairs<unsigned int, unsigned long long int>((unsigned int*)key_begin, (unsigned long long int *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I32 && value_type == nvcc::RelationalAlgebraKernel::I128)
	{
		typedef gpu128 type;
		mgpu::MergesortPairs<unsigned int, type>((unsigned int*)key_begin, (type *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I32 && value_type == nvcc::RelationalAlgebraKernel::I256)
	{
		typedef gpu256 type;
		mgpu::MergesortPairs<unsigned int, type>((unsigned int*)key_begin, (type *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I32 && value_type == nvcc::RelationalAlgebraKernel::I512)
	{
		typedef gpu512 type;
		mgpu::MergesortPairs<unsigned int, type>((unsigned int*)key_begin, (type *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I64 && value_type == nvcc::RelationalAlgebraKernel::I8)
	{
		mgpu::MergesortPairs<unsigned long long int, unsigned char>((unsigned long long int*)key_begin, (unsigned char *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I64 && value_type == nvcc::RelationalAlgebraKernel::I16)
	{
		mgpu::MergesortPairs<unsigned long long int, unsigned short>((unsigned long long int*)key_begin, (unsigned short *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I64 && value_type == nvcc::RelationalAlgebraKernel::I32)
	{
		mgpu::MergesortPairs<unsigned long long int, unsigned int>((unsigned long long int*)key_begin, (unsigned int *)value_begin, 
				size/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I128 && value_type == nvcc::RelationalAlgebraKernel::I16)
	{
		typedef gpu128 type;
		mgpu::MergesortPairs<type, unsigned short>((type*)key_begin, (unsigned short *)value_begin, 
				size, compare_sort_gpu128()/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I128 && value_type == nvcc::RelationalAlgebraKernel::I32)
	{
		typedef gpu128 type;
		mgpu::MergesortPairs<type, unsigned int>((type*)key_begin, (unsigned int *)value_begin, 
				size, compare_sort_gpu128()/*, *context*/);
	}
	else if (key_type == nvcc::RelationalAlgebraKernel::I128 && value_type == nvcc::RelationalAlgebraKernel::I64)
	{
		typedef gpu128 type;
		mgpu::MergesortPairs<type, unsigned long long int>((type*)key_begin, (unsigned long long int *)value_begin, 
				size, compare_sort_gpu128()/*, *context*/);
	}

//	exe_time += context->Split();

//        cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("sort %lf\n", exe_time);

//	check(cudaMemcpy(data_key, (unsigned int *)key_begin, 40,
//		cudaMemcpyDeviceToHost));
//
//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 80,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 10; ++i)
//		printf("%u %u %llx\n", i, data_key[i], data_value[i]);
}

void sort_string_key(void* key_begin, unsigned long long int size, 
	unsigned long long int type)
{
//	unsigned int *data_key = (unsigned int *)malloc(100 * 4);
//
//	check(cudaMemcpy(data_key, (unsigned int *)key_begin, 100*4,
//		cudaMemcpyDeviceToHost));

//	for(unsigned int i = 0; i < 100; ++i)
//		printf("%u %x \n", i, data_key[i]);

//	printf("%llu %p\n", size, key_begin);

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

//	size_t freeMem, totalMem;
//	cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";

//	mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);
//
//	context->Start();

		mgpu::MergesortKeys<unsigned long long int>((unsigned long long int*)key_begin, 
				size, compare_sort_string()/*, *context*/);

//	exe_time += context->Split();

//        cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("sort %lf\n", exe_time);

//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 10,
//		cudaMemcpyDeviceToHost));
//
//	check(cudaMemcpy(data_value, (double *)value_begin, 80,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 10; ++i)
//		printf("%u %x %lf\n", i, data_key[i], data_value[i]);
}

void sort_key(void* key_begin, unsigned long long int size, 
	unsigned long long int type)
{
//	unsigned int *data_key = (unsigned int *)malloc(100 * 4);
//
//	check(cudaMemcpy(data_key, (unsigned int *)key_begin, 100*4,
//		cudaMemcpyDeviceToHost));

//	for(unsigned int i = 0; i < 100; ++i)
//		printf("%u %x \n", i, data_key[i]);

//	printf("%llu %p\n", size, key_begin);

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

//	size_t freeMem, totalMem;
//	cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";

//	mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);
//
//	context->Start();

	if (type == nvcc::RelationalAlgebraKernel::I8)
	{
		mgpu::MergesortKeys<unsigned char>((unsigned char*)key_begin, 
				size/*, *context*/);
	}
	else if (type == nvcc::RelationalAlgebraKernel::I16)
	{
		mgpu::MergesortKeys<unsigned short>((unsigned short*)key_begin, 
				size/*, *context*/);
	}
	else if (type == nvcc::RelationalAlgebraKernel::I32)
	{
		mgpu::MergesortKeys<unsigned int>((unsigned int*)key_begin, 
				size/*, *context*/);
	}
	else if (type == nvcc::RelationalAlgebraKernel::I64)
	{
		mgpu::MergesortKeys<unsigned long long int>((unsigned long long int*)key_begin, 
				size/*, *context*/);
	}
	else if (type == nvcc::RelationalAlgebraKernel::I128)
	{
		typedef gpu128 type;
		mgpu::MergesortKeys<type>((type *)key_begin, 
			size, compare_sort_gpu128()/*, *context*/);
	}
	else if (type == nvcc::RelationalAlgebraKernel::I256)
	{
		typedef gpu256 type;
		mgpu::MergesortKeys<type>((type *)key_begin, 
			size, compare_sort_gpu256()/*, *context*/);
	}
	else if (type == nvcc::RelationalAlgebraKernel::I512)
	{
		typedef gpu512 type;
		mgpu::MergesortKeys<type>((type *)key_begin, 
			size, compare_sort_gpu512()/*, *context*/);
	}


//	exe_time += context->Split();

//        cudaMemGetInfo(&freeMem, &totalMem);
//	std::cout << freeMem << "\n";

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("sort %lf\n", exe_time);

//	check(cudaMemcpy(data_key, (unsigned char *)key_begin, 10,
//		cudaMemcpyDeviceToHost));
//
//	check(cudaMemcpy(data_value, (double *)value_begin, 80,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 10; ++i)
//		printf("%u %x %lf\n", i, data_key[i], data_value[i]);
}

}


