/*! \file Sort.cu
	\author Gregory Diamos <gregory.diamos>
	\date Wednesday December 1, 2010
	\brief The source file for the C interface to CUDA sorting routines.
*/

#ifndef SORT_CU_INCLUDED
#define SORT_CU_INCLUDED

// Redfox Includes
#include <redfox/nvcc/interface/RelationalAlgebraKernel.h>
#include <redfox/ra/interface/Sort.h>
#include <redfox/ra/interface/Tuple.h>

// Thrust Includes
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

// Hydrazine Includes
//#include <hydrazine/interface/debug.h>

#include <stdio.h>

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
  __host__ __device__
  bool operator()(ra::tuple::PackedNBytes<2> i, ra::tuple::PackedNBytes<2> j)
  {
   if (i.a[1] != j.a[1])
	return (i.a[1] < j.a[1]);

    return (i.a[0] < j.a[0]);
  }
};

struct compare_sort_gpu256
{
  __host__ __device__
  bool operator()(ra::tuple::PackedNBytes<4> i, ra::tuple::PackedNBytes<4> j)
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
  __host__ __device__
  bool operator()(ra::tuple::PackedNBytes<8> i, ra::tuple::PackedNBytes<8> j)
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

struct compare_sort_gpu1024
{
  __host__ __device__
  bool operator()(ra::tuple::PackedNBytes<16> i, ra::tuple::PackedNBytes<16> j)
  {
    if (i.a[15] != j.a[15])
	return (i.a[15] < j.a[15]);

    if (i.a[14] != j.a[14])
	return (i.a[14] < j.a[14]);

    if (i.a[13] != j.a[13])
	return (i.a[13] < j.a[13]);

    if (i.a[12] != j.a[12])
	return (i.a[12] < j.a[12]);

    if (i.a[11] != j.a[11])
	return (i.a[11] < j.a[11]);

    if (i.a[10] != j.a[10])
	return (i.a[10] < j.a[10]);

    if (i.a[9] != j.a[9])
	return (i.a[9] < j.a[9]);

    if (i.a[8] != j.a[8])
	return (i.a[8] < j.a[8]);

    if (i.a[7] != j.a[7])
	return (i.a[7] < j.a[7]);

    if (i.a[6] != j.a[6])
	return (i.a[6] < j.a[6]);

    if (i.a[5] != j.a[5])
	return (i.a[5] < j.a[5]);

    if (i.a[4] != j.a[4])
	return (i.a[4] < j.a[4]);

    if (i.a[3] != j.a[3])
	return (i.a[2] < j.a[2]);

    if (i.a[2] != j.a[2])
	return (i.a[2] < j.a[2]);

    if (i.a[1] != j.a[1])
	return (i.a[1] < j.a[1]);

    return (i.a[0] < j.a[0]);
  }
};

void sort_string(void* begin, void* end)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);


	typedef thrust::device_ptr<long long unsigned int> ptr;
	
	thrust::sort(
		ptr((long long unsigned int*)begin),
		ptr((long long unsigned int*)end), compare_sort_string());

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("sort %f\n", exe_time);
}

void sort(void* begin, void* end, unsigned long long int type)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float exe_time = 0.0f;
	cudaEventRecord(start,0);

	switch(type)
	{
	case nvcc::RelationalAlgebraKernel::I8:
	{
		thrust::sort(
			thrust::device_ptr<unsigned char>((unsigned char*)begin),
			thrust::device_ptr<unsigned char>((unsigned char*)end));
		break;
	}
	case nvcc::RelationalAlgebraKernel::I16:
	{
		thrust::sort(
			thrust::device_ptr<unsigned short>((unsigned short*)begin),
			thrust::device_ptr<unsigned short>((unsigned short*)end));
		break;
	}
	case nvcc::RelationalAlgebraKernel::I32:
	{
		thrust::sort(
			thrust::device_ptr<unsigned int>((unsigned int*)begin),
			thrust::device_ptr<unsigned int>((unsigned int*)end));
		break;
	}
	case nvcc::RelationalAlgebraKernel::I64:
	{
		typedef thrust::device_ptr<long long unsigned int> ptr;
		
		thrust::sort(
			ptr((long long unsigned int*)begin),
			ptr((long long unsigned int*)end));

		break;
	}
	case nvcc::RelationalAlgebraKernel::I128:
	{
		typedef thrust::device_ptr<ra::tuple::PackedNBytes<2> > ptr;

		thrust::sort(
			ptr((ra::tuple::PackedNBytes<2>*)begin),
			ptr((ra::tuple::PackedNBytes<2>*)end), compare_sort_gpu128());

		break;
	}
	case nvcc::RelationalAlgebraKernel::I256:
	{
		typedef thrust::device_ptr<ra::tuple::PackedNBytes<4> > ptr;

		thrust::sort(
			ptr((ra::tuple::PackedNBytes<4>*)begin),
			ptr((ra::tuple::PackedNBytes<4>*)end), compare_sort_gpu256());

		break;
	}
	case nvcc::RelationalAlgebraKernel::I512:
	{
		typedef thrust::device_ptr<ra::tuple::PackedNBytes<8> > ptr;

		thrust::sort(
			ptr((ra::tuple::PackedNBytes<8>*)begin),
			ptr((ra::tuple::PackedNBytes<8>*)end), compare_sort_gpu512());

		break;
	}
	case nvcc::RelationalAlgebraKernel::I1024:
	{
		typedef thrust::device_ptr<ra::tuple::PackedNBytes<16> > ptr;

		thrust::sort(
			ptr((ra::tuple::PackedNBytes<16>*)begin),
			ptr((ra::tuple::PackedNBytes<16>*)end), compare_sort_gpu1024());

		break;
	}
	default:
	{
		break;
	}
	}

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("sort %f\n", exe_time);
}

//void sort2(void* begin, void* end, unsigned long long int type)
//{
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start); cudaEventCreate(&stop);
//    float exe_time = 0.0f;
//    cudaEventRecord(start,0);
//
//	switch(type)
//	{
//	case nvcc::RelationalAlgebraKernel::I8:
//	{
//		thrust::stable_sort(
//			thrust::device_ptr<unsigned char>((unsigned char*)begin),
//			thrust::device_ptr<unsigned char>((unsigned char*)end));
//		break;
//	}
//	case nvcc::RelationalAlgebraKernel::I16:
//	{
//		thrust::stable_sort(
//			thrust::device_ptr<unsigned short>((unsigned short*)begin),
//			thrust::device_ptr<unsigned short>((unsigned short*)end));
//		break;
//	}
//	case nvcc::RelationalAlgebraKernel::I32:
//	{
//		thrust::stable_sort(
//			thrust::device_ptr<unsigned int>((unsigned int*)begin),
//			thrust::device_ptr<unsigned int>((unsigned int*)end));
//		break;
//	}
//	case nvcc::RelationalAlgebraKernel::I64:
//	{
//		typedef thrust::device_ptr<unsigned long long int> ptr;
//	
//		thrust::host_vector<unsigned long long int> h_vec0(1500000);
//
//		thrust::copy(ptr((unsigned long long int*)begin), ptr((unsigned long long int*)end), h_vec0.begin());
//
//		for(int i = 0; i < 1500000; ++i)
//		{
//			if (h_vec0[i] == 0)
//				printf("%d %llx\n", i, h_vec0[i]);
//		}
//	
//		thrust::stable_sort(
//			ptr((unsigned long long int*)begin),
//			ptr((unsigned long long int*)end));
//
//		thrust::host_vector<unsigned long long int> h_vec(1500000);
//
//		thrust::copy(ptr((unsigned long long int*)begin), ptr((unsigned long long int*)end), h_vec.begin());
//
//		for(int i = (1500000 - 1); i > (1500000 - 5); i--)
//			printf("***%d %llx\n", i, h_vec[i]);
//
//		break;
//	}
//	case nvcc::RelationalAlgebraKernel::I128:
//	{
//		typedef thrust::device_ptr<ra::tuple::PackedNBytes<2> > ptr;
//
//		ra::tuple::PackedNBytes<2> data[1];
//
//		check(cudaMemcpy(data, (ra::tuple::PackedNBytes<2> *)begin, 16,
//			cudaMemcpyDeviceToHost));
//
//		thrust::host_vector<ra::tuple::PackedNBytes<2> > h_vec(10000);
//		thrust::copy(ptr((ra::tuple::PackedNBytes<2>*)begin), ptr((ra::tuple::PackedNBytes<2>*)end), h_vec.begin());
//
//		for(int i = 0; i < 10000; i++)
//		{
//			ra::tuple::PackedNBytes<2> temp = h_vec[i] >> (unsigned int)14;
//			unsigned long long int temp2 = temp.a[0];
//			double *pointer = (double *)(&temp2);
//			double temp3 = *pointer;
//
//			if(temp3 == 1772627.25f) printf("%d\n find the data %llx %llx", i, h_vec[i].a[0], h_vec[i].a[1]);
//		}
////		printf("%llx, %llx\n", h_vec[0].a[0], h_vec[0].a[1]);
////		printf("%llx, %llx\n", h_vec[1].a[0], h_vec[1].a[1]);
////		printf("%llx, %llx\n", h_vec[8059].a[0], h_vec[8059].a[1]);
//	
//		thrust::stable_sort(
//			ptr((ra::tuple::PackedNBytes<2>*)begin),
//			ptr((ra::tuple::PackedNBytes<2>*)end), compare_sort_gpu128());
//
//		ra::tuple::PackedNBytes<2> data2[1];
//
//		check(cudaMemcpy(data2, (ra::tuple::PackedNBytes<2> *)begin, 16,
//			cudaMemcpyDeviceToHost));
//
//		thrust::host_vector<ra::tuple::PackedNBytes<2> > h_vec2(10000);
//		thrust::copy(ptr((ra::tuple::PackedNBytes<2>*)begin), ptr((ra::tuple::PackedNBytes<2>*)end), h_vec2.begin());
//
//		for(int i = 1; i < 10000; i++)
//		{
//			ra::tuple::PackedNBytes<2> temp = h_vec2[i] >> (unsigned int)14;
//			unsigned long long int temp2 = temp.a[0];
//			double *pointer = (double *)(&temp2);
//			double temp3 = *pointer;
//
//                        ra::tuple::PackedNBytes<2> temp4 = h_vec2[i - 1] >> (unsigned int)14;
//                        unsigned long long int temp5 = temp4.a[0];
//                        double *pointer2 = (double *)(&temp5);
//                        double temp6 = *pointer2;
//
//			if(temp3 < temp6)
//			printf("************after sort wrong %d, %f %llx %llx %f %llx %llx\n", i, temp3, h_vec2[i].a[0], h_vec2[i].a[1], temp6, h_vec2[i-1].a[0], h_vec2[i-1].a[1]);
//		}
//
////
////		thrust::host_vector<ra::tuple::PackedNBytes<2> > h_vec2(8060);
////		thrust::copy(ptr((ra::tuple::PackedNBytes<2>*)begin), ptr((ra::tuple::PackedNBytes<2>*)end), h_vec2.begin());
////
////		for(int i = 0; i < 8060; i++)
////		{
////			if(h_vec2[i].a[1] != 0x0 && h_vec2[i].a[1] != 0x1 && h_vec2[i].a[1] != 0x3 && h_vec2[i].a[1] != 0x5)
////				printf("after 2nd sort %d, %x\n", i, h_vec2[i].a[1]);
////		}
//
////		printf("%llx, %llx\n", h_vec2[0].a[0], h_vec2[0].a[1]);
////		printf("%llx, %llx\n", h_vec[1].a[0], h_vec[1].a[1]);
////		printf("%llx, %llx\n", h_vec[8059].a[0], h_vec[8059].a[1]);
//
//		break;
//	}
//	case nvcc::RelationalAlgebraKernel::I256:
//	{
//		typedef thrust::device_ptr<ra::tuple::PackedNBytes<4> > ptr;
//
////unsigned long long int data[16120];
////
////	check(cudaMemcpy(data, (char *)begin, 128960,
////		cudaMemcpyDeviceToHost));
////printf("before sort %llx, %llx\n", data[0], data[1]);
////		thrust::host_vector<ra::tuple::PackedNBytes<2> > h_vec(8060);
////		thrust::copy(ptr((ra::tuple::PackedNBytes<2>*)begin), ptr((ra::tuple::PackedNBytes<2>*)end), h_vec.begin());
////
////		for(int i = 0; i < 8060; i++)
////		{
////			if(h_vec[i].a[1] != 0x0 && h_vec[i].a[1] != 0x1 && h_vec[i].a[1] != 0x3 && h_vec[i].a[1] != 0x5)
////				printf("************before 2nd sort %d, %llx, %llx \n", i, h_vec[i].a[1], h_vec[i].a[0]);
////		}
////		printf("%llx, %llx\n", h_vec[0].a[0], h_vec[0].a[1]);
////		printf("%llx, %llx\n", h_vec[1].a[0], h_vec[1].a[1]);
////		printf("%llx, %llx\n", h_vec[8059].a[0], h_vec[8059].a[1]);
//	
//		thrust::stable_sort(
//			ptr((ra::tuple::PackedNBytes<4>*)begin),
//			ptr((ra::tuple::PackedNBytes<4>*)end), compare_sort_gpu256());
//
//		thrust::host_vector<ra::tuple::PackedNBytes<4> > h_vec2(158960);
//		thrust::copy(ptr((ra::tuple::PackedNBytes<4>*)begin), ptr((ra::tuple::PackedNBytes<4>*)end), h_vec2.begin());
//
//		for(int i = 1; i < 158960; i++)
//		{
////			typedef ra::tuple::Tuple<64, 3, 5, 14, 18, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0> Tuple;
//			typedef ra::tuple::Tuple<18, 64, 3, 5, 14, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0> Tuple;
//			unsigned int before = ra::tuple::extract<unsigned int, 0, Tuple>(h_vec2[i - 1]);
//			unsigned int after  = ra::tuple::extract<unsigned int, 0, Tuple>(h_vec2[i]);
//
//			if(before > after) printf("***ERR*** %d %u %u\n", i, before, after);
//		}
//		
//		printf("******after sort\n");
////		printf("%llx, %llx\n", h_vec2[0].a[0], h_vec2[0].a[1]);
////		printf("%llx, %llx\n", h_vec[1].a[0], h_vec[1].a[1]);
////		printf("%llx, %llx\n", h_vec[8059].a[0], h_vec[8059].a[1]);
//
//		break;
//	}
//	default:
//	{
//		printf("****************sort default********************\n");
//		break;
//	}
//	}
//
//    cudaEventRecord(stop,0);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&exe_time, start, stop);
//	cudaEventDestroy(start);
//	cudaEventDestroy(stop);
//    printf("sort %f\n", exe_time);
//}
}

#endif

