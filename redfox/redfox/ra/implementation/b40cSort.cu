/*! \file Sort.cu
	\author Gregory Diamos <gregory.diamos>
	\date Wednesday December 1, 2010
	\brief The source file for the C interface to CUDA sorting routines.
*/

// Redfox Includes
#include <redfox/nvcc/interface/RelationalAlgebraKernel.h>
#include <redfox/ra/interface/b40cSort.h>
#include <redfox/ra/interface/Tuple.h>

#include <redfox/ra/interface/b40c/radix_sort/enactor.cuh>
#include <redfox/ra/interface/b40c/util/multi_buffer.cuh>

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

void sort_pair(void* key_begin, void* value_begin, unsigned long long int size, 
	unsigned long long int key_bits, unsigned long long int value_type)
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

	static const b40c::radix_sort::ProblemSize PROBLEM_SIZE = b40c::radix_sort::LARGE_PROBLEM;

	// Create a reusable sorting enactor
	b40c::radix_sort::Enactor enactor;

	if (key_bits == 2 && value_type == nvcc::RelationalAlgebraKernel::I32)
	{
		typedef unsigned char key_type;
		typedef unsigned int value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 2, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 3 && value_type == nvcc::RelationalAlgebraKernel::I32)
	{
		typedef unsigned char key_type;
		typedef unsigned int value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 3, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 3 && value_type == nvcc::RelationalAlgebraKernel::I64)
	{
		typedef unsigned char key_type;
		typedef unsigned long long int value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 3, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 3 && value_type == nvcc::RelationalAlgebraKernel::I128)
	{
		typedef unsigned char key_type;
		typedef gpu128 value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 3, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 5 && value_type == nvcc::RelationalAlgebraKernel::I16)
	{
		typedef unsigned char key_type;
		typedef unsigned short value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 5, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 5 && value_type == nvcc::RelationalAlgebraKernel::I32)
	{
		typedef unsigned char key_type;
		typedef unsigned int value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 5, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 5 && value_type == nvcc::RelationalAlgebraKernel::I64)
	{
		typedef unsigned char key_type;
		typedef unsigned long long int value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 5, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 5 && value_type == nvcc::RelationalAlgebraKernel::I128)
	{
		typedef unsigned char key_type;
		typedef gpu128 value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 5, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 14 && value_type == nvcc::RelationalAlgebraKernel::I32)
	{
		typedef unsigned short key_type;
		typedef unsigned int value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 14, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 14 && value_type == nvcc::RelationalAlgebraKernel::I64)
	{
		typedef unsigned short key_type;
		typedef unsigned long long int value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 14, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 14 && value_type == nvcc::RelationalAlgebraKernel::I128)
	{
		typedef unsigned short key_type;
		typedef gpu128 value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 14, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 18 && value_type == nvcc::RelationalAlgebraKernel::I32)
	{
		typedef unsigned int key_type;
		typedef unsigned int value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 18, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 18 && value_type == nvcc::RelationalAlgebraKernel::I64)
	{
		typedef unsigned int key_type;
		typedef unsigned long long int value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 18, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 18 && value_type == nvcc::RelationalAlgebraKernel::I128)
	{
		typedef unsigned int key_type;
		typedef gpu128 value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 18, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 23 && value_type == nvcc::RelationalAlgebraKernel::I8)
	{
		typedef unsigned int key_type;
		typedef unsigned char value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 23, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 23 && value_type == nvcc::RelationalAlgebraKernel::I16)
	{
		typedef unsigned int key_type;
		typedef unsigned short value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 23, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 23 && value_type == nvcc::RelationalAlgebraKernel::I32)
	{
		typedef unsigned int key_type;
		typedef unsigned int value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 23, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 23 && value_type == nvcc::RelationalAlgebraKernel::I64)
	{
		typedef unsigned int key_type;
		typedef unsigned long long int value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 23, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 23 && value_type == nvcc::RelationalAlgebraKernel::I128)
	{
		typedef unsigned int key_type;
		typedef gpu128 value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 23, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 26 && value_type == nvcc::RelationalAlgebraKernel::I16)
	{
		typedef unsigned int key_type;
		typedef unsigned short value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 26, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 26 && value_type == nvcc::RelationalAlgebraKernel::I32)
	{
		typedef unsigned int key_type;
		typedef unsigned int value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 26, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 26 && value_type == nvcc::RelationalAlgebraKernel::I64)
	{
		typedef unsigned int key_type;
		typedef unsigned long long int value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 26, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 26 && value_type == nvcc::RelationalAlgebraKernel::I128)
	{
		typedef unsigned int key_type;
		typedef gpu128 value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 26, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 32 && value_type == nvcc::RelationalAlgebraKernel::I32)
	{
		typedef unsigned int key_type;
		typedef unsigned int value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 32, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}
	else if (key_bits == 32 && value_type == nvcc::RelationalAlgebraKernel::I64)
	{
		typedef unsigned int key_type;
		typedef unsigned long long int value_type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<key_type, value_type> double_buffer;

		double_buffer.d_keys[0] = (key_type *)key_begin;
		double_buffer.d_values[0] = (value_type *)value_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(key_type) * size));

        check(cudaMalloc((void**) &double_buffer.d_values[1], sizeof(value_type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 32, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(key_type) * size, cudaMemcpyDeviceToDevice));

			check(cudaMemcpy(double_buffer.d_values[0], double_buffer.d_values[1], sizeof(value_type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
		check(cudaFree(double_buffer.d_values[1]));
	}

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("radix_sort %lf\n", exe_time);

//	check(cudaMemcpy(data_key, (unsigned int *)key_begin, 40,
//		cudaMemcpyDeviceToHost));
//
//	check(cudaMemcpy(data_value, (unsigned long long int *)value_begin, 80,
//		cudaMemcpyDeviceToHost));
//
//	for(unsigned int i = 0; i < 10; ++i)
//		printf("%u %u %llx\n", i, data_key[i], data_value[i]);
}

void sort_key(void* key_begin, unsigned long long int size, 
	unsigned long long int bits)
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

	static const b40c::radix_sort::ProblemSize PROBLEM_SIZE = b40c::radix_sort::LARGE_PROBLEM;

	// Create a reusable sorting enactor
	b40c::radix_sort::Enactor enactor;

	if (bits == 3)
	{
		typedef unsigned char type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<type> double_buffer;

		double_buffer.d_keys[0] = (type *)key_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 3, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
	}
	else if (bits == 14)
	{
		typedef unsigned short type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<type> double_buffer;

		double_buffer.d_keys[0] = (type *)key_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 14, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
	}
	else if (bits == 18)
	{
		typedef unsigned int type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<type> double_buffer;

		double_buffer.d_keys[0] = (type *)key_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 18, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
	}
	else if (bits == 23)
	{
		typedef unsigned int type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<type> double_buffer;

		double_buffer.d_keys[0] = (type *)key_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 23, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
	}
	else if (bits == 26)
	{
		typedef unsigned int type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<type> double_buffer;

		double_buffer.d_keys[0] = (type *)key_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 26, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
	}
	else if (bits == 32)
	{
		typedef unsigned int type;

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<type> double_buffer;

		double_buffer.d_keys[0] = (type *)key_begin;

		// Allocate pong buffer
		check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(type) * size));

		// Sort
		enactor.Sort<PROBLEM_SIZE, 32, 0>(double_buffer, size);

		if(double_buffer.selector != 0)
		{
			check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(type) * size, cudaMemcpyDeviceToDevice));
		}

		check(cudaFree(double_buffer.d_keys[1]));
	}
    else if (bits == 37)
    {
        typedef unsigned long long int type;

        // Create ping-pong storage wrapper
        b40c::util::DoubleBuffer<type> double_buffer;

        double_buffer.d_keys[0] = (type *)key_begin;

        // Allocate pong buffer
        check(cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(type) * size));

        // Sort
        enactor.Sort<PROBLEM_SIZE, 37, 0>(double_buffer, size);

        if(double_buffer.selector != 0)
        {
            check(cudaMemcpy(double_buffer.d_keys[0], double_buffer.d_keys[1], sizeof(type) * size, cudaMemcpyDeviceToDevice));
        }

        check(cudaFree(double_buffer.d_keys[1]));
    }

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exe_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("radix_sort %lf\n", exe_time);

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


