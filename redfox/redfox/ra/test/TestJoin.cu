#include <iostream>
#include <redfox/ra/interface/Join.h>
#include <redfox/ra/interface/Tuple.h>

typedef unsigned int   out_type;
typedef unsigned short type;
const unsigned int elements = 1 << 15; // keep under 16 for shorts

const unsigned int threads = 64;
const unsigned int ctas    = 32;

__global__ void lowerBound(
	const type* leftBegin,
	const type* leftEnd,
	const type* rightBegin,
	const type* rightEnd,
	unsigned int* bounds)
{
	typedef ra::tuple::Tuple<sizeof(type) * 8> type;
	
	ra::cuda::lowerBound<type, type, 1>(
		(type::BasicType*) leftBegin,
		(type::BasicType*) leftEnd,
		(type::BasicType*) rightBegin,
		(type::BasicType*) rightEnd,
		bounds);
}

__global__ void join(
	const type* leftBegin,
	const type* leftEnd,
	const type* rightBegin,
	const type* rightEnd,
	out_type* output,
	unsigned int* bounds)
{
	typedef ra::tuple::Tuple<sizeof(type) * 8, sizeof(type) * 8> out_type;
	typedef ra::tuple::Tuple<sizeof(type) * 8>                   type;

	ra::cuda::join<type, type, out_type, 1, threads>(
		(type::BasicType*) leftBegin,
		(type::BasicType*) leftEnd,
		(type::BasicType*) rightBegin,
		(type::BasicType*) rightEnd,
		(out_type::BasicType*) output,
		bounds);
}

__global__ void exclusiveScan(unsigned int* values)
{
	unsigned int max = 0;
	values[threadIdx.x] = ra::cuda::exclusiveScan<512>(
		values[threadIdx.x], max);
}

__global__ void gather(
	out_type* begin,
	out_type* end,
	const out_type* inBegin,
	const out_type* inEnd,
	const unsigned int* histogram)
{
	typedef ra::tuple::Tuple<sizeof(type) * 8, sizeof(type) * 8> type;

	ra::cuda::gather<type>(begin, end, inBegin, inEnd, histogram);
}

void join(type* left, type* leftEnd, type* right, type* rightEnd, out_type* out)
{
	unsigned int elements   = leftEnd - left;
	unsigned int partitions = PARTITIONS;
	
	unsigned int* histogram = 0;
	cudaMalloc(&histogram, (1 + partitions) * sizeof(unsigned int));
	
	out_type* temp = 0;
	cudaMalloc(&temp, elements * sizeof(out_type));
	
	lowerBound<<<ctas, threads>>>(left, leftEnd,
		right, rightEnd, histogram);
	join<<<partitions, threads>>>(left, leftEnd,
		right, rightEnd, temp, histogram);
	exclusiveScan<<<1, partitions + 1>>>(histogram);
	gather<<<partitions, threads>>>(out, out + elements,
		temp, temp + elements, histogram);
	
	cudaFree(temp);
	cudaFree(histogram);
}

int main()
{
	cudaEvent_t start;
	cudaEvent_t finish;

	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	type*     left  = 0;
	type*     right = 0;
	out_type* out   = 0;
	
	cudaMalloc(&left,  sizeof(type)     * elements);
	cudaMalloc(&right, sizeof(type)     * elements);
	cudaMalloc(&out,   sizeof(out_type) * elements);
	
	type* data = new type[elements];

	for(unsigned int i = 0; i < elements; ++i) data[i] = i;

	cudaMemcpy(left,  data, sizeof(type) * elements, cudaMemcpyHostToDevice);
	cudaMemcpy(right, data, sizeof(type) * elements, cudaMemcpyHostToDevice);

	cudaEventRecord(start);

	join(left, left + elements, right, right + elements, out);

	cudaEventRecord(finish);
	cudaEventSynchronize(finish);

	float ms = 0.0f;

	cudaEventElapsedTime(&ms, start, finish);

	ms /= 1000.0f;

	std::cout << "elements  " << elements << " ("
		<< (sizeof(type) * elements) << " bytes)\n";
	std::cout << "time      " << ms << " seconds\n";
	std::cout << "bandwidth "
		<< (((sizeof(type) * 2 + sizeof(out_type))
			* elements / ms) / 1073741824.0f) << " GB/s\n";

	cudaEventDestroy(start);
	cudaEventDestroy(finish);

	out_type* out_data = new out_type[elements];

	cudaMemcpy(out_data, out, sizeof(out_type) * elements,
		cudaMemcpyDeviceToHost);

	unsigned int errors = 0;
	for(unsigned int i = 0; i < elements; ++i)
	{
		if(out_data[i] != (i << 8 * sizeof(type) | i))
		{
			if(errors < 20)
			{
				std::cout << "Result[" << i
					<< "] (" << out_data[i]
					<< ") does not match ref ("
					<< (i << 8 * sizeof(type) | i) << ")\n";
			}
			++errors;
		}
	}

	delete[] data;
	delete[] out_data;
	cudaFree(left);
	cudaFree(right);
	cudaFree(out);
	
	return 0;
}

