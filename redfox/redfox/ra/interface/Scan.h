/*! \file Scan.h
	\date Thursday December 2, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the scan family of functions.
*/

#ifndef SCAN_H_INCLUDED
#define SCAN_H_INCLUDED

namespace ra
{

namespace cuda
{

#if 0 
template<unsigned int CTA_SIZE, typename T>
__device__ T exclusiveScan(T val, T& max, T carryIn = 0)
{
	__shared__ T _array[CTA_SIZE + 1];
	if(threadIdx.x == 0) _array[0] = carryIn;

	T* array = _array + 1;
	
	array[threadIdx.x] = val;

	__syncthreads();

	if (CTA_SIZE >   1) { if(threadIdx.x >=   1) 
		{ unsigned int tmp = array[threadIdx.x -   1]; val = tmp + val; } 
			__syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
	if (CTA_SIZE >   2) { if(threadIdx.x >=   2) 
		{ unsigned int tmp = array[threadIdx.x -   2]; val = tmp + val; } 
			__syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
	if (CTA_SIZE >   4) { if(threadIdx.x >=   4) 
		{ unsigned int tmp = array[threadIdx.x -   4]; val = tmp + val; } 
			__syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
	if (CTA_SIZE >   8) { if(threadIdx.x >=   8) 
		{ unsigned int tmp = array[threadIdx.x -   8]; val = tmp + val; } 
			__syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
	if (CTA_SIZE >  16) { if(threadIdx.x >=  16) 
		{ unsigned int tmp = array[threadIdx.x -  16]; val = tmp + val; } 
			__syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
	if (CTA_SIZE >  32) { if(threadIdx.x >=  32) 
		{ unsigned int tmp = array[threadIdx.x -  32]; val = tmp + val; } 
			__syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
	if (CTA_SIZE >  64) { if(threadIdx.x >=  64) 
		{ unsigned int tmp = array[threadIdx.x -  64]; val = tmp + val; } 
			__syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
	if (CTA_SIZE > 128) { if(threadIdx.x >= 128) 
		{ unsigned int tmp = array[threadIdx.x - 128]; val = tmp + val; } 
			__syncthreads(); array[threadIdx.x] = val; __syncthreads(); }
	if (CTA_SIZE > 256) { if(threadIdx.x >= 256) 
		{ unsigned int tmp = array[threadIdx.x - 256]; val = tmp + val; }
			__syncthreads(); array[threadIdx.x] = val; __syncthreads(); }  
	if (CTA_SIZE > 512) { if(threadIdx.x >= 512) 
		{ unsigned int tmp = array[threadIdx.x - 512]; val = tmp + val; }
			__syncthreads(); array[threadIdx.x] = val; __syncthreads(); }  

	max = array[CTA_SIZE - 1];
	return _array[threadIdx.x];
}
#endif

#define MGPU_DEVICE __device__ __forceinline__
#define WARP_SIZE (32)

MGPU_DEVICE int shfl_add(int x, int offset, int width = WARP_SIZE) {
    int result = 0;
    int mask = (WARP_SIZE - width) << 8;
    asm(
        "{.reg .s32 r0;"
        ".reg .pred p;"
        "shfl.up.b32 r0|p, %1, %2, %3;"
        "@p add.s32 r0, r0, %4;"
        "mov.s32 %0, r0; }"
        : "=r"(result) : "r"(x), "r"(offset), "r"(mask), "r"(x));
    return result;
}

template<unsigned int CTA_SIZE, typename T>
MGPU_DEVICE T exclusiveScan(T val, T& max, T carryIn = 0)
{
//printf("%u %u %u\n", CTA_SIZE, gridDim.x, blockDim.x);
    enum { Size = CTA_SIZE, NumSegments = WARP_SIZE, SegSize = CTA_SIZE / NumSegments };
    enum { Capacity = NumSegments + 1 };
    __shared__ T shared[Capacity + 1];

    int tid = threadIdx.x;
    int lane = (SegSize - 1) & tid;
    int segment = tid / SegSize;

//printf("%u %u\n", CTA_SIZE, gridDim.x, blockDim.x);
    // Scan each segment using shfl_add.
    T scan = val;
    #pragma unroll
    for(int offset = 1; offset < SegSize; offset <<= 1)
        scan = shfl_add(scan, offset, SegSize);

//printf("%u %u\n", threadIdx.x, scan);
    // Store the reduction (last element) of each segment into storage.
    if(SegSize - 1 == lane) shared[segment] = scan;
    __syncthreads();

    // Warp 0 does a full shfl warp scan on the partials. The total is
    // stored to shared[NumSegments]. (NumSegments = WARP_SIZE)
    if(tid < NumSegments) {
        T y = shared[tid];
        T scan = y;
        #pragma unroll
        for(int offset = 1; offset < NumSegments; offset <<= 1)
            scan = shfl_add(scan, offset, NumSegments);
        shared[tid] = scan - y;
        if(NumSegments - 1 == tid) shared[NumSegments] = scan;
    }
    __syncthreads();

    // Add the scanned partials back in and convert to exclusive scan.
    scan += shared[segment];
    scan -= val;
    max = shared[NumSegments];
//    __syncthreads();

//printf("%u %u\n", threadIdx.x, scan);
    return scan;
}

template<unsigned int NT, typename T = unsigned int>
struct CTAScan {
	enum { Size = NT, NumSegments = WARP_SIZE, SegSize = NT / NumSegments };
	enum { Capacity = NumSegments + 1 };
	struct Storage { T shared[Capacity + 1]; };

	MGPU_DEVICE static T Scan(unsigned int tid, T x, Storage& storage, T* total) {
	
		// Define WARP_SIZE segments that are NT / WARP_SIZE large.
		// Each warp makes log(SegSize) shfl_add calls.
		// The spine makes log(WARP_SIZE) shfl_add calls.
		int lane = (SegSize - 1) & tid;
		int segment = tid / SegSize;

		// Scan each segment using shfl_add.
		int scan = x;
		#pragma unroll
		for(int offset = 1; offset < SegSize; offset *= 2)
			scan = shfl_add(scan, offset, SegSize);

		// Store the reduction (last element) of each segment into storage.
		if(SegSize - 1 == lane) storage.shared[segment] = scan;
		__syncthreads();

		// Warp 0 does a full shfl warp scan on the partials. The total is
		// stored to shared[NumSegments]. (NumSegments = WARP_SIZE)
		if(tid < NumSegments) {
			int y = storage.shared[tid];
			int scan = y;
			#pragma unroll
			for(int offset = 1; offset < NumSegments; offset *= 2)
				scan = shfl_add(scan, offset, NumSegments);
			storage.shared[tid] = scan - y;
			if(NumSegments - 1 == tid) storage.shared[NumSegments] = scan;
		}
		__syncthreads();

		// Add the scanned partials back in and convert to exclusive scan.
		scan += storage.shared[segment];
		scan -= x;
		*total = storage.shared[NumSegments];
		__syncthreads();

		return scan;
	}
};
}

}

#endif

