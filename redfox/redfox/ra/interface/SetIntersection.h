/*!
	\file SetIntersection.h
	\date Monday June 8, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the Set Intersection family of CUDA functions.
*/

#ifndef SET_INTERSECTION_H_INCLUDED
#define SET_INTERSECTION_H_INCLUDED

#include <redfox/ra/interface/SetCommon.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <hydrazine/cuda/Cuda.h>
#include <hydrazine/cuda/DeviceQueries.h>
#include <hydrazine/implementation/macros.h>
#include <hydrazine/implementation/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)

namespace gpu
{

namespace algorithms
{

namespace cuda
{
	template< bool sync >
	__device__ unsigned int ctaExclusiveReduction( unsigned int value, 
		unsigned int size, unsigned int& max )
	{
		const unsigned int n = 2 * THREADS;
		__shared__ unsigned int shared[ n + n/NUM_BANKS ];
		
		/*
		shared[ THREAD_ID() ] = value;
		
		__syncthreads();
		
		if( THREAD_ID() == 0 )
		{
			unsigned int count = 0;
			for( unsigned int i = 0; i < THREADS; ++i )
			{
				unsigned int temp = shared[i];
				shared[i] = count;
				count += temp;
			}
			shared[ THREADS ] = count;
		}
		
		__syncthreads();
		
		max = shared[ THREADS ];
		return shared[ THREAD_ID() ];
		*/
		
		/*
		unsigned int id = THREAD_ID();
		shared[id] = 0;
		id += size;
		if( sync ) __syncthreads();
		shared[id] = value;
		if( sync ) __syncthreads();

		for( unsigned int i = 1; i <= size; i *= 2 )
		{
			shared[ id ] += shared[ id - i ];
			if( sync ) __syncthreads();
		}
		
		max = shared[ 2 * size - 1 ];
		return shared[ id - 1 ];
		
		*/
		
		unsigned int thid = THREAD_ID();

		unsigned int ai = thid;
		unsigned int bi = thid + ( n / 2 );

		unsigned int bankOffsetA = CONFLICT_FREE_OFFSET( ai );
		unsigned int bankOffsetB = CONFLICT_FREE_OFFSET( bi );

		shared[ ai + bankOffsetA ] = value;
		shared[ bi + bankOffsetB ] = 0; 

		unsigned int offset = 1;

		for( unsigned int d = n / 2; d > 0; d >>= 1)
		{
			if( sync ) __syncthreads();

			if( thid < d )      
			{
				unsigned int ai = offset * ( 2 * thid + 1 ) - 1;
				unsigned int bi = offset * ( 2 * thid + 2 ) - 1;

				ai += CONFLICT_FREE_OFFSET( ai );
				bi += CONFLICT_FREE_OFFSET( bi );

				shared[ bi ] += shared[ ai ];
			}

			offset *= 2;
		}

		if (thid == 0)
		{
			unsigned int index = n - 1;
			index += CONFLICT_FREE_OFFSET( index );
			shared[ index ] = 0;
		}   

		for( unsigned int d = 1; d < n; d *= 2 )
		{
			offset /= 2;

			if( sync ) __syncthreads();

			if( thid < d )
			{
				unsigned int ai = offset * ( 2 * thid + 1 ) - 1;
				unsigned int bi = offset * ( 2 * thid + 2 ) - 1;

				ai += CONFLICT_FREE_OFFSET( ai) ;
				bi += CONFLICT_FREE_OFFSET( bi );

				unsigned int t  = shared[ ai ];
				shared[ ai ] = shared[ bi ];
				shared[ bi ] += t;
			}
		}

		if( sync ) __syncthreads();
		
		max = shared[ n - 1 + CONFLICT_FREE_OFFSET( n - 1 ) ];
		return shared[ ai + bankOffsetA ];
	}
	
	template< typename Key, typename Value, typename Compare >
	__device__ void findWithBinarySearch( POINTER begin, POINTER end, 
		const PAIR& element, POINTER& result, Compare comp )
	{
		size_t low = 0;
		size_t high = end - begin;
		
		while( low < high )
		{
			size_t median = ( low + ( high - low ) / 2 );
			if( comp( begin[ median ].first, element.first ) )
			{
				low = median + 1;
			}
			else
			{
				high = median;
			}
		}
		
		if( comp( begin[ low ].first, element.first ) 
			|| comp( element.first, begin[ low ].first ) )
		{
			low = end - begin;
		}
		
		result = begin + low;		
	}
	
	template< bool sync, typename Key, typename Value, 
		typename Operator, typename Compare >
	__device__ unsigned int sharedMemoryIntersection( POINTER results, 
		POINTER smaller, size_t smallerSize, POINTER larger, size_t largerSize, 
		bool swap, Operator op, Compare comp )
	{	
		PAIR element;
		POINTER match = larger;
		unsigned int index = 0;
		unsigned int maxIndex = 0;
		bool keysMatch = false;
		
		if( THREAD_ID() < smallerSize )
		{
			element = smaller[ THREAD_ID() ];
			findWithBinarySearch( larger, larger + largerSize, 
				element, match, comp );
			keysMatch = match != ( larger + largerSize );
		}

		index = ctaExclusiveReduction< sync >( keysMatch, smallerSize, 
			maxIndex );
		
		if( keysMatch )
		{		
			element = swap ? op( *match, element ) : op( element, *match );
			results[ index ] = element;
		}
		
		if( sync ) __syncthreads();
		
		return maxIndex;
	}
	
	template< bool sync, typename Key, typename Value, 
		typename Operator, typename Compare >
	__global__ void computeIntersection( POINTER _resultBegin, 
		SIZE_POINTER histogram, CONST_POINTER largerBegin, 
		SIZE_POINTER largerRange, CONST_POINTER smallerBegin, 
		SIZE_POINTER smallerRange, bool swap, Operator op, Compare comp )
	{
		__shared__ int _left[ THREADS * sizeof( PAIR ) / sizeof( int ) ];
		__shared__ int _right[ THREADS * sizeof( PAIR ) / sizeof( int ) ];		
		__shared__ int _result[ THREADS * sizeof( PAIR ) / sizeof( int ) ];
		
		POINTER left = ( POINTER ) _left;
		POINTER right = ( POINTER ) _right;
		POINTER result = ( POINTER ) _result;
		
		CONST_POINTER leftBegin = largerBegin + largerRange[ CTA_ID() ];
		CONST_POINTER leftEnd = largerBegin + largerRange[ CTA_ID() + 1 ];
		CONST_POINTER rightBegin = smallerBegin + smallerRange[ CTA_ID() ];
		CONST_POINTER rightEnd = smallerBegin + smallerRange[ CTA_ID() + 1 ];
				
		POINTER resultBegin = _resultBegin + smallerRange[ CTA_ID() ];
		
		if( leftBegin == leftEnd || rightBegin == rightEnd )
		{
			if( THREAD_ID() == 0 )
			{
				histogram[ CTA_ID() ] = 0;
			}
			return;
		}
		
		POINTER o = resultBegin;
		CONST_POINTER r = rightBegin;
		CONST_POINTER l = leftBegin;

		bool fetchLeft = true;
		bool fetchRight = true;
		
		size_t leftSize = 0;
		size_t rightSize = 0;

		while( true )
		{
			if( fetchLeft )
			{
				CONST_POINTER copyEnd = MIN( l + THREADS, leftEnd );
				leftSize = copyEnd - l;
				hydrazine::cuda::memcpyCta( left, l, 
					leftSize * sizeof( PAIR ) );
				l = copyEnd;
			}
			
			if( fetchRight )
			{
				CONST_POINTER copyEnd = MIN( r + THREADS, rightEnd );
				rightSize = copyEnd - r;
				hydrazine::cuda::memcpyCta( right, r, 
					rightSize * sizeof( PAIR ) );
				r = copyEnd;
			}
			
			if( sync ) __syncthreads();
			
			bool leftLarger = comp( right[ rightSize - 1 ].first, 
				left[ leftSize - 1 ].first );
			bool rightLarger = comp( left[ leftSize - 1 ].first, 
				right[ rightSize - 1 ].first );
			
			POINTER larger = ( leftLarger ) ? left : right;
			size_t largerSize = ( leftLarger ) ? leftSize : rightSize;
			POINTER smaller = ( leftLarger ) ? right : left;
			size_t smallerSize = ( leftLarger ) ? rightSize : leftSize;

			fetchLeft = !leftLarger;
			fetchRight = !rightLarger;
			bool swapInternal = swap ? !leftLarger : leftLarger;
			
			unsigned int joinedElements = sharedMemoryIntersection< sync >( 
				result, smaller, smallerSize, larger, largerSize, swapInternal, 
				op, comp );

			hydrazine::cuda::memcpyCta( o, result, 
				joinedElements * sizeof( PAIR ) );
			o += joinedElements;
			
			if( fetchLeft && l == leftEnd ) break;
			if( fetchRight && r == rightEnd ) break;
		}	
		
		if( THREAD_ID() == 0 )
		{
			size_t results = o - resultBegin;
			histogram[ CTA_ID() ] = results;
		}
	}
	
	template< typename Key, typename Value, 
		typename Operator, typename Compare >
	size_t setIntersection( POINTER resultBegin, POINTER resultEnd, 
		CONST_POINTER leftBegin, CONST_POINTER leftEnd, 
		CONST_POINTER rightBegin, CONST_POINTER rightEnd, Operator op, 
		Compare comp )
	{
		const size_t pairsPerCta = CEIL_DIV( BYTES_PER_CTA, sizeof( PAIR ) );
		size_t leftSize = leftEnd - leftBegin;
		size_t rightSize = rightEnd - rightBegin;
		size_t resultSize = resultEnd - resultBegin;
		
		report( "Running GPU set intersection (left " << leftSize << ") (right "
			<< rightSize << ") (results " << resultSize << ")" );
		report( " Left: " << hydrazine::toFormattedString( leftBegin, leftEnd, 
			FormatPair< Key, Value >() ) );
		report( " Right: " << hydrazine::toFormattedString( rightBegin, 
			rightEnd, FormatPair< Key, Value >() ) );
		
		size_t largerSize = std::max( leftSize, rightSize );
		size_t smallerSize = std::min( leftSize, rightSize );
		bool swap = leftSize < rightSize;
		
		if( smallerSize == 0 ){	return 0; }
		
		CONST_POINTER smallerBegin = ( leftSize < rightSize ) 
			? leftBegin : rightBegin;
		CONST_POINTER smallerEnd = ( leftSize < rightSize ) 
			? leftEnd : rightEnd;
		CONST_POINTER largerBegin = ( leftSize < rightSize ) 
			? rightBegin : leftBegin;
		CONST_POINTER largerEnd = ( leftSize < rightSize ) 
			? rightEnd : leftEnd;
		
		unsigned int ctas = CEIL_DIV( largerSize, pairsPerCta );
		const unsigned int threads = THREADS;
		
		report( " CTAs " << ctas << ", Threads " << threads );
		
		thrust::device_ptr< size_t > largerRange 
			= thrust::device_malloc< size_t >( ctas + 1 );
		thrust::device_ptr< size_t > smallerRange 
			= thrust::device_malloc< size_t >( ctas + 1 );
		thrust::device_ptr< size_t > resultRange 
			= thrust::device_malloc< size_t >( ctas + 1 );
		thrust::device_ptr< size_t > resultHistogram 
			= thrust::device_malloc< size_t >( ctas + 1 );
		thrust::device_ptr< PAIR > tempResults 
			= thrust::device_malloc< PAIR >( resultSize );
		
		determineRange( smallerRange.get(), largerRange.get(), smallerBegin, 
			smallerEnd, largerBegin, largerEnd, ctas, comp );	
				
		if( threads > hydrazine::cuda::warpSize() )
		{
			computeIntersection< true ><<< ctas, threads >>>( tempResults.get(), 
				resultHistogram.get(), largerBegin, largerRange.get(), 
				smallerBegin, smallerRange.get(), swap, op, comp );
		}
		else
		{
			computeIntersection< false ><<< ctas, threads >>>( 
				tempResults.get(), resultHistogram.get(), largerBegin, 
				largerRange.get(), smallerBegin, smallerRange.get(), 
				swap, op, comp );
		}
		
		hydrazine::cuda::check( cudaGetLastError() );
		report( " Result histogram is " << hydrazine::toString( resultHistogram,
			resultHistogram + ctas + 1 ) );
		
		thrust::exclusive_scan( resultHistogram, resultHistogram + ctas + 1, 
			resultRange );
		gatherResults<<< ctas, threads >>>( resultBegin, tempResults.get(), 
			smallerRange.get(), resultHistogram.get(), resultRange.get() );
		hydrazine::cuda::check( cudaGetLastError() );
		
		size_t intersectionSize = resultRange[ ctas ];
		
		report( " Result (" << intersectionSize << "): " 
			<< hydrazine::toFormattedString( resultBegin, 
			resultBegin + intersectionSize, FormatPair< Key, Value >() ) );
		
		thrust::device_free( largerRange );
		thrust::device_free( smallerRange );
		thrust::device_free( resultHistogram );
		thrust::device_free( resultRange );
		thrust::device_free( tempResults );
		
		return intersectionSize;
	}
		
}	

}

}

#undef NUM_BANKS
#undef LOG_NUM_BANKS
#undef CONFLICT_FREE_OFFSET

#endif

