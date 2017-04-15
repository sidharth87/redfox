/*!
	\file SetDifference.h
	\date Tuesday June 9, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the Set Difference family of CUDA functions.
*/

#ifndef SET_DIFFERENCE_H_INCLUDED
#define SET_DIFFERENCE_H_INCLUDED

#include <redfox/ra/interface/SetCommon.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <hydrazine/cuda/Cuda.h>
#include <hydrazine/implementation/macros.h>
#include <hydrazine/implementation/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace gpu
{

namespace algorithms
{

namespace cuda
{
	
	template< typename Key, typename Value, 
		typename Operator, typename Compare >
	__global__ void computeDifference( POINTER _resultBegin, 
		SIZE_POINTER histogram, CONST_POINTER _leftBegin, 
		SIZE_POINTER leftRange, CONST_POINTER _rightBegin, 
		SIZE_POINTER rightRange, Operator op, Compare comp )
	{
		CONST_POINTER leftBegin = _leftBegin + leftRange[ CTA_ID() ];
		CONST_POINTER leftEnd = _leftBegin + leftRange[ CTA_ID() + 1 ];
		CONST_POINTER rightBegin = _rightBegin + rightRange[ CTA_ID() ];
		CONST_POINTER rightEnd = _rightBegin + rightRange[ CTA_ID() + 1 ];
		
		POINTER resultBegin = _resultBegin + leftRange[ CTA_ID() ];
		
		POINTER o = resultBegin;
		CONST_POINTER r = rightBegin;
		CONST_POINTER l = leftBegin;
		
		if( THREAD_ID() == 0 )
		{
			while( l != leftEnd && r != rightEnd )
			{
				if( comp( l->first, r->first ) ){ *o = *l; ++l; ++o; continue; }
				if( comp( r->first, l->first ) ){ ++r; continue; }
				++l; ++r;;
			}
			
			for( ; l != leftEnd; ++l )
			{
				*o = *l;
				++o;
			}
			
			size_t results = o - resultBegin;
			histogram[ CTA_ID() ] = results;
		}
	}
	
	template< typename Key, typename Value, 
		typename Operator, typename Compare >
	size_t setDifference( POINTER resultBegin, POINTER resultEnd, 
		CONST_POINTER leftBegin, CONST_POINTER leftEnd, 
		CONST_POINTER rightBegin, CONST_POINTER rightEnd, Operator op, 
		Compare comp )
	{
		const size_t pairsPerCta = CEIL_DIV( BYTES_PER_CTA, sizeof( PAIR ) );
		size_t leftSize = leftEnd - leftBegin;
		size_t rightSize = rightEnd - rightBegin;
		size_t resultSize = resultEnd - resultBegin;
	
		report( "Running GPU set difference (left " << leftSize << ") (right "
			<< rightSize << ") (results " << resultSize << ")" );
		report( " Left: " << hydrazine::toFormattedString( leftBegin, leftEnd, 
			FormatPair< Key, Value >() ) );
		report( " Right: " << hydrazine::toFormattedString( rightBegin, 
			rightEnd, FormatPair< Key, Value >() ) );
		
		size_t largerSize = std::max( leftSize, rightSize );
		size_t smallerSize = std::min( leftSize, rightSize );
		
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
		
		SIZE_POINTER leftRange = ( leftSize < rightSize ) 
			? smallerRange.get() : largerRange.get();
		SIZE_POINTER rightRange = ( leftSize < rightSize ) 
			? largerRange.get() : smallerRange.get();
		
		computeDifference<<< ctas, threads >>>( tempResults.get(), 
			resultHistogram.get(), leftBegin, leftRange, 
			rightBegin, rightRange, op, comp );
		hydrazine::cuda::check( cudaGetLastError() );
		report( " Result histogram is " << hydrazine::toString( resultHistogram,
			resultHistogram + ctas + 1 ) );
			
		thrust::exclusive_scan( resultHistogram, resultHistogram + ctas + 1, 
			resultRange );
		gatherResults<<< ctas, threads >>>( resultBegin, tempResults.get(), 
			leftRange, resultHistogram.get(), resultRange.get() );
		hydrazine::cuda::check( cudaGetLastError() );
		
		size_t differenceSize = resultRange[ ctas ];
		
		report( " Result (" << differenceSize << "): " 
			<< hydrazine::toFormattedString( resultBegin, 
			resultBegin + differenceSize, FormatPair< Key, Value >() ) );
		
		thrust::device_free( largerRange );
		thrust::device_free( smallerRange );
		thrust::device_free( resultHistogram );
		thrust::device_free( resultRange );
		thrust::device_free( tempResults );
		
		return differenceSize;
	}
		
}	

}

}

#endif

