/*!
	\file MapOperator.h
	\date Wednesday July 8, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the Map family of cuda functions.
*/

#ifndef MAP_OPERATOR_H_INCLUDED
#define MAP_OPERATOR_H_INCLUDED

#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <hydrazine/cuda/Cuda.h>
#include <hydrazine/implementation/macros.h>
#include <hydrazine/implementation/debug.h>

#define PAIR std::pair< Key, Value >
#define POINTER PAIR*
#define CONST_POINTER const POINTER

#define LEFT_PAIR std::pair< Key1, Key2 >
#define LEFT_POINTER LEFT_PAIR*
#define CONST_LEFT_POINTER const LEFT_POINTER

#define SWAPPED_PAIR std::pair< Key2, Key1 >
#define SWAPPED_POINTER SWAPPED_PAIR*
#define CONST_SWAPPED_POINTER const SWAPPED_POINTER

#define RIGHT_PAIR std::pair< Key2, Value2 >
#define RIGHT_POINTER RIGHT_PAIR*
#define CONST_RIGHT_POINTER const RIGHT_POINTER

#define RESULT_PAIR std::pair< Key1, Value2 >
#define RESULT_POINTER RESULT_PAIR*
#define CONST_RESULT_POINTER const RESULT_POINTER

#define SIZE_POINTER size_t*

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
	
	template< typename Key, typename Value >
	class SwapKeyValue
	{
		public:
			PAIR operator()( const PAIR& p )
			{
				return PAIR( p.second, p.first );
			}
	};
	
	template< typename Key, typename Value, typename Compare >
	class ComparePair
	{
		public:
			Compare _comp;
	
		public:
			ComparePair( const Compare& comp ) : _comp( comp ) {}
	
			bool operator()( const PAIR& left, const PAIR& right )
			{
				return _comp( left.first, right.first );
			}
	};

	template< typename Key, typename Value, typename Compare >
	class EqualPair
	{
		public:
			Compare _comp;
	
		public:
			EqualPair( const Compare& comp ) : _comp( comp ) {}
	
			bool operator()( const PAIR& left, const PAIR& right )
			{
				return !_comp( left.first, right.first ) 
					&& !_comp( right.first, left.first );
			}
	};

	
	template< typename Key, typename Value >
	class CollectLeftKeyRightValue
	{
		public:
			PAIR operator()( const PAIR& left, const PAIR& right )
			{
				return PAIR( left.second, right.second );
			}
	};
	
	template< typename Key1, typename Key2, typename Value2, 
		typename Operator, typename Compare >
	__global__ void computeIntersectionWithDuplicates( 
		RESULT_POINTER _resultBegin, 
		SIZE_POINTER histogram, CONST_SWAPPED_POINTER _leftBegin, 
		SIZE_POINTER leftRange, CONST_RIGHT_POINTER _rightBegin, 
		SIZE_POINTER rightRange, Operator op, Compare comp )
	{
		if( THREAD_ID() != 0 ) return;
		
		CONST_SWAPPED_POINTER leftBegin = _leftBegin + leftRange[ CTA_ID() ];
		CONST_SWAPPED_POINTER leftEnd = _leftBegin + leftRange[ CTA_ID() + 1 ];
		CONST_RIGHT_POINTER rightBegin = _rightBegin + rightRange[ CTA_ID() ];
		CONST_RIGHT_POINTER rightEnd = _rightBegin + rightRange[ CTA_ID() + 1 ];
				
		RESULT_POINTER resultBegin = _resultBegin + leftRange[ CTA_ID() ];
		
		RESULT_POINTER o = resultBegin;
		CONST_RIGHT_POINTER r = rightBegin;
		CONST_SWAPPED_POINTER l = leftBegin;

		while( l != leftEnd && r != rightEnd )
		{
			if( comp( l->first, r->first ) )
			{
				++l;
				continue;
			}
			
			if( comp( r->first, l->first ) )
			{
				++r;
				continue;
			}
			
			*o = op( *l, *r);
			l = l + 1;
			++o;
		}

		size_t results = o - resultBegin;
		histogram[ CTA_ID() ] = results;
	}
	
	template< typename Key1, typename Key2, typename Value2, 
		typename Operator, typename Compare >
	size_t setIntersectionWithDuplicates( RESULT_POINTER resultBegin, 
		RESULT_POINTER resultEnd, CONST_SWAPPED_POINTER leftBegin, 
		CONST_SWAPPED_POINTER leftEnd, CONST_RIGHT_POINTER rightBegin, 
		CONST_RIGHT_POINTER rightEnd, Operator op, Compare comp )
	{
		const size_t pairsPerCta = CEIL_DIV( BYTES_PER_CTA, 
			sizeof( SWAPPED_PAIR ) );
		size_t leftSize = leftEnd - leftBegin;
		size_t rightSize = rightEnd - rightBegin;
		size_t resultSize = resultEnd - resultBegin;
	
		report( "Running GPU set intersection with duplicates allowed (left " 
			<< leftSize << ") (right " << rightSize << ") (results " 
			<< resultSize << ")" );
		report( " Left: " << hydrazine::toFormattedString( leftBegin, leftEnd, 
			FormatPair< Key2, Key1 >() ) );
		report( " Right: " << hydrazine::toFormattedString( rightBegin, 
			rightEnd, FormatPair< Key2, Value2 >() ) );
		
		if( leftSize == 0 || rightSize == 0 ){ return 0; }
		
		unsigned int ctas = CEIL_DIV( std::min( leftSize, rightSize ), 
			pairsPerCta );
		const unsigned int threads = THREADS;
		
		report( " CTAs " << ctas << ", Threads " << threads );
		
		thrust::device_ptr< size_t > leftRange 
			= thrust::device_malloc< size_t >( ctas + 1 );
		thrust::device_ptr< size_t > rightRange 
			= thrust::device_malloc< size_t >( ctas + 1 );
		thrust::device_ptr< size_t > resultRange 
			= thrust::device_malloc< size_t >( ctas + 1 );
		thrust::device_ptr< size_t > resultHistogram 
			= thrust::device_malloc< size_t >( ctas + 1 );
		thrust::device_ptr< RESULT_PAIR > tempResults 
			= thrust::device_malloc< RESULT_PAIR >( resultSize );
		
		determineRange( leftRange.get(), rightRange.get(), leftBegin, 
			leftEnd, rightBegin, rightEnd, ctas, comp );	
		
		computeIntersectionWithDuplicates<<< ctas, threads >>>( 
			tempResults.get(), resultHistogram.get(), leftBegin, 
			leftRange.get(), rightBegin, rightRange.get(), op, comp );
		
		hydrazine::cuda::check( cudaGetLastError() );
		report( " Result histogram is " << hydrazine::toString( resultHistogram,
			resultHistogram + ctas + 1 ) );
		
		thrust::exclusive_scan( resultHistogram, resultHistogram + ctas + 1, 
			resultRange );
		gatherResults<<< ctas, threads >>>( resultBegin, tempResults.get(), 
			leftRange.get(), resultHistogram.get(), resultRange.get() );
		hydrazine::cuda::check( cudaGetLastError() );
		
		size_t intersectionSize = resultRange[ ctas ];
		
		report( " Result (" << intersectionSize << "): " 
			<< hydrazine::toFormattedString( resultBegin, 
			resultBegin + intersectionSize, FormatPair< Key1, Value2 >() ) );
		
		thrust::device_free( leftRange );
		thrust::device_free( rightRange );
		thrust::device_free( resultHistogram );
		thrust::device_free( resultRange );
		thrust::device_free( tempResults );
		
		return intersectionSize;
	}
	
	template< typename Key1, typename Key2, typename Value2, typename Compare >
	size_t map( RESULT_POINTER resultBegin, RESULT_POINTER resultEnd, 
		CONST_LEFT_POINTER leftBegin, CONST_LEFT_POINTER leftEnd,
		CONST_RIGHT_POINTER rightBegin, CONST_RIGHT_POINTER rightEnd, 
		Compare comp )
	{
		typedef thrust::device_ptr< RESULT_PAIR > result_pointer;
		typedef thrust::device_ptr< LEFT_PAIR > pointer;
		typedef thrust::device_ptr< const LEFT_PAIR > const_pointer;
		typedef thrust::device_ptr< SWAPPED_PAIR > swapped;
		typedef thrust::device_ptr< const SWAPPED_PAIR > const_swapped;
		
		size_t leftSize = leftEnd - leftBegin;
		size_t rightSize = rightEnd - rightBegin;
		size_t resultSize = resultEnd - resultBegin;
		
		swapped tempLeft = thrust::device_malloc< SWAPPED_PAIR >( leftSize );
		
		report( "Running GPU map (left " << leftSize << ") (right "
			<< rightSize << ") (results " << resultSize << ")" );
		report( " Left: " << hydrazine::toFormattedString( leftBegin, leftEnd, 
			FormatPair< Key1, Key2 >() ) );
		report( " Right: " << hydrazine::toFormattedString( rightBegin, 
			rightEnd, FormatPair< Key2, Value2 >() ) );
		
		thrust::transform( const_pointer( leftBegin ), 
			const_pointer( leftEnd ), tempLeft, SwapKeyValue< Key1, Key2 >() );
		
		report( " Transformed Left: " << hydrazine::toFormattedString( 
			tempLeft.get(), tempLeft.get() + leftSize, 
			FormatPair< Key2, Key1 >() ) );
		
		thrust::sort( tempLeft, tempLeft + leftSize, 
			ComparePair< Key2, Key1, Compare >( comp ) );
		
		report( " Sorted Left: " << hydrazine::toFormattedString( 
			tempLeft.get(), tempLeft.get() + leftSize, 
			FormatPair< Key2, Key1 >() ) );
		
		size_t result = setIntersectionWithDuplicates( resultBegin, 
			resultEnd, tempLeft.get(), 
			tempLeft.get() + leftSize, rightBegin, rightEnd, 
			CollectLeftKeyRightValue< Key1, Value2 >(), comp );
		
		thrust::sort( result_pointer( resultBegin ), 
			result_pointer( resultBegin ) + result, 
			ComparePair< Key1, Value2, Compare >( comp ) );
				
		report( " Sorted Result (" << result << "): " 
			<< hydrazine::toFormattedString( resultBegin, 
			resultBegin + result, FormatPair< Key1, Value2 >() ) );
				
		thrust::device_free( tempLeft );
		
		return result;
	}
		
}	

}

}

#undef PAIR
#undef POINTER
#undef CONST_POINTER

#undef LEFT_PAIR
#undef LEFT_POINTER
#undef CONST_LEFT_POINTER

#undef SWAPPED_PAIR
#undef SWAPPED_POINTER
#undef CONST_SWAPPED_POINTER

#undef RIGHT_PAIR
#undef RIGHT_POINTER
#undef CONST_RIGHT_POINTER

#undef RESULT_PAIR
#undef RESULT_POINTER
#undef CONST_LEFT_POINTER

#undef SIZE_POINTER


#endif

