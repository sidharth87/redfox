/*!
	\file SetIntersectionWithDecompression.h
	\date Thursday July 16, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the Set Intersection family of CUDA functions
		with support for key decompression.
*/

#ifndef SET_INTERSECTION_WITH_DECOMPRESSION_H_INCLUDED
#define SET_INTERSECTION_WITH_DECOMPRESSION_H_INCLUDED

#include <hydrazine/implementation/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

#define PAIR std::pair< Key, Value >
#define POINTER PAIR*
#define CONST_POINTER const POINTER
#define SIZE_POINTER size_t*

namespace gpu
{

namespace algorithms
{

namespace cuda
{
	#ifndef NDEBUG

	template< typename Key, typename Value >
	class FormatCompressedPair
	{
		public:
			std::string operator()( CONST_POINTER _p )
			{
				POINTER p = const_cast< POINTER >( _p );
				std::stringstream stream;
				thrust::device_ptr< PAIR > pointer( p );
				PAIR pair = *pointer;
				stream << pair.first << ",(" << pair.second.first << "," 
					<< pair.second.second << ")";
				return stream.str();
			}
	};

	#endif
	
	template< typename Key, typename Value, typename Decompressor >
	class KeyDecompressor
	{
		public:
			typedef std::pair< Key, Value > Pair;
			typedef std::pair< Key, std::pair< Key, Value > > DecompressedPair;

		private:
			Decompressor _decomp;
			
		public:
			KeyDecompressor( const Decompressor& d ) : _decomp( d ) {}
			
			DecompressedPair operator()( const Pair& p )
			{
				return DecompressedPair( _decomp.decompress( p.first ), p );
			}	
	};
	
	template< typename Key, typename Value, typename Compare >
	class DecompressedCompare
	{
		public:
			typedef std::pair< Key, Value > Pair;
			typedef std::pair< Key, std::pair< Key, Value > > DecompressedPair;

		private:
			Compare _comp;
			
		public:
			DecompressedCompare( const Compare& c ) : _comp( c ) {}
			
			bool operator()( const DecompressedPair& one, 
				const DecompressedPair& two )
			{
				return _comp( one.first, two.first );
			}
	};
	
	template< typename Key, typename Value, typename Compare >
	class CompressedCompare
	{
		public:
			typedef std::pair< Key, Value > Pair;

		private:
			Compare _comp;
			
		public:
			CompressedCompare( const Compare& c ) : _comp( c ) {}
			
			bool operator()( const Pair& one, const Pair& two )
			{
				return _comp( one.first, two.first );
			}
	};
	
	template< typename Key, typename Value, typename Operator >
	class DecompressedOp
	{
		public:
			typedef std::pair< Key, Value > Pair;
			typedef std::pair< Key, std::pair< Key, Value > > DecompressedPair;

		private:
			Operator _op;
			
		public:
			DecompressedOp( const Operator& o ) : _op( o ) {}
			
			DecompressedPair operator()( const DecompressedPair& one, 
				const DecompressedPair& two )
			{
				return DecompressedPair( one.first, _op( one.second, 
					two.second ) );
			}
	};
	
	template< typename Key, typename Value >
	class KeyCompressor
	{
		public:
			typedef std::pair< Key, Value > Pair;
			typedef std::pair< Key, std::pair< Key, Value > > DecompressedPair;

		public:
			KeyCompressor( ) {}
			
			Pair operator()( const DecompressedPair& one )
			{
				return one.second;
			}
	};
	
	template< typename Key, typename Value, 
		typename Operator, typename Compare >
	__global__ void gpuIntersectionWithDecompression( 
		POINTER _resultBegin, SIZE_POINTER resultRange,
		SIZE_POINTER histogram, CONST_POINTER _leftBegin, 
		SIZE_POINTER leftRange, CONST_POINTER _rightBegin, 
		SIZE_POINTER rightRange, Operator op, Compare comp )
	{
		if( THREAD_ID() != 0 ) return;
		
		CONST_POINTER leftBegin = _leftBegin + leftRange[ CTA_ID() ];
		CONST_POINTER leftEnd = _leftBegin + leftRange[ CTA_ID() + 1 ];
		CONST_POINTER rightBegin = _rightBegin + rightRange[ CTA_ID() ];
		CONST_POINTER rightEnd = _rightBegin + rightRange[ CTA_ID() + 1 ];
		
		POINTER resultBegin = _resultBegin + resultRange[ CTA_ID() ];
		
		POINTER o = resultBegin;
		CONST_POINTER r = rightBegin;
		CONST_POINTER l = leftBegin;

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
			
			*o = op( *l, *r );
			++l;
			++o;
		}

		size_t results = o - resultBegin;
		histogram[ CTA_ID() ] = results;
	}
	
	template< typename Key, typename Value, 
		typename Operator, typename Compare >
	size_t computeIntersectionWithDecompression( POINTER resultBegin, 
		POINTER resultEnd, CONST_POINTER leftBegin, 
		CONST_POINTER leftEnd, CONST_POINTER rightBegin, 
		CONST_POINTER rightEnd, Operator op, Compare comp )
	{
		typedef thrust::device_ptr< size_t > SizePtr;
		
		const size_t pairsPerCta = CEIL_DIV( BYTES_PER_CTA, 
			sizeof( PAIR ) );
		size_t leftSize = leftEnd - leftBegin;
		size_t rightSize = rightEnd - rightBegin;
		size_t resultSize = resultEnd - resultBegin;
	
		if( leftSize == 0 || rightSize == 0 ){ return 0; }
		
		unsigned int ctas = std::max( std::min( leftSize, rightSize), 
			CEIL_DIV( std::max( leftSize, rightSize ), pairsPerCta ) );
		const unsigned int threads = THREADS;
		report( "  CTAs " << ctas );
		
		SizePtr leftRange = thrust::device_malloc< size_t >( ctas + 1 );
		SizePtr rightRange = thrust::device_malloc< size_t >( ctas + 1 );
		SizePtr resultRange = thrust::device_malloc< size_t >( ctas + 1 );
		SizePtr resultHistogram = thrust::device_malloc< size_t >( ctas + 1 );
		thrust::device_ptr< PAIR > tempResults 
			= thrust::device_malloc< PAIR >( resultSize );
		
		determineRange( leftRange.get(), rightRange.get(), leftBegin, 
			leftEnd, rightBegin, rightEnd, ctas, comp );	
		
		bool leftSmaller = leftSize < rightSize;
		thrust::device_ptr< size_t > tempResultRange = leftSmaller 
			? rightRange : leftRange;
		
		report( "  Result range is " << hydrazine::toString( 
			SizePtr( tempResultRange ), 
			SizePtr( tempResultRange + ctas ) + 1 ) );
		
		gpuIntersectionWithDecompression<<< ctas, threads >>>( 
			tempResults.get(), tempResultRange.get(), 
			resultHistogram.get(), leftBegin, 
			leftRange.get(), rightBegin, rightRange.get(), op, comp );
				
		hydrazine::cuda::check( cudaGetLastError() );
		report( " Result histogram is " << hydrazine::toString( resultHistogram,
			resultHistogram + ctas + 1 ) );
		
		thrust::exclusive_scan( resultHistogram, resultHistogram + ctas + 1, 
			resultRange );
		
		/*
		for( unsigned int i = 0; i < ctas; ++i )
		{
			report( " Copying from " << (tempResults + tempResultRange[i]).get()
				<< " to " << (resultBegin + resultRange[i]) << ", " 
				<< resultHistogram[i] * sizeof( PAIR ) << " bytes " );
			cudaMemcpy( resultBegin + resultRange[i], 
				(tempResults + tempResultRange[i]).get(), 
				resultHistogram[i] * sizeof( PAIR ), cudaMemcpyDeviceToDevice );
		}
		*/

		gatherResults<<< ctas, threads >>>( resultBegin, tempResults.get(), 
			tempResultRange.get(), resultHistogram.get(), resultRange.get() );
		hydrazine::cuda::check( cudaGetLastError() );
		
		size_t intersectionSize = resultRange[ ctas ];
				
		thrust::device_free( leftRange );
		thrust::device_free( rightRange );
		thrust::device_free( resultHistogram );
		thrust::device_free( resultRange );
		thrust::device_free( tempResults );
		
		return intersectionSize;
	}
	
	template< typename Key, typename Value, typename Operator, 
		typename Compare, typename LeftDecompressor, 
		typename RightDecompressor >
	size_t setIntersection( POINTER resultBegin, POINTER resultEnd, 
		CONST_POINTER leftBegin, CONST_POINTER leftEnd, 
		CONST_POINTER rightBegin, CONST_POINTER rightEnd, Operator op, 
		Compare comp, LeftDecompressor leftDecompressor, 
		RightDecompressor rightDecompressor )
	{
		typedef std::pair< Key, Value > Pair;
		typedef std::pair< Key, Pair > DecompressedPair;
		typedef thrust::device_ptr< DecompressedPair > DecompressedPtr;
		typedef thrust::device_ptr< Pair > PairPtr;
		typedef thrust::device_ptr< const Pair > ConstPairPtr;
		
		size_t leftSize = leftEnd - leftBegin;
		size_t rightSize = rightEnd - rightBegin;
		size_t resultSize = resultEnd - resultBegin;
		
		report( "Running GPU set intersection with decompression (left " 
			<< leftSize << ") (right " << rightSize << ") (results " 
			<< resultSize << ")" );
		report( " Left: " << hydrazine::toFormattedString( leftBegin, leftEnd, 
			FormatPair< Key, Value >(), " ", 80 ) );
		report( " Right: " << hydrazine::toFormattedString( rightBegin, 
			rightEnd, FormatPair< Key, Value >(), " ", 80 ) );

		if( leftSize == 0 || rightSize == 0 ){ return 0; }
				
		DecompressedPtr dLeft = thrust::device_malloc< 
			DecompressedPair >( leftSize );
		DecompressedPtr dRight = thrust::device_malloc< 
			DecompressedPair >( rightSize );
		DecompressedPtr dResult = thrust::device_malloc< 
			DecompressedPair >( resultSize );
		
		thrust::transform( ConstPairPtr( leftBegin ), 
			ConstPairPtr( leftEnd ), dLeft, 
			KeyDecompressor< Key, Value, LeftDecompressor >( 
			leftDecompressor ) );
		thrust::transform( ConstPairPtr( rightBegin ), 
			ConstPairPtr( rightEnd ), dRight, 
			KeyDecompressor< Key, Value, RightDecompressor >( 
			rightDecompressor ) );
		
		thrust::sort( dLeft, dLeft + leftSize, 
			DecompressedCompare< Key, Value, Compare >( comp ) );
		thrust::sort( dRight, dRight + rightSize,  
			DecompressedCompare< Key, Value, Compare >( comp ) );
		
		report( " Decompressed Left: " << 
			hydrazine::toFormattedString( dLeft.get(), 
			( dLeft + leftSize ).get(), 
			FormatCompressedPair< Key, Pair >(), " ", 80 ) );
		report( " Decompressed Right: " << 
			hydrazine::toFormattedString( dRight.get(), 
			( dRight + rightSize ).get(), 
			FormatCompressedPair< Key, Pair >(), " ", 80 ) );
		
		report( " Decompressed Left End: " << 
			hydrazine::toFormattedString( MAX( dLeft.get(), 
			( dLeft + leftSize - 10 ).get() ), 
			( dLeft + leftSize ).get(), 
			FormatCompressedPair< Key, Pair >(), " ", 80 ) );
		report( " Decompressed Right End: " << 
			hydrazine::toFormattedString( MAX( dRight.get(), 
			( dRight + rightSize - 10 ).get() ), 
			( dRight + rightSize ).get(), 
			FormatCompressedPair< Key, Pair >(), " ", 80 ) );
		
		resultSize = computeIntersectionWithDecompression( dResult.get(), 
			( dResult + resultSize ).get(), dLeft.get(), 
			( dLeft + leftSize ).get(), dRight.get(), 
			( dRight + rightSize ).get(), 
			DecompressedOp< Key, Value, Operator >( op ), comp );
		
		report( " Decompressed Results: " << 
			hydrazine::toFormattedString( dResult.get(), 
			( dResult + resultSize ).get(), 
			FormatCompressedPair< Key, Pair >(), " ", 80 ) );
		
		thrust::transform( dResult, dResult + resultSize, 
			PairPtr( resultBegin ), KeyCompressor< Key, Value >() );
		
		thrust::sort( PairPtr( resultBegin ), 
			PairPtr( resultBegin + resultSize ), 
			CompressedCompare< Key, Value, Compare >( comp ) );
		
		report( "Generated results (results " << resultSize << ")" );
		report( " Results: " << hydrazine::toFormattedString( resultBegin, 
			resultBegin + resultSize, FormatPair< Key, Value >(), " ", 80 ) );
		
		thrust::device_free( dLeft );
		thrust::device_free( dRight );
		thrust::device_free( dResult );
		
		return resultSize;
	}
		
}	

}

}

#undef SIZE_POINTER
#undef PAIR
#undef POINTER
#undef CONST_POINTER

#endif

