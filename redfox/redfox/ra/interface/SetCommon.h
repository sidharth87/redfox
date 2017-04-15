/*!
	\file SetCommon.h
	\date Tuesday June 9, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for a common set of functions used by datalog 
		Set functions.
*/

#ifndef SET_COMMON_H_INCLUDED
#define SET_COMMON_H_INCLUDED

#include <thrust/device_ptr.h>
#include <hydrazine/cuda/Cuda.h>
#include <hydrazine/cuda/Memory.h>
#include <hydrazine/implementation/macros.h>
#include <hydrazine/implementation/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

#define TREE_SIZE ( 16 / sizeof( Key ) )
#define COMMON_THREADS 128

namespace gpu
{

namespace algorithms
{

namespace cuda
{

	#ifndef NDEBUG
	
		template< typename Key, typename Value >
		class FormatPair
		{
			public:
				std::string operator()( CONST_POINTER _p )
				{
					POINTER p = const_cast< POINTER >( _p );
					std::stringstream stream;
					thrust::device_ptr< PAIR > pointer( p );
					PAIR pair = *pointer;
					stream << pair.first << "," << pair.second;
					return stream.str();
				}
		};
	
	#endif
	
	template< typename Key, typename Compare >
	__device__ void lowerBound( Key* begin, Key* end, 
		const Key& key, Compare comp, Key*& result )
	{
		size_t low = 0;
		size_t high = end - begin;
	
		while( low < high )
		{
			size_t middle = low + ( ( high - low ) / 2 );
			if( comp( begin[ middle ], key ) )
			{
				low = middle;
				++low;
			}
			else
			{
				high = middle;
			}
		}
		result = begin + low;	
	}
	
	template< typename Key, typename Value, typename Compare >
	__device__ void lowerBound( CONST_POINTER begin, CONST_POINTER end, 
		const Key& key, Compare comp, CONST_POINTER& result )
	{
		size_t low = 0;
		size_t high = end - begin;
	
		while( low < high )
		{
			size_t middle = low + ( ( high - low ) / 2 );
			if( comp( begin[ middle ].first, key ) )
			{
				low = middle;
				++low;
			}
			else
			{
				high = middle;
			}
		}
		result = begin + low;		
	}
	
	template< typename Key, typename Value >
	__device__ void loadSearchTree( Key* tree, CONST_POINTER begin, 
		CONST_POINTER end, const unsigned int treeSize, unsigned int& step )
	{
		size_t size = end - begin;
		step = CEIL_DIV( size, treeSize );
		
		for( unsigned int i = THREAD_ID(); i < treeSize; i += COMMON_THREADS )
		{
			size_t index = i * step;
			if( index < size )
			{
				tree[ i ] = begin[ index ].first;
			}
			else
			{
				tree[ i ] = begin[ size - 1 ].first;
			}
		}
		
		__syncthreads();
	}
	
	template< typename Key, typename Value, typename Compare >
	__device__ void lowerBound( SIZE_POINTER result, CONST_POINTER begin, 
		CONST_POINTER end, const Key& key, Key* searchTree, 
		const unsigned int treeSize, unsigned int treeStep, Compare comp )
	{
		const unsigned int size = end - begin;
							
		unsigned int minOffset = 0;
		unsigned int maxOffset = size;
		
		Key* lowerKey;
		lowerBound( searchTree, searchTree + treeSize, key, comp, 
			lowerKey );
	
		if( lowerKey != searchTree + treeSize )
		{
			if( lowerKey != searchTree && !comp( *lowerKey, key ) )
			{
				--lowerKey;
			}
		}
		else
		{
			--lowerKey;
		}
	
		minOffset += ( lowerKey - searchTree ) * treeStep;
		minOffset = MIN( minOffset, maxOffset );
		maxOffset = MIN( minOffset + treeStep, maxOffset );
	
		CONST_POINTER lowerPointer;
		lowerBound( begin + minOffset, begin + maxOffset, key, comp, 
			lowerPointer );
		
		size_t smallerIndex = lowerPointer - begin;
		
		result[ GLOBAL_ID() ] = ( GLOBAL_ID() == 0 ) ? 0 : smallerIndex;
	}
	
	template< typename Key, typename Value, typename Key1, 
		typename Value1, typename Compare >
	__global__ void determineRangeGlobal( SIZE_POINTER smallerRange, 
		SIZE_POINTER largerRange, CONST_POINTER smallerBegin, 
		CONST_POINTER smallerEnd, CONST_POINTER_ONE largerBegin, 
		CONST_POINTER_ONE largerEnd, unsigned int partitions, Compare comp )
	{
		const unsigned int treeSize = TREE_SIZE;
		unsigned int treeStep;
		__shared__ Key searchTree[ treeSize ];
		
		loadSearchTree( searchTree, smallerBegin, smallerEnd, 
			treeSize, treeStep );
				
		if( GLOBAL_ID() >= partitions ) return;
		
		const unsigned int partitionSize = CEIL_DIV( largerEnd - largerBegin, 
			partitions );
		size_t begin = GLOBAL_ID() * partitionSize;
		
		begin = ( begin < largerEnd - largerBegin ) 
			? begin : largerEnd - largerBegin - 1;
			
		size_t end = MIN( begin + partitionSize, largerEnd - largerBegin );

		largerRange[ GLOBAL_ID() ] = begin;
		Key key = largerBegin[ begin ].first;
		
		lowerBound( smallerRange, smallerBegin, smallerEnd, key, searchTree, 
			treeSize, treeStep, comp );
		
		if( GLOBAL_ID() == partitions - 1 )
		{
			smallerRange[ GLOBAL_ID() + 1 ] = smallerEnd - smallerBegin;
			largerRange[ GLOBAL_ID() + 1 ] = end;
		}
	}
	
	template< typename Key, typename Value, typename Key1, 
		typename Value1, typename Compare >
	void determineRange( SIZE_POINTER smallerRange, 
		SIZE_POINTER largerRange, CONST_POINTER smallerBegin, 
		CONST_POINTER smallerEnd, CONST_POINTER_ONE largerBegin, 
		CONST_POINTER_ONE largerEnd, unsigned int partitions, Compare comp )
	{
		typedef thrust::device_ptr< size_t > pointer;
		const unsigned int threads = COMMON_THREADS;
		unsigned int ctas = CEIL_DIV( partitions, threads );
		
		report( " Initializing ranges, creating " << partitions 
			<< " partitions." );
		report( "  CTAS " << ctas << ", Threads " << threads );
		
		determineRangeGlobal<<< ctas, threads >>>( smallerRange, largerRange, 
			smallerBegin, smallerEnd, largerBegin, largerEnd, partitions, 
			comp );
		hydrazine::cuda::check( cudaGetLastError() );
		
		report( "  Tree size is " << TREE_SIZE );

		report( "  Smaller range is " << hydrazine::toString( 
			pointer( smallerRange ), pointer( smallerRange ) 
			+ partitions + 1 ) );
		report( "  Larger range is " << hydrazine::toString( 
			pointer( largerRange ), pointer( largerRange + partitions ) + 1 ) );
		
		reportE( partitions > 10, "  Smaller range end is " 
			<< hydrazine::toString( 
			pointer( MAX( smallerRange, smallerRange + partitions - 10 ) ), 
			pointer( smallerRange ) + partitions + 1 ) );
		reportE( partitions > 10, "  Larger range end is " 
			<< hydrazine::toString( 
			pointer( MAX( largerRange, largerRange + partitions - 10 ) ), 
			pointer( largerRange + partitions ) + 1 ) );
	}
		
	template< typename Key, typename Value >
	__global__ void gatherResults( POINTER results, POINTER temp, 
		const SIZE_POINTER range, const SIZE_POINTER histogram, 
		const SIZE_POINTER scannedHistogram )
	{
		const size_t size = histogram[ CTA_ID() ];
		hydrazine::cuda::memcpyCta( results + scannedHistogram[ CTA_ID() ], 
			temp + range[ CTA_ID() ], sizeof( PAIR ) * size );
	}
	
}	

}

}

#undef COMMON_THREADS
#undef TREE_SIZE

#endif

