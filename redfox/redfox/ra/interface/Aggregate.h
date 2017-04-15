/*!
	\file Aggregate.h
	\date Thursday June 11, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the Aggregate family of CUDA functions.
*/

#ifndef AGGREGATE_H_INCLUDED
#define AGGREGATE_H_INCLUDED

#define MAX_BUCKETS THREADS

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
		
	template< typename Key, typename Value, typename Compare >
	__global__ void partitionAggregates( unsigned int* partition, 
		CONST_POINTER begin, size_t size, Compare comp )
	{
		if( GLOBAL_ID() >= size ) return;
		if( GLOBAL_ID() == 0 )
		{
			partition[ 0 ] = 0;
			return;
		}
		
		Key left = begin[ GLOBAL_ID() - 1 ].first;
		Key right = begin[ GLOBAL_ID() ].first;
		
		partition[ GLOBAL_ID() ] = comp( left, right );		 
	}
	
	template< typename Key, typename Value, typename Operator >
	__global__ void computeAggregate1( Value* result, CONST_POINTER input, 
		size_t size, Operator op )
	{
		__shared__ Value shared[ THREADS ];
		
		if( GLOBAL_ID() >= size )
		{ 
			shared[ THREAD_ID() ] = 0;
		}
		else
		{
			shared[ THREAD_ID() ] = input[ GLOBAL_ID() ].second;
		}
		
		for( unsigned int i = THREADS/2; i != 0; i /= 2 )
		{
			if( ( THREAD_ID() & i ) == 0 )
			{
				if( THREAD_ID() + i < THREADS )
				{
					shared[ THREAD_ID() ] = op( shared[ THREAD_ID() ], 
						shared[ THREAD_ID() + i ] );
				}
			}
		}
		
		if( THREAD_ID() == 0 )
		{
			result[ CTA_ID() ] = shared[ 0 ];
		}
		
	}
	
	template< typename Value, typename Operator >
	__global__ void computeAggregate2( Value* result, const Value* input, 
		size_t size, Operator op )
	{
		__shared__ Value shared[ THREADS ];
		
		if( GLOBAL_ID() >= size )
		{ 
			shared[ THREAD_ID() ] = 0;
		}
		else
		{
			shared[ THREAD_ID() ] = input[ GLOBAL_ID() ];
		}
		
		for( unsigned int i = THREADS/2; i != 0; i /= 2 )
		{
			if( ( THREAD_ID() & i ) == 0 )
			{
				if( THREAD_ID() < i )
				{
					shared[ THREAD_ID() ] = op( shared[ THREAD_ID() ],
						shared[ THREAD_ID() + i ] );
				}
			}
		}
		
		if( THREAD_ID() == 0 )
		{
			result[ CTA_ID() ] = shared[ 0 ];
		}
	}
	
	template< typename Key, typename Value, typename Operator >
	void aggregateSingle( POINTER result, CONST_POINTER inputBegin, 
		CONST_POINTER inputEnd, Operator op )
	{
		size_t inputSize = inputEnd - inputBegin;
		if( inputSize == 0 ) return;
		
		const size_t threads = THREADS;
		size_t intermediates = CEIL_DIV( inputSize, threads );
		
		size_t ctas = intermediates;
		thrust::device_ptr< Value > oldValues 
			= thrust::device_malloc< Value >( ctas );
		computeAggregate1<<< ctas, threads >>>( oldValues.get(), inputBegin, 
			inputSize, op );
		
		while( intermediates > 1 )
		{
			ctas = CEIL_DIV( intermediates, threads );
			thrust::device_ptr< Value > values 
				= thrust::device_malloc< Value >( ctas );
			computeAggregate2<<< ctas, threads >>>( values.get(), 
				oldValues.get(), intermediates, op );
			intermediates = ctas;
			thrust::device_free( oldValues );
			oldValues = values;
		}
					
		PAIR resultValue = *thrust::device_ptr< PAIR >( (POINTER) inputBegin );
		resultValue.second = *oldValues;

		*thrust::device_ptr< PAIR >( result ) = resultValue;

		thrust::device_free( oldValues );
	}
	
	template< typename Key, typename Value, typename Operator >
	__global__ void cudaAggregateParallel( POINTER result, 
		CONST_POINTER input, unsigned int* histogram, Operator op )
	{
		__shared__ Value shared[ THREADS ];
	
		unsigned int begin = histogram[ CTA_ID() ];
		unsigned int end = histogram[ CTA_ID() + 1 ];
		
		Value mine;
		
		unsigned int i = THREAD_ID() + begin;
		
		if( i < end )
		{
			mine = input[ i ].second;
		
			i += THREADS;
		
			for( ; i < end; i += THREADS )
			{
				mine = op( mine, input[ i ].second );
			}
		
			shared[ THREAD_ID() ] = mine;
		}
		
		__syncthreads();
		
		if( i < end )
		{
			if( THREADS < 256 )
			{
				if( THREAD_ID() < 128 )
				{
					shared[ THREAD_ID() ] = op( shared[ THREAD_ID() ], 
						shared[ THREAD_ID() + 128 ] );
				}
			}
		}

		__syncthreads();

		if( i < end )
		{
			if( THREADS < 128 )
			{
				if( THREAD_ID() < 64 )
				{
					shared[ THREAD_ID() ] = op( shared[ THREAD_ID() ], 
						shared[ THREAD_ID() + 64 ] );
				}
			}
		}

		__syncthreads();

		if( i < end )
		{
			if( THREADS < 64 )
			{
				if( THREAD_ID() < 32 )
				{
					shared[ THREAD_ID() ] = op( shared[ THREAD_ID() ], 
						shared[ THREAD_ID() + 32 ] );
				}
			}
		}

		__syncthreads();

		if( i < end )
		{
			if( THREADS < 32 )
			{
				if( THREAD_ID() < 16 )
				{
					shared[ THREAD_ID() ] = op( shared[ THREAD_ID() ], 
						shared[ THREAD_ID() + 16 ] );
				}
			}
		}

		__syncthreads();

		if( i < end )
		{
			if( THREADS < 16 )
			{
				if( THREAD_ID() < 8 )
				{
					shared[ THREAD_ID() ] = op( shared[ THREAD_ID() ], 
						shared[ THREAD_ID() + 8 ] );
				}
			}
		}

		__syncthreads();

		if( i < end )
		{
			if( THREADS < 8 )
			{
				if( THREAD_ID() < 4 )
				{
					shared[ THREAD_ID() ] = op( shared[ THREAD_ID() ], 
						shared[ THREAD_ID() + 4 ] );
				}
			}
		}

		__syncthreads();

		if( i < end )
		{
			if( THREADS < 4 )
			{
				if( THREAD_ID() < 2 )
				{
					shared[ THREAD_ID() ] = op( shared[ THREAD_ID() ], 
						shared[ THREAD_ID() + 2 ] );
				}
			}
		}

		__syncthreads();

		if( i < end )
		{
			if( THREADS < 2 )
			{
				if( THREAD_ID() < 1 )
				{
					shared[ THREAD_ID() ] = op( shared[ THREAD_ID() ], 
						shared[ THREAD_ID() + 1 ] );
				}
			}
		}

		__syncthreads();

		if( THREAD_ID() == 0 )
		{
			result[ CTA_ID() ].first = input[ 0 ].first;
			result[ CTA_ID() ].second = shared[ 0 ];
		}
		
	}
	
	template< typename Key, typename Value, typename Operator >
	void aggregateParallel( POINTER result, CONST_POINTER inputBegin, 
		thrust::device_ptr< unsigned int > histogram, unsigned int buckets, 
		Operator op )
	{
		for( unsigned int i = 0; i < buckets; i += 65535 )
		{
			unsigned int ctas = MIN( buckets - i, 65535 );
		
			cudaAggregateParallel<<< ctas, THREADS, 0 >>>( result + i, 
				inputBegin, histogram.get() + i, op );
		}
	}
	
	__global__ void fastHistogram( unsigned int* histogram, 
		unsigned int* partition, size_t size )
	{
		__shared__ unsigned int buckets[ MAX_BUCKETS ];
		
		if( GLOBAL_ID() >= size ) return;
		
		buckets[ THREAD_ID() ] = 0;
		__syncthreads();
		
		unsigned int element = partition[ GLOBAL_ID() ];
		atomicInc( buckets + element, 1 );
		
		__syncthreads();
		
		if( buckets[ THREAD_ID() ] != 0 )
		{
			atomicAdd( histogram + GLOBAL_ID(), buckets[ THREAD_ID() ] );
		}
	}
	
	__global__ void slowHistogram( unsigned int* histogram, 
		unsigned int* partition, size_t size )
	{
		if( GLOBAL_ID() >= size ) return;
		
		unsigned int element = partition[ GLOBAL_ID() ];
		atomicInc( histogram + element, 1 );
	}
	
	template< typename Key, typename Value, 
		typename Operator, typename Compare >
	size_t aggregate( POINTER resultBegin, POINTER resultEnd, 
		CONST_POINTER inputBegin, CONST_POINTER inputEnd, Operator op, 
		Compare comp )
	{
		if( inputEnd == inputBegin ) return 0;
		
		size_t inputSize = inputEnd - inputBegin;
		size_t resultSize = resultEnd - resultBegin;
		assert( resultSize >= inputSize );

		report( "Running GPU aggregation over input with " 
			<< inputSize << " elements." );
		report( " Input: " << hydrazine::toFormattedString( inputBegin, inputEnd, 
			FormatPair< Key, Value >() ) );
		
		thrust::device_ptr< unsigned int > partition 
			= thrust::device_malloc< unsigned int >( inputSize );
		
		const size_t threads = THREADS;
		const size_t ctas = CEIL_DIV( inputSize, threads );
		
		partitionAggregates<<< ctas, threads >>>( partition.get(), inputBegin, 
			inputSize, comp );
		hydrazine::cuda::check( cudaGetLastError() );
		thrust::inclusive_scan( partition, partition + inputSize, partition );
		unsigned int partitions = partition[ inputSize - 1 ] + 1;
		
		report( " Partitioned into " << partitions << " sections." );
		report( " Partitions: " << hydrazine::toString( partition, 
			partition + inputSize ) );
		
		thrust::device_ptr< unsigned int > histogram 
			= thrust::device_malloc< unsigned int >( partitions + 1 );
		thrust::fill( histogram, histogram + partitions + 1, 0 );
		
		if( partitions < MAX_BUCKETS )
		{
			fastHistogram<<< ctas, threads >>>( histogram.get(), 
				partition.get(), inputSize );
		}
		else
		{
			slowHistogram<<< ctas, threads >>>( histogram.get(), 
				partition.get(), inputSize );
		}
		hydrazine::cuda::check( cudaGetLastError() );
		
		report( " Histogram: " << hydrazine::toString( histogram, 
			histogram + partitions + 1 ) );
		
		thrust::exclusive_scan( histogram, histogram + partitions + 1, 
			histogram );
		
		report( " Ranges: " << hydrazine::toString( histogram, 
			histogram + partitions + 1 ) );
		
		if( partitions < THREADS )
		{
			for( unsigned int bucket = 0; bucket != partitions; ++bucket )
			{
				unsigned int begin = histogram[ bucket ];
				unsigned int end = histogram[ bucket + 1 ];
			
				aggregateSingle( resultBegin + bucket, inputBegin + begin, 
					inputBegin + end, op );
			}
		}
		else
		{
			aggregateParallel( resultBegin, inputBegin, 
				histogram, partitions, op );
		}
		
		thrust::device_free( histogram );
		thrust::device_free( partition );
		
		return partitions;
	}
		
}	

}

}

#undef MAX_BUCKETS

#endif

