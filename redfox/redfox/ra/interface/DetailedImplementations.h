/*!
	\file DetailedImplementations.h
	\date Tuesday June 2, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the set of GPU primitives needed to implement
		arbitrary datalog programs.
*/

#ifndef DETAILED_IMPLEMENTATIONS_H_INCLUDED
#define DETAILED_IMPLEMENTATIONS_H_INCLUDED

#include <hydrazine/implementation/Exception.h>
#include <hydrazine/interface/ValueCompare.h>
#include <algorithm>
#include <thrust/sort.h>

#define BYTES_PER_CTA 10000
#define PAIR std::pair< Key, Value >
#define POINTER PAIR*
#define CONST_POINTER const POINTER
#define PAIR_ONE std::pair< Key1, Value1 >
#define POINTER_ONE PAIR_ONE*
#define CONST_POINTER_ONE const POINTER_ONE
#define SIZE_POINTER size_t*
#define THREADS 128

#include <redfox/ra/interface/SetDifference.h>
#include <redfox/ra/interface/SetUnion.h>
#include <redfox/ra/interface/Aggregate.h>
#include <redfox/ra/interface/SetIntersection.h>

#undef SIZE_POINTER
#undef POINTER
#undef PAIR

#include <hydrazine/implementation/debug.h>
#include <redfox/ra/interface/MapOperator.h>
#include <redfox/ra/interface/Project.h>
#include <redfox/ra/interface/SetIntersectionWithDecompression.h>

#undef THREADS
#undef BYTES_PER_CTA

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace gpu
{
	namespace algorithms
	{
		namespace detail
		{
		
			#ifndef NDEBUG
	
				template< typename iterator >
				class FormatPair
				{
					public:
						std::string operator()( const iterator& p )
						{
							std::stringstream stream;
							stream << p->first << "," << p->second;
							return stream.str();
						}
				};
	
			#endif
		
			// Set intersection	with decompression
			template< typename Key, typename Value, typename Operator, 
				typename Compare, typename LeftDecompressor, 
				typename RightDecompressor >
			void set_intersection( host::types::Map< Key, Value >& result, 
				const host::types::Map< Key, Value >& left, 
				const host::types::Map< Key, Value >& right, 
				Operator op, Compare comp, 
				LeftDecompressor leftDecompressor, 
				RightDecompressor rightDecompressor )
			{
				typedef host::types::Map< Key, Value > DataStructure;
				typedef std::multimap< Key, std::pair< Key, Value > > MultiMap;
				typedef std::pair< typename MultiMap::const_iterator, 
					typename MultiMap::const_iterator > Range;
				assert(result.empty());
				
				MultiMap lookup;
				
				for( typename DataStructure::const_iterator li = left.begin(); 
					li != left.end(); ++li )
				{
					lookup.insert( std::make_pair( leftDecompressor.decompress( 
						li->first), *li ) );
				}
				
				for( typename DataStructure::const_iterator ri = right.begin(); 
					ri != right.end(); ++ri )
				{
					Range range = lookup.equal_range( 
						rightDecompressor.decompress( ri->first ) );
					for( typename MultiMap::const_iterator li = range.first; 
						li != range.second; ++li )
					{
						result.insert( op( li->second, *ri ) );
					}
				}				
			}

			template< typename Key, typename Value, typename Operator, 
				typename Compare, typename LeftDecompressor, 
				typename RightDecompressor >
			void set_intersection( 
				gpu::types::Map< Key, Value >& result, 
				const gpu::types::Map< Key, Value >& left, 
				const gpu::types::Map< Key, Value >& right, 
				Operator op, Compare comp, 
				LeftDecompressor leftDecompressor, 
				RightDecompressor rightDecompressor )
			{
				typedef std::pair< Key, Value >* cast;
				result.resize( std::max( left.size(), right.size() ) );
				size_t size = gpu::algorithms::cuda::setIntersection( 
					(cast) result.begin().base(), (cast) result.end().base(), 
					(cast) left.begin().base(), (cast) left.end().base(),
					(cast) right.begin().base(), (cast) right.end().base(), 
					op, comp, leftDecompressor, rightDecompressor );
				result.resize( size );
			}
		
			// Set intersection		
			template< typename Key, typename Value, typename Operator, 
				typename Compare >
			void set_intersection( host::types::Map< Key, Value >& result, 
				const host::types::Map< Key, Value >& left, 
				const host::types::Map< Key, Value >& right, 
				Operator op, Compare comp )
			{
				typedef host::types::Map< Key, Value > DataStructure;
				assert(result.empty());
				typename DataStructure::const_iterator li = left.begin();
				typename DataStructure::const_iterator ri = right.begin();
				
				while( ri != right.end() && li != left.end() )
				{
					if( comp( li->first, ri->first ) ){ ++li; continue; }
					if( comp( ri->first, li->first ) ){ ++ri; continue; }
					result.insert( result.end(), op( *li, *ri ) );
					++li; ++ri;
				}
			}

			template< typename Key, typename Value, typename Operator, 
				typename Compare >
			void set_intersection( 
				gpu::types::Map< Key, Value >& result, 
				const gpu::types::Map< Key, Value >& left, 
				const gpu::types::Map< Key, Value >& right, 
				Operator op, Compare comp )
			{
				typedef std::pair< Key, Value >* cast;
				result.resize( std::min( left.size(), right.size() ) );
				size_t size = gpu::algorithms::cuda::setIntersection( 
					(cast) result.begin().base(), (cast) result.end().base(), 
					(cast) left.begin().base(), (cast) left.end().base(),
					(cast) right.begin().base(), (cast) right.end().base(), 
					op, comp );
				result.resize( size );
			}
			
			// Set union
			template< typename Key, typename Value, typename Operator, 
				typename Compare >
			void set_union( host::types::Map< Key, Value >& result, 
				const host::types::Map< Key, Value >& left, 
				const host::types::Map< Key, Value >& right, 
				Operator op, Compare comp )
			{
				typedef host::types::Map< Key, Value > DataStructure;
				assert(result.empty());
				typename DataStructure::const_iterator li = left.begin();
				typename DataStructure::const_iterator ri = right.begin();
				
				while( li != left.end() && ri != right.end() )
				{
					if( comp( li->first, ri->first ) )
					{
						result.insert( result.end(), std::make_pair( li->first, 
							li->second ) );
						++li;
					}
					else if( comp( ri->first, li->first ) )
					{
						result.insert( result.end(), std::make_pair( ri->first, 
							ri->second ) );
						++ri;
					}
					else
					{
						result.insert( result.end(), op( *li, *ri ) );
						++li; ++ri;
					}
				}
				
				for( ; li != left.end(); ++li )
				{
					result.insert( result.end(), std::make_pair( li->first,
							li->second ) );
				}
				
				for( ; ri != right.end(); ++ri )
				{
					result.insert( result.end(), std::make_pair( ri->first,
							ri->second ) );
				}
			}

			template< typename Key, typename Value, typename Operator, 
				typename Compare >
			void set_union( 
				gpu::types::Map< Key, Value >& result, 
				const gpu::types::Map< Key, Value >& left, 
				const gpu::types::Map< Key, Value >& right, 
				Operator op, Compare comp )
			{
				result.resize( left.size() + right.size() );
				typedef std::pair< Key, Value >* cast;
				size_t size = gpu::algorithms::cuda::setUnion( 
					(cast) result.begin().base(), (cast) result.end().base(), 
					(cast) left.begin().base(), (cast) left.end().base(),
					(cast) right.begin().base(), (cast) right.end().base(), 
					op, comp );
				result.resize( size );
			}
			
			// Set difference
			template< typename Key, typename Value, typename Operator, 
				typename Compare >
			void set_difference( host::types::Map< Key, Value >& result, 
				const host::types::Map< Key, Value >& left, 
				const host::types::Map< Key, Value >& right, Operator op, 
				Compare comp )
			{
				typedef host::types::Map< Key, Value > DataStructure;
				assert(result.empty());
				typename DataStructure::const_iterator li = left.begin();
				typename DataStructure::const_iterator ri = right.begin();
				
				while( li != left.end() && ri != right.end() )
				{
					if( comp( li->first, ri->first ) )
					{
						result.insert( result.end(), std::make_pair( li->first, 
							li->second ) );
						++li;
					}
					else if( comp( ri->first, li->first ) )
					{
						++ri;
					}
					else
					{
						++li; ++ri;
					}
				}
				for( ; li != left.end(); ++li )
				{
					result.insert( result.end(), std::make_pair( li->first,
							li->second ) );
				}
			}

			template< typename Key, typename Value, typename Operator, 
				typename Compare >
			void set_difference( 
				gpu::types::Map< Key, Value >& result, 
				const gpu::types::Map< Key, Value >& left, 
				const gpu::types::Map< Key, Value >& right, 
				Operator op, Compare comp )
			{
				result.resize( left.size() );
				typedef std::pair< Key, Value >* cast;
				size_t size = gpu::algorithms::cuda::setDifference( 
					(cast) result.begin().base(), (cast) result.end().base(), 
					(cast) left.begin().base(), (cast) left.end().base(),
					(cast) right.begin().base(), (cast) right.end().base(), 
					op, comp );
				result.resize( size );
			}
		  	
		  	// Aggregate
			template< typename Key, typename Value, typename Operator, 
				typename Compare >
			void aggregate( host::types::Map< Key, Value >& result, 
				const host::types::Map< Key, Value >& input, 
				Operator op, Compare comp )
			{
				typedef host::types::Map< Key, Value > DataStructure;
				assert(result.empty());
				if( input.empty() ) return;
				
				typename DataStructure::const_iterator fi = input.begin();
				std::pair< Key, Value > current = *fi;
				++fi;
				
				for( ; fi != input.end(); ++fi )
				{
					assert( !comp( fi->first, current.first ) );
					if( comp( current.first, fi->first ) )
					{
						result.insert( result.end(), current );
						current = *fi;
					}
					else
					{
						current.second = op( current.second, fi->second );
					}
				}
				result.insert( result.end(), current );
			}

			template< typename Key, typename Value, typename Operator, 
				typename Compare >
			void aggregate( gpu::types::Map< Key, Value >& result, 
				const gpu::types::Map< Key, Value >& input, 
				Operator op, Compare comp )
			{
				result.resize( input.size() );
				typedef std::pair< Key, Value >* cast;
				size_t size = gpu::algorithms::cuda::aggregate( 
					(cast) result.begin().base(), (cast) result.end().base(), 
					(cast) input.begin().base(), (cast) input.end().base(), 
					op, comp );
				result.resize( size );
			}
			
			// Select
			template< typename Key, typename Value, typename Compare >
			void select( host::types::Map< Key, Value >& result, 
				const host::types::Map< Key, Value >& input, Compare pred )
			{
				typedef host::types::Map< Key, Value > DataStructure;
				assert(result.empty());
				if( input.empty() ) return;
				
				typename DataStructure::const_iterator fi = input.begin();
				for( ; fi != input.end(); ++fi )
				{
					if( pred( fi->first ) )
					{
						result.insert( result.end(), *fi );
					}
				}
			}

			template< typename Key, typename Value, typename Compare >
			void select( gpu::types::Map< Key, Value >& result, 
				const gpu::types::Map< Key, Value >& input, Compare pred )
			{
				report( "Running GPU selection." );
				typedef thrust::device_ptr< std::pair< Key, Value > > pointer;
				typedef std::pair< Key, Value >* cast;
				result.resize( input.size() );
				
				report( " Input: " << hydrazine::toFormattedString( 
					(cast) input.begin().base(), (cast) input.end().base(), 
					gpu::algorithms::cuda::FormatPair< Key, Value >() ) );
				report( " Predicate Limit: " << pred.value );
				
				pointer end = thrust::copy_if( 
					pointer( (cast) input.begin().base() ), 
					pointer( (cast) input.end().base() ), 
					pointer( (cast) result.begin().base() ), pred );
				result.resize( end.get() - (cast) result.begin().base() );
				
				report( " Result: " << hydrazine::toFormattedString( 
					(cast) result.begin().base(), (cast) result.end().base(), 
					gpu::algorithms::cuda::FormatPair< Key, Value >() ) );
				
			}

			template< typename Key1, typename Key2, typename Value2, 
				typename Compare >
			void map( host::types::Map< Key1, Value2 >& result, 
				const host::types::Map< Key1, Key2 >& left, 
				const host::types::Map< Key2, Value2 >& right, Compare pred )
			{
				report( "Runing host map." );
				
				typedef host::types::Map< Key1, Key2 > Left;
				typedef host::types::Map< Key2, Value2 > Right;
				typedef host::types::Map< Key1, Value2 > Result;
				for( typename Left::const_iterator li = left.begin(); 
					li != left.end(); ++li )
				{
					typename Right::const_iterator ri 
						= right.find( li->second );
					if( ri != right.end() )
					{
						result.insert( std::make_pair( li->first, ri->second ) );
					}
				}
				report( " Result: " << hydrazine::toFormattedString( 
					result.begin(), result.end(), 
					FormatPair< typename Result::iterator >() ) );
			}
			
			template< typename Key1, typename Key2, typename Value2, 
				typename Compare >
			void map( gpu::types::Map< Key1, Value2 >& result, 
				const gpu::types::Map< Key1, Key2 >& left, 
				const gpu::types::Map< Key2, Value2 >& right, Compare comp )
			{
				result.resize( std::min( left.size(), right.size() ) );
				typedef const std::pair< Key1, Key2 >* Left;
				typedef const std::pair< Key2, Value2 >* Right;
				typedef std::pair< Key1, Value2 >* Result;
				size_t size = gpu::algorithms::cuda::map( 
					(Result) result.begin().base(), 
					(Result) result.end().base(), 
					(Left) left.begin().base(), (Left) left.end().base(), 
					(Right) right.begin().base(), (Right) right.end().base(), 
					comp );
				result.resize( size );
			}
			
			template< typename Key, typename Value, 
				typename Operator, typename Compare >
			void transform( host::types::Map< Key, Value >& result, 
				const host::types::Map< Key, Value >& input, 
				Operator op, Compare comp )
			{
				typedef host::types::Map< Key, Value > DataStructure;
				assert(result.empty());
				if( input.empty() ) return;
				
				typename DataStructure::const_iterator fi = input.begin();
				for( ; fi != input.end(); ++fi )
				{
					result.insert( result.end(), op( *fi ) );
				}
			}
			
			template< typename Key, typename Value, 
				typename Operator, typename Compare >
			void transform( gpu::types::Map< Key, Value >& result, 
				const gpu::types::Map< Key, Value >& input, 
				Operator op, Compare comp )
			{
				report( "Running GPU transform." );
				typedef thrust::device_ptr< std::pair< Key, Value > > pointer;
				typedef std::pair< Key, Value >* cast;
				result.resize( input.size() );
				
				report( " Input: " << hydrazine::toFormattedString( 
					(cast) input.begin().base(), (cast) input.end().base(), 
					gpu::algorithms::cuda::FormatPair< Key, Value >() ) );
				
				pointer end = thrust::transform( 
					pointer( (cast) input.begin().base() ), 
					pointer( (cast) input.end().base() ), 
					pointer( (cast) result.begin().base() ), op );
				result.resize( end.get() - (cast) result.begin().base() );
				
				report( " Result: " << hydrazine::toFormattedString( 
					(cast) result.begin().base(), (cast) result.end().base(), 
					gpu::algorithms::cuda::FormatPair< Key, Value >() ) );
			}
			
			template< typename Key, typename Value, 
				typename Operator, typename Compare >
			void project( host::types::Map< Key, Value >& result, 
				const host::types::Map< Key, Value >& left, 
				const host::types::Map< Key, Value >& right, 
				Operator op, Compare comp )
			{
				report( "Running Host projection." );
				typedef host::types::Map< Key, Value > DataStructure;
				assert( result.empty() );
				
				typename DataStructure::const_iterator li = left.begin();
				for( ; li != left.end(); ++li )
				{
					typename DataStructure::const_iterator ri = right.begin();
					for( ; ri != right.end(); ++ri )
					{
						result.insert( result.end(), op( *li, *ri ) );
					}
				}
			}
			
			template< typename Key, typename Value, 
				typename Operator, typename Compare >
			void project( gpu::types::Map< Key, Value >& result, 
				const gpu::types::Map< Key, Value >& left, 
				const gpu::types::Map< Key, Value >& right, 
				Operator op, Compare comp )
			{
				report( "Running GPU project." );
				typedef thrust::device_ptr< std::pair< Key, Value > > pointer;
				typedef std::pair< Key, Value >* cast;
				result.resize( left.size() * right.size() );
				
				report( " Left: " << hydrazine::toFormattedString( 
					(cast) left.begin().base(), (cast) left.end().base(), 
					gpu::algorithms::cuda::FormatPair< Key, Value >() ) );
				report( " Right: " << hydrazine::toFormattedString( 
					(cast) right.begin().base(), (cast) right.end().base(), 
					gpu::algorithms::cuda::FormatPair< Key, Value >() ) );
				
				typedef const std::pair< Key, Value >* Left;
				typedef const std::pair< Key, Value >* Right;
				typedef std::pair< Key, Value >* Result;
				gpu::algorithms::cuda::project( 
					(Result) result.begin().base(), 
					(Result) result.end().base(), 
					(Left) left.begin().base(), (Left) left.end().base(), 
					(Right) right.begin().base(), (Right) right.end().base(), 
					op, comp );
								
				report( " Result: " << hydrazine::toFormattedString( 
					(cast) result.begin().base(), (cast) result.end().base(), 
					gpu::algorithms::cuda::FormatPair< Key, Value >() ) );
			}
			
		}
	}
}

#endif

