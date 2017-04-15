/*!
	\file DeviceVectorWrapper.cu
	\date Wednesday June 3, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the wrapper functions for thrust vector.
*/

#ifndef DEVICE_VECTOR_WRAPPER_CU_INCLUDED
#define DEVICE_VECTOR_WRAPPER_CU_INCLUDED

#include <thrust/device_vector.h>
#include <thrust/is_sorted.h>
#include <thrust/sort.h>
#include <redfox/ra/interface/DeviceVectorWrapper.h>
#include <hydrazine/implementation/Exception.h>

#include <hydrazine/implementation/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace gpu
{
	namespace types
	{
		#define PAIR std::pair< Key, Value >
		#define TYPE thrust::device_vector< PAIR >
		
		template< typename Key, typename Value, typename Owner >
		static typename TYPE::iterator convert( 
			const typename DeviceVector< Key, Value, Owner >::iterator& i )
		{
			return typename TYPE::iterator( thrust::device_ptr< PAIR >( 
				i.base() ) );
		}
		
		template< typename Key, typename Value, typename Owner >
		static typename DeviceVector< Key, Value, Owner >::iterator convert( 
			const typename TYPE::iterator& i )
		{
			return typename DeviceVector< Key, Value, Owner >::iterator( 
				i.base().get() );
		}

		template< typename Key, typename Value, typename Owner >
		static typename DeviceVector< Key, Value, Owner >::const_iterator 
			convert( const typename TYPE::const_iterator& i )
		{
			return typename DeviceVector< Key, Value, Owner >::const_iterator( 
				i.base().get() );
		}
		
		template< typename Key, typename Value, typename Owner >
		void* DeviceVector< Key, Value, Owner >::newVector()
		{
			return new TYPE;
		}

		template< typename Key, typename Value, typename Owner >
		void* DeviceVector< Key, Value, Owner >::newVector( const void* v )
		{
			const TYPE* vector = reinterpret_cast< const TYPE* >( v );
			return new TYPE( *vector );
		}

		template< typename Key, typename Value, typename Owner >
			template< typename Iterator >
		void* DeviceVector< Key, Value, Owner >::newVector( Iterator first, 
			Iterator last )
		{
			thrust::host_vector< PAIR > v( first, last );
			report( "Building vector with " 
				<< std::distance( first, last ) << " elements" );
			TYPE* vector = new TYPE( v.begin(), v.end() );
			report( "Checking if vector is sorted." );
			if( !thrust::is_sorted( vector->begin(), vector->end(), 
				typename Owner::value_compare( ) ) )
			{
				throw hydrazine::Exception( 
					"NVCC cannot compile the necessary code here." );
				//thrust::sort( vector->begin(), vector->end(), 
				//	typename Owner::value_compare( ) );
			}
			report( "Done." );
			return vector;
		}
		
		template< typename Key, typename Value, typename Owner >
		void DeviceVector< Key, Value, Owner >::destroyVector( void* v )
		{
			TYPE* vector = reinterpret_cast< TYPE* >( v );
			delete vector;
		}
				
		template< typename Key, typename Value, typename Owner >
		void DeviceVector< Key, Value, Owner >::copyVector( void* v1, 
			const void* v2 )
		{
			TYPE* vector1 = reinterpret_cast< TYPE* >( v1 );
			const TYPE* vector2 = reinterpret_cast< const TYPE* >( v2 );
			*vector1 = *vector2;
		}
		
		template< typename Key, typename Value, typename Owner >
		void DeviceVector< Key, Value, Owner >::clearVector( void* v )
		{
			TYPE* vector = reinterpret_cast< TYPE* >( v );
			vector->clear();
		}
		
		template< typename Key, typename Value, typename Owner >
		size_t DeviceVector< Key, Value, Owner >::vectorSize( const void* v )
		{
			const TYPE* vector = reinterpret_cast< const TYPE* >( v );
			return vector->size();
		}
		
		template< typename Key, typename Value, typename Owner >
		size_t DeviceVector< Key, Value, Owner >::vectorMaxSize( const void* v )
		{
			const TYPE* vector = reinterpret_cast< const TYPE* >( v );
			return vector->max_size();
		}
		
		template< typename Key, typename Value, typename Owner >
		bool DeviceVector< Key, Value, Owner >::vectorEmpty( const void* v )
		{
			const TYPE* vector = reinterpret_cast< const TYPE* >( v );
			return vector->empty();
		}
		
		template< typename Key, typename Value, typename Owner >
		typename DeviceVector< Key, Value, Owner >::iterator 
			DeviceVector< Key, Value, Owner >::vectorBegin( void* v )
		{
			TYPE* vector = reinterpret_cast< TYPE* >( v );
			return convert< Key, Value, Owner >( vector->begin() );
		}
		
		template< typename Key, typename Value, typename Owner >
		typename DeviceVector< Key, Value, Owner >::const_iterator 
			DeviceVector< Key, Value, Owner >::vectorBegin( const void* v )
		{
			const TYPE* vector = reinterpret_cast< const TYPE* >( v );
			return convert< Key, Value, Owner >( vector->begin() );
		}
		
		template< typename Key, typename Value, typename Owner >
		typename DeviceVector< Key, Value, Owner >::iterator 
			DeviceVector< Key, Value, Owner >::vectorEnd( void* v )
		{
			TYPE* vector = reinterpret_cast< TYPE* >( v );
			return convert< Key, Value, Owner >( vector->end() );
		}
		
		template< typename Key, typename Value, typename Owner >
		typename DeviceVector< Key, Value, Owner >::const_iterator 
			DeviceVector< Key, Value, Owner >::vectorEnd( const void* v )
		{
			const TYPE* vector = reinterpret_cast< const TYPE* >( v );
			return convert< Key, Value, Owner >( vector->end() );
		}
		
		template< typename Key, typename Value, typename Owner >
		typename DeviceVector< Key, Value, Owner >::iterator 
			DeviceVector< Key, Value, Owner >::vectorInsert( void* v, 
			typename DeviceVector::iterator i, const value_type& val )
		{
			TYPE* vector = reinterpret_cast< TYPE* >( v );
			return convert< Key, Value, Owner >( vector->insert( 
				convert< Key, Value, Owner >( i ), val ) );
		}
		
		template< typename Key, typename Value, typename Owner >
		typename DeviceVector< Key, Value, Owner >::iterator 
			DeviceVector< Key, Value, Owner >::vectorErase( void* v, 
			typename DeviceVector::iterator i )
		{
			TYPE* vector = reinterpret_cast< TYPE* >( v );
			return convert< Key, Value, Owner >( vector->erase( 
				convert< Key, Value, Owner >( i ) ) );
		}
		
		template< typename Key, typename Value, typename Owner >
		void DeviceVector< Key, Value, Owner >::vectorErase( void* v, 
			typename DeviceVector::iterator f, 
			typename DeviceVector::iterator l )
		{
			TYPE* vector = reinterpret_cast< TYPE* >( v );
			vector->erase( convert< Key, Value, Owner >( f ), 
				convert< Key, Value, Owner >( l ) );
		}
		
		template< typename Key, typename Value, typename Owner >
		void DeviceVector< Key, Value, Owner >::resize( void* v, size_t size )
		{
			TYPE* vector = reinterpret_cast< TYPE* >( v );
			vector->resize( size );
		}
		
		template< typename Key, typename Value, typename Owner >
		void DeviceVector< Key, Value, Owner >::reserve( void* v, size_t size )
		{
			TYPE* vector = reinterpret_cast< TYPE* >( v );
			vector->reserve( size );
		}
		
		#undef PAIR
		#undef TYPE
		
	}
}

#endif

