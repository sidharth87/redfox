/*!
	\file DeviceVectorWrapper.h
	\date Wednesdat June 3, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the wrapper functions for thrust vector.

	This should allow device vector to be treated as an opaque type that is
		compiled with nvcc and then managed from code compiled with a standard
		c++ compiler.
*/

#ifndef DEVICE_VECTOR_WRAPPER_H_INCLUDED
#define DEVICE_VECTOR_WRAPPER_H_INCLUDED

#include <hydrazine/cuda/DevicePointerIterator.h>

namespace gpu
{
	namespace types
	{
		template< typename Key, typename Value, typename Owner > 
		class DeviceVector
		{
			public:
				typedef std::pair< Key, Value > value_type;
				typedef value_type* pointer;
				typedef const value_type* const_pointer;
				typedef hydrazine::cuda::DevicePointerIterator< pointer, 
					Owner > iterator;
				typedef hydrazine::cuda::DevicePointerIterator< const_pointer, 
					Owner > const_iterator;
				typedef std::reverse_iterator< iterator > reverse_iterator;
				typedef std::reverse_iterator< const_iterator >
					const_reverse_iterator;
					
			public:
				static void* newVector();
				static void* newVector( const void* );
				
				template< typename Iterator >
				static void* newVector( Iterator first, Iterator last );
				static void destroyVector( void* );
				static void copyVector( void*, const void* );
				static void clearVector( void* );
			
				static size_t vectorSize( const void* );
				static size_t vectorMaxSize( const void* );
				static bool vectorEmpty( const void* );
				
				static iterator vectorBegin( void* );
				static const_iterator vectorBegin( const void* );
				static iterator vectorEnd( void* );
				static const_iterator vectorEnd( const void* );
				
				static iterator vectorInsert( void*, iterator, 
					const value_type& );
				static iterator vectorErase( void*, iterator );
				static void vectorErase( void*, iterator, iterator );
				
				static void resize( void*, size_t size );
				static void reserve( void*, size_t size );
		};
		
		
		
	}
}

#endif

