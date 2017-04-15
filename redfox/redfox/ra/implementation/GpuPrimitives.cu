/*!
	\file GpuPrimitives.cu
	\date Friday May 29, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The instantiation file for the set of GPU primitives needed to 
		implement arbitrary datalog programs.
	This is needed so that GPU algorithms can be compiled with NVCC and linked
		against code that uses it.
*/

#ifndef GPU_PRIMITIVES_CU_INCLUDED
#define GPU_PRIMITIVES_CU_INCLUDED

#include <redfox/ra/interface/Map.h>
#include <redfox/ra/implementation/GpuPrimitives.inl>
#include <redfox/ra/interface/Operators.h>
#include <redfox/ra/interface/Comparisons.h>
#include <redfox/ra/interface/DataTypes.h>
#include <redfox/ra/interface/DetailedImplementations.h>

#include <thrust/transform.h>
#include <thrust/device_ptr.h>

namespace gpu
{
	namespace algorithms
	{
	
		#define INSTANTIATE_SET_INTERSECTION_D( key, value, operator, \
			compare, type ) \
			template void set_intersection( \
				type< key, value >&, \
				const type< key, value >&, \
				const type< key, value >&, operator, compare, \
				lb::KeyCompressor::Decompressor, \
				lb::KeyCompressor::Decompressor );
			
		#define INSTANTIATE_TYPE( key, value, operator, compare, type ) \
			template void set_intersection( \
				type< key, value >&, \
				const type< key, value >&, \
				const type< key, value >&, operator, compare ); \
			INSTANTIATE_SET_INTERSECTION_D( key, value, operator, \
				compare, type ) \
			template void set_union( \
				type< key, value >&, \
				const type< key, value >&, \
				const type< key, value >&, operator, compare ); \
			template void set_difference( \
				type< key, value >&, \
				const type< key, value >&, \
				const type< key, value >&, operator, compare ); \
			template void aggregate( \
				type< key, value >&, \
				const type< key, value >&, operator, compare ); 
		
		#define INSTANTIATE_SELECT( key, value, compare, type ) \
			template void select( \
				type< key, value >&, \
				const type< key, value >&, compare );

		#define INSTANTIATE_TRANSFORM( key, value, operator, compare, type ) \
			template void transform( \
				type< key, value >&, \
				const type< key, value >&, operator, compare );
		
		#define INSTANTIATE_PROJECT( key, value, operator, compare, type ) \
			template void project( \
				type< key, value >&, \
				const type< key, value >&, const type< key, value >&, \
				operator, compare );
		
		#define INSTANTIATE_MAP_BODY( key1, key2, value2, compare, type ) \
			template void map( \
				type< key1, value2 >&, \
				const type< key1, key2 >&, \
				const type< key2, value2 >&, compare ); 
			
		#define INSTANTIATE( key, value, operator, compare ) \
			INSTANTIATE_TYPE( key, value, operator, compare, \
				gpu::types::Map )
				
		#define INSTANTIATE_UNARY( key, value, compare ) \
			INSTANTIATE_SELECT( key, value, \
				gpu::comparisons::unary< compare >, \
				gpu::types::Map )

		#define INSTANTIATE_UNARY_OP( key, value, op, compare ) \
			INSTANTIATE_TRANSFORM( key, value, op, \
				compare, \
				gpu::types::Map )
		
		#define INSTANTIATE_BINARY_OP( key, value, op, compare ) \
			INSTANTIATE_PROJECT( key, value, op, \
				compare, \
				gpu::types::Map )
		
				
		#define INSTANTIATE_MAP( key1, key2, value2, compare ) \
			INSTANTIATE_MAP_BODY( key1, key2, value2, \
				compare, gpu::types::Map )
				
		INSTANTIATE_MAP( gpu::types::uint32, gpu::types::uint32, 
			gpu::types::uint32, gpu::comparisons::lt< gpu::types::uint32 > )
		INSTANTIATE_MAP( gpu::types::uint32, gpu::types::uint32, 
			gpu::types::float32, gpu::comparisons::lt< gpu::types::uint32 > )
		
		INSTANTIATE_UNARY( gpu::types::uint32, gpu::types::uint32,
			gpu::comparisons::lt< gpu::types::uint32 > )
		
		typedef std::pair< gpu::types::uint32, 
			gpu::types::float32 > uint32float32;
		typedef gpu::operators::CompareDimensionToValue< uint32float32, 
			gpu::comparisons::le< gpu::types::uint32 > > 
			CompareDimensionToValueLe;
		
		INSTANTIATE_UNARY_OP( gpu::types::uint32, gpu::types::float32,
			CompareDimensionToValueLe,
			gpu::comparisons::lt< gpu::types::uint32 > )
		INSTANTIATE_UNARY_OP( gpu::types::uint32, gpu::types::float32,
			gpu::operators::RemoveDimension< uint32float32 >,
			gpu::comparisons::lt< gpu::types::uint32 > )
		INSTANTIATE_UNARY_OP( gpu::types::uint32, gpu::types::float32,
			gpu::operators::AddConstant< uint32float32 >,
			gpu::comparisons::lt< gpu::types::uint32 > )
			
		INSTANTIATE_BINARY_OP( gpu::types::uint32, gpu::types::float32,
			gpu::operators::ProjectDimension< uint32float32 >,
			gpu::comparisons::lt< gpu::types::uint32 > )
				
		INSTANTIATE( gpu::types::uint32, gpu::types::float32, 
			gpu::operators::multiply< gpu::types::float32 >, 
			gpu::comparisons::lt< gpu::types::uint32 > );
		INSTANTIATE( gpu::types::uint32, gpu::types::float32, 
			gpu::operators::divide< gpu::types::float32 >, 
			gpu::comparisons::lt< gpu::types::uint32 > );
		INSTANTIATE( gpu::types::uint32, gpu::types::uint32, 
			gpu::operators::add< gpu::types::uint32 >, 
			gpu::comparisons::lt< gpu::types::uint32 > );
		INSTANTIATE( gpu::types::uint32, gpu::types::float32, 
			gpu::operators::add< gpu::types::float32 >, 
			gpu::comparisons::lt< gpu::types::uint32 > );
		INSTANTIATE( gpu::types::uint32, gpu::types::float32, 
			gpu::operators::Max< gpu::types::float32 >, 
			gpu::comparisons::lt< gpu::types::uint32 > );
		INSTANTIATE( gpu::types::uint32, gpu::types::uint32, 
			gpu::operators::subtract< gpu::types::uint32 >, 
			gpu::comparisons::lt< gpu::types::uint32 > );
		INSTANTIATE( gpu::types::uint32, gpu::types::float32, 
			gpu::operators::subtract< gpu::types::float32 >, 
			gpu::comparisons::lt< gpu::types::uint32 > );
		INSTANTIATE( gpu::types::uint32, gpu::types::uint32, 
			gpu::operators::add< gpu::types::uint32 >, 
			gpu::comparisons::gt< gpu::types::uint32 > );
		
		INSTANTIATE_SET_INTERSECTION_D( gpu::types::uint32, 
			gpu::types::float32, 
			gpu::operators::SelectIf< gpu::types::float32 >, 
			gpu::comparisons::lt< gpu::types::uint32 >,
			gpu::types::Map );
		INSTANTIATE_SET_INTERSECTION_D( gpu::types::uint32, 
			gpu::types::float32, 
			gpu::operators::Min< gpu::types::float32 >, 
			gpu::comparisons::lt< gpu::types::uint32 >,
			gpu::types::Map );
		
		typedef gpu::comparisons::lt< gpu::types::float32 > LessThanFloat32;
		typedef gpu::comparisons::gt< gpu::types::float32 > GreaterThanFloat32;
		typedef gpu::comparisons::CompareValue< uint32float32, LessThanFloat32 >
			CompareValueLtUint32Float32;
		typedef gpu::comparisons::CompareValue< uint32float32, 
			GreaterThanFloat32 > CompareValueGtUint32Float32;
		
		INSTANTIATE_SELECT( gpu::types::uint32, gpu::types::float32, 
			CompareValueLtUint32Float32, gpu::types::Map )
		INSTANTIATE_SELECT( gpu::types::uint32, gpu::types::float32, 
			CompareValueGtUint32Float32, gpu::types::Map )
		
		
		#undef INSTANTIATE_SET_INTERSECTION_D
		#undef INSTANTIATE_TRANSFORM
		#undef INSTANTIATE_PROJECT
		#undef INSTANTIATE_SELECT
		#undef INSTANTIATE_MAP
		#undef INSTANTIATE_MAP_BODY
		#undef INSTANTIATE_TYPE
		#undef INSTANTIATE_UNARY
		#undef INSTANTIATE_UNARY_OP
		#undef INSTANTIATE_BINARY_OP
		#undef INSTANTIATE
	
	}
}

#endif

