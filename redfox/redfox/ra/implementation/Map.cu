/*!
	\file Map.cu
	\date Wednesday June 3, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The template instantiation file for a device vector.
*/

#ifndef GPU_MAP_CU_INCLUDED
#define GPU_MAP_CU_INCLUDED

#include <redfox/ra/interface/Map.h>
#include <redfox/ra/interface/DataTypes.h>
#include <redfox/ra/implementation/DeviceVectorWrapper.cu>
#include <lb/host/interface/Map.h>

namespace gpu
{

	namespace types
	{
		
		#define INSTANTIATE( key, value ) \
			template class Map< key, value >; \
			template class DeviceVector< key, value, \
				gpu::types::Map< key, value > >; \
			template void* DeviceVector< key, value, \
				gpu::types::Map< key, value > >::newVector( \
				host::types::Map< key, value >::iterator first, \
				host::types::Map< key, value >::iterator last );

		INSTANTIATE( gpu::types::uint32, gpu::types::uint32 )
		INSTANTIATE( gpu::types::uint32, gpu::types::float32 )

	}

}

#endif

