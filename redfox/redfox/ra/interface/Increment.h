/*! \file   Increment.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Saturday November 20, 2010
	\brief The header file for the increment series of utility functions
*/

#ifndef INCREMENT_H_INCLUDED
#define INCREMENT_H_INCLUDED

namespace ra
{

typedef long long unsigned int uint;

namespace cuda
{

template<unsigned int bytes, typename Type>
__device__ void increment(Type* out, const Type* in)
{
	*out += *in + bytes;
}

}

}

#endif

