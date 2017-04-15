/*! \file Unique.h
	\author Gregory Diamos <gregory.diamos>
	\date Wednesday December 8, 2010
	\brief The header file for the C interface to CUDA unique routines.
*/

#ifndef UNIQUE_H_INCLUDED
#define UNIQUE_H_INCLUDED

#include <redfox/ra/interface/Tuple.h>

namespace redfox
{

extern void unique(void* begin, unsigned long long int *size, unsigned long long int type);

}

#endif

