/*! \file Sort.h
	\author Gregory Diamos <gregory.diamos>
	\date Wednesday December 1, 2010
	\brief The header file for the C interface to CUDA sorting routines.
*/

#ifndef SORT_H_INCLUDED
#define SORT_H_INCLUDED

#include <redfox/ra/interface/Tuple.h>

namespace redfox
{

extern void sort(void* begin, void* end, unsigned long long int type);

extern void sort_string(void* begin, void* end);

//extern void sort2(void* begin, void* end, unsigned long long int type);
}

#endif

