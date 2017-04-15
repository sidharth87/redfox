/*! \file Sort.h
	\author Gregory Diamos <gregory.diamos>
	\date Wednesday December 1, 2010
	\brief The header file for the C interface to CUDA sorting routines.
*/

#ifndef MODERNGPU_SORT_H_INCLUDED
#define MODERNGPU_SORT_H_INCLUDED

#include <redfox/ra/interface/Tuple.h>

namespace redfox
{

extern void sort_pair(void* key_begin, void* value_begin, unsigned long long int size, unsigned long long int key_bits, unsigned long long int value_size);

extern void sort_key(void* key_begin, unsigned long long int size, unsigned long long int bits);

}

#endif

