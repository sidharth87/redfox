/*! \file Merge.h
	\author Gregory Diamos <gregory.diamos>
	\date Wednesday January 5, 2010
	\brief The header file for the C interface to CUDA merge routines.
*/

#ifndef MERGE_H_INCLUDED
#define MERGE_H_INCLUDED

namespace redfox
{

extern void set_union(void *result, unsigned long long int *size, void* lbegin, void* lend, 
	void* rbegin, void* rend, unsigned int type);

}

#endif

