/*! \file Assign.h
	\date Saturday November 6, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the assign hamily of RA kernels.
*/

#ifndef ASSIGN_H_INCLUDED
#define ASSIGN_H_INCLUDED

#include <stdio.h>

namespace ra
{

typedef long long unsigned int uint;

namespace cuda
{

template<typename Type>
__device__ void assign(Type* begin, const unsigned long long int* element, const Type value)
{
//printf("assign %llx\n", (unsigned long long int)begin);
//printf("assign %llu\n", *element);
//	char* address = (char*) begin;
//	address += *element;
//printf("assign %llx\n", address);
//	begin = (Type*) address;
//printf("assign %llx\n", begin);
//	*begin = *value;
//printf("assign %llx\n", begin);
	*begin = value;
}

}

}

#endif 

