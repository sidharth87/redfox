/*!
	\file DataTypes.h
	\date Friday June 5, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for Datalog basic types.
*/

#ifndef DATA_TYPES_H_INCLUDED
#define DATA_TYPES_H_INCLUDED

#include <string>

namespace gpu
{
	namespace types
	{
		typedef unsigned char uint8;
		typedef unsigned short uint16;
		typedef unsigned int uint32;
		typedef long long unsigned int uint64;
		typedef char int8;
		typedef short int16;
		typedef int int32;
		typedef long long int int64;
		typedef float float32;
		typedef double float64;
		typedef std::string string;
		typedef bool boolean;
	}
}

#endif

