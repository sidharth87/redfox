/*!	\file   ControlCall.cpp
	\date   Sunday October 25, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the hir::ControlCall class 
*/

// Harmony Includes
#include <harmony/hir/interface/ControlCall.h>

namespace harmony
{

namespace hir
{

ControlCall::ControlCall(const std::string& n, Type t, const std::string& p,
	OperandList l) : KernelCall(n, t, p, l)
{

}

}

}

