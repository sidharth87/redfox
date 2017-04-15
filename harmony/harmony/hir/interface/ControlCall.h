/*!	\file   ControlCall.h
	\date   Sunday October 25, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the hir::ControlCall class 
*/

#pragma once

// Harmony Includes
#include <harmony/hir/interface/KernelCall.h>

namespace harmony
{

namespace hir
{

/*! \brief A class for representing a Control parsed from a HIR file */
class ControlCall : public KernelCall
{
public:
	ControlCall(const std::string& n = "", Type t = UnconditionalBranch,
		const std::string& p = "", OperandList l = OperandList());

};

}

}

