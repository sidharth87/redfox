/*!	\file   KernelCall.cpp
	\date   Sunday October 25, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the hir::KernelCall class
*/

// Harmony Includes
#include <harmony/hir/interface/KernelCall.h>

namespace harmony
{

namespace hir
{

KernelCall::KernelCall(const std::string& n, Type t,
	const std::string& p, OperandList l) : _name(n), _type(t),
	_ptx(p), _operands(l)
{

}

const std::string& KernelCall::name() const
{
	return _name;
}

const KernelCall::Type& KernelCall::type() const
{
	return _type;
}

const std::string& KernelCall::ptx() const
{
	return _ptx;
}

const std::string& KernelCall::binary() const
{
	return _ptx;
}

const KernelCall::OperandList& KernelCall::operands() const
{
	return _operands;
}

bool KernelCall::isPTXKernel() const
{
	return type() == Kernel || type() == ControlDecision;
}

bool KernelCall::isExternalKernel() const
{
	return type() == ExternalKernel;
}

}

}

