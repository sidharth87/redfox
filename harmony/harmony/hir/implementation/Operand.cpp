/*!	\file   Operand.cpp
	\date   Sunday November 7, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the Operand class
*/

// Harmony Includes
#include <harmony/hir/interface/Operand.h>

namespace harmony
{

namespace hir
{

Operand::Operand(const VariableDeclaration* v, AccessMode m)
	: _variable(v), _mode(m)
{

}

const VariableDeclaration& Operand::variable() const
{
	return *_variable;
}

const Operand::AccessMode& Operand::mode() const
{
	return _mode;
}

bool Operand::isOut() const
{
	return mode() == Out || mode() == InOut;
}

bool Operand::isIn() const
{
	return mode() == In || mode() == InOut;
}

}

}

