/*!	\file   Operand.h
	\date   Sunday October 25, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Operand class
*/

#pragma once

// Forward Declarations
namespace harmony { namespace hir { class VariableDeclaration; } }

namespace harmony
{

namespace hir
{

/*! \brief A class for representing an operand to a Kernel call */
class Operand
{
	friend class Module;	
public:
	/*! \brief Possible access modes for an operand */
	enum AccessMode
	{
		Invalid = 0x0,
		In      = 0x1,
		Out     = 0x2,
		InOut   = 0x3
	};

public:
	/*! \brief The constructor for an operand */
	Operand(const VariableDeclaration* v, AccessMode m);

public:
	/*! \brief Get the underlying variable */
	const VariableDeclaration& variable() const;
	/*! \brief Get the access mode */
	const AccessMode& mode() const;

public:
	/*! \brief Is this an output? */
	bool isOut() const;
	/*! \brief Is this an input? */
	bool isIn() const;

private:
	const VariableDeclaration* _variable; /*! Variable reference */
	AccessMode                 _mode;     /*! Access mode for the operand */
};

}

}

