/*!	\file   KernelCall.h
	\date   Sunday October 25, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the KernelCall class
*/

#pragma once

// Harmony Includes
#include <harmony/hir/interface/Operand.h>

// Standard Library Includes
#include <vector>
#include <string>

namespace harmony
{

namespace hir
{

/*! \brief A class for representing a Kernel parsed from a HIR file */
class KernelCall
{
	friend class Module;
public:
	/*! \brief An ordered list of operands */
	typedef std::vector<Operand> OperandList;

	/*! \brief The type of kernel */
	enum Type
	{
		InvalidType,
		Kernel,
		ExternalKernel,
		Resize,
		GetSize,
		UpdateSize,
		ControlDecision,
		UnconditionalBranch,
		Exit
	};

public:
	/*! \brief Initializing constructor */
	KernelCall(const std::string& n, Type t, const std::string& p = "",
		OperandList l = OperandList());

public:
	/*! \brief Get the name of the kernel */
	const std::string& name() const;
	/*! \brief Get the type of the kernel */
	const Type& type() const;
	/*! \brief Get the PTX assembly for a PTX kernel */
	const std::string& ptx() const;
	/*! \brief Get the binary for an external kernel */
	const std::string& binary() const;
	/*! \brief Get the operand list */
	const OperandList& operands() const;

public:
	/*! \brief Is the kernel a PTX kernel ? */
	bool isPTXKernel() const;
	/*! \brief Is the kernel an external kernel ? */
	bool isExternalKernel() const;

private:
	std::string _name;     /*! The name of the kernel       */
	Type        _type;     /*! The type of kernel           */
	std::string _ptx;      /*! The source code for the call */
	OperandList _operands; /*! The kernel operands          */
};

}

}

