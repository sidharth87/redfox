/*! \file   LLVMKernel.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Saturday August 1, 2009
	\brief  The header file for the LLVMKernel class.
*/

#pragma once

// Harmony Includes
#include <harmony/llvm/interface/LLVMInstruction.h>
#include <harmony/llvm/interface/LLVMStatement.h>

// Standard Library Includes
#include <deque>

namespace harmony
{

namespace vm
{

/*! \brief A class containing a complete representation of an LLVM kernel */
class LLVMKernel
{
public:
	/*! \brief A vector of LLVM instructions */
	typedef std::vector< LLVMInstruction* > LLVMInstructionVector;
	/*! \brief A vector of LLVM Statements */
	typedef std::deque< LLVMStatement > LLVMStatementVector;

private:
	/*! \brief The assembled LLVM kernel */
	std::string _code;
	/*! \brief The set of statements representing the kernel */
	LLVMStatementVector _statements;
	
public:
	/*! \brief Sets the ISA */
	LLVMKernel();

public:
	/*! \brief Add a statement to the end */
	void push_back(const LLVMStatement& statement);
	/*! \brief Add a statement to the beginning */
	void push_front(const LLVMStatement& statement);

public:
	/*! \brief Assemble the LLVM kernel from the set of statements */
	void assemble();
	/*! \brief Is the kernel assembled? */
	bool assembled() const;
	/*! \brief Get the assembly code */
	const std::string& code() const;
	/*! \brief Get the assembly code with line numbers */
	std::string numberedCode() const;
	/*! \brief Get the set of statements */
	const LLVMStatementVector& llvmStatements() const;

public:
	std::string name;

};

}

}

