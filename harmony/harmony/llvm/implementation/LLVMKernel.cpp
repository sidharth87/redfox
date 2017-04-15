/*! \file   LLVMKernel.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Saturday August 1, 2009
	\brief  The source file for the LLVMKernel class.
*/

// Harmony Includes
#include <harmony/llvm/interface/LLVMKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/Version.h>
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/string.h>

// Configuration
#include <configure.h>

namespace harmony
{

namespace vm
{

LLVMKernel::LLVMKernel()
{

}

void LLVMKernel::push_back(const LLVMStatement& statement)
{
	_statements.push_back(statement);
}

void LLVMKernel::push_front(const LLVMStatement& statement)
{
	_statements.push_front(statement);
}

void LLVMKernel::assemble()
{
	_code.clear();
	hydrazine::Version version;
	
	_code += "; Code assembled by Harmony LLVMKernel " + version.toString() 
		+ "\n\n";
	
	for( LLVMStatementVector::const_iterator 
		statement = llvmStatements().begin(); 
		statement != llvmStatements().end(); ++statement )
	{
		if( statement->type == LLVMStatement::Instruction ) _code += "\t";
		_code += statement->toString() + "\n";
	}
}

bool LLVMKernel::assembled() const
{
	return !_code.empty();
}

const std::string& LLVMKernel::code() const
{
	return _code;
}

std::string LLVMKernel::numberedCode() const
{
	return hydrazine::addLineNumbers( _code );
}

const LLVMKernel::LLVMStatementVector& LLVMKernel::llvmStatements() const
{
	return _statements;
}

}

}

