/*! \file Module.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Tuesday November 24, 2009
	\brief The header file for the Module class.
*/

#pragma once

// Harmony Includes
#include <harmony/hir/interface/VariableDeclaration.h>
#include <harmony/hir/interface/ControlFlowGraph.h>
#include <harmony/hir/interface/DataflowInformation.h>

// Standard Library Includes
#include <string>
#include <iostream>
#include <map>

namespace harmony
{

namespace hir
{

/*! \brief A representation of complete harmony program */
class Module
{
public:
	/*! \brief A map from variable names to declarations */
	typedef std::map<unsigned int, VariableDeclaration> VariableDeclarationMap;

public:
	/*! \brief Create a new module */
	Module();	
	/*! \brief Deep copy constructor */
	Module(const Module& m);
	/*! \brief Deep assignment operator */
	Module& operator=(const Module& m);

public:
	/*! \brief Get a reference to the set of variables in the module */
	const VariableDeclarationMap& variables() const;
	/*! \brief Get a reference to the name of the module */
	const std::string& name() const;
	/*! \brief Get a reference to the module's control flow graph */
	ControlFlowGraph& cfg();
	/*! \brief Get a reference to the module's control flow graph */
	const ControlFlowGraph& cfg() const;
	/*! \brief Get a reference to the module's data flow information */
	DataflowInformation& dataflowInformation();
	/*! \brief Get a reference to the module's data flow information */
	const DataflowInformation& dataflowInformation() const;
	/*! \brief Get the number of tests */
	unsigned int testCount() const;

public:
	/*! \brief Clear all kernels, variables, and the CFG */
	void clear();

public:
	/*! \brief Write the module out to a stream */
	void write(std::ostream& stream) const;
	/*! \brief Read the module in from a stream */
	void read(std::istream& stream);

private:
	void _remapOperands();

private:
	std::string            _name;
	VariableDeclarationMap _variables;
	ControlFlowGraph       _cfg;
	DataflowInformation    _dataflowInformation;
	unsigned int           _testCount;
};

}

}

std::ostream& operator<<(std::ostream& s, const harmony::hir::Module& g);

