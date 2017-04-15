/*	\file   UnitTest.h
	\date   Wednesday December 15, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the unit test class
*/

#pragma once

// Harmony Includes
#include <harmony/hir/interface/Module.h>

// Standard Library Includes
#include <string>

namespace harmony
{

namespace hir
{

/*! \brief A class representing a single harmony unit test */
class UnitTest
{
public:
	typedef Module::VariableDeclarationMap DataMap;

public:
	/*! \brief Create a new test with a specific name, module, inputs */
	UnitTest(const std::string& name, const std::string& moduleName,
		const DataMap& inputs, const DataMap& outputs);
	/*! \brief Create a new test from a an input stream */
	UnitTest(std::istream& s);

public:
	/*! \brief Get the name of the test */
	const std::string& name() const;
	/*! \brief Get the name of the module used by the test */
	const std::string& moduleName() const;
	/*! \brief Get the inputs */
	const DataMap&     inputs() const;
	/*! \brief Get the outputs */
	const DataMap&     outputs() const;

private:
	std::string _name;
	std::string _moduleName;
	DataMap     _inputs;
	DataMap     _outputs;
};

}

}

