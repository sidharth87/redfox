/*! \file   HirManager.h 
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday December 4, 2009
	\brief  The header file for the HirManager class
*/

#pragma once

// Harmony Includes
#include <harmony/hir/interface/Module.h>
#include <harmony/hir/interface/UnitTest.h>

namespace harmony
{

namespace hir
{

/*! \brief A class for managing modules in a harmony runtime */
class HirManager
{
public:
	/*! \brief A map from a module name to the Module */
	typedef std::map<std::string, Module> ModuleMap;
	/*! \brief A map from test name to tests */
	typedef std::map<std::string, UnitTest> TestMap;

public:
	/*! \brief Loads a module from a file */
	void load(const std::string& name, bool loadTests);

	/*! \brief Loads a module from a stream */
	void load(std::istream& stream, bool loadTests);

	/*! \brief Loads a module directly */
	void load(const Module& module);

public:	
	/*! \brief Runs all loaded tests, reporting success/failure */
	bool runTests(std::ostream& results);
	
	/*! \brief Runs the loaded program */
	void runProgram();
	
private:
	/*! \brief The set of modules that should be added to the program */
	ModuleMap _modules;
	/*! \brief The set of tests in all modules */
	TestMap   _tests;
};

}

}

