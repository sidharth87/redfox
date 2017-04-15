/*! \file   HirManager.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Saturday December 12, 2009
	\brief  The source file for the HirManager class
*/

// Harmony Includes
#include <harmony/hir/interface/HirManager.h>
#include <harmony/hir/interface/Module.h>
#include <harmony/runtime/interface/Runtime.h>

// Hydrazine Includes
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/string.h>

// Standard Library Includes
#include <fstream>

// Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace harmony
{

namespace hir
{

void HirManager::load(const std::string& name, bool loadTests)
{
	std::ifstream file(name.c_str());
	
	if(!file.is_open())
	{
		throw hydrazine::Exception(
			"Could not open Harmony module in file: " + name);
	}
		
	load(file, loadTests);
}

void HirManager::load(std::istream& stream, bool loadTests)
{
	Module module;
	
	module.read(stream);
	
	_modules.insert(std::make_pair(module.name(), module));

	for(unsigned int t = 0; t != module.testCount(); ++t)
	{
		long long unsigned int words = 0;
		stream.read((char*)&words, sizeof(long long unsigned int));
		
		std::string buffer(words, ' ');
		stream.read((char*)buffer.c_str(), words);
		
		std::stringstream testStream(buffer);
		
		UnitTest test(testStream);
		_tests.insert(std::make_pair(test.name(), test));
	}
}

void HirManager::load(const Module& module)
{
	assert(_modules.count(module.name()) == 0);
	
	_modules.insert(std::make_pair(module.name(), module));
}

bool HirManager::runTests(std::ostream& results)
{
/*
	bool pass = true;
		
	std::stringstream message;
	
	for(TestMap::iterator test = _tests.begin(); test != _tests.end(); ++test)
	{
		ModuleMap::iterator module = _modules.find(test->second.moduleName());
		assert(module != _modules.end());
		
		runtime.clear();

		_addModule(module->second, runtime);

		report("Running unit test '" << test->second.name()
			<< "' against module '" << test->second.moduleName() << "'");

		report(" Setting up inputs.");
		for(Module::VariableDeclarationMap::const_iterator
			variable = test->second.inputs().begin();
			variable != test->second.inputs().end(); ++variable)
		{
			report("  " << variable->second.name() << " ( "
				<< variable->second.data().size() << " bytes)");
			harmony::Variable& reference = runtime.find(
				variable->second.name());
			
			reference.resize(variable->second.data().size());
			reference.assign(variable->second.data().data(),
				variable->second.data().size());
		}

		report(" Running test.");
		runtime.run();
		runtime.check();
		
		report(" Checking outputs.");
		for(Module::VariableDeclarationMap::const_iterator
			variable = test->second.outputs().begin();
			variable != test->second.outputs().end(); ++variable)
		{			
			harmony::Variable& computed = runtime.find(variable->second.name());

			report("  " << variable->second.name() << " (computed "
				<< computed.bytes() << " bytes)" << " (reference "
				<< variable->second.data().size() << " bytes)");
			
			if(variable->second.data().size() != computed.bytes())
			{
				message << "For variable " << variable->second.name()
					<< " computed size (" << computed.bytes() << " bytes)"
					<< " does not match reference size (reference "
					<< variable->second.data().size() << " bytes)\n";
				message << " computed data ( " 
					<< hydrazine::dataToString(computed.pointer(),
						std::min(computed.bytes(), (unsigned int)64)) << ")\n"
					<< " reference data ("
					<< hydrazine::dataToString(variable->second.data().data(),
						std::min((int)variable->second.data().size(), 64))
						<< " )\n";
				pass = false;
			}
			else if((computed.bytes() != 0) &&
				(std::string((char*)computed.pointer(), computed.bytes())
				!= variable->second.data()))
			{
				message << "For variable " << variable->second.name()
					<< " computed data (" 
					<< hydrazine::dataToString(computed.pointer(),
						std::min(computed.bytes(), (unsigned int)64)) << ")\n"
					<< " does not match reference data ( "
					<< hydrazine::dataToString(variable->second.data().data(),
						std::min((int)variable->second.data().size(), 64))
						<< ")\n";
				pass = false;
			}
		}
	}
	
	result = message.str();
	
	return pass;
	*/
	
	assertM(false, "Not implemented.");
}

void HirManager::runProgram()
{

	harmony::runtime::Runtime runtime;

	for(ModuleMap::iterator module = _modules.begin(); module != _modules.end(); ++module)
//	for(auto module : _modules)
	{
		runtime.load(&module->second);
		runtime.execute();
	}
}

}

}

