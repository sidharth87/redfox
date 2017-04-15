/*	\file   UnitTest.cpp
	\date   Wednesday December 15, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the unit test class
*/

// Harmony Includes
#include <harmony/hir/interface/UnitTest.h>
#include <harmony/hir/interface/KernelIR.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/Exception.h>

// Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace harmony
{

namespace hir
{

UnitTest::UnitTest(const std::string& name, const std::string& moduleName,
	const DataMap& inputs, const DataMap& outputs) : _name(name),
	_moduleName(moduleName), _inputs(inputs), _outputs(outputs)
{

}

UnitTest::UnitTest(std::istream& s)
{
	typedef google::protobuf::RepeatedPtrField<pb::Variable> RepeatedVariable;

	pb::Test test;
	
	if(!test.ParseFromIstream(&s))
	{
		throw hydrazine::Exception("Failed to parse protocol buffer "
			"containing Harmony IR Unit Test.");
	}
	
	_name       = test.name();
	_moduleName = test.programname();

	report("Creating unit test '" << name() << "'");

	report(test.DebugString());

	// Add inputs
	for(RepeatedVariable::const_iterator variable = test.inputs().begin();
		variable != test.inputs().end(); ++variable)
	{
		_inputs.insert(std::make_pair(variable->name(), 
			VariableDeclaration(variable->name(), variable->data())));
	}

	// Add outputs
	for(RepeatedVariable::const_iterator variable = test.outputs().begin();
		variable != test.outputs().end(); ++variable)
	{
		_outputs.insert(std::make_pair(variable->name(), 
			VariableDeclaration(variable->name(), variable->data())));
	}
}

const std::string& UnitTest::name() const
{
	return _name;
}

const std::string& UnitTest::moduleName() const
{
	return _moduleName;
}

const UnitTest::DataMap& UnitTest::inputs() const
{
	return _inputs;
}

const UnitTest::DataMap& UnitTest::outputs() const
{
	return _outputs;
}

}

}

