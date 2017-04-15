/*! \file   VariableDeclaration.h
	\date   Sunday October 25, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the hir::Variable class
*/

// Harmony Includes
#include <harmony/hir/interface/VariableDeclaration.h>
#include <harmony/hir/interface/KernelIR.h>

namespace harmony
{

namespace hir
{

VariableDeclaration::VariableDeclaration(Name name,
	const std::string& data) : _name(name), _data(data)
{

}

VariableDeclaration::VariableDeclaration(const pb::Variable& v)
	: _name(v.name()), _size(v.size()), _data(v.data()), _filename(v.filename()), 
	_isInput(v.input()), _isOutput(v.output())
{
	switch(v.type())
	{
	case pb::I8:
	{
		_type = i8;
		break; 
	}
	case pb::I16:
	{
		_type = i16;
		break; 
	}
	case pb::I32:
	{
		_type = i32;
		break; 
	}
	case pb::I64:
	{
		_type = i64;
		break; 
	}
	case pb::I128:
	{
		_type = i128;
		break; 
	}
	case pb::F32:
	{
		_type = f32;
		break; 
	}
	case pb::F64:
	{
		_type = f64;
		break; 
	}
	}
}

pb::Variable VariableDeclaration::pbIR() const
{
	pb::Variable variable;
	
	variable.set_name(name());
	variable.set_data(data());
	
	return variable;
}

VariableDeclaration::Name VariableDeclaration::name() const
{
	return _name;
}

uint64_t VariableDeclaration::size() const
{
	return _size;
}

const std::string& VariableDeclaration::data() const
{
	return _data;
}

const std::string& VariableDeclaration::filename() const
{
	return _filename;
}

VariableDeclaration::DataType VariableDeclaration::type() const
{
	return _type;
}

bool VariableDeclaration::isInput() const
{
	return _isInput;
}

bool VariableDeclaration::isOutput() const
{
	return _isOutput;
}

unsigned int VariableDeclaration::bytes(DataType t)
{
	switch(t)
	{
	case i8:  return 1;
	case i16: return 2;
	case i32: return 4;
	case i64: return 8;
	case i128: return 16;
	case f32: return 4;
	case f64: return 8;
	}
	
	return 1;
}

}

}

