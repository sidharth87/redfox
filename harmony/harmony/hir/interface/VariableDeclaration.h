/*! \file VariableDeclaration.h
	\date Sunday October 25, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the hir::Variable class
*/

#pragma once

// Standard Library Includes
#include <string>
#include <cstdint>

// Forward Declarations
namespace harmony { namespace hir { namespace pb { class Variable; } } }

namespace harmony
{

namespace hir
{

/*! \brief A class for representing a Harmony variable */
class VariableDeclaration
{
public:
	/*! \brief Harmony variable */
	typedef unsigned int Name;

	enum DataType
	{
		i8,
		i16,
		i32,
		i64,
		i128,
		f32,
		f64
	};

public:	
	static const Name InvalidName = -1;

public:
	/*! \brief Default Constructor */
	VariableDeclaration(Name name = InvalidName,
		const std::string& data = "");
	/*! \brief Construct an IR object from a protocol buffer equiavalent */
	explicit VariableDeclaration(const hir::pb::Variable& v);
	/*! \brief Convert an IR object into a protocol buffer equivalent */
	hir::pb::Variable pbIR() const;

public:
	/*! \brief Get the name of the variable */
	Name name() const;
	/*! \brief Get the size of the variable */
	uint64_t size() const;
	/*! \brief Get the initial contents or filename */
	const std::string& data() const;
	/*! \brief Get the initial contents or filename */
	const std::string& filename() const;

	/*! \brief Get the data type */
	DataType type() const;

public:	
	/*! \brief Is the variable a program input */
	bool isInput() const;
	/*! \brief Is the variable a program output */
	bool isOutput() const;

public:
	static unsigned int bytes(DataType t);

private:
	Name        _name; //! Variable name
	uint64_t    _size; //! Variable size in elements
	std::string _data; //! Initial variable contents
	std::string _filename;
	DataType    _type; //! Primitive data type
	bool	    _isInput;
	bool	    _isOutput;
};

}

}

