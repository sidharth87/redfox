/*! \file RelationalAlgebraCompiler.h
	\date Sunday October 31, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the RelationalAlgebraCompiler class.
*/

#ifndef RELATIONAL_ALGEBRA_COMPILER_H_INCLUDED
#define RELATIONAL_ALGEBRA_COMPILER_H_INCLUDED

// Standard Library Includes
#include <ostream>
#include <istream>

namespace nvcc
{

/*! \brief An interface to a compiler from a ra.proto to a hir.proto */
class RelationalAlgebraCompiler
{
public:
	/*! \brief Read in from a stream containing a RA pb, produce an HIR pb */
	void compile(std::ostream& hir, std::istream& ra) const;

	/*! \brief Read in a stream containing a RA pb, get a readable string */
	std::string getRAString(std::istream& ra) const;
	
	/*! \brief Read in a stream containing a HIR pb, get a readable string */
	std::string getHIRString(std::istream& hir) const;

};

}

#endif

