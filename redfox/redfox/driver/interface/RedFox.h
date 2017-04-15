/*! \file RedFox.h
	\date Sunday October 31, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the RedFox class.
*/

#ifndef RED_FOX_H_INCLUDED
#define RED_FOR_H_INCLUDED

// Standard Library Includes
#include <string>

/*! \brief A namespace for the RedFox compiler. */ 
namespace redfox
{

/*! \brief A class for controling the command line interface to RedFox. */
class RedFox
{
public:
	/*! \brief Set the input and output file names */
	RedFox(const std::string& ra, const std::string& hir,
		const std::string& dot);
	
	/*! \brief Compile the input into the output */
	void compile() const;
	/*! \brief Set whether ot not verbose mode is enabled */
	void setVerboseMode(bool isVerbose);
	/*! \brief Print out the contents of the input file */
	void printRelationalAlgebraFile() const;
	/*! \brief Print out the contents of the output file only */
	void printHarmonyIRFile() const;
	/*! \brief Print out the contents of the output file only */
	void printDOTFile() const;
	
private:
	std::string _relationalAlgebraSourceFileName;
	std::string _harmonyIrFileName;
	std::string _dotFileName;
	bool        _verbose;
};

}

int main(int argc, char** argv);

#endif

