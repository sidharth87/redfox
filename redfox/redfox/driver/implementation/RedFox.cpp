/*! \file RedFox.cpp
	\date Sunday October 31, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the RedFox class.
*/

#ifndef RED_FOX_CPP_INCLUDED
#define RED_FOR_CPP_INCLUDED

// Red Fox Includes
#include <redfox/driver/interface/RedFox.h>
#include <redfox/nvcc/interface/RelationalAlgebraCompiler.h>
#include <redfox/protocol/interface/HarmonyIRPrinter.h>

// Hydrazine Includes
#include <hydrazine/interface/ArgumentParser.h>
#include <hydrazine/interface/Exception.h>

// Standard Library Includes
#include <fstream>

namespace redfox
{

RedFox::RedFox(const std::string& ra, const std::string& hir,
	const std::string& dot) : _relationalAlgebraSourceFileName(ra),
	_harmonyIrFileName(hir), _dotFileName(dot), _verbose(false)
{

}

void RedFox::compile() const
{
	if(_relationalAlgebraSourceFileName.empty())
	{
		throw hydrazine::Exception("Relational algebra "
			"input file not specified.");
	}

	if(_harmonyIrFileName.empty())
	{
		throw hydrazine::Exception("Harmony IR output file not specified.");
	}

	std::ifstream raFile( _relationalAlgebraSourceFileName.c_str());
	std::ofstream hirFile(_harmonyIrFileName.c_str());
	
	if(!raFile.is_open())
	{
		throw hydrazine::Exception("Failed to open Relational Algebra "
			"protocol buffer file: '" + _relationalAlgebraSourceFileName
			+ "' for reading.");
	}
	
	if(!hirFile.is_open())
	{
		throw hydrazine::Exception("Failed to open Harmony IR "
			"protocol buffer file: '" + _harmonyIrFileName + "' for writing.");
	}
	
	nvcc::RelationalAlgebraCompiler compiler;
	
	compiler.compile(hirFile, raFile);
}

void RedFox::setVerboseMode(bool isVerbose)
{
	_verbose = isVerbose;
}

void RedFox::printRelationalAlgebraFile() const
{
	if(_relationalAlgebraSourceFileName.empty())
	{
		throw hydrazine::Exception("Relational algebra "
			"input file not specified.");
	}

	std::ifstream raFile( _relationalAlgebraSourceFileName.c_str());
	
	if(!raFile.is_open())
	{
		throw hydrazine::Exception("Failed to open Relational Algebra "
			"protocol buffer file: '" + _relationalAlgebraSourceFileName
			+ "' for reading.");
	}
	
	nvcc::RelationalAlgebraCompiler compiler;
	
	std::cout << compiler.getRAString(raFile);
}

void RedFox::printHarmonyIRFile() const
{
	if(_harmonyIrFileName.empty())
	{
		throw hydrazine::Exception("Harmony IR input file not specified.");
	}

	std::ifstream hirFile(_harmonyIrFileName.c_str());
	
	if(!hirFile.is_open())
	{
		throw hydrazine::Exception("Failed to open Harmony IR "
			"protocol buffer file: '" + _harmonyIrFileName + "' for reading.");
	}
	
	nvcc::RelationalAlgebraCompiler compiler;
	
	std::cout << compiler.getHIRString(hirFile);
}

void RedFox::printDOTFile() const
{
	if(_dotFileName.empty())
	{
		throw hydrazine::Exception("DOT output file not specified.");
	}

	std::ofstream file(_dotFileName.c_str());
	
	if(!file.is_open())
	{
		throw hydrazine::Exception("Failed to open DOT "
			"file: '" + _dotFileName + "' for writing.");
	}
	
	if(_harmonyIrFileName.empty())
	{
		throw hydrazine::Exception("Harmony IR input file not specified.");
	}

	std::ifstream hirFile(_harmonyIrFileName.c_str());
	
	if(!hirFile.is_open())
	{
		throw hydrazine::Exception("Failed to open Harmony IR "
			"protocol buffer file: '" + _harmonyIrFileName + "' for reading.");
	}
	
	pb::HarmonyIRPrinter printer(hirFile);
	
	printer.write(file);
}

}

int main(int argc, char** argv)
{
	hydrazine::ArgumentParser parser(argc, argv);

	parser.description("The RedFox compiler from a relational algebra protocol"
		" buffer format into a Harmony IR protocol buffer that can be executed"
		" by the Harmony Runtime.");

	std::string hir;
	std::string ra;
	std::string dot;
	bool verbose;
	bool compile;
	bool printRA;
	bool printHIR;
	bool graphviz;

	parser.parse("-v", "--verbose", verbose, false, 
		"Print out extra information during the compile.");
	parser.parse("-a", "--print-ra", printRA, false, 
		"Print out the contents of the RA file in a readable form.");
	parser.parse("-r", "--print-hir", printHIR, false, 
		"Print out the contents of the HIR file in a readable form.");
	parser.parse("-c", "--compile", compile, false, 
		"Compile the input RA file into an output HIR file.");
	parser.parse("-i", "--ra-file", ra, "", 
		"The input relational algebra protocol buffer file path.");
	parser.parse("-o", "--hir-file", hir, "", 
		"The output harmony ir protocol buffer file path.");
	parser.parse("-d", "--dot-file", dot, "", 
		"The output DOT file path.");
	parser.parse("-g", "--graphviz", graphviz, false, 
		"Write out a graphviz DOT file.");
	parser.parse();

	redfox::RedFox fox(ra, hir, dot);

	fox.setVerboseMode(verbose);

	try
	{
		if(compile)       fox.compile();
		else if(printRA)  fox.printRelationalAlgebraFile();
		else if(printHIR) fox.printHarmonyIRFile();
		else if(graphviz) fox.printDOTFile();
		else              std::cout << parser.help();
	}
	catch(const hydrazine::Exception& e)
	{
		std::cout << "\n[RedFox Error]: " << e.what() << "\n\n";
	}	
	
	return 0;
}

#endif

