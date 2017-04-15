/*! \file   harmony-runtime.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Monday July 2, 2012
	\brief  The header file for the harmony runtime class.
*/

// Harmony Includes
#include <harmony/hir/interface/HirManager.h>
#include <harmony/cuda/interface/CudaDriver.h>

// Hydrazine Includes
#include <hydrazine/interface/ArgumentParser.h>

namespace harmony
{

void execute(const std::string& hirFileName, bool runTests, bool verbose)
{
	cuda::CudaDriver::cuInit(0);

	hir::HirManager manager;
	try
	{
		manager.load(hirFileName, runTests);
	}
	catch(const std::exception& e)
	{
		std::cout << "Harmony runtime failed to load .hir file: "
			<< hirFileName << "\n";
		std::cout << "Error: " << e.what() << "\n";
	}
	
	try
	{
		if(runTests)
		{
			manager.runTests(std::cout);
		}
		else
		{
			manager.runProgram();
		}
	}
	catch(const std::exception& e)
	{
		std::cout << "Harmony runtime failed to run .hir file: "
			<< hirFileName << "\n";
		std::cout << "Error: " << e.what() << "\n";
	}
}

}

int main(int argc, char** argv)
{
	hydrazine::ArgumentParser parser( argc, argv );
	parser.description("The harmony runtime.");
	
	std::string hirFile;
	bool verbose  = false;
	bool runTests = false;
	
	parser.parse("-i", "--input", hirFile, "",
		"The Harmony-IR program being executed.");
	parser.parse("-t", "--test", runTests, false, 
		"Run all unit tests in the Harmony-IR file.");
	parser.parse("-v", "--verbose", verbose, false, 
		"Print out information as the program is executing.");
	parser.parse();
	
	harmony::execute(hirFile, runTests, verbose);

	return 0;
}



