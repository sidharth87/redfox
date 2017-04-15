/* 	\file CudaCompilerInterface.cpp
	\date Monday October 25, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the CudaCompilerInterface class.
*/

#ifndef CUDA_COMPILER_INTERFACE_CPP_INCLUDED
#define CUDA_COMPILER_INTERFACE_CPP_INCLUDED

// Red Fox Includes
#include <redfox/nvcc/interface/CudaCompilerInterface.h>
#include <redfox/nvcc/interface/RelationalAlgebraKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1    

#define REPORT_ALL_CUDA_SOURCE  0
#define REPORT_ALL_PTX_ASSEMBLY 0
#define REPORT_ALL_BIN 0

/*! \brief The namespace for wrapper classes around NVCC */
namespace nvcc
{

void Compiler::setArchitecture(Architecture architecture)
{
	_interface.setArchitecture(architecture);
}

void Compiler::setIncludePath(const std::string& includePath)
{
	_interface.setIncludePath(includePath);
}

void Compiler::compileToPTX(const std::string& cudaSource)
{
	_interface.compileToPTX(cudaSource);
}

void Compiler::compileToBIN(const std::string& cudaSource, 
	const std::string& external, const std::string EXTOBJFileName)
{
	_interface.compileToBIN(cudaSource, external, EXTOBJFileName);
}

void Compiler::compileToModernBIN(const std::string& cudaSource, 
	const std::string& external, const std::string EXTOBJFileName)
{
	_interface.compileToModernBIN(cudaSource, external, EXTOBJFileName);
}

void Compiler::compileToB40CBIN(const std::string& cudaSource, 
	const std::string& external, const std::string EXTOBJFileName)
{
	_interface.compileToB40CBIN(cudaSource, external, EXTOBJFileName);
}

const std::string& Compiler::compiledPTX()
{
	return _interface.compiledPTX();
}

const std::string& Compiler::compiledBIN()
{
	return _interface.compiledBIN();
}

std::string Compiler::flags(Architecture architecture)
{
	switch(architecture)
	{
	case sm_12: return "-arch sm_12";
	case sm_20: return "-arch sm_20";
	case sm_21: return "-arch sm_21";
	case sm_23: return "-arch sm_23";
	case sm_35: return "-gencode arch=compute_35,code=sm_35";
	default: break;
	}
	
	return "ARCHITECTURE_INVALID";
}

Compiler::CudaCompilerInterface::CudaCompilerInterface()
	: _architecture(sm_35), _include(".")
{

}

void Compiler::CudaCompilerInterface::setArchitecture(Architecture architecture)
{
	_architecture = architecture;	
}

void Compiler::CudaCompilerInterface::setIncludePath(const std::string& path)
{
	_include = path;
}

void Compiler::CudaCompilerInterface::compileToPTX(const std::string& cudaSource)
{
	if(_checkPTXCache(cudaSource)) return;

	reportE(REPORT_ALL_CUDA_SOURCE, "Compiling cuda source\n" << cudaSource);
	
	std::string tempPTXFileName  = "RedFox.tmp.ptx";
	std::string tempCUDAFileName = "RedFox.tmp.cu";
	
	std::ofstream cudaFile(tempCUDAFileName.c_str());
	cudaFile.write(cudaSource.c_str(), cudaSource.size());
	cudaFile.flush();
	
	std::string command = "nvcc --ptx -o " + tempPTXFileName + " "
		+ tempCUDAFileName + " " + flags(_architecture);
	
	if(!_include.empty()) command += " -I" + _include;
	
	report("Invoking NVCC with command '" << command << "'");
	int ret = std::system(command.c_str());

	std::ifstream ptxFile(tempPTXFileName);
	
	if(!ptxFile.is_open() || ret == -1)
	{
		throw hydrazine::Exception("NVCC failed to generate "
			"PTX assembly file '" + tempPTXFileName + "'.");
	}
	
//	ptxFile.seekg(0, std::ios::end);
//	size_t length = ptxFile.tellg();
//	ptxFile.seekg(0, std::ios::beg);
//	
//	_ptx.resize(length);
//	
//	ptxFile.read((char*)_ptx.c_str(), length);
	
	std::stringstream pss;
	pss << ptxFile.rdbuf();
	_ptx = pss.str();
	
	reportE(REPORT_ALL_PTX_ASSEMBLY,
		"Generated the following kernel: \n" << _ptx);
	
	std::remove(tempPTXFileName.c_str());
	std::remove(tempCUDAFileName.c_str());

	// insert it into the cache
	std::string key = cudaSource + flags(_architecture) + _include;

	_ptxcache.insert(std::make_pair(key, _ptx));	
}

void Compiler::CudaCompilerInterface::compileToBIN(const std::string& cudaSource,
	const std::string& external, const std::string EXTOBJFileName)
{
	if(_checkBINCache(cudaSource)) return;

	reportE(REPORT_ALL_CUDA_SOURCE, "Compiling cuda source\n" << cudaSource);
	
	std::string tempBINFileName  = "RedFox.tmp.so";
	std::string tempCUDAFileName = "RedFox.tmp.cu";
	std::string tempOBJFileName = "RedFox.tmp.o";
//	std::string tempEXTOBJFileName = "RedFox.external.tmp.o";
	
	std::ofstream cudaFile(tempCUDAFileName.c_str());
	cudaFile.write(cudaSource.c_str(), cudaSource.size());
	cudaFile.flush();
	
	std::string command1 = "nvcc -c -o " + tempOBJFileName + " -Xcompiler=-fPIC "
		+ tempCUDAFileName + " " + flags(_architecture);
	
	if(!_include.empty()) command1 += " -I" + _include;
	
	report("Invoking NVCC with command '" << command1 << "'");
	int ret_obj = std::system(command1.c_str());

	std::ifstream objFile(tempOBJFileName);
	
	if(!objFile.is_open() || ret_obj == -1)
	{
		throw hydrazine::Exception("NVCC failed to generate "
			"Object file '" + tempOBJFileName + "'.");
	}
	
//	std::string command2 = "nvcc -c -o " + tempEXTOBJFileName + " -Xcompiler=-fPIC "
//		+ external + " " + flags(_architecture);
//	
//	if(!_include.empty()) command2 += " -I" + _include;
//	
//	report("Invoking NVCC with command '" << command2 << "'");
//	int ret_ext = std::system(command2.c_str());
//
//	std::ifstream extFile(tempEXTOBJFileName);
//	
//	if(!extFile.is_open() || ret_ext == -1)
//	{
//		throw hydrazine::Exception("NVCC failed to generate "
//			"external Object file '" + tempEXTOBJFileName + "'.");
//	}

	std::string command3 = "nvcc -shared -o " + tempBINFileName + " -Xlinker=--no-as-needed "
		+ /*temp*/EXTOBJFileName + " " + tempOBJFileName + " " + flags(_architecture);
	
	report("Invoking NVCC with command '" << command3 << "'");
	int ret_bin = std::system(command3.c_str());
	
//	std::string temp = "nm " + tempBINFileName + ">> dump";
//	int ret_tmp = system(temp.c_str());
//
//	if(ret_tmp == -1)
//		printf("ret error\n");

	std::ifstream binFile(tempBINFileName);
	
	if(!binFile.is_open() || ret_bin == -1)
	{
		throw hydrazine::Exception("NVCC failed to generate "
			"external Object file '" + tempBINFileName + "'.");
	}

//	binFile.seekg(0, std::ios::end);
//	size_t length = binFile.tellg();
//	binFile.seekg(0, std::ios::beg);
//	
//	_bin.resize(length);
//	
//	binFile.read((char*)_bin.c_str(), length);
	std::stringstream bss;
	bss << binFile.rdbuf();
	_bin = bss.str();
	
	reportE(REPORT_ALL_BIN,
		"Generated the following kernel: \n" << _bin);
	
	std::remove(tempBINFileName.c_str());
	std::remove(tempCUDAFileName.c_str());
	std::remove(tempOBJFileName.c_str());
//	std::remove(tempEXTOBJFileName.c_str());

	// insert it into the cache
	std::string key = cudaSource + flags(_architecture) + _include;

	_bincache.insert(std::make_pair(key, _bin));	
}

void Compiler::CudaCompilerInterface::compileToModernBIN(const std::string& cudaSource,
	const std::string& external, const std::string EXTOBJFileName)
{
	if(_checkBINCache(cudaSource)) return;

	reportE(REPORT_ALL_CUDA_SOURCE, "Compiling cuda source\n" << cudaSource);
	
	std::string tempBINFileName  = "RedFox.tmp.so";
	std::string tempCUDAFileName = "RedFox.tmp.cu";
	std::string tempOBJFileName = "RedFox.tmp.o";
//	std::string tempEXTOBJFileName = "RedFox.external.tmp.o";
	
	std::ofstream cudaFile(tempCUDAFileName.c_str());
	cudaFile.write(cudaSource.c_str(), cudaSource.size());
	cudaFile.flush();
	
	std::string command1 = "nvcc -c -o " + tempOBJFileName + " -Xcompiler=-fPIC "
		+ tempCUDAFileName + " " + flags(_architecture);
	
	if(!_include.empty()) command1 += " -I" + _include;
	
	report("Invoking NVCC with command '" << command1 << "'");
	int ret_obj = std::system(command1.c_str());

	std::ifstream objFile(tempOBJFileName);
	
	if(!objFile.is_open() || ret_obj == -1)
	{
		throw hydrazine::Exception("NVCC failed to generate "
			"Object file '" + tempOBJFileName + "'.");
	}
	
//	std::string command2 = "nvcc -c -o " + tempEXTOBJFileName + " -Xcompiler=-fPIC "
//		+ external + " " + flags(_architecture);
//	
//	if(!_include.empty()) command2 += " -I" + _include;
//	
//	report("Invoking NVCC with command '" << command2 << "'");
//	int ret_ext = std::system(command2.c_str());
//
//	std::ifstream extFile(tempEXTOBJFileName);
//	
//	if(!extFile.is_open() || ret_ext == -1)
//	{
//		throw hydrazine::Exception("NVCC failed to generate "
//			"external Object file '" + tempEXTOBJFileName + "'.");
//	}

	std::string command3 = "nvcc -shared -o " + tempBINFileName + " -Xlinker=--no-as-needed "
		+ /*temp*/EXTOBJFileName + " " + tempOBJFileName + " format.o random.o mgpucontext.o " + flags(_architecture);
	
	report("Invoking NVCC with command '" << command3 << "'");
	int ret_bin = std::system(command3.c_str());
	
//	std::string temp = "nm " + tempBINFileName + ">> dump";
//	int ret_tmp = system(temp.c_str());
//
//	if(ret_tmp == -1)
//		printf("ret error\n");

	std::ifstream binFile(tempBINFileName);
	
	if(!binFile.is_open() || ret_bin == -1)
	{
		throw hydrazine::Exception("NVCC failed to generate "
			"external Object file '" + tempBINFileName + "'.");
	}

//	binFile.seekg(0, std::ios::end);
//	size_t length = binFile.tellg();
//	binFile.seekg(0, std::ios::beg);
//	
//	_bin.resize(length);
//	
//	binFile.read((char*)_bin.c_str(), length);
	std::stringstream bss;
	bss << binFile.rdbuf();
	_bin = bss.str();
	
	reportE(REPORT_ALL_BIN,
		"Generated the following kernel: \n" << _bin);
	
	std::remove(tempBINFileName.c_str());
	std::remove(tempCUDAFileName.c_str());
	std::remove(tempOBJFileName.c_str());
//	std::remove(tempEXTOBJFileName.c_str());

	// insert it into the cache
	std::string key = cudaSource + flags(_architecture) + _include;

	_bincache.insert(std::make_pair(key, _bin));	
}

void Compiler::CudaCompilerInterface::compileToB40CBIN(const std::string& cudaSource,
	const std::string& external, const std::string EXTOBJFileName)
{
	if(_checkBINCache(cudaSource)) return;

	reportE(REPORT_ALL_CUDA_SOURCE, "Compiling cuda source\n" << cudaSource);
	
	std::string tempBINFileName  = "RedFox.tmp.so";
	std::string tempCUDAFileName = "RedFox.tmp.cu";
	std::string tempOBJFileName = "RedFox.tmp.o";
//	std::string tempEXTOBJFileName = "RedFox.external.tmp.o";
	
	std::ofstream cudaFile(tempCUDAFileName.c_str());
	cudaFile.write(cudaSource.c_str(), cudaSource.size());
	cudaFile.flush();
	
	std::string command1 = "nvcc -c -o " + tempOBJFileName + " -Xcompiler=-fPIC "
		+ tempCUDAFileName + " " + flags(_architecture);
	
	if(!_include.empty()) command1 += " -I" + _include;
	
	report("Invoking NVCC with command '" << command1 << "'");
	int ret_obj = std::system(command1.c_str());

	std::ifstream objFile(tempOBJFileName);
	
	if(!objFile.is_open() || ret_obj == -1)
	{
		throw hydrazine::Exception("NVCC failed to generate "
			"Object file '" + tempOBJFileName + "'.");
	}
	
//	std::string command2 = "nvcc -c -o " + tempEXTOBJFileName + " -Xcompiler=-fPIC "
//		+ external + " " + flags(_architecture);
//	
//	if(!_include.empty()) command2 += " -I" + _include;
//	
//	report("Invoking NVCC with command '" << command2 << "'");
//	int ret_ext = std::system(command2.c_str());
//
//	std::ifstream extFile(tempEXTOBJFileName);
//	
//	if(!extFile.is_open() || ret_ext == -1)
//	{
//		throw hydrazine::Exception("NVCC failed to generate "
//			"external Object file '" + tempEXTOBJFileName + "'.");
//	}

	std::string command3 = "nvcc -shared -o " + tempBINFileName + " -Xlinker=--no-as-needed "
		+ /*temp*/EXTOBJFileName + " " + tempOBJFileName + " " + flags(_architecture);
	
	report("Invoking NVCC with command '" << command3 << "'");
	int ret_bin = std::system(command3.c_str());
	
//	std::string temp = "nm " + tempBINFileName + ">> dump";
//	int ret_tmp = system(temp.c_str());
//
//	if(ret_tmp == -1)
//		printf("ret error\n");

	std::ifstream binFile(tempBINFileName);
	
	if(!binFile.is_open() || ret_bin == -1)
	{
		throw hydrazine::Exception("NVCC failed to generate "
			"external Object file '" + tempBINFileName + "'.");
	}

//	binFile.seekg(0, std::ios::end);
//	size_t length = binFile.tellg();
//	binFile.seekg(0, std::ios::beg);
//	
//	_bin.resize(length);
//	
//	binFile.read((char*)_bin.c_str(), length);
	std::stringstream bss;
	bss << binFile.rdbuf();
	_bin = bss.str();
	
	reportE(REPORT_ALL_BIN,
		"Generated the following kernel: \n" << _bin);
	
	std::remove(tempBINFileName.c_str());
	std::remove(tempCUDAFileName.c_str());
	std::remove(tempOBJFileName.c_str());
//	std::remove(tempEXTOBJFileName.c_str());

	// insert it into the cache
	std::string key = cudaSource + flags(_architecture) + _include;

	_bincache.insert(std::make_pair(key, _bin));	
}

const std::string& Compiler::CudaCompilerInterface::compiledPTX() const
{
	return _ptx;
}

const std::string& Compiler::CudaCompilerInterface::compiledBIN() const
{
	return _bin;
}

bool Compiler::CudaCompilerInterface::_checkPTXCache(const std::string& cudaSource)
{
	std::string key = cudaSource + flags(_architecture) + _include;
	
	CodeCache::iterator entry = _ptxcache.find(key);
	
	if(entry == _ptxcache.end()) return false;
	
	_ptx = entry->second;
	
	return true;
}

bool Compiler::CudaCompilerInterface::_checkBINCache(const std::string& cudaSource)
{
	std::string key = cudaSource + flags(_architecture) + _include;
	
	CodeCache::iterator entry = _bincache.find(key);
	
	if(entry == _bincache.end()) return false;
	
	_bin = entry->second;
	
	return true;
}

Compiler::CudaCompilerInterface Compiler::_interface;
}

#endif

