/* 	\file CudaCompilerInterface.h
	\date Monday October 25, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the CudaCompilerInterface class.
*/

#ifndef CUDA_COMPILER_INTERFACE_H_INCLUDED
#define CUDA_COMPILER_INTERFACE_H_INCLUDED

// Standard Library Includes
#include <string>
#include <unordered_map>

/*! \brief The namespace for wrapper classes around NVCC */
namespace nvcc
{

class Compiler
{
public:
	enum Architecture
	{
		sm_12,
		sm_20,
		sm_21,
		sm_23,
		sm_35,
		InvalidArchitecture
	};

public:
	/*! \brief Set the architecture to compile for */
	static void setArchitecture(Architecture architecture);

	/*! \brief Set the include directory */
	static void setIncludePath(const std::string& includePath);

	/*! \brief Compile the source code conatined in the input string */		
	static void compileToPTX(const std::string& cudaSource);

	/*! \brief Compile the source code conatined in the input string */		
	static void compileToBIN(const std::string& cudaSource, const std::string& external, const std::string EXTOBJFileName);

	/*! \brief Compile the source code conatined in the input string */		
	static void compileToModernBIN(const std::string& cudaSource, const std::string& external, const std::string EXTOBJFileName);

	/*! \brief Compile the source code conatined in the input string */		
	static void compileToB40CBIN(const std::string& cudaSource, const std::string& external, const std::string EXTOBJFileName);


	/*! \brief Get the PTX assembly for the most recently compiled kernel */
	static const std::string& compiledPTX();

	/*! \brief Get the PTX assembly for the most recently compiled kernel */
	static const std::string& compiledBIN();

	/*! \brief Get a string representation of the architecture */
	static std::string flags(Architecture architecture);

private:
	/*! \brief An interface for invoking nvcc on some source code */
	class CudaCompilerInterface
	{
	public:
		/*! \brief Constructor sets the default values */
		CudaCompilerInterface();
	
		/*! \brief Set the architecture to compile for */
		void setArchitecture(Architecture architecture);

		/*! \brief Set the include directory */
		void setIncludePath(const std::string& includePath);

		/*! \brief Compile the source code conatined in the input string */		
		void compileToPTX(const std::string& cudaSource);
	
		/*! \brief Compile the source code conatined in the input string */		
		void compileToBIN(const std::string& cudaSource, const std::string& external, const std::string EXTOBJFileName);

		/*! \brief Compile the source code conatined in the input string */		
		void compileToModernBIN(const std::string& cudaSource, const std::string& external, const std::string EXTOBJFileName);

		/*! \brief Compile the source code conatined in the input string */		
		void compileToB40CBIN(const std::string& cudaSource, const std::string& external, const std::string EXTOBJFileName);


		/*! \brief Get the PTX assembly for the most recently compiled kernel */
		const std::string& compiledPTX() const;

		/*! \brief Get the PTX assembly for the most recently compiled kernel */
		const std::string& compiledBIN() const;

	private:
		/*! \brief Check if the source is in the cache, set the ptx if it is */
		bool _checkPTXCache(const std::string& cudaSource);

		/*! \brief Check if the source is in the cache, set the ptx if it is */
		bool _checkBINCache(const std::string& cudaSource);

	private:
		/*! \brief A cache from source code to compiled assembly */
		typedef std::unordered_map<std::string, std::string> CodeCache;

	private:
		/*! \brief The assembly code for the most recently compiled kernel */
		std::string  _ptx;
		/*! \brief The assembly code for the most recently compiled kernel */
		std::string  _bin;
		/*! \brief The target architecture */
		Architecture _architecture;
		/*! \brief The include path */
		std::string  _include;
		/*! \brief A cache of the most recently compiled programs */
		CodeCache    _ptxcache;
		/*! \brief A cache of the most recently compiled programs */
		CodeCache    _bincache;
	};

private:
	static CudaCompilerInterface _interface;
};

}

#endif

