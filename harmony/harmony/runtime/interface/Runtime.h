/*! \file   Runtime.h 
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Monday July 2, 2012
	\brief  The header file for the Runtime class
*/

#pragma once

// Harmony Inlcludes
#include <harmony/hir/interface/VariableDeclaration.h>

// CUDA Includes
#include <harmony/cuda/include/cuda.h>

// Standard Library Includes
#include <cstdint>
#include <vector>
#include <map>

// Forward Declarations
namespace harmony { namespace hir { class Module;     } }
namespace harmony { namespace hir { class KernelCall; } }
namespace harmony { namespace hir { class Block;      } }

namespace llvm { class Module;   }
namespace llvm { class Function; }

namespace harmony
{

namespace runtime
{

/*! \brief A runtime for executing Harmony IR programs */
class Runtime
{
public:
	/*! \brief Load a module into the runtime */
	void load(const hir::Module* module);

public:
	/*! \brief Begin execution of the module */
	void execute();

public:
	uint64_t getKernelParameter(const hir::KernelCall& call,
		unsigned int index);
	uint64_t getVariable(hir::VariableDeclaration::Name variable);

private:
	typedef                           uint64_t DevicePointer;
	typedef hir::VariableDeclaration::DataType DataType;
	typedef hir::VariableDeclaration::Name     VariableName;
		
	class Operation
	{
	public:
		Operation(Runtime* runtime);
	
	public:
		virtual bool execute(hir::Block*&) = 0;
	
	protected:
		Runtime* _runtime;
	
	};

	class Operand
	{
	public:
		DataType      type;
		DevicePointer pointer;
	};
	
	typedef std::vector<Operand> OperandVector;

	class KernelLaunch : public Operation
	{
	public:
		KernelLaunch(Runtime* runtime, const hir::KernelCall* kernel);
	
	public:
		const hir::KernelCall* kernel;
		OperandVector          inputs;
		OperandVector          outputs; 
	
	};
	
	class PTXKernelLaunch : public KernelLaunch
	{
	public:
		typedef CUfunction Handle;
		
	public:
		PTXKernelLaunch(Runtime* runtime,
			const hir::KernelCall* kernel, Handle handle);
	
	public:
		virtual bool execute(hir::Block*&);
		
	public:
		Handle kernelHandle;
	};
	
	class ControlLaunch: public PTXKernelLaunch
	{
	public:
		ControlLaunch(Runtime* runtime,
			const hir::KernelCall* kernel, Handle handle);
	
	public:
		virtual bool execute(hir::Block*&);
	};

	typedef void (*ExternalKernel)(void*);
	
	class BinaryKernelLaunch : public KernelLaunch
	{
	public:
		BinaryKernelLaunch(Runtime* runtime,
			const hir::KernelCall* kernel, void* functionPointer);
		
	public:
		virtual bool execute(hir::Block*&);
		
	public:
		void* functionPointer;
	};

	class VariableUpdate : public Operation
	{
	public:
		VariableUpdate(Runtime* runtime, VariableName variable);
				
	public:
		VariableName name;
	
	};

	class VariableGetsize : public VariableUpdate
	{
	public:
		VariableGetsize(Runtime* runtime,
			VariableName variable, VariableName hostVariable);
		
	public:
		virtual bool execute(hir::Block*&);
		
	public:
		VariableName hostVariable;
	};

	class VariableAllocation : public VariableUpdate
	{
	public:
		VariableAllocation(Runtime* runtime,
			VariableName variable, uint64_t size);
		
	public:
		virtual bool execute(hir::Block*&);
		
	public:
		uint64_t size;

	};

	class VariableResize : public VariableUpdate
	{
	public:
		VariableResize(Runtime* runtime,
			VariableName variable, VariableName size);
		
	public:
		virtual bool execute(hir::Block*&);
		
	public:
		VariableName size;

	};
	
	class VariableUpdateSize : public VariableUpdate
	{
	public:
		VariableUpdateSize(Runtime* runtime,
			VariableName variable, VariableName size);
		
	public:
		virtual bool execute(hir::Block*&);
		
	public:
		VariableName size;

	};

	class VariableFree : public VariableUpdate
	{
	public:
		VariableFree(Runtime* runtime,
			VariableName variable);
		
	public:
		virtual bool execute(hir::Block*&);

	public:
		
	};

	class LoadedKernel
	{
	public:
		LoadedKernel(const std::string& name);
		
	public:
		virtual void load(const std::string& binary) = 0;
	
	public:
		std::string name;
	};
	
	class LoadedBinaryKernel : public LoadedKernel
	{
	public:
		LoadedBinaryKernel(const hir::KernelCall* kernel);
		~LoadedBinaryKernel();

	public:
		llvm::Function* jitFunction(llvm::Module* m);
		void load(const std::string& data);

	public:
		void* functionPointer;
		void* originalFunctionPointer;

	public:
		std::string mangledName();

	private:
		std::string            _fileName;
		const hir::KernelCall* _kernel;
		void*                  _libraryHandle;
		
	private:
		llvm::Module* _module;
	};
	
	class LoadedPTXKernel : public LoadedKernel
	{
	public:
		LoadedPTXKernel(const std::string& name);
		~LoadedPTXKernel();

	public:
		void load(const std::string& data);
	
	public:
		PTXKernelLaunch::Handle handle;

	private:
		CUmodule _moduleHandle;
	};

private:
	class VariableDescriptor
	{
	public:
		VariableDescriptor(CUdeviceptr address, DataType type,
			uint64_t sizeInElements);
	
	public:
		CUdeviceptr address;
		DataType    type;
		uint64_t    sizeInElements;
		uint64_t    capacityInElements;
	};

	typedef std::map<VariableName, VariableDescriptor> VariableMap;
	typedef std::map<std::string,  LoadedKernel*>      KernelMap;
	
	typedef std::vector<Operation*> OperationVector;

	class BasicBlockSchedule
	{
	public:
		BasicBlockSchedule(const OperationVector&);
	
	public:
		OperationVector scheduledOperations;
	};
	
	typedef unsigned int BasicBlockId;

	typedef std::map<BasicBlockId, BasicBlockSchedule>
		LabelToBasicBlockScheduleMap;
		
private:
	void _destroyCudaContext();
	void _loadCudaContext();
	void _loadKernels();
	void _loadVariables();
	bool _runBlock(hir::Block*);
	const BasicBlockSchedule& _scheduleBasicBlock(hir::Block*);

private:
	void _loadPTXKernel(const hir::KernelCall& kernel);
	void _loadExternalKernel(const hir::KernelCall& kernel);

private:
	const hir::Module* _module;
	VariableMap        _variables;
	KernelMap          _kernels;
	CUcontext          _context;
	CUdevice           _device;
	
	LabelToBasicBlockScheduleMap _scheduledBlocks;

};

}

}


