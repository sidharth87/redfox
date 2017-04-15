/*! \file   Runtime.cpp 
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Tuesday July 3, 2012
	\brief  The source file for the Runtime class
*/

// Harmony Includes
#include <harmony/runtime/interface/Runtime.h>
#include <harmony/hir/interface/Module.h>
#include <harmony/cuda/interface/CudaDriver.h>

#include <harmony/llvm/interface/LLVMKernel.h>
#include <harmony/llvm/interface/LLVMStatement.h>
#include <harmony/llvm/interface/LLVMInstruction.h>
#include <harmony/llvm/interface/LLVMState.h>

// Hydrazine Includes
#include <hydrazine/interface/Casts.h>
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/string.h>

// Configuration
//#include <configure.h>

// LLVM Includes
#if HAVE_LLVM
#include <llvm/Transforms/Scalar.h>
#include <llvm/PassManager.h>
//#include <llvm/Target/TargetData.h>
//#include <llvm/IR/DataLayout.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Assembly/Parser.h>
#include <llvm/Analysis/Verifier.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#endif

// Linux Includes
#include <dlfcn.h>

// Standard Library Includes
#include <cassert>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <fstream>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1 


namespace harmony
{

namespace runtime
{

void Runtime::load(const hir::Module* module)
{
	_module = module;
}

void Runtime::execute()
{
	_loadCudaContext();
	_loadKernels();
	_loadVariables();
	
	// main loop
//	hir::Block*  block  = 0;
	bool         exited = false;

	for(auto block : _module->cfg())
//	while(!exited)
	{
		exited = _runBlock(&block);

		if(exited) break;
	}

	_destroyCudaContext();
//	system("rm *.library");
}

uint64_t Runtime::getKernelParameter(const hir::KernelCall& call,
	unsigned int index)
{
	assert(index < call.operands().size());
	return getVariable(call.operands()[index].variable().name());
}

uint64_t Runtime::getVariable(hir::VariableDeclaration::Name variable)
{
	auto descriptor = _variables.find(variable);
	assert(descriptor != _variables.end());
	
	uint64_t result = 0;
	CUresult status = cuda::CudaDriver::cuMemcpyDtoH(&result,
		descriptor->second.address, sizeof(uint64_t));	

	if(status != CUDA_SUCCESS)
	{	
		std::stringstream message;

		message << "Failed to get variable from " << descriptor->second.address << "\n";		
		message << "Error Message: " << cuda::CudaDriver::toString(status) << "\n";
		throw std::runtime_error(message.str());
	}

	return result;
}

void Runtime::_loadCudaContext()
{
	int devices = 0;
	cuda::CudaDriver::cuDeviceGetCount(&devices);

	assert(devices > 0);

	cuda::CudaDriver::cuDeviceGet(&_device, 0);
	cuda::CudaDriver::cuCtxCreate(&_context, 0, _device);
}

void Runtime::_destroyCudaContext()
{
	/*if(_context)*/ cuda::CudaDriver::cuCtxDestroy(_context);
}

void Runtime::_loadKernels()
{
        cuda::CudaDriver driver;
	driver._interface.load();

	for(auto block : _module->cfg())
	{
		for(auto kernel : block.kernels)
		{
			if(kernel.isPTXKernel())
			{
				_loadPTXKernel(kernel);
			}
			else if(kernel.isExternalKernel())
			{
				_loadExternalKernel(kernel);
			}
			else
			{
				// No need to load the built-in kernels
			}
		}
	}
}

void Runtime::_loadVariables()
{
report("XXXX Starting load variables: " << __LINE__);
	for(auto variable : _module->variables())
	{
		report("loading variable " <<variable.second.name() 
			<< " needs " << variable.second.size() << " Bytes "
			<< "isInput " << variable.second.isInput()
			<< " isOutput " << variable.second.isOutput());

		CUdeviceptr newAddress  = 0;
		size_t sizeInBytes = variable.second.size();

		if(variable.second.isInput())
		{
			CUresult status_alloc = cuda::CudaDriver::cuMemAlloc(&newAddress, sizeInBytes);
		
			if(status_alloc != CUDA_SUCCESS)
			{	
				std::stringstream message;
		
				message << "Failed to allocate " << sizeInBytes
					<< " for variable with name " << variable.second.name() << "\n";		
			
				throw std::runtime_error(message.str());
			}

//			char* data = (char *)malloc(sizeInBytes);

			char* data;
			
			CUresult status_host_alloc = cuda::CudaDriver::cuMemHostAlloc((void **)&data, sizeInBytes, CU_MEMHOSTALLOC_WRITECOMBINED);		

			if(status_host_alloc != CUDA_SUCCESS)
			{	
				std::stringstream message;
		
				message << "Failed to allocate PIN host memory " << sizeInBytes
					<< " for variable with name " << variable.second.name() << "\n";		
			
				throw std::runtime_error(message.str());
			}

			std::string filename = variable.second.filename();

			if(filename.size() > 0)
			{
				std::ifstream file;
		
		  		file.open(filename, std::ios::binary);
		
			        if(!file.is_open())
			        {
			                throw hydrazine::Exception(
			                        "Could not open Harmony module in file: " + filename);
			        }
		
				file.read(data, sizeInBytes);
				file.close();
			}
			else
			{
				memcpy(data, variable.second.data().c_str(), sizeInBytes);

				if(sizeInBytes == 8)	
					report("initial value " << *((unsigned int *)data));	
			}
			
			CUevent start, stop;
			float transfer_time = 0.0f;
			cuda::CudaDriver::cuEventCreate(&start, CU_EVENT_DEFAULT);
			cuda::CudaDriver::cuEventCreate(&stop, CU_EVENT_DEFAULT);
			cuda::CudaDriver::cuEventRecord(start, 0);

			CUresult status_h2d = cuda::CudaDriver::cuMemcpyHtoD(newAddress,
				(void *)data, sizeInBytes);

			cuda::CudaDriver::cuEventRecord(stop, 0);
			cuda::CudaDriver::cuEventSynchronize(stop);
			cuda::CudaDriver::cuEventElapsedTime(&transfer_time, start, stop);
		
			cuda::CudaDriver::cuEventDestroy(start);	
			cuda::CudaDriver::cuEventDestroy(stop);	

			std::cout << sizeInBytes << " " << transfer_time << "\n";

			report("gpu address " << newAddress);	

			if(status_h2d != CUDA_SUCCESS)
			{	
				std::stringstream message;
		
				message << "Failed to resize (copy) " << sizeInBytes
					<< " for variable with name " << variable.second.name() << "\n";		
			
				throw std::runtime_error(message.str());
			}

//			if(sizeInBytes == 8)
//			{
//				unsigned int value_out;
//
//				CUresult status_d2h = cuda::CudaDriver::cuMemcpyDtoH(&value_out,
//					newAddress, sizeof(uint64_t));	
//			
//				if(status_d2h != CUDA_SUCCESS)
//				{	
//					std::stringstream message;
//			
//					message << "Failed to get variable from " << newAddress << "\n";		
//					message << "Error Message: " << cuda::CudaDriver::toString(status_d2h) << "\n";
//					throw std::runtime_error(message.str());
//				}
//		
//				report("DtoH " << value_out);
//			}
		}
		
		_variables.insert(std::make_pair(variable.second.name(),
			VariableDescriptor(newAddress, variable.second.type(), sizeInBytes)));
	}
}

bool Runtime::_runBlock(hir::Block* block)
{
printf("START Scheduling 1\n");
report("START Scheduling 1\n");
	auto schedule = _scheduleBasicBlock(block);
printf("END Scheduling 1\n");
report("END Scheduling 1\n");
	for(auto operation : schedule.scheduledOperations)
	{
		report("\n\n\nexecute 1 operation");
		bool exit = operation->execute(block);
		
		if(exit) return true;
	}
	
	report("\n\n\nafter execution all operations");
	return false;
}

const Runtime::BasicBlockSchedule&
	Runtime::_scheduleBasicBlock(hir::Block* block)
{
//	typedef hir::Block::KernelList KernelList;
report("Scheduling 1\n");
	auto schedule = _scheduledBlocks.find(block->id());

	if(schedule != _scheduledBlocks.end()) return schedule->second;

	// schedule the block now

	// Ideally we would schedule the kernels here, for now we just do 
	// naive in order scheduling
//	KernelList kernels = block->kernels;
	
	// Walk the block in reverse order, allocating, initializing, and deleting
	// unused variables
	OperationVector operations;

	auto blockDataflowInformation =
		_module->dataflowInformation().getLiveInOutSet(block->id());
	
	auto live = blockDataflowInformation.liveOut;

report("Scheduling 2\n");
	
	// TODO handle the control decision
	
	hir::DataflowInformation::VariableSet freeList;
	hir::DataflowInformation::VariableSet allocateList;

	for(auto kernel = block->kernels.rbegin(); kernel != block->kernels.rend(); ++kernel)
	{
		// kill defs
		for(auto write = kernel->operands().begin();
			write != kernel->operands().end(); ++write)
		{
			if(!write->isIn() && !write->variable().isInput())
			{
				live.erase(write->variable().name());

				if(kernel->type() != hir::KernelCall::Resize && kernel->type() != hir::KernelCall::UpdateSize)
					allocateList.insert(write->variable().name());
			}
//			if(!write->isIn() || write->variable().isInput()) continue;
//			
//			live.erase(write->variable().name());
//			allocateList.insert(write->variable().name());
		}
		
		// create uses
		for(auto read = kernel->operands().begin();
			read != kernel->operands().end(); ++read)
		{
			if(!read->isOut() && !read->variable().isOutput())
			{
				if(live.insert(read->variable().name()).second)
				{
					if(!read->variable().isInput())
						freeList.insert(read->variable().name());
				}
			}

//			if(!read->isOut()) continue;
//
//			// true if a new element was inserted or false 
//			// if an element with the same value existed.		
//			if(live.insert(read->variable().name()).second)
//			{
//				freeList.insert(read->variable().name());
//			}
		}

		// free unused variables
		for(auto variable : freeList)
		{
			auto variableDeclaration = _module->variables().find(variable);
			assert(variableDeclaration != _module->variables().end());
			operations.push_back(new VariableFree(this,
				variableDeclaration->second.name()));
		}
	
		// Add the kernel
		if(kernel->isExternalKernel())
		{
			auto externalKernel = _kernels.find(kernel->name());
			assert(externalKernel != _kernels.end());
			operations.push_back(new BinaryKernelLaunch(this, &*kernel,
				static_cast<LoadedBinaryKernel*>(
				externalKernel->second)->functionPointer));
		}
		else if(kernel->isPTXKernel())
		{
			auto ptxKernel = _kernels.find(kernel->name());
			assert(ptxKernel != _kernels.end());
			operations.push_back(new PTXKernelLaunch(this, &*kernel,
				static_cast<LoadedPTXKernel*>(ptxKernel->second)->handle));
		}
		else
		{
			if(kernel->type() == hir::KernelCall::GetSize)
			{
				operations.push_back(new VariableGetsize(this,
					kernel->operands()[0].variable().name(),
					kernel->operands()[1].variable().name()));
			}
			else if(kernel->type() == hir::KernelCall::Resize)
			{
				operations.push_back(new VariableResize(this,
					kernel->operands()[0].variable().name(),
					kernel->operands()[1].variable().name()));
			}
			else if(kernel->type() == hir::KernelCall::UpdateSize)
			{
				operations.push_back(new VariableUpdateSize(this,
					kernel->operands()[0].variable().name(),
					kernel->operands()[1].variable().name()));
			}
		}

		// allocate new variables
		for(auto variable : allocateList)
		{
			auto variableDeclaration = _module->variables().find(variable);
			assert(variableDeclaration != _module->variables().end());

			operations.push_back(new VariableAllocation(this,
				variableDeclaration->second.name(),
				variableDeclaration->second.size()));
		}
	
		freeList.clear();
		allocateList.clear();
	}

report("Scheduling 3\n");
	schedule = _scheduledBlocks.insert(std::make_pair(block->id(),
		BasicBlockSchedule(OperationVector(operations.rbegin(),
		operations.rend())))).first;
		
report("Scheduling 4\n");
	return schedule->second;
}

void Runtime::_loadPTXKernel(const hir::KernelCall& kernel)
{
	if(_kernels.count(kernel.name()) != 0) return;

	auto ptxKernel = _kernels.insert(std::make_pair(kernel.name(),
		new LoadedPTXKernel(kernel.name()))).first;
	ptxKernel->second->load(kernel.binary());
}

void Runtime::_loadExternalKernel(const hir::KernelCall& kernel)
{
	if(_kernels.count(kernel.name()) != 0) return;
	
	auto externalKernel = _kernels.insert(std::make_pair(kernel.name(),
		new LoadedBinaryKernel(&kernel))).first;

	// eager compilation
	externalKernel->second->load(kernel.binary());
}
	
Runtime::Operation::Operation(Runtime* runtime)
: _runtime(runtime)
{
	
}

Runtime::KernelLaunch::KernelLaunch(Runtime* runtime,
	const hir::KernelCall* k)
: Operation(runtime), kernel(k)
{

}

Runtime::PTXKernelLaunch::PTXKernelLaunch(Runtime* runtime,
	const hir::KernelCall* kernel, Handle h)
: KernelLaunch(runtime, kernel), kernelHandle(h)
{

}

class PackedArgument
{
public:
	PackedArgument(uint64_t a = 0/*, uint64_t s = 0*/) : address(a)/*, size(s)*/ {}

public:
	uint64_t address;
//	uint64_t size;
};

class SingleArgument
{
public:
	SingleArgument(uint64_t a = 0) : value(a) {}

public:
	uint64_t value;
};

bool Runtime::PTXKernelLaunch::execute(hir::Block*& block)
{
	report("PTXKernelLaunch execute " << kernel->name());
	
	// shared size, thread count, cta count
	uint64_t ctaCount  = _runtime->getKernelParameter(*kernel, 0);
	uint64_t threadCount = _runtime->getKernelParameter(*kernel, 1);
	uint64_t sharedSize    = _runtime->getKernelParameter(*kernel, 2);

	report("PTXKernel lanch info " << ctaCount << " " << threadCount << " " << sharedSize);

	CUresult status1 = cuda::CudaDriver::cuFuncSetSharedSize(kernelHandle, sharedSize);

	if(status1 != CUDA_SUCCESS)
	{	
		std::stringstream message;

		message << status1 << "\n";		
	
		throw std::runtime_error(message.str());
	}
	
	size_t free, total;
	cuda::CudaDriver::cuMemGetInfo(&free, &total);
	
	// convert arguments into (pointer, size) pairs
	PackedArgument arguments[kernel->operands().size()];
	
	unsigned int parameterBytes = (kernel->operands().size() - 3) 
		* sizeof(PackedArgument);
	
	CUresult status = cuda::CudaDriver::cuParamSetSize(kernelHandle,
		parameterBytes);
	
	if(status != CUDA_SUCCESS)
	{	
		std::stringstream message;

		message << "Failed to allocate " << parameterBytes
			<< " bytes of parameter memory for kernel with name "
			<< kernel->name() <<  " " << status << "\n";		
	
		throw std::runtime_error(message.str());
	}
	
	// Initialize parameters
	unsigned int argumentIndex = 0;
	unsigned int i = 0;

	for(auto operand : kernel->operands())
	{
		i++;

		if (i > 3) 
		{
			auto variable = _runtime->_variables.find(operand.variable().name());
			
			if(variable == _runtime->_variables.end())
			{
				std::stringstream message;
	
				message << "Kernel "
					<< kernel->name() << " referenced undeclared variable "
					<< operand.variable().name() << "\n";		
		
				throw std::runtime_error(message.str());
			}
			
			arguments[argumentIndex++] = PackedArgument(variable->second.address/*,
				variable->second.sizeInElements*/);
		}
	}
	
	status = cuda::CudaDriver::cuParamSetv(kernelHandle, 0,
		arguments, parameterBytes);
	
	if(status != CUDA_SUCCESS)
	{	
		std::stringstream message;

		message << "Failed to initialize " << parameterBytes
			<< " bytes of parameter memory for kernel with name "
			<< kernel->name() << "\n";		
	
		throw std::runtime_error(message.str());
	}
	
	// block shape
	status = cuda::CudaDriver::cuFuncSetBlockShape(kernelHandle,
		threadCount, 1, 1);
	
	if(status != CUDA_SUCCESS)
	{	
		std::stringstream message;

		message << "Failed to set block size to " << threadCount
			<< " for kernel with name "
			<< kernel->name() << "\n";		
	
		throw std::runtime_error(message.str());
	}
	
	// launch
	CUevent start, stop;
	float exe_time = 0.0f;
	cuda::CudaDriver::cuEventCreate(&start, CU_EVENT_DEFAULT);
	cuda::CudaDriver::cuEventCreate(&stop, CU_EVENT_DEFAULT);
	cuda::CudaDriver::cuEventRecord(start, 0);
	
	status = cuda::CudaDriver::cuLaunchGrid(kernelHandle, ctaCount, 1);

	cuda::CudaDriver::cuEventRecord(stop, 0);
	cuda::CudaDriver::cuEventSynchronize(stop);
	cuda::CudaDriver::cuEventElapsedTime(&exe_time, start, stop);

	cuda::CudaDriver::cuEventDestroy(start);	
	cuda::CudaDriver::cuEventDestroy(stop);	

	if(status != CUDA_SUCCESS)
	{	
		std::stringstream message;

		message << "Failed to launch kernel with " << ctaCount
			<< " CTAs for kernel with name "
			<< kernel->name() << "\n"		
			<< "Error message: " << cuda::CudaDriver::toString(status) << "\n";	
		throw std::runtime_error(message.str());
	}

	std::cout << kernel->name() << " " << exe_time << "\n";

	//check result
//	report("checking ptx results: ");
//	if(kernel->name().compare("join_get_size") == 0)
//	{
//		uint64_t check =  _runtime->getVariable(kernel->operands()[3].variable().name()); 
//		report("   " << (check >> 32));
//	}
	
	return false; // Don't exit
}

Runtime::ControlLaunch::ControlLaunch(Runtime* runtime,
	const hir::KernelCall* kernel, Handle handle)
: PTXKernelLaunch(runtime, kernel, handle)
{

}

bool Runtime::ControlLaunch::execute(hir::Block*& nextBlock)
{
	report("ControlLaunch execute");
	PTXKernelLaunch::execute(nextBlock);
	
	assert(false); // not implemented
	
	return false;
}

Runtime::BinaryKernelLaunch::BinaryKernelLaunch(Runtime* runtime,
	const hir::KernelCall* kernel, void* f)
: KernelLaunch(runtime, kernel), functionPointer(f)
{
	
}

bool Runtime::BinaryKernelLaunch::execute(hir::Block*& block)
{
	report("BinaryKernelLaunch execute " << kernel->name());
	unsigned int argumentCount = kernel->operands().size();
	
	PackedArgument arguments[argumentCount - 1];
	
	unsigned int argumentIndex = 0;
	
	if(kernel->name().compare(0, 8, "set_sort") == 0)
	{
		for(auto operand : kernel->operands())
		{
			auto variable = _runtime->_variables.find(operand.variable().name());
			
			if(variable == _runtime->_variables.end())
			{
				std::stringstream message;
	
				message << "Kernel "
					<< kernel->name() << " referenced undeclared variable "
					<< operand.variable().name() << "\n";		
		
				throw std::runtime_error(message.str());
			}
		
			if(argumentIndex == 0)
			{
				report(" binary argument " << argumentIndex << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(variable->second.address/*,
					variable->second.sizeInElements*/);
			}
			else
			{
				uint64_t value = _runtime->getVariable(operand.variable().name());
				report(" binary argument " << argumentIndex << " " << value << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(value);
			}
		}
	}
	else if(kernel->name().compare(0, 16, "set_mgpusortpair") == 0)
	{
		for(auto operand : kernel->operands())
		{
			auto variable = _runtime->_variables.find(operand.variable().name());
			
			if(variable == _runtime->_variables.end())
			{
				std::stringstream message;
	
				message << "Kernel "
					<< kernel->name() << " referenced undeclared variable "
					<< operand.variable().name() << "\n";		
		
				throw std::runtime_error(message.str());
			}
		
			if(argumentIndex == 0 || argumentIndex == 1)
			{
				report(" binary argument " << argumentIndex << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(variable->second.address/*,
					variable->second.sizeInElements*/);
			}
			else
			{
				uint64_t value = _runtime->getVariable(operand.variable().name());
				report(" binary argument " << argumentIndex << " " << value << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(value);
			}
		}
	}
	else if(kernel->name().compare(0, 15, "set_mgpusortkey") == 0)
	{
		for(auto operand : kernel->operands())
		{
			auto variable = _runtime->_variables.find(operand.variable().name());
			
			if(variable == _runtime->_variables.end())
			{
				std::stringstream message;
	
				message << "Kernel "
					<< kernel->name() << " referenced undeclared variable "
					<< operand.variable().name() << "\n";		
		
				throw std::runtime_error(message.str());
			}
		
			if(argumentIndex == 0)
			{
				report(" binary argument " << argumentIndex << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(variable->second.address/*,
					variable->second.sizeInElements*/);
			}
			else
			{
				uint64_t value = _runtime->getVariable(operand.variable().name());
				report(" binary argument " << argumentIndex << " " << value << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(value);
			}
		}
	}
	else if(kernel->name().compare(0, 16, "set_b40csortpair") == 0)
	{
		for(auto operand : kernel->operands())
		{
			auto variable = _runtime->_variables.find(operand.variable().name());
			
			if(variable == _runtime->_variables.end())
			{
				std::stringstream message;
	
				message << "Kernel "
					<< kernel->name() << " referenced undeclared variable "
					<< operand.variable().name() << "\n";		
		
				throw std::runtime_error(message.str());
			}
		
			if(argumentIndex == 0 || argumentIndex == 1)
			{
				report(" binary argument " << argumentIndex << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(variable->second.address/*,
					variable->second.sizeInElements*/);
			}
			else
			{
				uint64_t value = _runtime->getVariable(operand.variable().name());
				report(" binary argument " << argumentIndex << " " << value << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(value);
			}
		}
	}
	else if(kernel->name().compare(0, 15, "set_b40csortkey") == 0)
	{
		for(auto operand : kernel->operands())
		{
			auto variable = _runtime->_variables.find(operand.variable().name());
			
			if(variable == _runtime->_variables.end())
			{
				std::stringstream message;
	
				message << "Kernel "
					<< kernel->name() << " referenced undeclared variable "
					<< operand.variable().name() << "\n";		
		
				throw std::runtime_error(message.str());
			}
		
			if(argumentIndex == 0)
			{
				report(" binary argument " << argumentIndex << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(variable->second.address/*,
					variable->second.sizeInElements*/);
			}
			else
			{
				uint64_t value = _runtime->getVariable(operand.variable().name());
				report(" binary argument " << argumentIndex << " " << value << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(value);
			}
		}
	}
	else if(kernel->name().compare(0, 21, "mgpu_join_find_bounds") == 0)
	{
		for(auto operand : kernel->operands())
		{
			auto variable = _runtime->_variables.find(operand.variable().name());
			
			if(variable == _runtime->_variables.end())
			{
				std::stringstream message;
	
				message << "Kernel "
					<< kernel->name() << " referenced undeclared variable "
					<< operand.variable().name() << "\n";		
		
				throw std::runtime_error(message.str());
			}
		
			if(argumentIndex == 0 || argumentIndex == 1 || argumentIndex == 2 || argumentIndex == 3 || argumentIndex == 5)
			{
				report(" binary argument " << argumentIndex << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(variable->second.address/*,
					variable->second.sizeInElements*/);
			}
			else
			{
				uint64_t value = _runtime->getVariable(operand.variable().name());
				report(" binary argument " << argumentIndex << " " << value << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(value);
			}
		}
	}
	else if(kernel->name().compare(0, 14, "mgpu_join_main") == 0)
	{
		for(auto operand : kernel->operands())
		{
			auto variable = _runtime->_variables.find(operand.variable().name());
			
			if(variable == _runtime->_variables.end())
			{
				std::stringstream message;
	
				message << "Kernel "
					<< kernel->name() << " referenced undeclared variable "
					<< operand.variable().name() << "\n";		
		
				throw std::runtime_error(message.str());
			}
		
			if(argumentIndex == 0 || argumentIndex == 1 || argumentIndex == 3 || argumentIndex == 4)
			{
				report(" binary argument " << argumentIndex << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(variable->second.address/*,
					variable->second.sizeInElements*/);
			}
			else
			{
				uint64_t value = _runtime->getVariable(operand.variable().name());
				report(" binary argument " << argumentIndex << " " << value << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(value);
			}
		}
	}
	else if(kernel->name().compare(0, 9, "set_union") == 0 || kernel->name().compare(0, 14, "set_difference") == 0 )
	{
		for(auto operand : kernel->operands())
		{
			auto variable = _runtime->_variables.find(operand.variable().name());
			
			if(variable == _runtime->_variables.end())
			{
				std::stringstream message;
	
				message << "Kernel "
					<< kernel->name() << " referenced undeclared variable "
					<< operand.variable().name() << "\n";		
		
				throw std::runtime_error(message.str());
			}
		
			if(argumentIndex == 0 || argumentIndex == 1 || argumentIndex == 2 || argumentIndex == 4)
			{
				report(" binary argument " << argumentIndex << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(variable->second.address/*,
					variable->second.sizeInElements*/);
			}
			else
			{
				uint64_t value = _runtime->getVariable(operand.variable().name());
				report(" binary argument " << argumentIndex << " " << value << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(value);
			}
		}
	}
	else if(kernel->name().compare(0, 17, "set_single_reduce") == 0)
	{
		for(auto operand : kernel->operands())
		{
			auto variable = _runtime->_variables.find(operand.variable().name());
			
			if(variable == _runtime->_variables.end())
			{
				std::stringstream message;
	
				message << "Kernel "
					<< kernel->name() << " referenced undeclared variable "
					<< operand.variable().name() << "\n";		
		
				throw std::runtime_error(message.str());
			}
		
			if(argumentIndex == 0 || argumentIndex == 1)
			{
				report(" binary argument " << argumentIndex << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(variable->second.address/*,
					variable->second.sizeInElements*/);
			}
			else
			{
				uint64_t value = _runtime->getVariable(operand.variable().name());
				report(" binary argument " << argumentIndex << " " << value << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(value);
			}
		}
	}
	if(kernel->name().compare(0, 10, "set_unique") == 0)
	{
		for(auto operand : kernel->operands())
		{
			auto variable = _runtime->_variables.find(operand.variable().name());
			
			if(variable == _runtime->_variables.end())
			{
				std::stringstream message;
	
				message << "Kernel "
					<< kernel->name() << " referenced undeclared variable "
					<< operand.variable().name() << "\n";		
		
				throw std::runtime_error(message.str());
			}
		
			if(argumentIndex == 0 || argumentIndex == 1)
			{
				report(" binary argument " << argumentIndex << " " << variable->second.address);	
				arguments[argumentIndex++] = PackedArgument(variable->second.address/*,
					variable->second.sizeInElements*/);
			}
			else
			{
				uint64_t value = _runtime->getVariable(operand.variable().name());
				report(" binary argument " << argumentIndex << " " << value);	
				arguments[argumentIndex++] = PackedArgument(value);
			}
		}
	}
	else if(kernel->name().compare(0, 10, "set_reduce") == 0 || kernel->name().compare(0, 9, "set_count") == 0)
	{
		for(auto operand : kernel->operands())
		{
			auto variable = _runtime->_variables.find(operand.variable().name());
			
			if(variable == _runtime->_variables.end())
			{
				std::stringstream message;
	
				message << "Kernel "
					<< kernel->name() << " referenced undeclared variable "
					<< operand.variable().name() << "\n";		
		
				throw std::runtime_error(message.str());
			}
		
			report(" binary argument " << argumentIndex << " " << variable->second.address);	
			arguments[argumentIndex++] = PackedArgument(variable->second.address/*,
				variable->second.sizeInElements*/);
		}
	}

	ExternalKernel pointer = (ExternalKernel)functionPointer;

//	CUcontext tmpContext;
//	if(kernel->name().compare(0, 12, "set_mgpusort") == 0 || 
//		kernel->name().compare(0, 21, "mgpu_join_find_bounds") == 0 || kernel->name().compare(0, 14, "mgpu_join_main") == 0)
//	{
//		cuda::CudaDriver::cuCtxPopCurrent(&tmpContext);
//		std::cout << "poped\n";
//	}
	
	report("Running external ...")
	pointer(arguments);

	//check result
//	uint64_t check =  _runtime->getVariable(kernel->operands()[0].variable().name()); 
//	report("Binary kernel result is " << check);

//	if(kernel->name().compare(0, 12, "set_mgpusort") == 0 
//		|| kernel->name().compare(0, 21, "mgpu_join_find_bounds") == 0 || kernel->name().compare(0, 14, "mgpu_join_main") == 0)
//	{
//		cuda::CudaDriver::cuCtxPushCurrent(tmpContext);
//		std::cout << "pushed\n";
//	}
	return false;
}

Runtime::VariableUpdate::VariableUpdate(Runtime* runtime,
	VariableName n)
: Operation(runtime), name(n)
{
	
}

Runtime::VariableGetsize::VariableGetsize(Runtime* r,
	VariableName n, VariableName h)
: VariableUpdate(r, n), hostVariable(h)
{

}

bool Runtime::VariableGetsize::execute(hir::Block*& block)
{
	report("Variable Getsize " << hostVariable << 
		" to " << name << " in block " << block->id());

	auto variable = _runtime->_variables.find(name);

	if(variable == _runtime->_variables.end())
	{
		throw std::runtime_error("Tried to assign size to unknown variable.");
	}

	if(variable->second.address == 0)
	{
		throw std::runtime_error("Tried to assign size to non-allocated variable");
	}

	assert(variable->second.sizeInElements > 0);

	auto variable_h = _runtime->_variables.find(hostVariable);

	if(variable_h == _runtime->_variables.end())
	{
		throw std::runtime_error("Tried to get size of unknown variable.");
	}

	if(variable_h->second.address == 0)
	{
		throw std::runtime_error("Tried to get size of non-allocated variable.");
	}

	assert(variable_h->second.sizeInElements > 0);

	uint64_t sizeInBytes = variable->second.sizeInElements;

	report("Getsize: the size of host is " << variable_h->second.sizeInElements << " byte size is " << sizeInBytes << " address is " << variable->second.address);
	
	CUresult status = cuda::CudaDriver::cuMemcpyHtoD(variable->second.address,
		&variable_h->second.sizeInElements, sizeInBytes);

	if(status != CUDA_SUCCESS)
	{	
		std::stringstream message;

		message << "Failed to get size from " << hostVariable
			<< " to " << name << "\n";		
	
		throw std::runtime_error(message.str());
	}

	//check 
//	uint64_t check = 0;
//	CUresult status_check = cuda::CudaDriver::cuMemcpyDtoH(&check, variable->second.address,
//		sizeInBytes);
//
//	if(status_check != CUDA_SUCCESS)
//	{
//		std::stringstream message;
//
//		message << "check Failed to get size from \n";		
//	
//		throw std::runtime_error(message.str());
//	}
//
//	report("Getsize check result " << check);
	
	return false;
}

Runtime::VariableAllocation::VariableAllocation(Runtime* r,
	VariableName n, uint64_t s)
: VariableUpdate(r, n), size(s)
{

}

bool Runtime::VariableAllocation::execute(hir::Block*& block)
{
	report("Variable Allocation " << name << " " << size << " bytes in block " << block->id());

	auto variable = _runtime->_variables.find(name);

	if(variable == _runtime->_variables.end())
	{
		throw std::runtime_error("Tried to allocate unknown variable.");
	}

	if(variable->second.address != 0)
	{
		throw std::runtime_error("Tried to re-allocate previously"
			" allocated variable.");
	}

	assert(variable->second.capacityInElements == 0);

	variable->second.sizeInElements     = size;
	variable->second.capacityInElements = size;

	uint64_t sizeInBytes =
		/*hir::VariableDeclaration::bytes(variable->second.type) **/ size;
	
	CUresult status = cuda::CudaDriver::cuMemAlloc(
		&variable->second.address, sizeInBytes);

	report("Allocated address is " << variable->second.address);

	if(status != CUDA_SUCCESS)
	{	
		std::stringstream message;

		message << "Failed to allocate " << sizeInBytes
			<< " for variable with name " << name << "\n";		
	
		throw std::runtime_error(message.str());
	}
	
	return false;
}

Runtime::VariableResize::VariableResize(Runtime* runtime,
	VariableName n, VariableName s)
: VariableUpdate(runtime, n), size(s)
{
	
}

bool Runtime::VariableResize::execute(hir::Block*& block)
{
	report("VariableResize execute " << name);
	auto variable = _runtime->_variables.find(name);

	if(variable == _runtime->_variables.end())
	{
		throw std::runtime_error("Tried to allocate unknown variable.");
	}

	if(variable->second.address != 0)
	{
		throw std::runtime_error("Tried to re-allocate previously"
			" allocated variable.");
	}

	assert(variable->second.capacityInElements == 0);

	uint64_t previousSizeInElements = variable->second.sizeInElements;
	uint64_t sizeInElements         = _runtime->getVariable(size);
	
	variable->second.sizeInElements = sizeInElements;

	report("VariableResize previous " << previousSizeInElements);

	report("VariableResize current " << sizeInElements);

	// trigger a resize only if the capacity is exceeded
	if(variable->second.sizeInElements <= variable->second.capacityInElements)
	{
		return false;
	}
	
	variable->second.capacityInElements = sizeInElements;

	CUdeviceptr newAddress  = 0;
	uint64_t    sizeInBytes =
		hir::VariableDeclaration::bytes(variable->second.type) * sizeInElements;
	
	CUresult status = cuda::CudaDriver::cuMemAlloc(&newAddress, sizeInBytes);
	
	if(status != CUDA_SUCCESS)
	{	
		std::stringstream message;

		message << "Failed to resize (allocate) " << sizeInBytes
			<< " for variable with name " << name << "\n";		
	
		throw std::runtime_error(message.str());
	}
	
	report("Resize allocate address " << newAddress << " with " << sizeInBytes << " bytes");
	report("variable->second.address " << variable->second.address);

	status = cuda::CudaDriver::cuMemcpyDtoD(newAddress,
		variable->second.address, previousSizeInElements);
	
	if(status != CUDA_SUCCESS)
	{	
		std::stringstream message;

		message << "Failed to resize (copy) " << sizeInBytes
			<< " for variable with name " << name << "\n";		
	
		throw std::runtime_error(message.str());
	}
	
	status = cuda::CudaDriver::cuMemFree(variable->second.address);
	
	if(status != CUDA_SUCCESS)
	{	
		std::stringstream message;

		message << "Failed to resize (free) " << sizeInBytes
			<< " for variable with name " << name << "\n";		
	
		throw std::runtime_error(message.str());
	}
	
	variable->second.address = newAddress;
	
	return false;
}

Runtime::VariableUpdateSize::VariableUpdateSize(Runtime* runtime,
	VariableName n, VariableName s)
: VariableUpdate(runtime, n), size(s)
{
	
}

bool Runtime::VariableUpdateSize::execute(hir::Block*& block)
{
	report("VariableUpdateSize execute " << name);
	auto variable = _runtime->_variables.find(name);

	if(variable == _runtime->_variables.end())
	{
		throw std::runtime_error("Tried to allocate unknown variable.");
	}

	uint64_t sizeInElements         = _runtime->getVariable(size);
	
	variable->second.sizeInElements = sizeInElements;

	report("VariableUpdateSize current " << sizeInElements);
	
	variable->second.capacityInElements = sizeInElements;

	return false;
}

Runtime::VariableFree::VariableFree(Runtime* runtime, VariableName n)
: VariableUpdate(runtime, n)
{
	
}

bool Runtime::VariableFree::execute(hir::Block*& block)
{
	report("VariableFree execute " << name << " in block " << block->id());	
	auto variable = _runtime->_variables.find(name);

	if(variable == _runtime->_variables.end())
	{
		throw std::runtime_error("Tried to free unknown variable.");
	}
	
	variable->second.sizeInElements     = 0;
	variable->second.capacityInElements = 0;
	
	cuda::CudaDriver::cuMemFree(variable->second.address);

	return false;
}

Runtime::LoadedKernel::LoadedKernel(const std::string& n)
: name(n)
{

}

Runtime::LoadedBinaryKernel::LoadedBinaryKernel(const hir::KernelCall* kernel)
: LoadedKernel(kernel->name()), functionPointer(nullptr),
	originalFunctionPointer(nullptr), _kernel(kernel)
{
	#ifdef HAVE_LLVM
	_module = new llvm::Module("_ZHarmonyExternalFunctionModule",
		llvm::getGlobalContext());
	#endif
}

Runtime::LoadedBinaryKernel::~LoadedBinaryKernel()
{
	dlclose(_libraryHandle);
	std::remove(_fileName.c_str());
	
	#ifdef HAVE_LLVM
	llvm::Function* function = _module->getFunction(mangledName());
	
	if(function != 0)
	{
		vm::LLVMState::jit()->freeMachineCodeForFunction(function);
	}
	
	vm::LLVMState::jit()->removeModule(_module);

	if(_module) delete _module;

	#endif
}

#ifdef HAVE_LLVM

static std::string getValueString(unsigned int value)
{
	std::stringstream stream;
	
	stream << "%r" << value;
	
	return stream.str();
}

static vm::LLVMInstruction::DataType translateType(
	hir::VariableDeclaration::DataType type)
{
	switch(type)
	{
	case hir::VariableDeclaration::i8:
	{
		return vm::LLVMInstruction::I8;
		break;
	}
	case hir::VariableDeclaration::i16:
	{
		return vm::LLVMInstruction::I16;
		break;
	}
	case hir::VariableDeclaration::i32:
	{
		return vm::LLVMInstruction::I32;
		break;
	}
	case hir::VariableDeclaration::i64:
	{
		return vm::LLVMInstruction::I64;
		break;
	}
	case hir::VariableDeclaration::f32:
	{
		return vm::LLVMInstruction::F32;
		break;
	}
	case hir::VariableDeclaration::f64:
	{
		return vm::LLVMInstruction::F64;
		break;
	}
	default:
	{
		break;
	}
	}
	
	return vm::LLVMInstruction::InvalidDataType;
}
#endif

llvm::Function* Runtime::LoadedBinaryKernel::jitFunction(llvm::Module* m)
{
#ifdef HAVE_LLVM
	vm::LLVMKernel llvmKernel;

	// Add a prototype for the function
	report("Add a prototype for the function");
	vm::LLVMStatement proto(vm::LLVMStatement::FunctionDeclaration);

	proto.label      = name;
	proto.linkage    = vm::LLVMStatement::InvalidLinkage;
	proto.convention = vm::LLVMInstruction::DefaultCallingConvention;
	proto.visibility = vm::LLVMStatement::Default;

	report("binary kernel operands " << _kernel->operands().size());

//	unsigned int argumentCount = _kernel->operands().size();
	unsigned int argumentIndex = 0;

	if(name.compare(0, 8, "set_sort") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			report("operand " << argument.variable().name() << " " << argument.variable().type());
	
			if(argumentIndex == 0)
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Pointer);
			}
			else
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Element);
			}
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 16, "set_mgpusortpair") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			report("operand " << argument.variable().name() << " " << argument.variable().type());
	
			if(argumentIndex == 0 || argumentIndex == 1)
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Pointer);
			}
			else
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Element);
			}
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 16, "set_b40csortpair") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			report("operand " << argument.variable().name() << " " << argument.variable().type());
	
			if(argumentIndex == 0 || argumentIndex == 1)
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Pointer);
			}
			else
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Element);
			}
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 21, "mgpu_join_find_bounds") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			report("operand " << argument.variable().name() << " " << argument.variable().type());
	
			if(argumentIndex == 0 || argumentIndex == 1 || argumentIndex == 2 || argumentIndex == 3 || argumentIndex == 5)
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Pointer);
			}
			else
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Element);
			}
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 14, "mgpu_join_main") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			report("operand " << argument.variable().name() << " " << argument.variable().type());
	
			if(argumentIndex == 0 || argumentIndex == 1 || argumentIndex == 3 || argumentIndex == 4)
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Pointer);
			}
			else
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Element);
			}
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 15, "set_mgpusortkey") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			report("operand " << argument.variable().name() << " " << argument.variable().type());
	
			if(argumentIndex == 0)
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Pointer);
			}
			else
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Element);
			}
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 15, "set_b40csortkey") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			report("operand " << argument.variable().name() << " " << argument.variable().type());
	
			if(argumentIndex == 0)
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Pointer);
			}
			else
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Element);
			}
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 9, "set_union") == 0 || name.compare(0, 14, "set_difference") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			report("operand " << argument.variable().name() << " " << argument.variable().type());
	
			if(argumentIndex == 0 || argumentIndex == 1 || argumentIndex == 2 || argumentIndex == 4)
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Pointer);
			}
			else
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Element);
			}
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 17, "set_single_reduce") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			report("operand " << argument.variable().name() << " " << argument.variable().type());
	
			if(argumentIndex == 0 || argumentIndex == 1)
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Pointer);
			}
			else
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Element);
			}
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 10, "set_unique") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			report("operand " << argument.variable().name() << " " << argument.variable().type());
	
			if(argumentIndex == 0 || argumentIndex == 1)
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Pointer);
			}
			else
			{
				proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
				proto.parameters.back().type = vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
					vm::LLVMInstruction::Type::Element);
			}
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 10, "set_reduce") == 0 || name.compare(0, 9, "set_count") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			report("operand " << argument.variable().name() << " " << argument.variable().type());
	
			proto.parameters.push_back(vm::LLVMInstruction::Parameter());
		
			proto.parameters.back().type = vm::LLVMInstruction::Type(
				translateType(argument.variable().type()),
				vm::LLVMInstruction::Type::Pointer);
	
			argumentIndex++;
		}
	}
	
	llvmKernel.push_back(proto);
	
	// Add the function
	report("Add the function");
	vm::LLVMStatement func(vm::LLVMStatement::FunctionDefinition);

	func.label              = mangledName();
	func.linkage            = vm::LLVMStatement::InvalidLinkage;
	func.convention         = vm::LLVMInstruction::DefaultCallingConvention;
	func.visibility         = vm::LLVMStatement::Default;
	func.functionAttributes = vm::LLVMInstruction::NoUnwind;

	func.parameters.resize(1);

	func.parameters[0].attribute     = vm::LLVMInstruction::NoAlias;
	func.parameters[0].type.type     = vm::LLVMInstruction::I8;
	func.parameters[0].type.category = vm::LLVMInstruction::Type::Pointer;
	func.parameters[0].name          = "%parameters";

	llvmKernel.push_back(func);

	llvmKernel.push_back(vm::LLVMStatement(
		vm::LLVMStatement::BeginFunctionBody));

	llvmKernel.push_back(vm::LLVMStatement("entry"));

	unsigned int offset = 0;
	unsigned int value  = 0;

	vm::LLVMCall call;

	call.name = "@" + name;
		
	// load and translate the operands
	
	report("load and translate the operands");

	argumentIndex = 0;

	if(name.compare(0, 8, "set_sort") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			// Loop over the variable pointer, then size
	
			vm::LLVMInstruction::Operand operand = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer = vm::LLVMInstruction::Operand(
				operand.name + "p",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer2 = vm::LLVMInstruction::Operand(
				operand.name + "pp",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Element));
	
			if(argumentIndex == 0)
			{
				call.parameters.push_back(operand);
			}
			else
			{
				call.parameters.push_back(pointer2);
			}
	
			vm::LLVMGetelementptr get;
		
			get.d = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
			get.a = vm::LLVMInstruction::Operand("%parameters",
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
		
			get.indices.push_back(offset);
		
			llvmKernel.push_back(vm::LLVMStatement(get));
		
			vm::LLVMBitcast bitcast;
		
			bitcast.d = pointer;
			bitcast.a = get.d;
		
			llvmKernel.push_back(vm::LLVMStatement(bitcast));
		
			vm::LLVMLoad load;
		
			load.d = pointer2;
			load.a = pointer;
		
			llvmKernel.push_back(vm::LLVMStatement(load));
	
			if(argumentIndex == 0)
			{
				vm::LLVMInttoptr int2ptr;
		
				int2ptr.d = operand;
				int2ptr.a = pointer2;
			
				llvmKernel.push_back(vm::LLVMStatement(int2ptr));
			}
		
			offset += sizeof(uint64_t);
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 16, "set_mgpusortpair") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			// Loop over the variable pointer, then size
	
			vm::LLVMInstruction::Operand operand = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer = vm::LLVMInstruction::Operand(
				operand.name + "p",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer2 = vm::LLVMInstruction::Operand(
				operand.name + "pp",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Element));
	
			if(argumentIndex == 0 || argumentIndex == 1)
			{
				call.parameters.push_back(operand);
			}
			else
			{
				call.parameters.push_back(pointer2);
			}
	
			vm::LLVMGetelementptr get;
		
			get.d = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
			get.a = vm::LLVMInstruction::Operand("%parameters",
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
		
			get.indices.push_back(offset);
		
			llvmKernel.push_back(vm::LLVMStatement(get));
		
			vm::LLVMBitcast bitcast;
		
			bitcast.d = pointer;
			bitcast.a = get.d;
		
			llvmKernel.push_back(vm::LLVMStatement(bitcast));
		
			vm::LLVMLoad load;
		
			load.d = pointer2;
			load.a = pointer;
		
			llvmKernel.push_back(vm::LLVMStatement(load));
	
			if(argumentIndex == 0 || argumentIndex == 1)
			{
				vm::LLVMInttoptr int2ptr;
		
				int2ptr.d = operand;
				int2ptr.a = pointer2;
			
				llvmKernel.push_back(vm::LLVMStatement(int2ptr));
			}
		
			offset += sizeof(uint64_t);
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 16, "set_b40csortpair") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			// Loop over the variable pointer, then size
	
			vm::LLVMInstruction::Operand operand = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer = vm::LLVMInstruction::Operand(
				operand.name + "p",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer2 = vm::LLVMInstruction::Operand(
				operand.name + "pp",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Element));
	
			if(argumentIndex == 0 || argumentIndex == 1)
			{
				call.parameters.push_back(operand);
			}
			else
			{
				call.parameters.push_back(pointer2);
			}
	
			vm::LLVMGetelementptr get;
		
			get.d = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
			get.a = vm::LLVMInstruction::Operand("%parameters",
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
		
			get.indices.push_back(offset);
		
			llvmKernel.push_back(vm::LLVMStatement(get));
		
			vm::LLVMBitcast bitcast;
		
			bitcast.d = pointer;
			bitcast.a = get.d;
		
			llvmKernel.push_back(vm::LLVMStatement(bitcast));
		
			vm::LLVMLoad load;
		
			load.d = pointer2;
			load.a = pointer;
		
			llvmKernel.push_back(vm::LLVMStatement(load));
	
			if(argumentIndex == 0 || argumentIndex == 1)
			{
				vm::LLVMInttoptr int2ptr;
		
				int2ptr.d = operand;
				int2ptr.a = pointer2;
			
				llvmKernel.push_back(vm::LLVMStatement(int2ptr));
			}
		
			offset += sizeof(uint64_t);
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 21, "mgpu_join_find_bounds") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			// Loop over the variable pointer, then size
	
			vm::LLVMInstruction::Operand operand = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer = vm::LLVMInstruction::Operand(
				operand.name + "p",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer2 = vm::LLVMInstruction::Operand(
				operand.name + "pp",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Element));
	
			if(argumentIndex == 0 || argumentIndex == 1 || argumentIndex == 2 || argumentIndex == 3 || argumentIndex == 5)
			{
				call.parameters.push_back(operand);
			}
			else
			{
				call.parameters.push_back(pointer2);
			}
	
			vm::LLVMGetelementptr get;
		
			get.d = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
			get.a = vm::LLVMInstruction::Operand("%parameters",
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
		
			get.indices.push_back(offset);
		
			llvmKernel.push_back(vm::LLVMStatement(get));
		
			vm::LLVMBitcast bitcast;
		
			bitcast.d = pointer;
			bitcast.a = get.d;
		
			llvmKernel.push_back(vm::LLVMStatement(bitcast));
		
			vm::LLVMLoad load;
		
			load.d = pointer2;
			load.a = pointer;
		
			llvmKernel.push_back(vm::LLVMStatement(load));
	
			if(argumentIndex == 0 || argumentIndex == 1 || argumentIndex == 2 || argumentIndex == 3 || argumentIndex == 5)
			{
				vm::LLVMInttoptr int2ptr;
		
				int2ptr.d = operand;
				int2ptr.a = pointer2;
			
				llvmKernel.push_back(vm::LLVMStatement(int2ptr));
			}
		
			offset += sizeof(uint64_t);
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 14, "mgpu_join_main") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			// Loop over the variable pointer, then size
	
			vm::LLVMInstruction::Operand operand = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer = vm::LLVMInstruction::Operand(
				operand.name + "p",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer2 = vm::LLVMInstruction::Operand(
				operand.name + "pp",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Element));
	
			if(argumentIndex == 0 || argumentIndex == 1 || argumentIndex == 3 || argumentIndex == 4)
			{
				call.parameters.push_back(operand);
			}
			else
			{
				call.parameters.push_back(pointer2);
			}
	
			vm::LLVMGetelementptr get;
		
			get.d = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
			get.a = vm::LLVMInstruction::Operand("%parameters",
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
		
			get.indices.push_back(offset);
		
			llvmKernel.push_back(vm::LLVMStatement(get));
		
			vm::LLVMBitcast bitcast;
		
			bitcast.d = pointer;
			bitcast.a = get.d;
		
			llvmKernel.push_back(vm::LLVMStatement(bitcast));
		
			vm::LLVMLoad load;
		
			load.d = pointer2;
			load.a = pointer;
		
			llvmKernel.push_back(vm::LLVMStatement(load));
	
			if(argumentIndex == 0 || argumentIndex == 1 || argumentIndex == 3 || argumentIndex == 4)
			{
				vm::LLVMInttoptr int2ptr;
		
				int2ptr.d = operand;
				int2ptr.a = pointer2;
			
				llvmKernel.push_back(vm::LLVMStatement(int2ptr));
			}
		
			offset += sizeof(uint64_t);
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 15, "set_mgpusortkey") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			// Loop over the variable pointer, then size
	
			vm::LLVMInstruction::Operand operand = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer = vm::LLVMInstruction::Operand(
				operand.name + "p",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer2 = vm::LLVMInstruction::Operand(
				operand.name + "pp",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Element));
	
			if(argumentIndex == 0)
			{
				call.parameters.push_back(operand);
			}
			else
			{
				call.parameters.push_back(pointer2);
			}
	
			vm::LLVMGetelementptr get;
		
			get.d = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
			get.a = vm::LLVMInstruction::Operand("%parameters",
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
		
			get.indices.push_back(offset);
		
			llvmKernel.push_back(vm::LLVMStatement(get));
		
			vm::LLVMBitcast bitcast;
		
			bitcast.d = pointer;
			bitcast.a = get.d;
		
			llvmKernel.push_back(vm::LLVMStatement(bitcast));
		
			vm::LLVMLoad load;
		
			load.d = pointer2;
			load.a = pointer;
		
			llvmKernel.push_back(vm::LLVMStatement(load));
	
			if(argumentIndex == 0)
			{
				vm::LLVMInttoptr int2ptr;
		
				int2ptr.d = operand;
				int2ptr.a = pointer2;
			
				llvmKernel.push_back(vm::LLVMStatement(int2ptr));
			}
		
			offset += sizeof(uint64_t);
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 15, "set_b40csortkey") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			// Loop over the variable pointer, then size
	
			vm::LLVMInstruction::Operand operand = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer = vm::LLVMInstruction::Operand(
				operand.name + "p",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer2 = vm::LLVMInstruction::Operand(
				operand.name + "pp",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Element));
	
			if(argumentIndex == 0)
			{
				call.parameters.push_back(operand);
			}
			else
			{
				call.parameters.push_back(pointer2);
			}
	
			vm::LLVMGetelementptr get;
		
			get.d = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
			get.a = vm::LLVMInstruction::Operand("%parameters",
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
		
			get.indices.push_back(offset);
		
			llvmKernel.push_back(vm::LLVMStatement(get));
		
			vm::LLVMBitcast bitcast;
		
			bitcast.d = pointer;
			bitcast.a = get.d;
		
			llvmKernel.push_back(vm::LLVMStatement(bitcast));
		
			vm::LLVMLoad load;
		
			load.d = pointer2;
			load.a = pointer;
		
			llvmKernel.push_back(vm::LLVMStatement(load));
	
			if(argumentIndex == 0)
			{
				vm::LLVMInttoptr int2ptr;
		
				int2ptr.d = operand;
				int2ptr.a = pointer2;
			
				llvmKernel.push_back(vm::LLVMStatement(int2ptr));
			}
		
			offset += sizeof(uint64_t);
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 9, "set_union") == 0 || name.compare(0, 14, "set_difference") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			// Loop over the variable pointer, then size
	
			vm::LLVMInstruction::Operand operand = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer = vm::LLVMInstruction::Operand(
				operand.name + "p",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer2 = vm::LLVMInstruction::Operand(
				operand.name + "pp",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Element));
	
			if(argumentIndex == 0 || argumentIndex == 1 || argumentIndex == 2 || argumentIndex == 4)
			{
				call.parameters.push_back(operand);
			}
			else
			{
				call.parameters.push_back(pointer2);
			}
	
			vm::LLVMGetelementptr get;
		
			get.d = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
			get.a = vm::LLVMInstruction::Operand("%parameters",
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
		
			get.indices.push_back(offset);
		
			llvmKernel.push_back(vm::LLVMStatement(get));
		
			vm::LLVMBitcast bitcast;
		
			bitcast.d = pointer;
			bitcast.a = get.d;
		
			llvmKernel.push_back(vm::LLVMStatement(bitcast));
		
			vm::LLVMLoad load;
		
			load.d = pointer2;
			load.a = pointer;
		
			llvmKernel.push_back(vm::LLVMStatement(load));
	
			if(argumentIndex == 0 || argumentIndex == 1 || argumentIndex == 2 || argumentIndex == 4)
			{
				vm::LLVMInttoptr int2ptr;
		
				int2ptr.d = operand;
				int2ptr.a = pointer2;
			
				llvmKernel.push_back(vm::LLVMStatement(int2ptr));
			}
		
			offset += sizeof(uint64_t);
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 17, "set_single_reduce") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			// Loop over the variable pointer, then size
	
			vm::LLVMInstruction::Operand operand = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer = vm::LLVMInstruction::Operand(
				operand.name + "p",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer2 = vm::LLVMInstruction::Operand(
				operand.name + "pp",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Element));
	
			if(argumentIndex == 0 || argumentIndex == 1)
			{
				call.parameters.push_back(operand);
			}
			else
			{
				call.parameters.push_back(pointer2);
			}
	
			vm::LLVMGetelementptr get;
		
			get.d = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
			get.a = vm::LLVMInstruction::Operand("%parameters",
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
		
			get.indices.push_back(offset);
		
			llvmKernel.push_back(vm::LLVMStatement(get));
		
			vm::LLVMBitcast bitcast;
		
			bitcast.d = pointer;
			bitcast.a = get.d;
		
			llvmKernel.push_back(vm::LLVMStatement(bitcast));
		
			vm::LLVMLoad load;
		
			load.d = pointer2;
			load.a = pointer;
		
			llvmKernel.push_back(vm::LLVMStatement(load));
	
			if(argumentIndex == 0 || argumentIndex == 1)
			{
				vm::LLVMInttoptr int2ptr;
		
				int2ptr.d = operand;
				int2ptr.a = pointer2;
			
				llvmKernel.push_back(vm::LLVMStatement(int2ptr));
			}
		
			offset += sizeof(uint64_t);
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 10, "set_unique") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			// Loop over the variable pointer, then size
	
			vm::LLVMInstruction::Operand operand = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer = vm::LLVMInstruction::Operand(
				operand.name + "p",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer2 = vm::LLVMInstruction::Operand(
				operand.name + "pp",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Element));
	
			if(argumentIndex == 0 || argumentIndex == 1)
			{
				call.parameters.push_back(operand);
			}
			else
			{
				call.parameters.push_back(pointer2);
			}
	
			vm::LLVMGetelementptr get;
		
			get.d = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
			get.a = vm::LLVMInstruction::Operand("%parameters",
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
		
			get.indices.push_back(offset);
		
			llvmKernel.push_back(vm::LLVMStatement(get));
		
			vm::LLVMBitcast bitcast;
		
			bitcast.d = pointer;
			bitcast.a = get.d;
		
			llvmKernel.push_back(vm::LLVMStatement(bitcast));
		
			vm::LLVMLoad load;
		
			load.d = pointer2;
			load.a = pointer;
		
			llvmKernel.push_back(vm::LLVMStatement(load));
	
			if(argumentIndex == 0 || argumentIndex == 1)
			{
				vm::LLVMInttoptr int2ptr;
		
				int2ptr.d = operand;
				int2ptr.a = pointer2;
			
				llvmKernel.push_back(vm::LLVMStatement(int2ptr));
			}
		
			offset += sizeof(uint64_t);
	
			argumentIndex++;
		}
	}
	else if(name.compare(0, 10, "set_reduce") == 0 || name.compare(0, 9, "set_count") == 0)
	{
		for(auto argument : _kernel->operands())
		{
			// Loop over the variable pointer, then size
	
			vm::LLVMInstruction::Operand operand = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(
					translateType(argument.variable().type()),
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer = vm::LLVMInstruction::Operand(
				operand.name + "p",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Pointer));
	
			vm::LLVMInstruction::Operand pointer2 = vm::LLVMInstruction::Operand(
				operand.name + "pp",
				vm::LLVMInstruction::Type(
					vm::LLVMInstruction::I64,
				vm::LLVMInstruction::Type::Element));
	
			call.parameters.push_back(operand);
	
			vm::LLVMGetelementptr get;
		
			get.d = vm::LLVMInstruction::Operand(
				getValueString(value++), 
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
			get.a = vm::LLVMInstruction::Operand("%parameters",
				vm::LLVMInstruction::Type(vm::LLVMInstruction::I8,
				vm::LLVMInstruction::Type::Pointer));
		
			get.indices.push_back(offset);
		
			llvmKernel.push_back(vm::LLVMStatement(get));
		
			vm::LLVMBitcast bitcast;
		
			bitcast.d = pointer;
			bitcast.a = get.d;
		
			llvmKernel.push_back(vm::LLVMStatement(bitcast));
		
			vm::LLVMLoad load;
		
			load.d = pointer2;
			load.a = pointer;
		
			llvmKernel.push_back(vm::LLVMStatement(load));
	
			vm::LLVMInttoptr int2ptr;
	
			int2ptr.d = operand;
			int2ptr.a = pointer2;
		
			llvmKernel.push_back(vm::LLVMStatement(int2ptr));
		
			offset += sizeof(uint64_t);
	
			argumentIndex++;
		}
	}

	llvmKernel.push_back(vm::LLVMStatement(call));

	// return void
	report("return void");
	llvmKernel.push_back(vm::LLVMStatement(vm::LLVMRet()));

	llvmKernel.push_back(vm::LLVMStatement(vm::LLVMStatement::EndFunctionBody));

	// assemble the function
	report("assemble the function");
	llvmKernel.assemble();

	// parse the function
	report("parse the function");
	llvm::SMDiagnostic error;
	
	m = llvm::ParseAssemblyString(llvmKernel.code().c_str(), 
		m, error, llvm::getGlobalContext());

	report(		"dumping code:\n" << llvmKernel.numberedCode());

	if(m == 0)
	{
		report("   Generating interface to external function failed, "
			"dumping code:\n" << llvmKernel.numberedCode());
		std::string string;
		llvm::raw_string_ostream message(string);
		message << "LLVM Parser failed: ";
		error.print(llvmKernel.name.c_str(), message);

		throw std::runtime_error(message.str());
	}
	
	// verify the function
	report("verify the function");
	std::string verifyError;
	
	if(llvm::verifyModule(*m, llvm::ReturnStatusAction, &verifyError))
	{
		report("   Checking kernel failed, dumping code:\n" 
			<< llvmKernel.numberedCode());
		throw std::runtime_error("LLVM Verifier failed for kernel: " 
			+ llvmKernel.name + " : \"" + verifyError + "\"");
	}
	
	llvm::GlobalValue* global = m->getNamedValue(name);
	assertM(global != 0, "Global function " << name
		<< " not found in llvm module.");
	vm::LLVMState::jit()->addGlobalMapping(global,
		originalFunctionPointer);
	
	// done, the function is now in the module
	report("done, the function is now in the module");
	return m->getFunction(mangledName());
	#else
	return 0;
	#endif
}

void Runtime::LoadedBinaryKernel::load(const std::string& data)
{
	report("load binary kernel");
	// write the binary to a temp file
	_fileName = "./" + name + ".library";
	
	// write the data to a file
	report("write the data to a file");
	std::ofstream binary(_fileName.c_str(), std::ios::binary);

	binary.write((char*)data.data(), data.size());
	
	binary.close();

	// Get a function pointer to the function in the binary
	report("Get a function pointer to the function in the binary");
	_libraryHandle = dlopen(_fileName.c_str(), RTLD_LAZY);
	
	if(_libraryHandle == nullptr)
	{
		throw std::runtime_error("Failed to open library '"
			+ _fileName + "' with dlopen:\n" + dlerror() + "\n");
	}
	
	originalFunctionPointer = dlsym(_libraryHandle,
		name.c_str());

	if(originalFunctionPointer == nullptr)
	{
		throw std::runtime_error("Failed to load function '"
			+ name + "' with dlsym\n" + dlerror() + "\n");
	}

	#if HAVE_LLVM
	functionPointer = vm::LLVMState::jit()->getPointerToFunction(jitFunction(_module));
	#else
	functionPointer = nullptr;
	#endif
}

std::string Runtime::LoadedBinaryKernel::mangledName()
{
	return "_Z_thunk_" + _kernel->name();
}
	
Runtime::LoadedPTXKernel::LoadedPTXKernel(const std::string& name)
: LoadedKernel(name), handle(nullptr), _moduleHandle(nullptr)
{
	
}

Runtime::LoadedPTXKernel::~LoadedPTXKernel()
{
	report("load ptx kernel");
	cuda::CudaDriver::cuModuleUnload(_moduleHandle);
	report("load ptx kernel 2");
}

void Runtime::LoadedPTXKernel::load(const std::string& data)
{
	report("PTXKernel"/* << data*/);
	CUjit_option options[] = {
		CU_JIT_TARGET,
		CU_JIT_ERROR_LOG_BUFFER, 
		CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, 
	};
	
	const uint32_t errorLogSize       = 2048;
	uint32_t       errorLogActualSize = errorLogSize - 1;

	uint8_t errorLogBuffer[errorLogSize];

	std::memset(errorLogBuffer, 0, errorLogSize);

	void* optionValues[3] = {
		(void*)CU_TARGET_COMPUTE_35,
		(void*)errorLogBuffer, 
		hydrazine::bit_cast<void*>(errorLogActualSize), 
	};

	report("call cuModuleLoadDataEx");

	CUresult result = cuda::CudaDriver::cuModuleLoadDataEx(&_moduleHandle, 
		data.c_str(), 3, options, optionValues);

	if(result != CUDA_SUCCESS)
	{
		throw std::runtime_error(" Failed to load module with error: "
			+ cuda::CudaDriver::toString(result) + "\n Compiler Error: "
			+ std::string((const char*)errorLogBuffer));
	}

	report("call cuModuleGetFunction " << name);
	result = cuda::CudaDriver::cuModuleGetFunction(&handle,
		_moduleHandle, name.c_str());
	
	report("call cuModuleGetFunction " << handle);

	if(result != CUDA_SUCCESS)
	{
		throw std::runtime_error("Failed to get kernel function '" + name
			+ "' from CUDA module." + "\n Compiler Error: " + cuda::CudaDriver::toString(result) + "\n");
	}
}

Runtime::BasicBlockSchedule::BasicBlockSchedule(const OperationVector& ops)
: scheduledOperations(ops)
{

}

Runtime::VariableDescriptor::VariableDescriptor(CUdeviceptr a,
	DataType t, uint64_t s)
: address(a), type(t), sizeInElements(s), capacityInElements(0)
{

}

}

}

