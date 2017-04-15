/*! \file   Module.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Tuesday November 24, 2009
	\brief  The source file for the Module class.
*/

// Harmony Includes
#include <harmony/hir/interface/Module.h>

#include <harmony/hir/interface/KernelIR.h>

// Hydrazine Includes
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <unordered_map>

// Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0 

namespace harmony
{

namespace hir
{

static KernelCall::Type translateType(pb::KernelType type)
{
	switch(type)
	{
	case pb::ComputeKernel:
	{
		return KernelCall::Kernel;
	}
	case pb::ControlDecision:
	{
		return KernelCall::ControlDecision;
	}
	case pb::UnconditionalBranch:
	{
		return KernelCall::UnconditionalBranch;
	}
	case pb::GetSize:
	{
		return KernelCall::GetSize;
	}
	case pb::BinaryKernel:
	{
		return KernelCall::ExternalKernel;
	}
	case pb::Resize:
	{
		return KernelCall::Resize;
	}
	case pb::UpdateSize:
	{
		return KernelCall::UpdateSize;
	}
	case pb::Exit:
	{
	 	return KernelCall::Exit;
	}
	default:
	{
		return KernelCall::InvalidType;
	}
	}
	return KernelCall::InvalidType;
}

typedef google::protobuf::RepeatedPtrField<pb::Operand> RepeatedOperand;

static KernelCall::OperandList translateOperands(
	const RepeatedOperand& operands,
	const Module::VariableDeclarationMap& variables)
{
	KernelCall::OperandList results;
	
	for(RepeatedOperand::const_iterator operand = operands.begin();
		operand != operands.end(); ++operand)
	{
		Operand::AccessMode mode = Operand::Invalid;
		
		switch(operand->mode())
		{
		case pb::In:
		{
			report("    " << operand->variable() << " (in)");
			mode = Operand::In;
			break;
		}
		case pb::Out:
		{
			report("    " << operand->variable() << " (out)");
			mode = Operand::Out;
			break;
		}
		case pb::InOut:
		{
			report("    " << operand->variable() << " (in) (out)");
			mode = Operand::InOut;
			break;
		}
		}
		
		Module::VariableDeclarationMap::const_iterator 
			variable = variables.find(operand->variable());
		assert(variable != variables.end());
		results.push_back(Operand(&variable->second, mode));
	}

	return results;
}

Module::Module()
: _name("::unloaded::"), _cfg(*this), _dataflowInformation(this)
{

}

Module::Module(const Module& m)
: _name(m.name()), _variables(m._variables),
  _cfg(*this), _dataflowInformation(this)
{
	_cfg = m.cfg();
	_dataflowInformation = m.dataflowInformation();
	_remapOperands();
}

Module& Module::operator=(const Module& m)
{
	_name      = m.name();
	_variables = m.variables();
	_cfg       = m.cfg();
	
	_remapOperands();
	
	return *this;
}

const Module::VariableDeclarationMap& Module::variables() const
{
	return _variables;
}

const std::string& Module::name() const
{
	return _name;
}

ControlFlowGraph& Module::cfg()
{
	return _cfg;
}

const ControlFlowGraph& Module::cfg() const
{
	return _cfg;
}

DataflowInformation& Module::dataflowInformation()
{
	return _dataflowInformation;
}

const DataflowInformation& Module::dataflowInformation() const
{
	return _dataflowInformation;
}

unsigned int Module::testCount() const
{
	return _testCount;
}

void Module::clear()
{
	_cfg.clear();
	_variables.clear();
	_name = "::unloaded::";
}

void Module::write(std::ostream& stream) const
{
	assertM(false, "Not Implemented");
}

void Module::read(std::istream& input)
{
	typedef google::protobuf::RepeatedPtrField<pb::Variable>   RepeatedVariable;
	typedef google::protobuf::RepeatedPtrField<pb::BasicBlock> RepeatedBlock;
	typedef google::protobuf::RepeatedPtrField<pb::Kernel>     RepeatedKernel;
	typedef google::protobuf::RepeatedField<unsigned int>      RepeatedUint32;

	typedef std::vector<ControlFlowGraph::iterator> BlockVector;
	typedef std::unordered_map<unsigned int, ControlFlowGraph::iterator> IdMap;
	
//	long long unsigned int messageSize = 0;
//	input.read((char*)&messageSize, sizeof(long long unsigned int));
//	
//	std::string cfgMessageBuffer(messageSize, ' ');
//	
//	input.read((char*)cfgMessageBuffer.c_str(), messageSize);
//	
//	std::stringstream stream(cfgMessageBuffer);
//
//	pb::KernelControlFlowGraph ir;
//	
//	if(!ir.ParseFromIstream(&stream))

	google::protobuf::io::ZeroCopyInputStream* raw_input = new google::protobuf::io::IstreamInputStream(&input);
	google::protobuf::io::CodedInputStream *coded_input = new google::protobuf::io::CodedInputStream(raw_input);

//	unsigned int messageSize = 0;
//	coded_input->ReadVarint32(&messageSize);
//	coded_input->ReadVarint32(&messageSize);
	coded_input->Skip(8);
	coded_input->SetTotalBytesLimit(400 << 20, 0);
	pb::KernelControlFlowGraph ir;
	
	if(!ir.ParseFromCodedStream(coded_input))
	{
		throw hydrazine::Exception("Failed to parse protocol buffer "
			"containing Harmony IR.");
	}
	
	report("Loading harmony IR file containing program: '" << ir.name() << "'");
	_name      = ir.name();
	_testCount = ir.testcount();
	
	report(" Loading variables...");
	
	for(RepeatedVariable::const_iterator variable = ir.variables().begin();
		variable != ir.variables().end(); ++variable)
	{
		report("  name " << variable->name() << " input " << variable->input() << " output " << variable->output() 
			<< " filename " << variable->filename() << " size " << variable->size());
		_variables.insert(std::make_pair(variable->name(), 
			VariableDeclaration(*variable)));
	}
	
	BlockVector blocks;
	IdMap       ids;

	report(" Loading kernels...");	
	for(RepeatedBlock::const_iterator block = ir.blocks().begin();
		block != ir.blocks().end(); ++block)
	{
		report("  basic block " << block->id());
		std::stringstream blockName;
		blockName << "KernelBasicBlock_" << block->id();
		
		ControlFlowGraph::iterator newBlock = _cfg.insert(blockName.str());
		blocks.push_back(newBlock);
		ids.insert(std::make_pair(block->id(), newBlock));
		
		report("   kernels...");
		for(RepeatedKernel::const_iterator kernel = block->kernels().begin();
			kernel != block->kernels().end(); ++kernel)
		{
			report("    " << kernel->name());
			
			newBlock->kernels.push_back(KernelCall(kernel->name(),
				translateType(kernel->type()), kernel->code(),
				translateOperands(kernel->operands(), _variables)));

		}

		report("   control '" << block->control().name() << "'");
		
		newBlock->control = ControlCall(block->control().name(),
			translateType(block->control().type()),
			block->control().code(),
			translateOperands(block->control().operands(), _variables));
	}

	report(" Connecting blocks...");
	IdMap::const_iterator entryMapping = ids.find(ir.entry());
	assert(entryMapping != ids.end());
	
	_cfg.connect(_cfg.entry(), entryMapping->second);
	
	IdMap::const_iterator exitMapping = ids.find(ir.exit());
	assert(exitMapping != ids.end());
	
	_cfg.connect(exitMapping->second, _cfg.exit());
	
	for(RepeatedBlock::const_iterator block = ir.blocks().begin();
		block != ir.blocks().end(); ++block)
	{

		ControlFlowGraph::iterator predecessor = blocks[std::distance(
			ir.blocks().begin(), block)];
		for(RepeatedUint32::const_iterator 
			target = block->control().targets().begin();
			target != block->control().targets().end(); ++target)
		{
			IdMap::const_iterator targetMapping = ids.find(*target);
			assert(targetMapping != ids.end());
			
			report("  " << block->id() << " -> " << *target);
			_cfg.connect(predecessor, targetMapping->second);
		}
	}

	unsigned int id = 0;

	for(ControlFlowGraph::iterator block = _cfg.begin(); block != _cfg.end(); ++block)
	{
		block->set_id(id++);
	}

	_dataflowInformation.computeLiveInOutSets();
}

void Module::_remapOperands()
{
	for(ControlFlowGraph::iterator block = _cfg.begin();
		block != _cfg.end(); ++block)
	{
		for(Block::KernelList::iterator kernel = block->kernels.begin();
			kernel != block->kernels.end(); ++kernel)
		{
			for(KernelCall::OperandList::iterator
				operand = kernel->_operands.begin();
				operand != kernel->_operands.end(); ++operand)
			{
				VariableDeclarationMap::iterator variable = _variables.find(
					operand->variable().name());
				assert(variable != _variables.end());
				operand->_variable = &variable->second;
			}
		}
		
		for(KernelCall::OperandList::iterator
			operand = block->control._operands.begin();
			operand != block->control._operands.end(); ++operand)
		{
			VariableDeclarationMap::iterator variable = _variables.find(
				operand->variable().name());
			assert(variable != _variables.end());
			operand->_variable = &variable->second;
		}
	}

}

}

std::ostream& operator<<(std::ostream& s, const hir::Module& g)
{
	g.write(s);
	return s;
}

}

