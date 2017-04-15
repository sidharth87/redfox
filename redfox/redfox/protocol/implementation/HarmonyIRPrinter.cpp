/*! \file HarmoynIRPrinter.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Thursday November 4, 2010
	\brief The source file for the HarmonyIRPrinter class.
*/

#ifndef HARMONY_IR_PRINTER_CPP_INCLUDED
#define HARMONY_IR_PRINTER_CPP_INCLUDED

// RedFox Includes
#include <redfox/protocol/interface/HarmonyIRPrinter.h>
#include <redfox/protocol/interface/HarmonyIR.pb.h>

// Hydrazine Includes
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/string.h>

// Typedefs
typedef google::protobuf::RepeatedPtrField<hir::pb::BasicBlock> RepeatedBlock;
typedef google::protobuf::RepeatedPtrField<hir::pb::Kernel>     RepeatedKernel;
typedef google::protobuf::RepeatedPtrField<hir::pb::Operand>    RepeatedOperand;
typedef google::protobuf::RepeatedField<
	google::protobuf::uint32> RepeatedUnsigned;

// Helper Functions
static void writeKernel(std::ostream& stream, const hir::pb::Kernel* kernel)
{
	stream << kernel->name() << "(";

	if(kernel->operands().size() == 0)
	{
		stream << ")";
		return;
	}

	RepeatedOperand::const_iterator operand = kernel->operands().begin();

	if(operand->mode() == hir::pb::In)
	{
		stream << "in ";
	}
	else if(operand->mode() == hir::pb::Out)
	{
		stream << "out ";
	}
	else
	{
		stream << "inout ";
	}

	stream << "variable_" << operand->variable();

	++operand;

	for( ; operand != kernel->operands().end(); ++operand)
	{
		stream << ", ";
		
		if(operand->mode() == hir::pb::In)
		{
			stream << "in ";
		}
		else if(operand->mode() == hir::pb::Out)
		{
			stream << "out ";
		}
		else
		{
			stream << "inout ";
		}

		stream << "variable_" << operand->variable();
	}
	
	stream << ")";
}

namespace pb
{

HarmonyIRPrinter::HarmonyIRPrinter(std::istream& hir)
	: _graph(new hir::pb::KernelControlFlowGraph)
{	
	long long unsigned int bytes = 0;
	hir.read((char*)&bytes, sizeof(long long unsigned int));

	std::string message(bytes, ' ');

	hir.read((char*)message.c_str(), bytes);
	
	std::stringstream stream(message);
	
	if(!_graph->ParseFromIstream(&stream))
	{
		throw hydrazine::Exception("Failed to parse protocol buffer "
			"containing Harmony IR.");
	}
}

HarmonyIRPrinter::HarmonyIRPrinter(const hir::pb::KernelControlFlowGraph& graph)
	: _graph(new hir::pb::KernelControlFlowGraph(graph))
{

}

HarmonyIRPrinter::~HarmonyIRPrinter()
{
	delete _graph;
}

void HarmonyIRPrinter::write(std::ostream& stream) const
{	
	stream << "digraph HarmonyKernelControlFlowGraph {\n";

	// write basic blocks
	stream << "\n\n  // basic blocks\n\n";
	stream << "  block_0 [shape=Mdiamond,label=\"entry\"];\n";
	stream << "  block_1 [shape=Msquare,label=\"exit\"];\n";

	unsigned int id = 2;
	for(RepeatedBlock::const_iterator block = _graph->blocks().begin();
		block != _graph->blocks().end(); ++block, ++id)
	{
		stream << "  block_" << (block->id()+2)
			<< " [shape=record,label=\"{BasicBlock_" << (block->id()+2);
		
		for(RepeatedKernel::const_iterator kernel = block->kernels().begin();
			kernel != block->kernels().end(); ++kernel)
		{
			stream << " | ";
			writeKernel(stream, &*kernel);
		}

		stream << " | ";
		writeKernel(stream, &block->control());
		
		stream << "}\"];\n";
	}
	
	// write control edges
	stream << "\n\n  // edges\n\n";
	
	stream << "  block_0 -> block_" << (_graph->entry() + 2) << "\n";
	
	id = 2;
	for(RepeatedBlock::const_iterator block = _graph->blocks().begin();
		block != _graph->blocks().end(); ++block, ++id)
	{
		for(RepeatedUnsigned::const_iterator 
			target = block->control().targets().begin();
			target != block->control().targets().end(); ++target)
		{
			stream << "  block_" << (block->id() + 2)
				<< " -> block_" << (*target + 2) << "\n";
		}
	}
	
	stream << "  block_" << (_graph->exit() + 2) << " -> block_1\n";
	
	stream << "}\n";
}

}

#endif

