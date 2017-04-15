/*! \file HarmoynIRPrinter.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Thursday November 4, 2010
	\brief The header file for the HarmonyIRPrinter class.
*/

#ifndef HARMONY_IR_PRINTER_H_INCLUDED
#define HARMONY_IR_PRINTER_H_INCLUDED

// Standard Library Inludes
#include<iostream>

// Forward Declarations
namespace hir { namespace pb { class KernelControlFlowGraph; } }

/*! \brief A namespace for redfox protocol buffer related classes */
namespace pb
{

/*! \brief A class for producing a graphviz representation of a HIR program */
class HarmonyIRPrinter
{
public:
	/*! \brief Create a harmony IR printer from a GPU graph in an istream */
	HarmonyIRPrinter(std::istream& stream);

	/*! \brief Create a harmony IR printer from a GPU graph */
	HarmonyIRPrinter(const hir::pb::KernelControlFlowGraph& graph);

	/*! \brief Destroy the printer */
	~HarmonyIRPrinter();

	/*! \brief Write a graphviz representation of the IR to an ostream */
	void write(std::ostream& stream) const;

private:
	hir::pb::KernelControlFlowGraph* _graph;
};

}

#endif

