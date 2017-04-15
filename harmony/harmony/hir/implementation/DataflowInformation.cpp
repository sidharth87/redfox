/*!	\file   DataflowInformation.h
	\date   Friday July 6, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the DataflowInformation class
*/

// Harmony Includes
#include <harmony/hir/interface/DataflowInformation.h>
#include <harmony/hir/interface/Module.h>

// Standard Library Includes
#include <cassert>

namespace harmony
{

namespace hir
{


DataflowInformation::DataflowInformation(Module* module)
: _module(module)
{

}

void DataflowInformation::computeLiveInOutSets()
{
	typedef std::vector<ControlFlowGraph::iterator> BlockQueue;

	// Create an empty set for each basic block
	// Create a worklist for each basic block
	BlockQueue changedBlocks;

	for(auto block = _module->cfg().begin();
		block != _module->cfg().end(); ++block)
	{
		_liveInOutSets.insert(std::make_pair(block->id(), LiveInOutSet()));
	
		changedBlocks.push_back(block);
	}
	
	// Propagate uses and defs until there is no change
	bool changed = true;

	BlockQueue previouslyChangedBlocks;
	
	while(changed)
	{
		previouslyChangedBlocks.swap(changedBlocks);
		changed = false;
		
		for(auto block : previouslyChangedBlocks)
		{
			auto liveInOutSet = _liveInOutSets.find(block->id());
			assert(liveInOutSet != _liveInOutSets.end());
			
			LiveInOutSet newLiveInOutSet;

			// new live-out is the union of the live-ins of the successors
			for(auto successor : block->targets())
			{
				auto successorLiveInOutSet =
					_liveInOutSets.find(successor->id());
				assert(successorLiveInOutSet != _liveInOutSets.end());
				
				for(auto variableName : successorLiveInOutSet->second.liveIn)
				{
					newLiveInOutSet.liveOut.insert(variableName);
				}
			}

			newLiveInOutSet.liveIn = newLiveInOutSet.liveOut;

			// Walk backwards, updating the live-in set
			for(auto kernel = block->kernels.rbegin();
				kernel != block->kernels.rend(); ++kernel)
			{
				// remove defs
				for(auto operand : kernel->operands())
				{
					if(!operand.isOut()) continue;
					
					newLiveInOutSet.liveIn.erase(operand.variable().name());
				}
				
				// add uses
				for(auto operand : kernel->operands())
				{
					if(!operand.isIn()) continue;
					
					newLiveInOutSet.liveIn.insert(operand.variable().name());
				}
			}
			
			// If the live in set is different, propagate the change to the
			// predecessors
			if(newLiveInOutSet.liveIn != liveInOutSet->second.liveIn)
			{
				changed = true;
				liveInOutSet->second.liveIn = newLiveInOutSet.liveIn;
				
				for(auto predecessor : block->predecessors())
				{
					changedBlocks.push_back(predecessor);
				}
			}
			
			liveInOutSet->second.liveOut = newLiveInOutSet.liveOut;
		}
		
		previouslyChangedBlocks.clear();
	}
}

const DataflowInformation::LiveInOutSet&
	DataflowInformation::getLiveInOutSet(Block::Id id) const
{
	auto set = _liveInOutSets.find(id);
	
	assert(set != _liveInOutSets.end());
	
	return set->second;
}

}

}


