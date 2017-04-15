/*!	\file   DataflowInformation.h
	\date   Thursday July 5, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the DataflowInformation class
*/

#pragma once

// Harmony Includes
#include <harmony/hir/interface/VariableDeclaration.h>
#include <harmony/hir/interface/ControlFlowGraph.h>

// Standard Library Includes
#include <set>
#include <unordered_map>

namespace harmony
{

namespace hir
{

/*! \brief Dataflow meta-data for a function control flow graph */
class DataflowInformation
{
public:
	typedef std::set<VariableDeclaration::Name> VariableSet;

	class LiveInOutSet
	{
	public:
		VariableSet liveIn;
		VariableSet liveOut;
	};

	typedef std::unordered_map<Block::Id, LiveInOutSet> BlockToLiveInOutMap;

public:
	DataflowInformation(Module* module);

public:
	void computeLiveInOutSets();

public:
	const LiveInOutSet& getLiveInOutSet(Block::Id id) const;

private:
	Module*             _module;
	BlockToLiveInOutMap _liveInOutSets;
};

}

}


