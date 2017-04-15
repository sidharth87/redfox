/*! \file   ControlFlowGraph.h
	\date   Tuesday November 24, 2009
	\author Gregory Diamos <gregory.diamso@gatech.edu>
	\brief  The header file for the ControlFlowGraph class
*/

#pragma once

// Harmony Includes
#include <harmony/hir/interface/ControlCall.h>
#include <harmony/hir/interface/KernelCall.h>

// Standard Library Includes
#include <list>
#include <vector>
#include <unordered_map>

// Forward Declarations
namespace harmony { namespace hir { class Module;           } }
namespace harmony { namespace hir { class ControlFlowGraph; } }

namespace harmony
{

namespace hir
{

/*! \brief A basic block */
class Block
{
public:
	friend class ControlFlowGraph;

public:
	/*! \brief A list of basic blocks */
	typedef std::list<Block> BlockList;
	/*! \brief An iterator over basic blocks */
	typedef BlockList::iterator iterator;
	/*! \brief An constant iterator over basic blocks */
	typedef BlockList::const_iterator const_iterator;
	/*! \brief A set of pointers to basic blocks */
	typedef std::vector<iterator> BlockPointerVector;
	/*! \brief An iterator to an iterator */
	typedef BlockPointerVector::iterator pointer_iterator;
	/*! \brief A const iterator to an iterator */
	typedef BlockPointerVector::const_iterator const_pointer_iterator;
	/*! \brief A list of kernels */
	typedef std::list<KernelCall> KernelList;
	/*! \brief A kernel list iterator */
	typedef KernelList::iterator kernel_iterator;
	/*! \brief A const kernel list iterator */
	typedef KernelList::const_iterator const_kernel_iterator;
	/*! \brief A unique identifier for a basic block */
	typedef unsigned int Id;
	
public:
	/*! \brief Construct a new basic block */
	Block(ControlFlowGraph* c, const std::string& label = "");

public:
	/*! \brief Get the id of the basic block */
	Id id() const;
	
	void set_id(Id id);
	/*! \brief Get the label of the basic block */
	const std::string& label() const;
	/*! \brief Get the set of targets in the block */
	const BlockPointerVector& targets() const;
	/*! \brief Get the set of predecessors of the block */
	const BlockPointerVector& predecessors() const;
	/*! \brief Get the owning control flow graph */
	const ControlFlowGraph& cfg() const;

public:
	/*! \brief Get a pointer iterator to a predecessor */
	pointer_iterator findPredecessor(iterator p);
	/*! \brief Get a const pointer iterator to a predecessor */
	const_pointer_iterator findPredecessor(iterator p) const;
	/*! \brief Get a pointer iterator to a target */
	pointer_iterator findTarget(iterator t);
	/*! \brief Get a const pointer iterator to a target */
	const_pointer_iterator findTarget(iterator t) const;

public:
	/*! \brief The sequence of kernels in the bb */
	KernelList  kernels;
	/*! \brief The ending control in the block */
	ControlCall control;
	
private:
	/*! \brief The id of the basic block */
	Id                 _id;
	/*! \brief The label of the basic block */
	std::string        _label;
	/*! \brief The set of targets in the block */
	BlockPointerVector _targets;
	/*! \brief The set of predecessors of the block */
	BlockPointerVector _predecessors;
	/*! \brief Pointer to the owning control flow graph */
	ControlFlowGraph*  _cfg;
};
	
/*! \brief A control flow graph of kernels */
class ControlFlowGraph
{
public:
	typedef Block::BlockList              BlockList;
	typedef Block::BlockPointerVector     BlockPointerVector;
	typedef Block::iterator               iterator;
	typedef Block::const_iterator         const_iterator;
	typedef Block::pointer_iterator       pointer_iterator;
	typedef Block::const_pointer_iterator const_pointer_iterator;
	typedef Block::kernel_iterator        kernel_iterator;
	typedef Block::const_kernel_iterator  const_kernel_iterator;

public:
	/*! \brief Construct a new control flow graph from an existing module */
	ControlFlowGraph(const Module& m);
	/*! \brief Destroy the current CFG */
	~ControlFlowGraph();

	/*! \brief Deep copy constructor */
	ControlFlowGraph(const ControlFlowGraph& g);

	/*! \brief Deep assignment operator */
	ControlFlowGraph& operator=(const ControlFlowGraph& g);

public:
	/*! \brief Get an iterator to a random block in the graph */
	iterator begin();
	/*! \brief Get a const iterator to a random block in the graph */
	const_iterator begin() const;

	/*! \brief Get the end iterator */
	iterator end();
	/*! \brief Get the end const iterator */
	const_iterator end() const;

	/*! \brief The total number of blocks in the graph, including entry/exit */
	size_t size() const;
	/*! \brief Does the graph have any blocks other than the entry/exit */
	bool empty() const;
	
	/*! \brief Get an iterator to the entry block */
	iterator entry();
	/*! \brief Get a const iterator to the entry block */
	const_iterator entry() const;

	/*! \brief Get an iterator to the exit block */
	iterator exit();
	/*! \brief Get a const iterator to the exit block */
	const_iterator exit() const;

public:
	iterator insert(const std::string& label, 
		iterator before, iterator after);
	iterator insert(const std::string& label);
	void connect(iterator from, iterator to);
	void disconnect(iterator from, iterator to);
	void erase(iterator b);

public:
	/*! \brief Clear all blocks other than the entry/exit */
	void clear();
	
public:
	const Module& module() const;

private:
	static void _copy(ControlFlowGraph& g, const ControlFlowGraph& c);
	void _reset();

private:
	const Module* _module;
	BlockList     _blocks;
	iterator      _entry;
	iterator      _exit;

};

}

}

namespace std
{

template<>
struct hash<harmony::hir::ControlFlowGraph::iterator>
{
	inline size_t operator()(
		const harmony::hir::ControlFlowGraph::iterator& it) const
	{
		return (size_t)it->id();
	}
};

template<>
struct hash<harmony::hir::ControlFlowGraph::const_iterator>
{
	inline size_t operator()(
		const harmony::hir::ControlFlowGraph::const_iterator& it) const
	{
		return (size_t)it->id();
	}
};

}



