/*! \file   ControlFlowGraph.cpp
	\date   Tuesday November 24, 2009
	\author Gregory Diamos <gregory.diamso@gatech.edu>
	\brief  The header file for the ControlFlowGraph class
*/

// Harmony Includes
#include <harmony/hir/interface/ControlFlowGraph.h>
#include <harmony/hir/interface/Module.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0 

namespace harmony
{

namespace hir
{

////////////////////////////////////////////////////////////////////////////////
// Basic Block Members
Block::Block(ControlFlowGraph* c, const std::string& l)
: _id(0), _label(l), _cfg(c)
{

}

Block::Id Block::id() const
{
	return _id;
}

void Block::set_id(Block::Id id)
{
	_id = id;
}
	
const std::string& Block::label() const
{
	return _label;
}

const Block::BlockPointerVector& Block::targets() const
{
	return _targets;
}

const Block::BlockPointerVector& Block::predecessors() const
{
	return _predecessors;
}

const ControlFlowGraph& Block::cfg() const
{
	return *_cfg;
}

Block::pointer_iterator Block::findPredecessor(iterator p)
{
	pointer_iterator predecessor = _predecessors.begin();

	for( ; predecessor != _predecessors.end(); ++predecessor)
	{
		if(*predecessor == p) break;
	}
	
	return predecessor;
}

Block::const_pointer_iterator Block::findPredecessor(iterator p) const
{
	const_pointer_iterator predecessor = _predecessors.begin();

	for( ; predecessor != _predecessors.end(); ++predecessor)
	{
		if(*predecessor == p) break;
	}
	
	return predecessor;
}

Block::pointer_iterator Block::findTarget(iterator t)
{
	pointer_iterator target = _targets.begin();

	for( ; target != _targets.end(); ++target)
	{
		if(*target == t) break;
	}
	
	return target;
}

Block::const_pointer_iterator Block::findTarget(iterator t) const
{
	const_pointer_iterator target = _targets.begin();

	for( ; target != _targets.end(); ++target)
	{
		if(*target == t) break;
	}
	
	return target;
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Control Flow Graph Members
ControlFlowGraph::ControlFlowGraph(const Module& m) : _module(&m)
{
	_reset();
}

ControlFlowGraph::~ControlFlowGraph()
{

}

ControlFlowGraph::ControlFlowGraph(const ControlFlowGraph& g) 
	: _module(g._module)
{
	_copy(*this, g);
}

ControlFlowGraph& ControlFlowGraph::operator=(const ControlFlowGraph& g)
{
	_module = g._module;
	
	_blocks.clear();
	_copy(*this, g);

	return *this;
}

ControlFlowGraph::iterator ControlFlowGraph::begin()
{
	return _blocks.begin();
}

ControlFlowGraph::const_iterator ControlFlowGraph::begin() const
{
	return _blocks.begin();
}

ControlFlowGraph::iterator ControlFlowGraph::end()
{
	return _blocks.end();
}

ControlFlowGraph::const_iterator ControlFlowGraph::end() const
{
	return _blocks.end();
}

size_t ControlFlowGraph::size() const
{
	return _blocks.size();
}

bool ControlFlowGraph::empty() const
{
	return size() <= 2;
}

ControlFlowGraph::iterator ControlFlowGraph::entry()
{
	return _entry;
}

ControlFlowGraph::const_iterator ControlFlowGraph::entry() const
{
	return const_iterator(_entry);
}

ControlFlowGraph::iterator ControlFlowGraph::exit()
{
	return _exit;
}

ControlFlowGraph::const_iterator ControlFlowGraph::exit() const
{
	return const_iterator(_exit);
}

ControlFlowGraph::iterator ControlFlowGraph::insert(const std::string& label,
	iterator before, iterator after)
{
	report("Inserting block " << label << " between " << before->label()
		<< " and " << after->label());

	iterator inserted = _blocks.insert(after, Block(this, label));
	
	pointer_iterator connection = before->findTarget(after);
	assertM(connection != before->targets().end(), 
		"No connection between " << before->label() 
		<< " and " << after->label());
	before->_targets.erase(connection);
	before->_targets.push_back(inserted);
	
	pointer_iterator reverseConnection = after->findPredecessor(before);
	assertM(reverseConnection != after->predecessors().end(), 
		"No back connection between " << before->label()
		<< " and " << after->label());
	after->_predecessors.erase(reverseConnection);
	after->_predecessors.push_back(inserted);
	
	inserted->_targets.push_back(after);
	inserted->_predecessors.push_back(before);
	
	return inserted;
}

ControlFlowGraph::iterator ControlFlowGraph::insert(const std::string& label)
{
	report( "Inserting block '" << label << "'." );

	return _blocks.insert(end(), Block(this, label));
}

void ControlFlowGraph::connect(iterator from, iterator to)
{
	report( "Connecting block " << from->label() << " to " << to->label() );

	assertM(from->findTarget(to) == from->targets().end(), 
		"Tried to add duplicate connection between " 
		<< from->label() << " and " << to->label() );
	from->_targets.push_back(to);
	
	assertM(to->findPredecessor(from) == to->predecessors().end(),
		"Malformed connection, back link but no forward link between " 
		<< from->label() << " and " << to->label() );
	to->_predecessors.push_back(from);
}

void ControlFlowGraph::disconnect(iterator from, iterator to)
{
	report("Disconnecting block " << from->label() << " from " << to->label());

	pointer_iterator connection = from->findTarget(to);
	assertM(connection != from->targets().end(), 
		"No connection between " << from->label() << " and " << to->label());
	from->_targets.erase(connection);

	pointer_iterator reverse = to->findPredecessor(from);
	assertM(reverse != to->predecessors().end(), 
		"Malformed connection, forward link but no backward link between " 
		<< from->label() << " and " << to->label());
	to->_predecessors.erase(reverse);
}

void ControlFlowGraph::erase(iterator b)
{
	report("Erasing block - " << b->label());

	assertM(b != _entry, "Cannot erase Entry block.");
	assertM(b != _exit,  "Cannot erase Exit block.");
	
	for(pointer_iterator from = b->_predecessors.begin();
		from != b->_predecessors.end(); ++from)
	{
		pointer_iterator connection = (*from)->findTarget(b);
		assert(connection != (*from)->targets().end());
		(*from)->_targets.erase(connection);
	}

	for(pointer_iterator to = b->_targets.begin();
		to != b->_targets.end(); ++to)
	{
		pointer_iterator connection = (*to)->findPredecessor(b);
		assert(connection != (*to)->predecessors().end());
		(*to)->_predecessors.erase(connection);
	}
	
	_blocks.erase(b);
}

void ControlFlowGraph::clear()
{
	report("Clearing CFG - " << module().name());
	_blocks.clear();
	_reset();
}

const Module& ControlFlowGraph::module() const
{
	return *_module;
}

void ControlFlowGraph::_copy(ControlFlowGraph& g, const ControlFlowGraph& c)
{
	typedef std::unordered_map<const_iterator, iterator> BlockMap;
	g._reset();
	
	BlockMap map;

	map.insert(std::make_pair(c.entry(), g.entry()));
	map.insert(std::make_pair(c.exit(),  g.exit()));
	
	for(const_iterator block = c.begin(); block != c.end(); ++block)
	{
		if(block == c.entry()) continue;
		if(block == c.exit())  continue;
		iterator newBlock = g.insert(block->label());

		newBlock->kernels = block->kernels;
		newBlock->control = block->control;
		newBlock->set_id(block->id());
		
		map.insert(std::make_pair(block, newBlock));
	}
	
	for(const_iterator block = c.begin(); block != c.end(); ++block)
	{
		BlockMap::const_iterator sourceMapping = map.find(block);
		assert(sourceMapping != map.end());

		for(const_pointer_iterator target = block->targets().begin();
			target != block->targets().end(); ++target)
		{
			BlockMap::const_iterator targetMapping = map.find(*target);
			assert(targetMapping != map.end());
			
			g.connect(sourceMapping->second, targetMapping->second);
		}
	}
}

void ControlFlowGraph::_reset()
{
	report("Resetting CFG " << module().name());
	
	_entry = insert("_ZEntry");
	_exit  = insert("_ZExit");
}

}

}


