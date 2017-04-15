/*! \file RelationalAlgebraKernel.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Sunday October 31, 2010
	\brief The source file for the RelationalAlgebraKernel class.
*/

#ifndef RELATIONAL_ALGEBRA_KERNEL_CPP_INCLUDED
#define RELATIONAL_ALGEBRA_KERNEL_CPP_INCLUDED

// RedFox Includes
#include <redfox/nvcc/interface/RelationalAlgebraKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <algorithm>

#include<string.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1 

namespace nvcc
{

// Helper Functions
static std::string elementType(
	const RelationalAlgebraKernel::Variable& variable)
{
	assertM(variable.size() <= 16, "Only 16-dimension tuples are supported.");
	
	std::stringstream stream;
	
	stream << "ra::tuple::Tuple<";
	
	for(unsigned int i = 0; i < 16; ++i)
	{
		if(i != 0) stream << ", ";
		if(i < variable.size())
		{
			stream << variable[i].limit;
		}
		else
		{
			stream << "0";
		}
	}
	
	stream << ">";
	
	return stream.str();
}

static unsigned int bytes(
	const RelationalAlgebraKernel::Variable& variable)
{
	assertM(variable.size() <= 16, "Only 16-dimension tuples are supported.");
	
	unsigned int ret = 0;
	
	for(unsigned int i = 0; i < variable.size(); ++i)
	{
		if(i < variable.size())
		{
			ret += variable[i].limit;
		}
	}
	
	return ((ret + 7) / 8);

}

std::string RelationalAlgebraKernel::mapValueType(DataType type)
{
	switch(type)
	{
	case RelationalAlgebraKernel::InvalidDataType:
	{
		assert(false);
		break;
	}
	case RelationalAlgebraKernel::I8:
	{
		return "unsigned char";
	}
	case RelationalAlgebraKernel::I16:
	{
		return "unsigned short";
	}
	case RelationalAlgebraKernel::I32:
	{
		return "unsigned int";
	}
	case RelationalAlgebraKernel::I64:
	{
		return "long long unsigned int";
	}
	case RelationalAlgebraKernel::F32:
	{
		return "float";
	}
	case RelationalAlgebraKernel::F64:
	{
		return "double";
	}
	case RelationalAlgebraKernel::Pointer:
	{
		return "char *";
	}
	default: break;
	}

	return "ERROR";
}

static std::string comparisonValueType(
	const RelationalAlgebraKernel::ComparisonExpression& expression,
	const RelationalAlgebraKernel::Variable& variable)
{
	const RelationalAlgebraKernel::ComparisonOperand* op = &expression.a;
	if(op->type != RelationalAlgebraKernel::VariableIndex)
	{
		op = &expression.b;
	}
	
	assert(op->type == RelationalAlgebraKernel::VariableIndex);

	unsigned int index = op->variableIndex;
	assert(index < variable.size());
	
	return RelationalAlgebraKernel::mapValueType(variable[index].type);
}

static std::string comparisonType(
	const RelationalAlgebraKernel::ComparisonExpression& expression,
	const RelationalAlgebraKernel::Variable& variable)
{
	std::string type;

	if(expression.a.type != RelationalAlgebraKernel::VariableIndex && expression.b.type == RelationalAlgebraKernel::VariableIndex)
	{
	switch(expression.comparison)
	{
	case RelationalAlgebraKernel::Eq:
	{
		type = "ra::comparisons::eq";
		break;
	}
	case RelationalAlgebraKernel::EqString:
	{
		type = "ra::comparisons::eqstring";
		break;
	}
	case RelationalAlgebraKernel::Ne:
	{
		type = "ra::comparisons::ne";
		break;
	}
	case RelationalAlgebraKernel::Slt:
	{
		type = "ra::comparisons::sgt";
		break;
	}
	case RelationalAlgebraKernel::Lt:
	{
		type = "ra::comparisons::gt";
		break;
	}
	case RelationalAlgebraKernel::Sle:
	{
		type = "ra::comparisons::sge";
		break;
	}
	case RelationalAlgebraKernel::Le:
	{
		type = "ra::comparisons::ge";
		break;
	}
	case RelationalAlgebraKernel::Sgt:
	{
		type = "ra::comparisons::slt";
		break;
	}
	case RelationalAlgebraKernel::Gt:
	{
		type = "ra::comparisons::lt";
		break;
	}
	case RelationalAlgebraKernel::Sge:
	{
		type = "ra::comparisons::sle";
		break;
	}
	case RelationalAlgebraKernel::Ge:
	{
		type = "ra::comparisons::le";
		break;
	}
	case RelationalAlgebraKernel::Like:
	{
		type = "ra::comparisons::like";
		break;
	}
	case RelationalAlgebraKernel::NotLike:
	{
		type = "ra::comparisons::notlike";
		break;
	}
	case RelationalAlgebraKernel::InvalidComparison:
	{
		assertM(false, "Invalid comparison type.");
	}
	}
	}
	else
	{
	switch(expression.comparison)
	{
	case RelationalAlgebraKernel::Eq:
	{
		type = "ra::comparisons::eq";
		break;
	}
	case RelationalAlgebraKernel::EqString:
	{
		type = "ra::comparisons::eqstring";
		break;
	}
	case RelationalAlgebraKernel::Ne:
	{
		type = "ra::comparisons::ne";
		break;
	}
	case RelationalAlgebraKernel::Slt:
	{
		type = "ra::comparisons::slt";
		break;
	}
	case RelationalAlgebraKernel::Lt:
	{
		type = "ra::comparisons::lt";
		break;
	}
	case RelationalAlgebraKernel::Sle:
	{
		type = "ra::comparisons::sle";
		break;
	}
	case RelationalAlgebraKernel::Le:
	{
		type = "ra::comparisons::le";
		break;
	}
	case RelationalAlgebraKernel::Sgt:
	{
		type = "ra::comparisons::sgt";
		break;
	}
	case RelationalAlgebraKernel::Gt:
	{
		type = "ra::comparisons::gt";
		break;
	}
	case RelationalAlgebraKernel::Sge:
	{
		type = "ra::comparisons::sge";
		break;
	}
	case RelationalAlgebraKernel::Ge:
	{
		type = "ra::comparisons::ge";
		break;
	}
	case RelationalAlgebraKernel::Like:
	{
		type = "ra::comparisons::like";
		break;
	}
	case RelationalAlgebraKernel::NotLike:
	{
		type = "ra::comparisons::notlike";
		break;
	}
	case RelationalAlgebraKernel::InvalidComparison:
	{
		assertM(false, "Invalid comparison type.");
	}
	}
	}

	std::string tmp = comparisonValueType(expression, variable);

	if(tmp.compare("double") != 0)
		return type + "<" + tmp + ">";
	else
		return type + tmp;
}

static bool isCompareWithConstantVariable(
	const RelationalAlgebraKernel::ComparisonExpression& expression)
{
	return expression.a.type == RelationalAlgebraKernel::ConstantVariable
		|| expression.b.type == RelationalAlgebraKernel::ConstantVariable;
}

static bool isCompareWithConstant(
	const RelationalAlgebraKernel::ComparisonExpression& expression)
{
	return expression.a.type == RelationalAlgebraKernel::Constant
		|| expression.b.type == RelationalAlgebraKernel::Constant;
}

static bool isCompareWithIndex(
	const RelationalAlgebraKernel::ComparisonExpression& expression)
{
	return expression.a.type == RelationalAlgebraKernel::VariableIndex
		&& expression.b.type == RelationalAlgebraKernel::VariableIndex;
}

static bool isCompareWithArith(
	const RelationalAlgebraKernel::ComparisonExpression& expression)
{
	return expression.a.type == RelationalAlgebraKernel::ConstantArith
		|| expression.b.type == RelationalAlgebraKernel::ConstantArith;
}

static std::string comparisonFieldOne(
	const RelationalAlgebraKernel::ComparisonExpression& expression)
{
	const RelationalAlgebraKernel::ComparisonOperand* op = &expression.a;

	if(op->type != RelationalAlgebraKernel::VariableIndex)
	{
		op = &expression.b;
	}

	assert(op->type == RelationalAlgebraKernel::VariableIndex);

	std::stringstream stream;
	
	stream << op->variableIndex;
	
	return stream.str();
}

static std::string comparisonFieldTwo(
	const RelationalAlgebraKernel::ComparisonExpression& expression)
{
	const RelationalAlgebraKernel::ComparisonOperand* op = &expression.b;

	if(op->type != RelationalAlgebraKernel::VariableIndex)
	{
		op = &expression.a;
	}

	assert(op->type == RelationalAlgebraKernel::VariableIndex);

	std::stringstream stream;
	
	stream << op->variableIndex;
	
	return stream.str();
}

static std::string comparisonConstant(
	const RelationalAlgebraKernel::ComparisonExpression& expression,
	const RelationalAlgebraKernel::Variable& variable)
{
	const RelationalAlgebraKernel::ComparisonOperand* constant = &expression.a;
	const RelationalAlgebraKernel::ComparisonOperand* var      = &expression.b;

	if(constant->type != RelationalAlgebraKernel::Constant)
	{
		constant = &expression.b;
		var      = &expression.a;
	}

	assert(var->type      == RelationalAlgebraKernel::VariableIndex);
	assert(constant->type == RelationalAlgebraKernel::Constant);
	assert(var->variableIndex < variable.size());

	std::stringstream stream;
	
	switch(variable[var->variableIndex].type)
	{
	case RelationalAlgebraKernel::InvalidDataType: assert(false);
	case RelationalAlgebraKernel::I8:
	{
		stream << (int)((unsigned char)constant->intValue);
		break;
	}
	case RelationalAlgebraKernel::I16:
	{
		stream << (int)((unsigned short)constant->intValue);
		break;
	}
	case RelationalAlgebraKernel::I32:
	{
		stream << (unsigned int)constant->intValue;
		break;
	}
	case RelationalAlgebraKernel::I64:
	{
		stream << constant->intValue;
		break;
	}
	case RelationalAlgebraKernel::F32:
	{
		stream << (float)constant->floatValue;
		break;
	}
	case RelationalAlgebraKernel::F64:
	{
		stream << constant->floatValue;
		break;
	}
	case RelationalAlgebraKernel::Pointer:
	{
		stream << constant->stringId;
		break;
	}
	default: assertM(false, "Invalid constant type.");
	}
	
	return stream.str();
}

static bool isCompareWithString(
	const RelationalAlgebraKernel::ComparisonExpression& expression,
	const RelationalAlgebraKernel::Variable& variable)
{
	const RelationalAlgebraKernel::ComparisonOperand* constant = &expression.a;
	const RelationalAlgebraKernel::ComparisonOperand* var      = &expression.b;

	if(constant->type != RelationalAlgebraKernel::Constant)
	{
		constant = &expression.b;
		var      = &expression.a;
	}

	assert(var->type      == RelationalAlgebraKernel::VariableIndex);
	assert(constant->type == RelationalAlgebraKernel::Constant);
	assert(var->variableIndex < variable.size());

	std::stringstream stream;
	
	switch(variable[var->variableIndex].type)
	{
	case RelationalAlgebraKernel::InvalidDataType: assert(false);
	case RelationalAlgebraKernel::I8:
	case RelationalAlgebraKernel::I16:
	case RelationalAlgebraKernel::I32:
	case RelationalAlgebraKernel::I64:
	case RelationalAlgebraKernel::F32:
	case RelationalAlgebraKernel::F64:
		return false;
	case RelationalAlgebraKernel::Pointer:
		return true;
	default: assertM(false, "Invalid constant type.");
	}
	
	return false;
}

static bool isCompareWithFloat(
	const RelationalAlgebraKernel::ComparisonExpression& expression,
	const RelationalAlgebraKernel::Variable& variable)
{
	const RelationalAlgebraKernel::ComparisonOperand* constant = &expression.a;
	const RelationalAlgebraKernel::ComparisonOperand* var      = &expression.b;

	if(constant->type != RelationalAlgebraKernel::Constant)
	{
		constant = &expression.b;
		var      = &expression.a;
	}

	assert(var->type      == RelationalAlgebraKernel::VariableIndex);
	assert(constant->type == RelationalAlgebraKernel::Constant);
	assert(var->variableIndex < variable.size());

	std::stringstream stream;
	
	switch(variable[var->variableIndex].type)
	{
	case RelationalAlgebraKernel::InvalidDataType: assert(false);
	case RelationalAlgebraKernel::I8:
	case RelationalAlgebraKernel::I16:
	case RelationalAlgebraKernel::I32:
	case RelationalAlgebraKernel::I64:
		return false;
	case RelationalAlgebraKernel::F32:
	case RelationalAlgebraKernel::F64:
		return true;
	case RelationalAlgebraKernel::Pointer:
		return false;
	default: assertM(false, "Invalid constant type.");
	}
	
	return false;
}

static std::string comparisonArithLeft(
	const RelationalAlgebraKernel::ComparisonExpression& expression,
	const RelationalAlgebraKernel::Variable& variable)
{
	const RelationalAlgebraKernel::ComparisonOperand* constant = &expression.a;
	const RelationalAlgebraKernel::ComparisonOperand* var      = &expression.b;

	if(constant->type != RelationalAlgebraKernel::ConstantArith)
	{
		constant = &expression.b;
		var      = &expression.a;
	}

	assert(var->type      == RelationalAlgebraKernel::VariableIndex);
	assert(constant->type == RelationalAlgebraKernel::ConstantArith);
	assert(var->variableIndex < variable.size());

	RelationalAlgebraKernel::ArithExpNode _arithExp = constant->arith;
	std::stringstream left;
	left << _arithExp.left->i;

	return left.str();
}

static std::string comparisonArithRight(
	const RelationalAlgebraKernel::ComparisonExpression& expression,
	const RelationalAlgebraKernel::Variable& variable)
{
	const RelationalAlgebraKernel::ComparisonOperand* constant = &expression.a;
	const RelationalAlgebraKernel::ComparisonOperand* var      = &expression.b;

	if(constant->type != RelationalAlgebraKernel::ConstantArith)
	{
		constant = &expression.b;
		var      = &expression.a;
	}

	assert(var->type      == RelationalAlgebraKernel::VariableIndex);
	assert(constant->type == RelationalAlgebraKernel::ConstantArith);
	assert(var->variableIndex < variable.size());

	RelationalAlgebraKernel::ArithExpNode _arithExp = constant->arith;
	std::stringstream right;

	if(_arithExp.right->type == RelationalAlgebraKernel::F32 || _arithExp.right->type == RelationalAlgebraKernel::F64)
		right << _arithExp.right->f;
	else
		right << _arithExp.right->i;

	return right.str();
}
RelationalAlgebraKernel::Element::Element(
	DataType t, long long unsigned int l) : type(t), limit(l)
{

}

std::string RelationalAlgebraKernel::toString(Operator op, unsigned int id)
{
	std::stringstream stream;
	
	stream << id;

	switch(op)
	{
	case InvalidOperator:      return "INVALID_OP" + stream.str();
	case Union:                return "set_union" + stream.str();
	case Intersection:         return "set_intersection" + stream.str();
	case ProductGetResultSize: return "product_get_result_size" + stream.str();
	case Product:              return "set_product" + stream.str();
	case Difference:           return "set_difference" + stream.str();
	case DifferenceGetResultSize:           return "difference_get_result_size" + stream.str();
	case JoinFindBounds:       return "join_find_bounds" + stream.str();
	case ModernGPUJoinFindBounds:       return "mgpu_join_find_bounds" + stream.str();
	case JoinTempSize:         return "join_get_temp_size" + stream.str();
	case ModernGPUJoinTempSize:         return "join_get_temp_size" + stream.str();
	case ModernGPUJoinResultSize:         return "join_get_result_size" + stream.str();
	case JoinMain:             return "set_join" + stream.str();
	case ModernGPUJoinMain:             return "mgpu_join_main" + stream.str();
	case ModernGPUJoinGather:             return "mgpu_join_gather" + stream.str();
	case Scan:                 return "scan" + stream.str();
	case JoinGetSize:          return "join_get_size" + stream.str();
	case JoinGather:           return "join_gather" + stream.str();
	case Project:              return "set_project" + stream.str();
	case ProjectGetResultSize: return "project_get_result_size" + stream.str();
	case SelectMain:           return "set_select" + stream.str();
	case SelectGetResultSize:  return "select_get_result_size" + stream.str();
	case SelectGather:         return "select_gather" + stream.str();
	case AssignValue:          return "assign_value" + stream.str();
	case Arithmetic:	   return "set_arith" + stream.str();
	case ArithGetResultSize:   return "arith_get_result_size" + stream.str();
	case AppendStringGetResultSize:   return "appendstring_get_result_size" + stream.str();
	case Split:   	   	   return "set_split" + stream.str();
	case SplitKey: 	   	   return "set_splitkey" + stream.str();
	case SplitGetResultSize:   return "split_get_result_size" + stream.str();
	case SplitKeyGetResultSize:   return "split_key_get_result_size" + stream.str();
	case Merge:	   	   return "set_merge" + stream.str();
	case MergeGetResultSize:   return "merge_get_result_size" + stream.str();
	case Sort:   	  	   return "set_sort" + stream.str();
	case ModernGPUSortPair:    return "set_mgpusortpair" + stream.str();
	case ModernGPUSortKey:     return "set_mgpusortkey" + stream.str();
	case RadixSortPair:    return "set_b40csortpair" + stream.str();
	case RadixSortKey:     return "set_b40csortkey" + stream.str();
	case Unique:   	  	   return "set_unique" + stream.str();
	case Total:   	  	   return "set_reduce" + stream.str();
	case SingleTotal:  	   return "set_single_reduce" + stream.str();
	case SingleCount:  	   return "set_single_reduce" + stream.str();
	case SingleMax:  	   return "set_single_reduce" + stream.str();
	case Max:   	  	   return "set_reduce" + stream.str();
	case Min:   	  	   return "set_reduce" + stream.str();
	case Count:   	  	   return "set_count" + stream.str();
	case Generate: 	  	   return "set_generate" + stream.str();
	case GenerateGetResultSize: return "generate_get_result_size" + stream.str();
	case ConvGetResultSize:    return "conv_get_result_size" + stream.str();
	case Convert:   	   return "set_convert" + stream.str();
	case SetString:   	   return "set_setstring" + stream.str();
	case UnionGetResultSize: return "union_get_result_size" + stream.str();
	case SubStringGetResultSize: return "substring_get_result_size" + stream.str();
	case SubString:	  	   return "set_substring" + stream.str();
	}

	return "";
}

RelationalAlgebraKernel::RelationalAlgebraKernel(Operator op, const Variable& d,
	const Variable& a, const Variable& b, const Variable& c) : destination(d),
	sourceA(a), sourceB(b), sourceC(c), _operator(op)
{
	if(op == Total || op == Max || op == Count || op == Min || 
		op == ModernGPUSortPair || op == RadixSortPair|| op == ModernGPUJoinFindBounds || op == ModernGPUJoinMain)
	{
		destination = d;
		destination_1 = a;
		sourceA = b;
		sourceB = c;
	}
	else
	{
		destination = d;
		sourceA = a;
		sourceB = b;
		sourceC = c;
	}
}

RelationalAlgebraKernel::RelationalAlgebraKernel(const Variable& d,
	const Variable& a, const Variable& b, Operator op) : destination(d),
	destination_1(a), sourceA(b), _operator(op)
{
	if(op == SplitGetResultSize)
	{
		destination = d;
		destination_1 = a;
		sourceA = b;
	}
//	else if(op == MergeGetResultSize)
//	{
//		destination = d;
//		sourceA = a;
//		sourceB = b;
//	}
}

RelationalAlgebraKernel::RelationalAlgebraKernel(const Variable& d,
	const Variable& a, Operator op) : destination(d),
	 sourceA(a), _operator(op)
{

}

RelationalAlgebraKernel::RelationalAlgebraKernel(
	const Variable& a, Operator op)  
{
	if(op == Total) 
	{
		sourceA = a;
		_operator = SingleTotal;
	}
	else if(op == Max) 
	{
		sourceA = a;
		_operator = SingleMax;
	}
	else if(op == Count) 
	{
		sourceA = a;
		_operator = SingleCount;
	}
}

RelationalAlgebraKernel::RelationalAlgebraKernel(Operator op, const Variable& d,
	const Variable& a, unsigned int t)  : destination(d),
	sourceA(a), _operator(op), _threadCount(t)
{

}

RelationalAlgebraKernel::RelationalAlgebraKernel(Operator op, const Variable& d,
	const Variable& a, const Variable& b, const Variable& c, unsigned int f,
	unsigned int t) : destination(d), sourceA(a), sourceB(b), sourceC(c),
	_operator(op), _keyFields(f), _threadCount(t)
{

}

//RelationalAlgebraKernel::RelationalAlgebraKernel(const ComparisonVector& c, 
//	const Variable& d, const Variable& a) : destination(d), sourceA(a),
//	_operator(SelectMain), _comparisons(c)
//{
//	
//}

RelationalAlgebraKernel::RelationalAlgebraKernel(const ComparisonVector& c, 
	const Variable& d, const Variable& a, const Variable& b) : destination(d), sourceA(a),
	sourceB(b), _operator(SelectMain), _comparisons(c)
{
	
}

RelationalAlgebraKernel::RelationalAlgebraKernel(const Variable& d,
	const Value& v) : destination(d), _operator(AssignValue), _value(v)
{

}

RelationalAlgebraKernel::RelationalAlgebraKernel(const Variable& d,
	const Variable& a, const IndexVector& i) : destination(d), sourceA(a),
	_operator(Project), _indicies(i)
{

}

RelationalAlgebraKernel::RelationalAlgebraKernel(const ArithExpNode& arith, 
	const Variable& d, const Variable& a) : destination(d), sourceA(a), 
	_operator(Arithmetic), _arithExp(arith)
{

}

RelationalAlgebraKernel::RelationalAlgebraKernel(Operator op, 
	const Variable& d, const Variable& a, 
	const int& domains) : destination(d), sourceA(a), _operator(op), _domains(domains) 
{

}

RelationalAlgebraKernel::RelationalAlgebraKernel(Operator op, 
	const Variable& a, const Variable& b, 
	const Variable& c, const int& domains) : _operator(op) 
{
	if(op == Split)
	{
		destination = a;
		destination_1 = b;
		sourceA = c; 	
	}
	else if(op == JoinFindBounds)
	{
		destination = a;
		sourceA = b;
		sourceB = c;
		_keyFields = domains;
	}
	else if(op == Merge)
	{
		destination = a;
		sourceA = b;
		sourceB = c;
	}
	
	_domains = domains;
}

RelationalAlgebraKernel::RelationalAlgebraKernel(Operator op, 
	const Variable& a, const Variable& b, 
	const Variable& c, const int& domains,
	const Variable& d, const Variable& e) : _operator(op) 
{
	if(op == ModernGPUJoinGather)
	{
		destination = a;
		sourceA = b;
		sourceB = c;
		_keyFields = domains;
		sourceC = d;
		sourceD = e;
	}
}

RelationalAlgebraKernel::RelationalAlgebraKernel(const DataType type,
	const unsigned int offset, 
	const Variable& d, const Variable& a) : 
	destination(d), sourceA(a), _operator(Convert), _type(type), _offset(offset)

{

}

RelationalAlgebraKernel::RelationalAlgebraKernel(const Variable& a, const unsigned int index) : 
	sourceA(a), _operator(SetString),  _index(index)
{

}

RelationalAlgebraKernel::RelationalAlgebraKernel(const Variable&d, const Variable& a, const unsigned int index) : 
	destination(d), sourceA(a), _operator(SubString),  _index(index)
{

}

void RelationalAlgebraKernel::setCtaCount(unsigned int ctas)
{
	_ctaCount = ctas;
}

void RelationalAlgebraKernel::setThreadCount(unsigned int threads)
{
	_threadCount = threads;
}

void RelationalAlgebraKernel::set_id(unsigned int id)
{
	_id = id;
}

RelationalAlgebraKernel::Operator RelationalAlgebraKernel::op() const
{
	return _operator;
}

unsigned int RelationalAlgebraKernel::id() const
{
	return _id;
}

const RelationalAlgebraKernel::ComparisonVector& 
	RelationalAlgebraKernel::comparisons() const
{
	return _comparisons;
}

std::string RelationalAlgebraKernel::cudaSourceRepresentation() const
{
	switch(op())
	{
	case Union:                return _unionSource();
	case Intersection:         return _intersectionSource();
	case ProductGetResultSize: return _productGetResultSizeSource();
	case Product:              return _productSource();
	case Difference:           return _differenceSource();
	case DifferenceGetResultSize: return _differenceGetResultSizeSource();
	case JoinFindBounds:       return _joinFindBoundsSource();
	case ModernGPUJoinFindBounds:       return _moderngpuJoinFindBoundsSource();
	case ModernGPUJoinGather:       return _moderngpuJoinGatherSource();
	case JoinTempSize:         return _joinTempSizeSource();
	case ModernGPUJoinTempSize:         return _moderngpuJoinTempSizeSource();
	case ModernGPUJoinResultSize:         return _moderngpuJoinResultSizeSource();
	case JoinMain:             return _joinMainSource();
	case ModernGPUJoinMain:             return _moderngpuJoinMainSource();
	case Scan:                 return _scanSource();
	case JoinGetSize:          return _joinGetSizeSource();
	case JoinGather:           return _joinGatherSource();
	case Project:              return _projectSource();
	case ProjectGetResultSize: return _projectGetResultSizeSource();
	case SelectMain:           return _selectMainSource();
	case SelectGetResultSize:  return _selectGetResultSizeSource();
	case SelectGather:         return _selectGatherSource();
	case AssignValue:          return _assignSource();
	case Arithmetic:   	   return _arithSource();
	case ArithGetResultSize:   return _arithGetResultSizeSource();
	case AppendStringGetResultSize:   return _appendStringGetResultSizeSource();
	case Split:   	   	   return _splitSource();
	case SplitKey: 	   	   return _splitkeySource();
	case SplitGetResultSize:   return _splitGetResultSizeSource();
	case SplitKeyGetResultSize:   return _splitKeyGetResultSizeSource();
	case Merge:   	   	   return _mergeSource();
	case MergeGetResultSize:   return _mergeGetResultSizeSource();
	case Sort:   	  	   return _sortSource();
	case ModernGPUSortPair:	   return _moderngpuSortPairSource();
	case ModernGPUSortKey:     return _moderngpuSortKeySource();
	case RadixSortPair:	   return _b40cSortPairSource();
	case RadixSortKey:     return _b40cSortKeySource();
 	case Unique:   	  	   return _uniqueSource();
	case Count:   	  	   return _countSource();
	case Total:   	  	   return _reduceSource(Total);
	case SingleTotal:   	   return _singleReduceSource();
	case SingleCount:   	   return _singleReduceSource();
	case SingleMax:   	   return _singleReduceSource();
	case Max:   	  	   return _reduceSource(Max);
	case Min:   	  	   return _reduceSource(Min);
	case Generate:  	   return _generateSource();
	case GenerateGetResultSize: return _generateGetResultSizeSource();
	case Convert:   	   return _convertSource();
	case ConvGetResultSize:    return _convGetResultSizeSource();
	case SetString: 	   return _setStringSource();
	case UnionGetResultSize: return _unionGetResultSizeSource();
	case SubStringGetResultSize: return _substringGetResultSizeSource();
	case SubString: 	     return _substringSource();

	default: assertM(false, "Invalid operation.");
	}
}

std::string RelationalAlgebraKernel::name() const
{
	// TODO fill this out
	std::string instantiation = toString(op(), id());

	return instantiation;
}

std::string RelationalAlgebraKernel::_intersectionSource() const
{
	assertM(false, "Not implemented.");
}

std::string RelationalAlgebraKernel::_productGetResultSizeSource() const
{
	report("Getting CUDA source for product get result size ptx kernel.");

	std::string destType  = elementType(destination);
	std::string leftType  = elementType(sourceA);
	std::string rightType = elementType(sourceB);

	std::stringstream idstream;
	idstream << _id;

	return "#include <redfox/ra/interface/Product.h>\n"
		"typedef " + destType  + " DestTuple;\n"
		"typedef " + leftType  + " LeftTuple;\n"
		"typedef " + rightType + " RightTuple;\n"
		"typedef LeftTuple::BasicType LeftType;\n"
		"typedef RightTuple::BasicType RightType;\n"
		"extern \"C\" __global__ void product_get_result_size" + idstream.str() + "("
			"long long unsigned int* size, "
			"const long long unsigned int* leftSize, "
			"const long long unsigned int* rightSize)\n"
		"{\n"
		"    ra::cuda::resultSize<DestTuple, LeftTuple, RightTuple>(size, "
			"*leftSize/sizeof(LeftType), "
			"*rightSize/sizeof(RightType));\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_productSource() const
{
	report("Getting CUDA source for product ptx kernel.");

	std::string destType  = elementType(destination);
	std::string leftType  = elementType(sourceA);
	std::string rightType = elementType(sourceB);

	std::stringstream idstream;
	idstream << _id;
std::string tmp = "";
//if(_id == 51) tmp = "2";

	return "#include <redfox/ra/interface/Product.h>\n"
		"typedef " + destType  + " DestTuple;\n"
		"typedef " + leftType  + " LeftTuple;\n"
		"typedef " + rightType + " RightTuple;\n"
		"typedef DestTuple::BasicType DestType;\n"
		"typedef LeftTuple::BasicType LeftType;\n"
		"typedef RightTuple::BasicType RightType;\n"
		"extern \"C\" __global__ void set_product" + idstream.str() + "("
			"DestType* result,"
			"const LeftType* left,   const long long unsigned int* leftSize,"
			"const RightType* right, const long long unsigned int* rightSize)\n"
		"{\n"
		"    ra::cuda::product" + tmp + "<DestTuple, LeftTuple, RightTuple>(result, "
			"left, left + (*leftSize/sizeof(LeftType)), "
			"right, right + (*rightSize/sizeof(RightType)));\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_differenceSource() const
{
	report("Getting CUDA source for merge ptx kernel.");

	std::string type = elementType(destination);

	std::stringstream idstream;
	idstream << _id;

	return "#include <stdio.h>\n"
		"#include <redfox/ra/interface/Tuple.h>\n"
		"#include <redfox/ra/interface/Difference.h>\n"
		"typedef " + type    + " Tuple;\n"
		"typedef Tuple::BasicType TupleType;\n"
		"extern \"C\" void set_difference" + idstream.str() + "(TupleType* result, long long unsigned int* size,"
		"TupleType* left_begin, const long long unsigned int left_size, "
		"TupleType* right_begin, const long long unsigned int right_size,"
		"unsigned long long int type)\n"
		"{\n"
			"    redfox::difference(result, size, left_begin, left_begin + (left_size / sizeof(TupleType)),"
				"right_begin, right_begin + (right_size / sizeof(TupleType)), type);\n"

		"}\n";

}

std::string RelationalAlgebraKernel::_joinFindBoundsSource() const
{
	report("Getting CUDA source for join find bounds ptx kernel.");

	std::string leftType  = elementType(sourceA);
	std::string rightType = elementType(sourceB);
	std::string keyFields = _fields();
	std::string ctaCount  = _ctas();

	std::stringstream idstream;
	idstream << _id;
std::string tmp = "";
//if(_id == 54) tmp = "2";

	if(sourceA[ _keyFields - 1].type == Pointer)
	{
		std::stringstream s;

		s << (_keyFields - 1);
		keyFields = s.str();
	}
		
	return 	"#include <redfox/ra/interface/Join.h>\n"
		"typedef " + leftType  + " LeftTuple;\n"
		"typedef " + rightType + " RightTuple;\n"
		"typedef LeftTuple::BasicType LeftType;\n"
		"typedef RightTuple::BasicType RightType;\n"
		"extern \"C\" __global__ void join_find_bounds" + idstream.str() + "("
			"unsigned int* lowerBounds,"
			"unsigned int* upperBounds,"
			"unsigned int* outBounds,"
			"const LeftType* leftBegin, "
			"const long long unsigned int* leftSize, "
			"const RightType* rightBegin, "
			"const long long unsigned int* rightSize)\n"
		"{\n"
			"ra::cuda::findBounds" + tmp + "<LeftTuple, RightTuple, " 
			+ keyFields + ">(leftBegin,"
			"leftBegin + ((*leftSize)/sizeof(LeftType)), rightBegin, "
			"rightBegin + ((*rightSize)/sizeof(RightType)), lowerBounds, "
			"upperBounds, outBounds);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_joinTempSizeSource() const
{
	report("Getting CUDA source for the join temp size ptx kernel.");

	std::string destType  = elementType(destination);
	std::string leftType  = elementType(sourceA);
	std::string rightType = elementType(sourceB);
	std::string ctaCount  = _ctas();

	std::stringstream idstream;
	idstream << _id;

	return "#define PARTITIONS " + ctaCount + "\n"
		"#include <redfox/ra/interface/Join.h>\n"
		"typedef " + destType  + "::BasicType DestType;\n"
		"typedef " + leftType  + "::BasicType LeftType;\n"
		"typedef " + rightType + "::BasicType RightType;\n"
		"extern \"C\" __global__ void join_get_temp_size" + idstream.str() + "("
			"long long unsigned int* tempSize,"
			"const long long unsigned int* leftSize,"
			"const long long unsigned int* rightSize)\n"
		"{\n"
		"    ra::cuda::getTempSize<DestType, LeftType, RightType>(tempSize, "
			"leftSize, rightSize);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_moderngpuJoinTempSizeSource() const
{
	report("Getting CUDA source for the join temp size ptx kernel.");

	std::string leftType  = elementType(destination);
	std::string ctaCount  = _ctas();

	std::stringstream idstream;
	idstream << _id;

	return "#define PARTITIONS " + ctaCount + "\n"
		"#include <redfox/ra/interface/Tuple.h>\n"
		"typedef " + leftType  + "::BasicType LeftType;\n"
		"extern \"C\" __global__ void join_get_temp_size" + idstream.str() + "("
			"unsigned long long int* lowerBoundSize,"
			"const unsigned long long int* leftSize)\n"
		"{\n"
		"    *lowerBoundSize = (*leftSize)/sizeof(LeftType) * 4;\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_moderngpuJoinGatherSource() const
{
	report("Getting CUDA source for the join temp size ptx kernel.");

	std::string Type  = elementType(destination);
	std::string LeftKeyType  = elementType(sourceA);
	std::string RightKeyType = elementType(sourceB);
	std::string LeftValueType  = elementType(sourceC);
	std::string RightValueType = elementType(sourceD);

	std::string keyFields = _fields();

	std::stringstream idstream;
	idstream << _id;

	if(0 == sourceC.size() && 0 == sourceD.size())
	return  "#include <redfox/ra/interface/Tuple.h>\n"
		"#include <redfox/ra/interface/ModernGPUJoin.h>\n"
		"typedef " + Type  + " Tuple;\n"
		"typedef Tuple::BasicType Type;\n"
		"typedef " + LeftKeyType  + " LeftKeyTuple;\n"
		"typedef LeftKeyTuple::BasicType LeftKeyType;\n"
		"typedef " + RightKeyType + " RightKeyTuple;\n"
		"typedef RightKeyTuple::BasicType RightKeyType;\n"
		"extern \"C\" __global__ void mgpu_join_gather" + idstream.str() + "("
			"Type* result, LeftKeyType *left_key, int* left_index, RightKeyType *right_key, int* right_index," 
			"unsigned long long int* size)\n"
		"{\n"
		"    redfox::gather_key_key<Tuple, LeftKeyTuple, 256, " + keyFields + ">(result, left_key, left_index, (*size/4));\n"
		"}\n";
	else if(0 == sourceC.size() && 0 < sourceD.size())
	return  "#include <redfox/ra/interface/Tuple.h>\n"
		"#include <redfox/ra/interface/ModernGPUJoin.h>\n"
		"typedef " + Type  + " Tuple;\n"
		"typedef Tuple::BasicType Type;\n"
		"typedef " + LeftKeyType  + " LeftKeyTuple;\n"
		"typedef LeftKeyTuple::BasicType LeftKeyType;\n"
		"typedef " + RightKeyType + " RightKeyTuple;\n"
		"typedef RightKeyTuple::BasicType RightKeyType;\n"
		"typedef " + RightValueType + " RightValueTuple;\n"
		"typedef RightValueTuple::BasicType RightValueType;\n"
		"extern \"C\" __global__ void mgpu_join_gather" + idstream.str() + "("
			"Type* result, LeftKeyType *left_key, int* left_index, RightKeyType *right_key, int* right_index," 
			"unsigned long long int* size, RightValueType *right_value)\n"
		"{\n"
		"    redfox::gather_key_value<Tuple, LeftKeyTuple, RightValueTuple, 256, " + keyFields + ">(result, left_key, left_index, right_value, right_index, (*size/4));\n"
		"}\n";
	else if(0 < sourceC.size() && 0 == sourceD.size())
	return  "#include <redfox/ra/interface/Tuple.h>\n"
		"#include <redfox/ra/interface/ModernGPUJoin.h>\n"
		"typedef " + Type  + " Tuple;\n"
		"typedef Tuple::BasicType Type;\n"
		"typedef " + LeftKeyType  + " LeftKeyTuple;\n"
		"typedef LeftKeyTuple::BasicType LeftKeyType;\n"
		"typedef " + LeftValueType  + " LeftValueTuple;\n"
		"typedef LeftValueTuple::BasicType LeftValueType;\n"
		"typedef " + RightKeyType + " RightKeyTuple;\n"
		"typedef RightKeyTuple::BasicType RightKeyType;\n"
		"extern \"C\" __global__ void mgpu_join_gather" + idstream.str() + "("
			"Type* result, LeftKeyType *left_key, int* left_index, RightKeyType *right_key, int* right_index," 
			"unsigned long long int* size, LeftValueType *left_value)\n"
		"{\n"
		"    redfox::gather_value_key<Tuple, LeftKeyTuple, LeftValueTuple, 256, " + keyFields + ">(result, left_key, left_value, left_index, (*size/4));\n"
		"}\n";
	else if(0 < sourceC.size() && 0 < sourceD.size())
	return  "#include <redfox/ra/interface/Tuple.h>\n"
		"#include <redfox/ra/interface/ModernGPUJoin.h>\n"
		"typedef " + Type  + " Tuple;\n"
		"typedef Tuple::BasicType Type;\n"
		"typedef " + LeftKeyType  + " LeftKeyTuple;\n"
		"typedef LeftKeyTuple::BasicType LeftKeyType;\n"
		"typedef " + LeftValueType  + " LeftValueTuple;\n"
		"typedef LeftValueTuple::BasicType LeftValueType;\n"
		"typedef " + RightKeyType + " RightKeyTuple;\n"
		"typedef RightKeyTuple::BasicType RightKeyType;\n"
		"typedef " + RightValueType + " RightValueTuple;\n"
		"typedef RightValueTuple::BasicType RightValueType;\n"
		"extern \"C\" __global__ void mgpu_join_gather" + idstream.str() + "("
			"Type* result, LeftKeyType *left_key, int* left_index, RightKeyType *right_key, int* right_index," 
			"unsigned long long int* size, LeftValueType *left_value, RightValueType *right_value)\n"
		"{\n"
		"    redfox::gather_value_value<Tuple, LeftKeyTuple, LeftValueTuple, RightValueTuple, 256, " + keyFields + ">(result, left_key, left_value, left_index, right_value, right_index, (*size/4));\n"
		"}\n";

	return "";
}

std::string RelationalAlgebraKernel::_moderngpuJoinResultSizeSource() const
{
	report("Getting CUDA source for the join temp size ptx kernel.");

	std::string type  = elementType(destination);
	std::string ctaCount  = _ctas();

	std::stringstream idstream;
	idstream << _id;

	return "#define PARTITIONS " + ctaCount + "\n"
		"#include <redfox/ra/interface/Tuple.h>\n"
		"typedef " + type  + "::BasicType Type;\n"
		"extern \"C\" __global__ void join_get_result_size" + idstream.str() + "("
			"unsigned long long int* index_size,"
			"unsigned long long int* result_size)\n"
		"{\n"
		"    *index_size = (*result_size) * 4;\n"
		"    *result_size = (*result_size) * sizeof(Type);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_moderngpuJoinFindBoundsSource() const
{
	report("Getting CUDA source for sort ptx kernel.");

	std::string type = elementType(sourceA);

	std::stringstream idstream;
	idstream << _id;
std::string tmp = "";
//if(_id == 52) tmp ="2";

if(sourceA.size() == 1 && sourceA[0].type == Pointer)
	return "#include <redfox/ra/interface/ModernGPUJoin.h>\n"
		"#include <stdio.h>\n"
		"typedef " + type    + " Tuple;\n"
		"typedef Tuple::BasicType TupleType;\n"
		"extern \"C\" void mgpu_join_find_bounds" + idstream.str() +"(int* lower_bound, "
		"int* left_count, unsigned long long int* result_size, "
		"char** left_key, unsigned long long int left_size,"
		"char** right_key, unsigned long long int right_size)\n"
		"{\n"
		"    redfox::find_bounds_string" + tmp + "(lower_bound, left_count,"
			"result_size, left_key, left_size/8, right_key, right_size/8);\n"
		"}\n";
if(bytes(sourceA) > 8 && bytes(sourceA) <= 16)
	return "#include <redfox/ra/interface/ModernGPUJoin.h>\n"
		"#include <stdio.h>\n"
		"typedef " + type    + " Tuple;\n"
		"typedef Tuple::BasicType TupleType;\n"
		"extern \"C\" void mgpu_join_find_bounds" + idstream.str() +"(int* lower_bound, "
		"int* left_count, unsigned long long int* result_size, "
		"char** left_key, unsigned long long int left_size,"
		"char** right_key, unsigned long long int right_size)\n"
		"{\n"
		"    redfox::find_bounds_128" + tmp + "(lower_bound, left_count,"
			"result_size, left_key, left_size/16, right_key, right_size/16);\n"
		"}\n";
if(bytes(sourceA) > 4 && bytes(sourceA) <= 8)
	return "#include <redfox/ra/interface/ModernGPUJoin.h>\n"
		"#include <stdio.h>\n"
		"typedef " + type    + " Tuple;\n"
		"typedef Tuple::BasicType TupleType;\n"
		"extern \"C\" void mgpu_join_find_bounds" + idstream.str() +"(int* lower_bound, "
		"int* left_count, unsigned long long int* result_size, "
		"char** left_key, unsigned long long int left_size,"
		"char** right_key, unsigned long long int right_size)\n"
		"{\n"
		"    redfox::find_bounds_64" + tmp + "(lower_bound, left_count,"
			"result_size, left_key, left_size/8, right_key, right_size/8);\n"
		"}\n";
else if(bytes(sourceA) == 4 || bytes(sourceA) == 3)
	return "#include <redfox/ra/interface/ModernGPUJoin.h>\n"
		"#include <stdio.h>\n"
		"typedef " + type    + " Tuple;\n"
		"typedef Tuple::BasicType TupleType;\n"
		"extern \"C\" void mgpu_join_find_bounds" + idstream.str() +"(int* lower_bound, "
		"int* left_count, unsigned long long int* result_size, "
		"TupleType *left_key, unsigned long long int left_size,"
		"TupleType *right_key, unsigned long long int right_size)\n"
		"{\n"
		"    redfox::find_bounds_32" + tmp + "(lower_bound, left_count,"
			"result_size, left_key, left_size/4, right_key, right_size/4);\n"
		"}\n";
else if(bytes(sourceA) == 2)
	return "#include <redfox/ra/interface/ModernGPUJoin.h>\n"
		"#include <stdio.h>\n"
		"typedef " + type    + " Tuple;\n"
		"typedef Tuple::BasicType TupleType;\n"
		"extern \"C\" void mgpu_join_find_bounds" + idstream.str() +"(int* lower_bound, "
		"int* left_count, unsigned long long int* result_size, "
		"TupleType *left_key, unsigned long long int left_size,"
		"TupleType *right_key, unsigned long long int right_size)\n"
		"{\n"
		"    redfox::find_bounds_16" + tmp + "(lower_bound, left_count,"
			"result_size, left_key, left_size/2, right_key, right_size/2);\n"
		"}\n";
else if(bytes(sourceA) == 1)
	return "#include <redfox/ra/interface/ModernGPUJoin.h>\n"
		"#include <stdio.h>\n"
		"typedef " + type    + " Tuple;\n"
		"typedef Tuple::BasicType TupleType;\n"
		"extern \"C\" void mgpu_join_find_bounds" + idstream.str() +"(int* lower_bound, "
		"int* left_count, unsigned long long int* result_size, "
		"TupleType *left_key, unsigned long long int left_size,"
		"TupleType *right_key, unsigned long long int right_size)\n"
		"{\n"
		"    redfox::find_bounds_8" + tmp + "(lower_bound, left_count,"
			"result_size, left_key, left_size, right_key, right_size);\n"
		"}\n";

	return "";
}

std::string RelationalAlgebraKernel::_joinMainSource() const
{
	report("Getting CUDA source for the main join ptx kernel.");

	std::string destType  = elementType(destination);
	std::string leftType  = elementType(sourceA);
	std::string rightType = elementType(sourceB);
	std::string keyFields = _fields();
	std::string threads   = _threads();
	std::string ctaCount  = _ctas();

	std::stringstream idstream;
	idstream << _id;

	std::string tmp = "";
//	if(_id == 45)
//		tmp = "2";

	if(sourceA[_keyFields - 1].type == Pointer)
	{
		return "#define PARTITIONS " + ctaCount + "\n"
		"#include <redfox/ra/interface/Join.h>\n"
		"typedef " + leftType  + " LeftTuple;\n"
		"typedef " + rightType + " RightTuple;\n"
		"typedef " + destType + "  DestTuple;\n"
		"typedef LeftTuple::BasicType  LeftType;\n"
		"typedef RightTuple::BasicType RightType;\n"
		"typedef DestTuple::BasicType  DestType;\n"
		"extern \"C\" __global__ void set_join" + idstream.str() + "("
			"DestType* temp, "
			"unsigned int* histogram, "
			"unsigned int* lowerBounds, "
			"unsigned int* upperBounds, "
			"unsigned int* outBounds, "
			"const LeftType* leftBegin, "
			"const long long unsigned int* leftSize, "
			"const RightType* rightBegin, "
			"const long long unsigned int* rightSize)\n"
		"{\n"
		"    ra::cuda::join_string" + tmp + "<LeftTuple, RightTuple, DestTuple, " 
			+ keyFields + ", " + threads 
			+ ">(leftBegin, leftBegin + (*leftSize / sizeof(LeftType)),"
			"rightBegin, rightBegin + (*rightSize / sizeof(RightType)),"
			"temp, histogram, lowerBounds, upperBounds, outBounds);\n"
		"}\n";
	}

	return "#define PARTITIONS " + ctaCount + "\n"
		"#include <redfox/ra/interface/Join.h>\n"
		"typedef " + leftType  + " LeftTuple;\n"
		"typedef " + rightType + " RightTuple;\n"
		"typedef " + destType + "  DestTuple;\n"
		"typedef LeftTuple::BasicType  LeftType;\n"
		"typedef RightTuple::BasicType RightType;\n"
		"typedef DestTuple::BasicType  DestType;\n"
		"extern \"C\" __global__ void set_join" + idstream.str() + "("
			"DestType* temp, "
			"unsigned int* histogram, "
			"unsigned int* lowerBounds, "
			"unsigned int* upperBounds, "
			"unsigned int* outBounds, "
			"const LeftType* leftBegin, "
			"const long long unsigned int* leftSize, "
			"const RightType* rightBegin, "
			"const long long unsigned int* rightSize)\n"
		"{\n"
		"    ra::cuda::join" + tmp + "<LeftTuple, RightTuple, DestTuple, " 
			+ keyFields + ", " + threads 
			+ ">(leftBegin, leftBegin + (*leftSize / sizeof(LeftType)),"
			"rightBegin, rightBegin + (*rightSize / sizeof(RightType)),"
			"temp, histogram, lowerBounds, upperBounds, outBounds);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_moderngpuJoinMainSource() const
{
	report("Getting CUDA source for sort ptx kernel.");

	std::stringstream idstream;
	idstream << _id;

std::string tmp = "";
//if(_id == 52) tmp ="2";

	return "#include <redfox/ra/interface/ModernGPUJoin.h>\n"
		"#include <stdio.h>\n"
		"extern \"C\" void mgpu_join_main" + idstream.str() +"(int *left_indices,"
		"int* right_indices, unsigned long long int result_size, "
		"int *lowerBound, int* leftCount,"
		"unsigned long long int input_size)\n"
		"{\n"
		"    redfox::join" + tmp + "(left_indices, right_indices, result_size/4,"
			"lowerBound, leftCount, input_size/4);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_scanSource() const
{
	report("Getting CUDA source for the scan ptx kernel.");

	std::stringstream idstream;
	idstream << _id;

	return  "#include <stdio.h>\n"
		"#include <redfox/ra/interface/Join.h>\n"
		"extern \"C\" __global__ void scan" + idstream.str() + "(unsigned int* histogram)\n"
		"{\n"
		"    unsigned int max = 0;\n"
		"    histogram[threadIdx.x] = ra::cuda::exclusiveScan<128, unsigned int>("
				"histogram[threadIdx.x], max);\n"
//		"    if(threadIdx.x == 0) printf(\"scan max %u\\n\",  max);\n"
//		"    if(threadIdx.x == 0) printf(\"scan 125 %u\\n\",  histogram[125]);\n"
//		"    if(threadIdx.x == 0) printf(\"scan 126 %u\\n\",  histogram[126]);\n"
//		"    if(threadIdx.x == 0) printf(\"scan 127 %u\\n\",  histogram[127]);\n"
//		"    if(threadIdx.x == 0) printf(\"scan 128 %u\\n\",  histogram[128]);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_joinGetSizeSource() const
{
	report("Getting CUDA source for the join get size ptx kernel.");

	std::string destType = elementType(destination);
	std::string ctaCount = _ctas();

	std::stringstream idstream;
	idstream << _id;

	return "#define PARTITIONS " + ctaCount + "\n"
		"#include <redfox/ra/interface/Join.h>\n"
		"typedef " + destType + "::BasicType DestType;\n"
		"extern \"C\" __global__ void join_get_size" + idstream.str() + "("
			"long long unsigned int* size, const unsigned int* histogram)\n"
		"{\n"
		"    ra::cuda::getResultSize<DestType, "
			+ ctaCount + ">(size, histogram);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_joinGatherSource() const
{
	report("Getting CUDA source for the join gather ptx kernel.");

	std::string destType   = elementType(destination);
	std::string ctaCount   = _ctas();

	std::stringstream idstream;
	idstream << _id;
std::string check = "";
//if(_id == 48) check = "2";
	return "#define PARTITIONS " + ctaCount + "\n"
		"#include <redfox/ra/interface/Join.h>\n"
		"typedef " + destType + " DestTuple;\n"
		"typedef DestTuple::BasicType DestType;\n"
		"extern \"C\" __global__ void join_gather" + idstream.str() + "("
			"DestType* dest, long long unsigned int *destSize,"
			"const DestType* temp, long long unsigned int *tempSize,"
			"const unsigned int* outBounds,"
			"const unsigned int* histogram)\n"
		"{\n"
		"    ra::cuda::gather" + check + "<DestTuple>(dest, "
				"dest + (*destSize/sizeof(DestType)), "
				"temp, temp + (*tempSize/sizeof(DestType)), "
				"outBounds, histogram);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_projectSource() const
{
	report("Getting CUDA source for project ptx kernel.");

	std::string resulttype = elementType(destination);
	std::string sourcetype = elementType(sourceA);
	
	std::stringstream stream;
	
	stream << "ra::cuda::PermuteMap<";
	
	for(unsigned int index = 0; index != 16; ++index)
	{
		if(index != 0) stream << ", ";
		if(index < _indicies.size())
		{
			stream << _indicies[index];
		}
		else
		{
			stream << "100";
		}
	}
	
	stream << ">";
	
	std::string dimensionlist = stream.str();
	
	std::stringstream idstream;
	idstream << _id;

std::string check = "";
//if(_id == 90)
//	check = "if(threadIdx.x == 0 && blockIdx.x == 0) printf(\"before projection %llx %llx %llx %llx %llx %llx %llx %llx %llx %llx %llx %llx %llx %llx %llx %llx \\n\", begin[0].a[0], begin[0].a[1], begin[0].a[2], begin[0].a[3], begin[0].a[4], begin[0].a[5], begin[0].a[6], begin[0].a[7], begin[0].a[8], begin[0].a[9], begin[0].a[10], begin[0].a[11], begin[0].a[12], begin[0].a[13], begin[0].a[14], begin[0].a[15]);\n";
//check = "2";
	return "#include <redfox/ra/interface/Project.h>\n"
		"typedef " + resulttype    + " ResultTuple;\n"
		"typedef " + sourcetype    + " SourceTuple;\n"
		"typedef " + dimensionlist + " PermuteMap;\n"
		"typedef ResultTuple::BasicType ResultType;\n"
		"typedef SourceTuple::BasicType SourceType;\n"
		"extern \"C\" __global__ void set_project" + idstream.str() + "(ResultType* result, "
		"const SourceType* begin, const long long unsigned int* size)\n"
		"{\n"
		"    ra::cuda::permute" + check + "<ResultTuple, SourceTuple, "
			"PermuteMap>(result, begin, "
			"begin + (*size / sizeof(ResultType)));\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_setStringSource() const
{
	report("Getting CUDA source for set string ptx kernel.");

	std::string sourcetype = elementType(sourceA);

	std::stringstream idstream;
	idstream << _id;

	std::stringstream index;
	index << _index;

	return "#include <stdio.h>\n"
		"#include <redfox/ra/interface/Tuple.h>\n"
		"typedef " + sourcetype    + " SourceTuple;\n"
		"typedef SourceTuple::BasicType SourceType;\n"
		"extern \"C\" __global__ void set_setstring" + idstream.str() + "(char* string_begin, "
		"SourceType* begin, const long long unsigned int* size)\n"
		"{\n"
		"	unsigned int step     = gridDim.x * blockDim.x;\n"
		"	unsigned int start    = threadIdx.x + blockIdx.x * blockDim.x;\n"
		"	unsigned int elements = (*size)/sizeof(SourceType);\n"
		"	for(unsigned int i = start; i < elements; i += step)\n"
		"	{\n"
		"	    if(i < elements)\n"
		"           {\n"
		"	    	unsigned long long int tmp1 = ra::tuple::extract<unsigned long long int, " + index.str() + ", SourceTuple>(begin[i]);\n"
//		"		if(i == elements - 1) printf(\"tmp1 %llu\\n\", tmp1);\n"
//		"		if(i == 0) printf(\"begin %llx %llx\\n\", begin[i].a[0], begin[i].a[1]);\n"
		"		char* tmp2 = string_begin + tmp1;\n"  
//		"		if(i == elements - 1) printf(\"tmp2 %s\\n\", tmp2);\n"
//		"		if(i == elements - 1) printf(\"tmp2 %llx\\n\", tmp2);\n"
 		"		begin[i] = ra::tuple::insert<unsigned long long int, " + index.str() + ", SourceTuple>(begin[i], (unsigned long long int)tmp2);\n"
//		"		if(i == 0) printf(\"%llx %llx\\n\", begin[i].a[0], begin[i].a[1]);\n"
		"           }\n"
		"	}\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_projectGetResultSizeSource() const
{
	std::string resulttype = elementType(destination);
	std::string sourcetype = elementType(sourceA);

	std::stringstream idstream;
	idstream << _id;

	return "#include <redfox/ra/interface/Project.h>\n"
		"typedef " + resulttype    + " ResultTuple;\n"
		"typedef " + sourcetype    + " SourceTuple;\n"
		"typedef ResultTuple::BasicType ResultType;\n"
		"typedef SourceTuple::BasicType SourceType;\n"
		"extern \"C\" __global__ void project_get_result_size" + idstream.str() + "("
			"long long unsigned int* size)\n"
		"{\n"
		"    ra::cuda::getResultSize<ResultType, SourceType>(size);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_generateGetResultSizeSource() const
{
	std::string sourcetype = elementType(sourceA);

	std::stringstream idstream;
	idstream << _id;

	return "#include <redfox/ra/interface/Split_Merge.h>\n"
		"typedef " + sourcetype    + " SourceTuple;\n"
		"extern \"C\" __global__ void generate_get_result_size" + idstream.str() + "("
			"long long unsigned int* generate_size, long long unsigned int *size)\n"
		"{\n"
		"    ra::cuda::getGenerateResultSize<SourceTuple>(generate_size, size);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_splitSource() const
{
	report("Getting CUDA source for project ptx kernel.");

	std::string keytype = elementType(destination);
	std::string valuetype = elementType(destination_1);
	std::string sourcetype = elementType(sourceA);

	std::stringstream stream;

	stream << _domains;

	std::string domain_string = stream.str();
	
	std::stringstream idstream;
	idstream << _id;

std::string tmp = "";
//if(_id == 102) tmp = "2";

	return "#include <redfox/ra/interface/Split_Merge.h>\n"
		"typedef " + keytype    + " KeyTuple;\n"
		"typedef " + valuetype    + " ValueTuple;\n"
		"typedef " + sourcetype    + " SourceTuple;\n"
		"typedef KeyTuple::BasicType KeyType;\n"
		"typedef ValueTuple::BasicType ValueType;\n"
		"typedef SourceTuple::BasicType SourceType;\n"
		"extern \"C\" __global__ void set_split" + idstream.str() + "(KeyType* key, "
		"ValueType* value, const SourceType* begin, "
		"const long long unsigned int* size)\n"
		"{\n"
		"    ra::cuda::split" + tmp + "<KeyTuple, ValueTuple, SourceTuple, "
		+ domain_string +">(key, value, begin, "
			"begin + (*size / sizeof(SourceType)));\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_splitkeySource() const
{
	report("Getting CUDA source for project ptx kernel.");

	std::string keytype = elementType(destination);
	std::string sourcetype = elementType(sourceA);

	std::stringstream stream;

	stream << _domains;

	std::string domain_string = stream.str();
	
	std::stringstream idstream;
	idstream << _id;
std::string tmp = "";
//if(_id == 54) tmp = "2";

	return "#include <redfox/ra/interface/Split_Merge.h>\n"
		"typedef " + keytype    + " KeyTuple;\n"
		"typedef " + sourcetype    + " SourceTuple;\n"
		"typedef KeyTuple::BasicType KeyType;\n"
		"typedef SourceTuple::BasicType SourceType;\n"
		"extern \"C\" __global__ void set_splitkey" + idstream.str() + "(KeyType* key, "
		"const SourceType* begin, "
		"const long long unsigned int* size)\n"
		"{\n"
//		"    if(threadIdx.x == 0 && blockIdx.x == 0) printf(\"%llu %llu\\n\", *size, sizeof(SourceType));\n"
		"    ra::cuda::split_key" + tmp + "<KeyTuple, SourceTuple, "
		+ domain_string +">(key, begin, "
			"begin + (*size / sizeof(SourceType)));\n"
		"}\n";
}
std::string RelationalAlgebraKernel::_generateSource() const
{
	report("Getting CUDA source for project ptx kernel.");
	
	std::stringstream idstream;
	idstream << _id;

	return "#include <redfox/ra/interface/Split_Merge.h>\n"
		"extern \"C\" __global__ void set_generate" + idstream.str() + "(unsigned int* begin, "
		"const long long unsigned int* size)\n"
		"{\n"
		"    ra::cuda::generate(begin, "
			"begin + (*size / sizeof(unsigned int)));\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_splitGetResultSizeSource() const
{
	std::string keytype = elementType(destination);
	std::string valuetype = elementType(destination_1);
	std::string sourcetype = elementType(sourceA);

	std::stringstream idstream;
	idstream << _id;

	return "#include <redfox/ra/interface/Split_Merge.h>\n"
		"typedef " + keytype    + " KeyTuple;\n"
		"typedef " + valuetype    + " ValueTuple;\n"
		"typedef " + sourcetype    + " SourceTuple;\n"
		"extern \"C\" __global__ void split_get_result_size" + idstream.str() +"("
			"long long unsigned int* size, "
			"long long unsigned int* key_size,"
			"long long unsigned int* value_size)\n"
		"{\n"
		"    ra::cuda::getSplitResultSize<KeyTuple, ValueTuple, SourceTuple>(size, key_size, value_size);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_splitKeyGetResultSizeSource() const
{
	std::string keytype = elementType(destination);
	std::string sourcetype = elementType(sourceA);

	std::stringstream idstream;
	idstream << _id;

	return "#include <redfox/ra/interface/Split_Merge.h>\n"
		"typedef " + keytype    + " KeyTuple;\n"
		"typedef " + keytype    + "::BasicType KeyType;\n"
		"typedef " + sourcetype    + " SourceTuple;\n"
		"typedef " + sourcetype    + "::BasicType SourceType;\n"
		"extern \"C\" __global__ void split_key_get_result_size" + idstream.str() +"("
			"long long unsigned int* key_size, "
			"long long unsigned int* size)\n"
		"{\n"
		"    *key_size = *size * sizeof(KeyType) / sizeof(SourceType);\n"
//		"if(threadIdx.x == 0 && blockIdx.x == 0) printf(\"%llu %llu %llu %llu\\n\", *key_size, *size, sizeof(KeyType), sizeof(SourceType));\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_mergeSource() const
{
	report("Getting CUDA source for merge ptx kernel.");

	std::string resulttype = elementType(destination);
	std::string keytype = elementType(sourceA);
	std::string valuetype = elementType(sourceB);

	std::stringstream stream;
	stream << _domains;
	
	std::stringstream idstream;
	idstream << _id;
std::string check="";
//if(_id == 105) check = "2";

	return "#include <redfox/ra/interface/Split_Merge.h>\n"
		"#include <stdio.h>\n"
		"typedef " + keytype    + " KeyTuple;\n"
		"typedef " + valuetype    + " ValueTuple;\n"
		"typedef " + resulttype    + " ResultTuple;\n"
		"typedef KeyTuple::BasicType KeyType;\n"
		"typedef ValueTuple::BasicType ValueType;\n"
		"typedef ResultTuple::BasicType ResultType;\n"
		"extern \"C\" __global__ void set_merge" + idstream.str() + "(ResultType* result, "
		"const KeyType* key_begin, "
		"const long long unsigned int* key_size, "
		"const ValueType* value_begin, "
		"const long long unsigned int* value_size)\n"
		"{\n"
//		"if(threadIdx.x == 0 && blockIdx.x == 0) printf(\"%u %u\\n\", gridDim.x, blockDim.x);\n"
		"    ra::cuda::merge" + check + "<KeyTuple, ValueTuple, ResultTuple, "
		+ stream.str() + ">(result, key_begin, "
			"key_begin + (*key_size / sizeof(KeyType)), "
			"value_begin, value_begin + (*value_size / sizeof(ValueType)));\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_mergeGetResultSizeSource() const
{
	std::string resulttype = elementType(destination);
	std::string valuetype = elementType(sourceA);

	std::stringstream idstream;
	idstream << _id;

	return "#include <redfox/ra/interface/Split_Merge.h>\n"
		"typedef " + resulttype    + " ResultTuple;\n"
		"typedef " + valuetype    + " ValueTuple;\n"
		"typedef ResultTuple::BasicType ResultType;\n"
		"typedef ValueTuple::BasicType ValueType;\n"
		"extern \"C\" __global__ void merge_get_result_size" + idstream.str() + "("
			"long long unsigned int* size,"
			"long long unsigned int* value_size)\n"
		"{\n"
		"    ra::cuda::getMergeResultSize<ResultType, ValueType>(size, value_size);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::arithOpToString(ArithmeticOp op)
{
	switch(op)
	{
	case Add:
	{
		return " + ";
		break;
	}
	case Multiply:
	{
		return " * ";
		break;
	}
	case Subtract:
	{
		return " - ";
		break;
	}
	case Divide:
	{
		return " / ";
		break;
	}
	case Mod:
	{
		return " % ";
		break;
	}
	default:
	{
		break;
	}
	}

	return "";
}

void RelationalAlgebraKernel::translateArithNode(std::stringstream& stream, 
	ArithExpNode* node, bool isLeft, std::string tupleName, std::string sourceName)
{
	if(node->kind == OperatorNode)
	{
		report("###### operator");
		stream << "(";

		translateArithNode(stream, node->left, true, tupleName, sourceName);
		
		stream << arithOpToString(node->op);

		translateArithNode(stream, node->right, false, tupleName, sourceName);

		stream << ")";
	}
	else if(node->kind == ConstantNode)
	{
		report("###### ConstantNode " << node->type << " " << node->f);
		switch(node->type)
		{
                case InvalidDataType:
		{
			assert(false);
			break;
		}
                case I8:
                case I16:
                case I32:
                case I64:
                case I128:
                case I256:
                case I512:
                case I1024:
		{
			stream << node->i;
			break;
		}
                case F32:
                case F64:
		{
			stream << node->f;
			break;
		}
                case Pointer:
		{
			assert(false);
			break;
		}
		}
	}
	else if(node->kind == IndexNode)
	{
		report("###### IndexNode");
		if(node->type == RelationalAlgebraKernel::F64)
		{	
			stream << "ra::tuple::extract_double<";
			int index = node->i;
			stream << index << ", " << tupleName << ">(" << sourceName << ")";
		} 
		else
		{
			stream << "ra::tuple::extract<" << mapValueType(node->type) << ", ";
			int index = node->i;
			stream << index << ", " << tupleName << ">(" << sourceName << ")";
		}
	}
}

std::string RelationalAlgebraKernel::arithToString(ArithExpNode _arithExp, 
	std::string tupleName, std::string sourceName)
{
	std::stringstream stream;

	if(_arithExp.kind == ConstantNode)
	{
		switch(_arithExp.type)
		{
                case InvalidDataType:
		{
			assert(false);
			break;
		}
                case I8:
                case I16:
                case I32:
                case I64:
                case I128:
                case I256:
                case I512:
                case I1024:
		{
			stream << _arithExp.i;
			break;
		}
                case F32:
                case F64:
		{
			stream << _arithExp.f;
			break;
		}
                case Pointer:
		{
			assert(false);
			break;
		}
		}
		return stream.str();
	}

	stream << "(";
	translateArithNode(stream, _arithExp.left, true, tupleName, sourceName);
	stream << arithOpToString(_arithExp.op); 
	translateArithNode(stream, _arithExp.right, false, tupleName, sourceName);
	stream << ")";

	return stream.str();
}

std::string RelationalAlgebraKernel::_substringSource() const
{
	report("Getting CUDA source for substring ptx kernel.");

	std::string resulttype = elementType(destination);
	std::string sourcetype = elementType(sourceA);
	
	std::stringstream idstream;
	idstream << _id;

	std::stringstream extract_stream;
	extract_stream << "ra::tuple::extract<unsigned long long int, " << _index << ", SourceTuple>(begin[i])";
	std::string extract;
	extract = extract_stream.str();

//	std::string search;
//	search = "unsigned int j;\n"
//		 "for (j = 0; j < (*StringTableSize); j += 128)\n"
//		 "{\n"
//		 "	char *string2 = StringTable + j;\n"
//		 " 	if(comp((char *)substring, string2))\n"
//		 "		break;\n"
//		 "}\n";

	std::stringstream pos;
	pos << (destination.size() - 1);

	std::string insert;
	insert = "ra::tuple::insert<unsigned long long int, " + pos.str() + ", ResultTuple>";

	return "#include <stdio.h>\n"
		"#include <redfox/ra/interface/Tuple.h>\n"
		"#include <redfox/ra/interface/Comparisons.h>\n"
		"typedef " + resulttype    + " ResultTuple;\n"
		"typedef " + sourcetype    + " SourceTuple;\n"
		"typedef ResultTuple::BasicType ResultType;\n"
		"typedef SourceTuple::BasicType SourceType;\n"
		"typedef ra::comparisons::eqstring<char *> Compare;\n"
		"extern \"C\" __global__ void set_substring" + idstream.str() + "(ResultType* result, "
		"const SourceType* begin, const long long unsigned int* size, char *StringTable, const long long unsigned int *StringTableSize)\n"
		"{\n"
		"	unsigned int step     = gridDim.x * blockDim.x;\n"
		"	unsigned int start    = threadIdx.x + blockIdx.x * blockDim.x;\n"
		"	unsigned int elements = (*size)/sizeof(ResultType);\n"
		"	for(unsigned int i = start; i < elements; i += step)\n"
		"	{\n"
		"	    if(i < elements)\n"
		"           {\n"
		"		    	ResultType tmp1 = (ResultType)begin[i] << (unsigned int)64;\n"
		"		        char *string1 = (char *)" + extract + ";\n"  
		"		        char *substring = StringTable + i * 128;\n"  
		"		        substring[0] = string1[0];\n"  
		"		        substring[1] = string1[1];\n"  
		"		        substring[2] = \'\\0\';\n"
		"		    	result[i] = " + insert + "(tmp1, (unsigned long long int)(substring));\n"
		"           }\n"
		"	}\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_arithSource() const
{
	report("Getting CUDA source for arithmetic ptx kernel.");

	std::string resulttype = elementType(destination);
	std::stringstream stream1;

	stream1 << destination.rbegin()->limit;

	std::string shift = stream1.str();

	std::stringstream stream2;

	stream2 << (destination.size() - 1);

	std::string pos = stream2.str();

	std::string sourcetype = elementType(sourceA);
	
	std::stringstream idstream;
	idstream << _id;

std::string check1 = "";
std::string check2 = "";

//if(_id >= 121)
//	check1 = "printf(\"%d, %lf\\n\", i, tmp2);\n";

	std::string insert;

	if(destination.rbegin()->type == RelationalAlgebraKernel::F64)
		insert = "ra::tuple::insert_double<";
	else
		insert = "ra::tuple::insert<" + mapValueType(destination.rbegin()->type) + ", ";

	if(_arithExp.op == AddMonth || _arithExp.op == AddYear) 
	{
		std::stringstream left, right;
		left << _arithExp.left->i;
		right << _arithExp.right->i;

		std::string extract = "ra::tuple::extract<long long unsigned int, " + left.str() + ", SourceTuple>"; 
		std::string arith;
		std::string wrap = "";

		if(_arithExp.op == AddMonth)
		{
			arith = "month += " + right.str();
			wrap = "ra::cuda::wrapdate(year, month);\n"; 
		}
		else if(_arithExp.op == AddYear)
			arith = "year += " + right.str();

		return "#include <stdio.h>\n"
			"#include <redfox/ra/interface/Tuple.h>\n"
			"#include <redfox/ra/interface/Date.h>\n"
			"typedef " + resulttype    + " ResultTuple;\n"
			"typedef " + sourcetype    + " SourceTuple;\n"
			"typedef ResultTuple::BasicType ResultType;\n"
			"typedef SourceTuple::BasicType SourceType;\n"
			"extern \"C\" __global__ void set_arith" + idstream.str() + "(ResultType* result, "
			"const SourceType* begin, const long long unsigned int* size)\n"
			"{\n"
			"	unsigned int step     = gridDim.x * blockDim.x;\n"
			"	unsigned int start    = threadIdx.x + blockIdx.x * blockDim.x;\n"
			"	unsigned int elements = (*size)/sizeof(ResultType);\n"
			"	for(unsigned int i = start; i < elements; i += step)\n"
			"	{\n"
			"	    if(i < elements)\n"
			"           {\n"
			"		    	ResultType tmp1 = (ResultType)begin[i] << (unsigned int)" + shift + ";\n"
			"		        long long unsigned int tmp2 = " + extract + "(begin[i]);\n"  
			"		        int year, month, date;\n"  
			"		        ra::cuda::int2date(tmp2, year, month, date);\n" 
			"                       " + arith + ";\n" 
			"                       " + wrap + ""
			"                       tmp2 = ra::cuda::date2int(year, month, date);\n" 
			"		    	result[i] = " + insert + pos +  
			", ResultTuple>(tmp1, tmp2);\n" + check1 +
//			"if(threadIdx.x == 0 & blockIdx.x == 0) printf(\"%llu\\n\", tmp2);\n"
			"           }\n"
			"	}\n"+ check2 +
			"}\n";
	}
	else if(_arithExp.op == PartYear) 
	{
		std::stringstream left;
		left << _arithExp.left->i;

		std::string extract = "ra::tuple::extract<long long unsigned int, " + left.str() + ", SourceTuple>"; 

		return "#include <stdio.h>\n"
			"#include <redfox/ra/interface/Tuple.h>\n"
			"#include <redfox/ra/interface/Date.h>\n"
			"typedef " + resulttype    + " ResultTuple;\n"
			"typedef " + sourcetype    + " SourceTuple;\n"
			"typedef ResultTuple::BasicType ResultType;\n"
			"typedef SourceTuple::BasicType SourceType;\n"
			"extern \"C\" __global__ void set_arith" + idstream.str() + "(ResultType* result, "
			"const SourceType* begin, const long long unsigned int* size)\n"
			"{\n"
			"	unsigned int step     = gridDim.x * blockDim.x;\n"
			"	unsigned int start    = threadIdx.x + blockIdx.x * blockDim.x;\n"
			"	unsigned int elements = (*size)/sizeof(ResultType);\n"
			"	for(unsigned int i = start; i < elements; i += step)\n"
			"	{\n"
			"	    if(i < elements)\n"
			"           {\n"
			"		  ResultType tmp1 = (ResultType)begin[i] << (unsigned int)" + shift + ";\n"
//			"		  if(i == 0) printf(\"begin %llx %llx %llx\\n\", begin[i].a[0], begin[i].a[1], begin[i].a[2]);\n"
//			"		  if(i == 0) printf(\"tmp1 %llx %llx %llx %llx\\n\", tmp1.a[0], tmp1.a[1], tmp1.a[2], tmp1.a[3]);\n"
			"		  long long unsigned int tmp2 = " + extract + "(begin[i]);\n"  
//			"		  if(i == 0) printf(\"tmp2 %llu\\n\", tmp2);\n"
			"		  int year, month, date;\n"  
			"		  ra::cuda::int2date(tmp2, year, month, date);\n" 
//			"		  if(i == 0) printf(\"year %llu\\n\", year);\n"
//			"		  if(year == 1994) printf(\"year %u, %d %llu\\n\", i, year, tmp2);\n"
//			"		  if(year == 1997) printf(\"year %u, %d %llu\\n\", i, year, tmp2);\n"
			"		  result[i] = " + insert + pos +  ", ResultTuple>(tmp1, year);\n" 
//			"		  if(i == 0) printf(\"result %llx %llx %llx %llx\\n\", result[i].a[0], result[i].a[1], result[i].a[2],result[i].a[3]);\n"
			"           }\n"
			"	}\n"+ check2 +
			"}\n";
	}
	else 
		return "#include <stdio.h>\n"
			"#include <redfox/ra/interface/Tuple.h>\n"
			"typedef " + resulttype    + " ResultTuple;\n"
			"typedef " + sourcetype    + " SourceTuple;\n"
			"typedef ResultTuple::BasicType ResultType;\n"
			"typedef SourceTuple::BasicType SourceType;\n"
			"extern \"C\" __global__ void set_arith" + idstream.str() + "(ResultType* result, "
			"const SourceType* begin, const long long unsigned int* size)\n"
			"{\n"
			"	unsigned int step     = gridDim.x * blockDim.x;\n"
			"	unsigned int start    = threadIdx.x + blockIdx.x * blockDim.x;\n"
			"	unsigned int elements = (*size)/sizeof(ResultType);\n"
			"	for(unsigned int i = start; i < elements; i += step)\n"
			"	{\n"
			"	    if(i < elements)\n"
			"           {\n"
			"		    	ResultType tmp1 = (ResultType)begin[i] << (unsigned int)" + shift + ";\n"
			"		        " + mapValueType(destination.rbegin()->type) + 
			" tmp2 = " + arithToString(_arithExp, "SourceTuple", "begin[i]") + ";\n"  
			"		    	result[i] = " + insert + pos +  ", ResultTuple>(tmp1, tmp2);\n"
			"           }\n"
			"	}\n"
//			"if(threadIdx.x == 0 && blockIdx.x == 0) printf(\"arith %llx\\n\", result[0].a[0]);\n"
			"}\n";
}

std::string RelationalAlgebraKernel::_convertSource() const
{
	report("Getting CUDA source for convert ptx kernel.");

	std::string resulttype = elementType(destination);
	std::stringstream stream1;

	stream1 << destination.rbegin()->limit;

	std::string shift = stream1.str();

	std::stringstream stream2;

	stream2 << (destination.size() - 1);

	std::string pos = stream2.str();

	std::stringstream stream3;

	stream3 << (_offset);

	std::string offset = stream3.str();

	std::string sourcetype = elementType(sourceA);
	
	std::stringstream idstream;
	idstream << _id;


	std::string insert;

	if(destination.rbegin()->type == RelationalAlgebraKernel::F64)
		insert = "ra::tuple::insert_double<";
	else
		insert = "ra::tuple::insert<" + mapValueType(destination.rbegin()->type) + ", ";

std::string check = "";

//if(_id == 110)
//	check = "if(threadIdx.x == 0 && blockIdx.x == 0) printf(\"%llx, %llx\\n\", result[0].a[0], result[0].a[1]);\n";
//else if(_id == 112)
//	check = "if(threadIdx.x == 0 && blockIdx.x == 0) printf(\"%llx, %llx, %llx, %llx\\n\", result[0].a[0], result[0].a[1], result[0].a[2], result[0].a[3]);\n";
//
	return "#include <stdio.h>\n"
		"#include <redfox/ra/interface/Tuple.h>\n"
		"typedef " + resulttype    + " ResultTuple;\n"
		"typedef " + sourcetype    + " SourceTuple;\n"
		"typedef ResultTuple::BasicType ResultType;\n"
		"typedef SourceTuple::BasicType SourceType;\n"
		"extern \"C\" __global__ void set_convert" + idstream.str() + "(ResultType* result, "
		"const SourceType* begin, const long long unsigned int* size)\n"
		"{\n"
		"	unsigned int step     = gridDim.x * blockDim.x;\n"
		"	unsigned int start    = threadIdx.x + blockIdx.x * blockDim.x;\n"
		"	unsigned int elements = (*size)/sizeof(ResultType);\n"
		"	for(unsigned int i = start; i < elements; i += step)\n"
		"	{\n"
		"	        if(i < elements)\n"
		"           {\n"
		"		    ResultType tmp1 = (ResultType)begin[i] << (unsigned int)" + shift + ";\n"
		"		    " + mapValueType(_type) + 
		" tmp2 = (" + mapValueType(_type) + ")ra::tuple::extract<unsigned long long int, " +  offset + ", SourceTuple>(begin[i]);\n"  
		"		    result[i] = " + insert + pos +  
		", ResultTuple>(tmp1, tmp2);\n"
		"           }\n"
		"	}\n"+ check +
		"}\n";
}

std::string RelationalAlgebraKernel::_arithGetResultSizeSource() const
{
	std::string resulttype = elementType(destination);
	std::string sourcetype = elementType(sourceA);

	std::stringstream idstream;
	idstream << _id;

	return "#include <redfox/ra/interface/Project.h>\n"
		"typedef " + resulttype    + " ResultTuple;\n"
		"typedef " + sourcetype    + " SourceTuple;\n"
		"typedef ResultTuple::BasicType ResultType;\n"
		"typedef SourceTuple::BasicType SourceType;\n"
		"extern \"C\" __global__ void arith_get_result_size" + idstream.str() +"("
			"long long unsigned int* size)\n"
		"{\n"
		"    ra::cuda::getResultSize<ResultType, SourceType>(size);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_appendStringGetResultSizeSource() const
{
	std::string resulttype = elementType(destination);
	std::string sourcetype = elementType(sourceA);

	std::stringstream idstream;
	idstream << _id;

	return "#include <redfox/ra/interface/Tuple.h>\n"
		"typedef " + resulttype    + " ResultTuple;\n"
		"typedef " + sourcetype    + " SourceTuple;\n"
		"typedef ResultTuple::BasicType ResultType;\n"
		"typedef SourceTuple::BasicType SourceType;\n"
		"extern \"C\" __global__ void appendstring_get_result_size" + idstream.str() +"("
			"long long unsigned int* size, long long unsigned int* stringsize)\n"
		"{\n"
		"    *stringsize = (*size * 128)/ sizeof(SourceType);\n"
		"    *size = (*size * sizeof(ResultType)/sizeof(SourceType));\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_convGetResultSizeSource() const
{
	std::string resulttype = elementType(destination);
	std::string sourcetype = elementType(sourceA);

	std::stringstream idstream;
	idstream << _id;

	return "#include <redfox/ra/interface/Project.h>\n"
		"typedef " + resulttype    + " ResultTuple;\n"
		"typedef " + sourcetype    + " SourceTuple;\n"
		"typedef ResultTuple::BasicType ResultType;\n"
		"typedef SourceTuple::BasicType SourceType;\n"
		"extern \"C\" __global__ void conv_get_result_size" + idstream.str() +"("
			"long long unsigned int* size)\n"
		"{\n"
		"    ra::cuda::getResultSize<ResultType, SourceType>(size);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_substringGetResultSizeSource() const
{
	std::string resulttype = elementType(destination);
	std::string sourcetype = elementType(sourceA);

	std::stringstream idstream;
	idstream << _id;

	return "#include <redfox/ra/interface/Tuple.h>\n"
		"typedef " + resulttype    + " ResultTuple;\n"
		"typedef " + sourcetype    + " SourceTuple;\n"
		"typedef ResultTuple::BasicType ResultType;\n"
		"typedef SourceTuple::BasicType SourceType;\n"
		"extern \"C\" __global__ void substring_get_result_size" + idstream.str() +"("
			"long long unsigned int* size)\n"
		"{\n"
		"    *size = (*size * sizeof(ResultType)/sizeof(SourceType));\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_selectMainSource() const
{
//	assertM(comparisons().size() == 1,
//		"Only one comparison supported per select so far..");
	std::string resulttype     = elementType(destination);
	std::string sourcetype     = elementType(sourceA);
	std::string compare        = comparisonType(comparisons()[0], sourceA);
	std::string valueType      = comparisonValueType(comparisons()[0], sourceA);
	std::string ctaCount       = _ctas();
	std::string threadCount    = _threads();
	std::string valueField     = comparisonFieldOne(comparisons()[0]);

	std::stringstream idstream;
	idstream << _id;

	std::string tmp = "";
//	if(_id == 36) tmp = "2";

	if(comparisons().size() == 2)
	{
		if(isCompareWithConstant(comparisons()[0]) && isCompareWithConstant(comparisons()[1]))
		{
		std::string compare2      = comparisonType(comparisons()[1], sourceA);
		std::string constant0     = comparisonConstant(comparisons()[0], sourceA);
		std::string constant1     = comparisonConstant(comparisons()[1], sourceA);
		std::string valueField2     = comparisonFieldOne(comparisons()[1]);

		return "#define PARTITIONS " + ctaCount + "\n"
			"#include <redfox/ra/interface/Select.h>\n"
			"typedef " + resulttype + " ResultTuple;\n"
			"typedef " + valueType  + " ValueType;\n"
			"typedef " + compare    + " Comparison0;\n"
			"typedef " + compare2    + " Comparison1;\n"
			"typedef ResultTuple::BasicType ResultType;\n"
			"extern \"C\" __global__ void set_select" + idstream.str() + "("
				"ResultType* output, unsigned int* histogram, "
				"const ResultType* input, "
				"const long long unsigned int* bytes)\n"
			"{\n"
			"    " + valueType + " constant0 = " + constant0 + ";\n"
			"    " + valueType + " constant1 = " + constant1 + ";\n"
			"    ra::cuda::select_field_constant_2<ResultTuple, "
				"Comparison0, Comparison1, ValueType, " + threadCount + ", " + valueField + ", " + valueField2 + ">(output, histogram, input, "
				"input + (*bytes/sizeof(ResultType)), constant0, constant1);\n"
			"}\n";
		}
		else if((isCompareWithArith(comparisons()[0]))&& (isCompareWithIndex(comparisons()[1])))
		{
		std::string constant0_left  = comparisonArithLeft(comparisons()[0], sourceA);
		std::string constant0_right = comparisonArithRight(comparisons()[0], sourceA);
		std::string compare2      = comparisonType(comparisons()[1], sourceA);
		std::string valueField2     = comparisonFieldOne(comparisons()[1]);
		std::string secondField2 = comparisonFieldTwo(comparisons()[1]);

		if(comparisons()[0].a.arith.op == AddMonth || comparisons()[0].b.arith.op == AddMonth)
		return "#define PARTITIONS " + ctaCount + "\n"
			"#include <redfox/ra/interface/Select.h>\n"
			"typedef " + resulttype + " ResultTuple;\n"
			"typedef " + valueType  + " ValueType;\n"
			"typedef " + compare    + " Comparison0;\n"
			"typedef " + compare2    + " Comparison1;\n"
			"typedef ResultTuple::BasicType ResultType;\n"
			"extern \"C\" __global__ void set_select" + idstream.str() + "("
				"ResultType* output, unsigned int* histogram, "
				"const ResultType* input, "
				"const long long unsigned int* bytes)\n"
			"{\n"
			"    ra::cuda::select_addmonth_field_field<ResultTuple, "
				"Comparison0, Comparison1, ValueType, " + threadCount + ", " + valueField + ", " + constant0_left + ", " + constant0_right + ", " + valueField2 + ", " + secondField2 + ">(output, histogram, input, "
				"input + (*bytes/sizeof(ResultType)));\n"
			"}\n";
		else
		return "#define PARTITIONS " + ctaCount + "\n"
			"#include <redfox/ra/interface/Select.h>\n"
			"typedef " + resulttype + " ResultTuple;\n"
			"typedef " + valueType  + " ValueType;\n"
			"typedef " + compare    + " Comparison0;\n"
			"typedef " + compare2    + " Comparison1;\n"
			"typedef ResultTuple::BasicType ResultType;\n"
			"extern \"C\" __global__ void set_select" + idstream.str() + "("
				"ResultType* output, unsigned int* histogram, "
				"const ResultType* input, "
				"const long long unsigned int* bytes)\n"
			"{\n"
			"    ra::cuda::select_add_field_field<ResultTuple, "
				"Comparison0, Comparison1, ValueType, " + threadCount + ", " + valueField + ", " + constant0_left + ", " + constant0_right + ", " + valueField2 + ", " + secondField2 + ">(output, histogram, input, "
				"input + (*bytes/sizeof(ResultType)));\n"
			"}\n";

		}
		std::string secondField = comparisonFieldTwo(comparisons()[0]);
		std::string compare2      = comparisonType(comparisons()[1], sourceA);
		std::string valueField2     = comparisonFieldOne(comparisons()[1]);
		std::string secondField2 = comparisonFieldTwo(comparisons()[1]);

		return "#define PARTITIONS " + ctaCount + "\n"
			"#include <redfox/ra/interface/Select.h>\n"
			"typedef " + resulttype + " ResultTuple;\n"
			"typedef " + valueType  + " ValueType;\n"
			"typedef " + compare    + " Comparison0;\n"
			"typedef " + compare2    + " Comparison1;\n"
			"typedef ResultTuple::BasicType ResultType;\n"
			"extern \"C\" __global__ void set_select" + idstream.str() +"("
				"ResultType* output, unsigned int* histogram, "
				"const ResultType* input, "
				"const long long unsigned int* bytes)\n"
			"{\n"
			"    ra::cuda::select_field_field_2<ResultTuple, "
				"Comparison0, Comparison1, ValueType, " + threadCount + ", " + valueField +
				", " + secondField + ", " + valueField2 + ", " + secondField2 + ">(output, histogram, input, "
				"input + (*bytes/sizeof(ResultType)));\n"
			"}\n";
	}

	if(isCompareWithConstantVariable(comparisons()[0]))
	{
		return "#define PARTITIONS " + ctaCount + "\n"
			"#include <redfox/ra/interface/Select.h>\n"
			"typedef " + resulttype + " ResultTuple;\n"
			"typedef " + valueType  + " ValueType;\n"
			"typedef " + compare    + " Comparison;\n"
			"typedef ResultTuple::BasicType ResultType;\n"
			"extern \"C\" __global__ void set_select" + idstream.str() + "("
				"ResultType* output, unsigned int* histogram, "
				"const ResultType* input, "
				"const long long unsigned int* bytes, "
				"" + valueType + "* value)\n"
			"{\n"
			"    " + valueType + " constant = *value;\n"
			"    ra::cuda::select_field_constant" + tmp + "<ResultTuple, "
				"Comparison, ValueType, " + threadCount + ", " + valueField +
				">(output, histogram, input, "
				"input + (*bytes/sizeof(ResultType)), constant);\n"
			"}\n";
	}
	else if(isCompareWithConstant(comparisons()[0]))
	{
		if(isCompareWithString(comparisons()[0], sourceA))
		{
		std::string id     = comparisonConstant(comparisons()[0], sourceA);

		return "#define PARTITIONS " + ctaCount + "\n"
			"#include <redfox/ra/interface/Select.h>\n"
			"typedef " + resulttype + " ResultTuple;\n"
			"typedef " + valueType  + " ValueType;\n"
			"typedef " + compare    + " Comparison;\n"
			"typedef ResultTuple::BasicType ResultType;\n"
			"extern \"C\" __global__ void set_select" + idstream.str() + "("
				"ResultType* output, unsigned int* histogram, "
				"const ResultType* input, "
				"const long long unsigned int* bytes, "
				"" + valueType + " StringTable)\n"
			"{\n"
			"    " + valueType + " constant = StringTable + 128 * " + id + ";\n"
			"    ra::cuda::select_field_constant" + tmp + "<ResultTuple, "
				"Comparison, ValueType, " + threadCount + ", " + valueField +
				">(output, histogram, input, "
				"input + (*bytes/sizeof(ResultType)), constant);\n"
			"}\n";
		}
		else if(isCompareWithFloat(comparisons()[0], sourceA))
		{
		std::string val     = comparisonConstant(comparisons()[0], sourceA);

		return "#define PARTITIONS " + ctaCount + "\n"
			"#include <redfox/ra/interface/Select.h>\n"
			"typedef " + resulttype + " ResultTuple;\n"
			"typedef " + valueType  + " ValueType;\n"
			"typedef " + compare    + " Comparison;\n"
			"typedef ResultTuple::BasicType ResultType;\n"
			"extern \"C\" __global__ void set_select" + idstream.str() + "("
				"ResultType* output, unsigned int* histogram, "
				"const ResultType* input, "
				"const long long unsigned int* bytes)\n"
			"{\n"
			"    " + valueType + " constant = (double)" + val + ";\n"
			"    ra::cuda::select_field_floatconstant" + tmp + "<ResultTuple, "
				"Comparison, ValueType, " + threadCount + ", " + valueField +
				">(output, histogram, input, "
				"input + (*bytes/sizeof(ResultType)), constant);\n"
			"}\n";

		}
		else
		{
		std::string val     = comparisonConstant(comparisons()[0], sourceA);

		return "#define PARTITIONS " + ctaCount + "\n"
			"#include <redfox/ra/interface/Select.h>\n"
			"typedef " + resulttype + " ResultTuple;\n"
			"typedef " + valueType  + " ValueType;\n"
			"typedef " + compare    + " Comparison;\n"
			"typedef ResultTuple::BasicType ResultType;\n"
			"extern \"C\" __global__ void set_select" + idstream.str() + "("
				"ResultType* output, unsigned int* histogram, "
				"const ResultType* input, "
				"const long long unsigned int* bytes)\n"
			"{\n"
			"    " + valueType + " constant = (ValueType)" + val + ";\n"
			"    ra::cuda::select_field_constant" + tmp + "<ResultTuple, "
				"Comparison, ValueType, " + threadCount + ", " + valueField +
				">(output, histogram, input, "
				"input + (*bytes/sizeof(ResultType)), constant);\n"
			"}\n";

		}
	}

	std::string secondField = comparisonFieldTwo(comparisons()[0]);

	if(sourceA[comparisons()[0].a.variableIndex].type == F64 || sourceA[comparisons()[0].a.variableIndex].type == F32)
	return "#define PARTITIONS " + ctaCount + "\n"
		"#include <redfox/ra/interface/Select.h>\n"
		"typedef " + resulttype + " ResultTuple;\n"
		"typedef " + valueType  + " ValueType;\n"
		"typedef " + compare    + " Comparison;\n"
		"typedef ResultTuple::BasicType ResultType;\n"
		"extern \"C\" __global__ void set_select" + idstream.str() +"("
			"ResultType* output, unsigned int* histogram, "
			"const ResultType* input, "
			"const long long unsigned int* bytes)\n"
		"{\n"
		"    ra::cuda::select_fielddouble_fielddouble<ResultTuple, "
			"Comparison, " + threadCount + ", " + valueField +
			", " + secondField + ">(output, histogram, input, "
			"input + (*bytes/sizeof(ResultType)));\n"
		"}\n";
	else
	return "#define PARTITIONS " + ctaCount + "\n"
		"#include <redfox/ra/interface/Select.h>\n"
		"typedef " + resulttype + " ResultTuple;\n"
		"typedef " + valueType  + " ValueType;\n"
		"typedef " + compare    + " Comparison;\n"
		"typedef ResultTuple::BasicType ResultType;\n"
		"extern \"C\" __global__ void set_select" + idstream.str() +"("
			"ResultType* output, unsigned int* histogram, "
			"const ResultType* input, "
			"const long long unsigned int* bytes)\n"
		"{\n"
		"    ra::cuda::select_field_field<ResultTuple, "
			"Comparison, ValueType, " + threadCount + ", " + valueField +
			", " + secondField + ">(output, histogram, input, "
			"input + (*bytes/sizeof(ResultType)));\n"
		"}\n";

	return "";
}

std::string RelationalAlgebraKernel::_selectGetResultSizeSource() const
{
	std::string resulttype = elementType(destination);
	std::string ctaCount   = _ctas();

	std::stringstream idstream;
	idstream << _id;

	return "#include <redfox/ra/interface/Select.h>\n"
		"typedef " + resulttype    + " ResultTuple;\n"
		"typedef ResultTuple::BasicType ResultType;\n"
		"extern \"C\" __global__ void select_get_result_size" + idstream.str() + "("
			"long long unsigned int* size, unsigned int* histogram)\n"
		"{\n"
		"    ra::cuda::getResultSize<ResultType, "
			+ ctaCount + ">(size, histogram);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_selectGatherSource() const
{
	report("Getting CUDA source for the select gather ptx kernel.");

	std::string destType   = elementType(destination);
	std::string ctaCount   = _ctas();
	std::string threadCount    = _threads();

	std::stringstream idstream;
	idstream << _id;

	return "#define PARTITIONS " + ctaCount + "\n"
		"#include <redfox/ra/interface/Select.h>\n"
		"typedef " + destType + " DestTuple;\n"
		"typedef DestTuple::BasicType DestType;\n"
		"extern \"C\" __global__ void select_gather" + idstream.str() + "("
			"DestType* dest,"
			"const DestType* temp, long long unsigned int *tempSize,"
			"const unsigned int* histogram)\n"
		"{\n"
		"    ra::cuda::gather<DestTuple" + ", " + threadCount + ">(dest, "
				"temp, temp + (*tempSize/sizeof(DestType)), "
				"histogram);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_assignSource() const
{
	report("Getting CUDA source for assign ptx kernel.");
	
	std::string datatype = elementType(destination);

	std::stringstream idstream;
	idstream << _id;

	return "#include <redfox/ra/interface/Assign.h>\n"
		"#include <redfox/ra/interface/Tuple.h>\n"
		"#include <stdio.h>\n"
		"typedef " + datatype + "::BasicType Element;\n"
		"extern \"C\" __global__ void assign_value" + idstream.str() + "(Element* begin, "
		"const long long unsigned int* element, const Element* value, char *StringTable)\n"
		"{\n"
		"    char * data = StringTable + (*value) * 128;\n"
//		"    printf(\"%llu\\n\", *value);\n"
//		"    printf(\"%llx\\n\", (unsigned long long int)StringTable);\n"
//		"    printf(\"%llx\\n\", (unsigned long long int)data);\n"
//		"    printf(\"%s\\n\", data);\n"
		"    ra::cuda::assign(begin, element, (unsigned long long int)data);\n"
//		"    printf(\"%llx\\n\", (*begin));\n"
//		"    printf(\"%s\\n\", (char *)(*begin));\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_sortSource() const
{
	report("Getting CUDA source for sort ptx kernel.");

	std::string type = elementType(destination);

	std::stringstream idstream;
	idstream << _id;
std::string tmp = "";
//if(_id == 52) tmp ="2";

	if(destination.size() == 1 && destination[0].type == RelationalAlgebraKernel::Pointer)
		return "#include <redfox/ra/interface/Sort.h>\n"
			"#include <stdio.h>\n"
			"typedef " + type    + " Tuple;\n"
			"typedef Tuple::BasicType TupleType;\n"
			"extern \"C\" void set_sort" + idstream.str() +"(TupleType* begin, "
			"const long long unsigned int size, unsigned long long int type)\n"
			"{\n"
			"    redfox::sort_string" + tmp + "(begin,"
				"begin + (size / sizeof(TupleType)));\n"
			"}\n";
	else
		return "#include <redfox/ra/interface/Sort.h>\n"
			"#include <stdio.h>\n"
			"typedef " + type    + " Tuple;\n"
			"typedef Tuple::BasicType TupleType;\n"
			"extern \"C\" void set_sort" + idstream.str() +"(TupleType* begin, "
			"const long long unsigned int size, unsigned long long int type)\n"
			"{\n"
			"    if(size != sizeof(TupleType))\n"
			"    redfox::sort" + tmp + "(begin,"
				"begin + (size / sizeof(TupleType)), type);\n"
			"}\n";
}

std::string RelationalAlgebraKernel::_moderngpuSortPairSource() const
{
	report("Getting CUDA source for sort ptx kernel.");

	std::string type_key = elementType(destination);
	std::string type_value = elementType(destination_1);

	std::stringstream idstream;
	idstream << _id;
std::string tmp = "";
//if(_id == 52) tmp ="2";

	if(destination.size() == 1 && destination[0].type == Pointer)
	return "#include <redfox/ra/interface/ModernGPUSort.h>\n"
		"#include <stdio.h>\n"
		"typedef " + type_key    + " Tuple_key;\n"
		"typedef Tuple_key::BasicType TupleType_key;\n"
		"typedef " + type_value    + " Tuple_value;\n"
		"typedef Tuple_value::BasicType TupleType_value;\n"

		"extern \"C\" void set_mgpusortpair" + idstream.str() +"(TupleType_key* key_begin, "
		"TupleType_value* value_begin, const long long unsigned int key_size, "
		"unsigned long long int key_type, unsigned long long int value_type)\n"
		"{\n"
		"    if(key_size != sizeof(TupleType_key))\n"
		"    redfox::sort_string_pair" + tmp + "(key_begin, value_begin,"
			"(key_size / sizeof(TupleType_key)), key_type, value_type);\n"
		"}\n";
	else
	return "#include <redfox/ra/interface/ModernGPUSort.h>\n"
		"#include <stdio.h>\n"
		"typedef " + type_key    + " Tuple_key;\n"
		"typedef Tuple_key::BasicType TupleType_key;\n"
		"typedef " + type_value    + " Tuple_value;\n"
		"typedef Tuple_value::BasicType TupleType_value;\n"

		"extern \"C\" void set_mgpusortpair" + idstream.str() +"(TupleType_key* key_begin, "
		"TupleType_value* value_begin, const long long unsigned int key_size, "
		"unsigned long long int key_type, unsigned long long int value_type)\n"
		"{\n"
		"    if(key_size != sizeof(TupleType_key))\n"
		"    redfox::sort_pair" + tmp + "(key_begin, value_begin,"
			"(key_size / sizeof(TupleType_key)), key_type, value_type);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_moderngpuSortKeySource() const
{
	report("Getting CUDA source for sort ptx kernel.");

	std::string type = elementType(destination);

	std::stringstream idstream;
	idstream << _id;
std::string tmp = "";
//if(_id == 52) tmp ="2";

	if(destination.size() == 1 && destination[0].type == Pointer)
	return "#include <redfox/ra/interface/ModernGPUSort.h>\n"
		"#include <stdio.h>\n"
		"typedef " + type    + " Tuple;\n"
		"typedef Tuple::BasicType TupleType;\n"

		"extern \"C\" void set_mgpusortkey" + idstream.str() +"(TupleType* key_begin, "
		"const long long unsigned int size, "
		"unsigned long long int type)\n"
		"{\n"
		"    if(size != sizeof(TupleType))\n"
		"    redfox::sort_string_key" + tmp + "(key_begin,"
			"(size / sizeof(TupleType)), type);\n"
		"}\n";
	else
	return "#include <redfox/ra/interface/ModernGPUSort.h>\n"
		"#include <stdio.h>\n"
		"typedef " + type    + " Tuple;\n"
		"typedef Tuple::BasicType TupleType;\n"

		"extern \"C\" void set_mgpusortkey" + idstream.str() +"(TupleType* key_begin, "
		"const long long unsigned int size, "
		"unsigned long long int type)\n"
		"{\n"
		"    if(size != sizeof(TupleType))\n"
		"    redfox::sort_key" + tmp + "(key_begin,"
			"(size / sizeof(TupleType)), type);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_b40cSortPairSource() const
{
	report("Getting CUDA source for sort ptx kernel.");

	std::string type_key = elementType(destination);
	std::string type_value = elementType(destination_1);

	std::stringstream idstream;
	idstream << _id;
std::string tmp = "";
//if(_id == 52) tmp ="2";

	return "#include <redfox/ra/interface/b40cSort.h>\n"
		"#include <stdio.h>\n"
		"typedef " + type_key    + " Tuple_key;\n"
		"typedef Tuple_key::BasicType TupleType_key;\n"
		"typedef " + type_value    + " Tuple_value;\n"
		"typedef Tuple_value::BasicType TupleType_value;\n"

		"extern \"C\" void set_b40csortpair" + idstream.str() +"(TupleType_key* key_begin, "
		"TupleType_value* value_begin, const long long unsigned int key_size, "
		"unsigned long long int key_bits, unsigned long long int value_type)\n"
		"{\n"
		"    if(key_size != sizeof(TupleType_key))\n"
		"    redfox::sort_pair" + tmp + "(key_begin, value_begin,"
			"(key_size / sizeof(TupleType_key)), key_bits, value_type);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_b40cSortKeySource() const
{
	report("Getting CUDA source for sort ptx kernel.");

	std::string type = elementType(destination);

	std::stringstream idstream;
	idstream << _id;
std::string tmp = "";
//if(_id == 52) tmp ="2";

	return "#include <redfox/ra/interface/b40cSort.h>\n"
		"#include <stdio.h>\n"
		"typedef " + type    + " Tuple;\n"
		"typedef Tuple::BasicType TupleType;\n"

		"extern \"C\" void set_b40csortkey" + idstream.str() +"(TupleType* key_begin, "
		"const long long unsigned int size, "
		"unsigned long long int bits)\n"
		"{\n"
		"    if(size != sizeof(TupleType))\n"
		"    redfox::sort_key" + tmp + "(key_begin,"
			"(size / sizeof(TupleType)), bits);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_reduceSource(Operator op) const
{
	report("Getting CUDA source for reduce ptx kernel.");
	
	std::stringstream idstream;
	idstream << _id;
	if(op == Total && sourceA.size() == 1 && sourceA[0].type == Pointer && sourceB[0].type == F64)
		return "#include <redfox/ra/interface/Reduce.h>\n"
			"extern \"C\" void set_reduce" + idstream.str() +"(char** result_key_begin, "
			"double* result_value_begin, char** key_begin, "
			"unsigned long long int *key_size, double* value_begin, " 
			"unsigned long long int *value_size)\n"
			"{\n"
			"    redfox::total_string_double(key_begin, "
			"         value_begin, result_key_begin, " 
			"         result_value_begin, key_size, value_size);\n"
			"}\n";
	else if(op == Total && sourceA.size() == 1 && sourceA[0].type == Pointer && sourceB[0].type == I32)
		return "#include <redfox/ra/interface/Reduce.h>\n"
			"extern \"C\" void set_reduce" + idstream.str() +"(char** result_key_begin, "
			"unsigned int* result_value_begin, char** key_begin, "
			"unsigned long long int *key_size, unsigned int* value_begin, " 
			"unsigned long long int *value_size)\n"
			"{\n"
			"    redfox::total_string_32(key_begin, "
			"         value_begin, result_key_begin, " 
			"         result_value_begin, key_size, value_size);\n"
			"}\n";
	else if(op == Total && bytes(sourceA) == 1 && sourceB[0].type == F64) 
		return "#include <redfox/ra/interface/Reduce.h>\n"
			"extern \"C\" void set_reduce" + idstream.str() +"(unsigned char* result_key_begin, "
			"double* result_value_begin, unsigned char* key_begin, "
			"unsigned long long int *key_size, double* value_begin, " 
			"unsigned long long int *value_size)\n"
			"{\n"
			"    redfox::total_8_double(key_begin, "
			"         value_begin, result_key_begin, " 
			"         result_value_begin, key_size, value_size);\n"
			"}\n";
	else if(op == Total && bytes(sourceA) == 2 && sourceB[0].type == F64) 
		return "#include <redfox/ra/interface/Reduce.h>\n"
			"extern \"C\" void set_reduce" + idstream.str() +"(unsigned char* result_key_begin, "
			"double* result_value_begin, unsigned char* key_begin, "
			"unsigned long long int *key_size, double* value_begin, " 
			"unsigned long long int *value_size)\n"
			"{\n"
			"    redfox::total_16_double(key_begin, "
			"         value_begin, result_key_begin, " 
			"         result_value_begin, key_size, value_size);\n"
			"}\n";
	else if(op == Total && (bytes(sourceA) == 3 || bytes(sourceA) == 4) && sourceB[0].type == F64) 
		return "#include <redfox/ra/interface/Reduce.h>\n"
			"extern \"C\" void set_reduce" + idstream.str() +"(unsigned char* result_key_begin, "
			"double* result_value_begin, unsigned char* key_begin, "
			"unsigned long long int *key_size, double* value_begin, " 
			"unsigned long long int *value_size)\n"
			"{\n"
			"    redfox::total_32_double(key_begin, "
			"         value_begin, result_key_begin, " 
			"         result_value_begin, key_size, value_size);\n"
			"}\n";
	else if(op == Total && (bytes(sourceA) >= 5 || bytes(sourceA) <= 8) && sourceB[0].type == I32) 
		return "#include <redfox/ra/interface/Reduce.h>\n"
			"extern \"C\" void set_reduce" + idstream.str() +"(unsigned long long int* result_key_begin, "
			"unsigned int* result_value_begin, unsigned long long int* key_begin, "
			"unsigned long long int *key_size, unsigned int* value_begin, " 
			"unsigned long long int *value_size)\n"
			"{\n"
			"    redfox::total_64_32(key_begin, "
			"         value_begin, result_key_begin, " 
			"         result_value_begin, key_size, value_size);\n"
			"}\n";
	else if(op == Total && (bytes(sourceA) >= 5 && bytes(sourceA) <= 8) && sourceB[0].type == F64) 
		return "#include <redfox/ra/interface/Reduce.h>\n"
			"extern \"C\" void set_reduce" + idstream.str() +"(unsigned char* result_key_begin, "
			"double* result_value_begin, unsigned char* key_begin, "
			"unsigned long long int *key_size, double* value_begin, " 
			"unsigned long long int *value_size)\n"
			"{\n"
			"    redfox::total_64_double(key_begin, "
			"         value_begin, result_key_begin, " 
			"         result_value_begin, key_size, value_size);\n"
			"}\n";
	else if(op == Total && (bytes(sourceA) >= 9 && bytes(sourceA) <= 16) && sourceB[0].type == F64) 
		return "#include <redfox/ra/interface/Reduce.h>\n"
			"extern \"C\" void set_reduce" + idstream.str() +"(unsigned char* result_key_begin, "
			"double* result_value_begin, unsigned char* key_begin, "
			"unsigned long long int *key_size, double* value_begin, " 
			"unsigned long long int *value_size)\n"
			"{\n"
			"    redfox::total_128_double(key_begin, "
			"         value_begin, result_key_begin, " 
			"         result_value_begin, key_size, value_size);\n"
			"}\n";

	else if(op == Min && (bytes(sourceA) == 4 || bytes(sourceA) == 3) && sourceB[0].type == F64)
		return "#include <redfox/ra/interface/Reduce.h>\n"
			"extern \"C\" void set_reduce" + idstream.str() +"(unsigned int* result_key_begin, "
			"double* result_value_begin, unsigned int* key_begin, "
			"unsigned long long int *key_size, double* value_begin, " 
			"unsigned long long int *value_size)\n"
			"{\n"
			"    redfox::min_32_double(key_begin, \n"
			"         value_begin, result_key_begin\n, " 
			"         result_value_begin, key_size, value_size);\n"
			"}\n";

	return "";
}

std::string RelationalAlgebraKernel::_singleReduceSource() const
{
	report("Getting CUDA source for single reduce ptx kernel.");
	
	std::stringstream idstream;
	idstream << _id;

	if(_operator == SingleTotal && sourceA[0].type == F64) 
		return "#include <redfox/ra/interface/Reduce.h>\n"
			"#include <stdio.h>\n"
			"extern \"C\" void set_single_reduce" + idstream.str() +"(double* result, "
			"double* begin, unsigned long long int size)\n"
			"{\n"
//			"    printf(\"*****%llu\\n\", size);\n"
			"    redfox::total_double(result, begin, begin + size / sizeof(double));\n"
			"}\n";
	else if(_operator == SingleMax && sourceA[0].type == F64) 
		return "#include <redfox/ra/interface/Reduce.h>\n"
			"#include <stdio.h>\n"
			"extern \"C\" void set_single_reduce" + idstream.str() +"(unsigned long long int* result, "
			"unsigned long long int* begin, unsigned long long int size)\n"
			"{\n"
//			"    printf(\"size *****%llu\\n\", size);\n"
			"    redfox::max_double(result, begin, begin + size / sizeof(unsigned long long int));\n"
			"}\n";
	else if(_operator == SingleCount && sourceA[0].type == I32) 
		return "#include <redfox/ra/interface/Reduce.h>\n"
			"#include <stdio.h>\n"
			"extern \"C\" void set_single_reduce" + idstream.str() +"(unsigned int* result, "
			"unsigned int* begin, unsigned long long int size)\n"
			"{\n"
//			"    printf(\"size *****%llu\\n\", size);\n"
			"    redfox::count(result, begin, begin + size / sizeof(unsigned int));\n"
			"}\n";

	return "";
}

std::string RelationalAlgebraKernel::_uniqueSource() const
{
	report("Getting CUDA source for unique ptx kernel.");

	std::string type = elementType(destination);

	std::stringstream idstream;
	idstream << _id;

	std::string tmp = "";

	return "#include <redfox/ra/interface/Unique.h>\n"
		"#include <stdio.h>\n"
		"typedef " + type    + " Tuple;\n"
		"typedef Tuple::BasicType TupleType;\n"
		"extern \"C\" void set_unique" + idstream.str() + "(TupleType* begin, "
		"unsigned long long int *size, unsigned long long int type)\n"
		"{\n"
		"   redfox::unique" + tmp + "(begin, size, type);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_countSource() const
{
	report("Getting CUDA source for count ptx kernel.");
	
	std::stringstream idstream;
	idstream << _id;

	if(sourceA.size() == 1 && sourceA[0].type == Pointer)
		return "#include <redfox/ra/interface/Reduce.h>\n"
			"extern \"C\" void set_count" + idstream.str() +"(char** result_key_begin, "
			"unsigned int* result_value_begin, char** key_begin, "
			"unsigned long long int *key_size, unsigned int* value_begin, " 
			"unsigned long long int *value_size)\n"
			"{\n"
			"    redfox::count_string(key_begin, "
			"    value_begin, result_key_begin, " 
			"    result_value_begin, key_size, value_size);\n"
			"}\n";
	else if(bytes(sourceA) >= 5 && bytes(sourceA) <= 16)
		return "#include <redfox/ra/interface/Reduce.h>\n"
			"extern \"C\" void set_count" + idstream.str() +"(unsigned char* result_key_begin, "
			"unsigned int* result_value_begin, unsigned char* key_begin, "
			"unsigned long long int *key_size, unsigned int* value_begin, " 
			"unsigned long long int *value_size)\n"
			"{\n"
			"    redfox::count_128(key_begin, "
			"    value_begin, result_key_begin, " 
			"    result_value_begin, key_size, value_size);\n"
			"}\n";
	else if(bytes(sourceA) == 1)
		return "#include <redfox/ra/interface/Reduce.h>\n"
			"extern \"C\" void set_count" + idstream.str() +"(unsigned char* result_key_begin, "
			"unsigned int* result_value_begin, unsigned char* key_begin, "
			"unsigned long long int *key_size, unsigned int* value_begin, " 
			"unsigned long long int *value_size)\n"
			"{\n"
			"    redfox::count_8(key_begin, "
			"    value_begin, result_key_begin, " 
			"    result_value_begin, key_size, value_size);\n"
			"}\n";
	else if(bytes(sourceA) == 2)
		return "#include <redfox/ra/interface/Reduce.h>\n"
			"extern \"C\" void set_count" + idstream.str() +"(unsigned char* result_key_begin, "
			"unsigned int* result_value_begin, unsigned char* key_begin, "
			"unsigned long long int *key_size, unsigned int* value_begin, " 
			"unsigned long long int *value_size)\n"
			"{\n"
			"    redfox::count_16(key_begin, "
			"    value_begin, result_key_begin, " 
			"    result_value_begin, key_size, value_size);\n"
			"}\n";
	else if(bytes(sourceA) == 3 || bytes(sourceA) == 4)
		return "#include <redfox/ra/interface/Reduce.h>\n"
			"extern \"C\" void set_count" + idstream.str() +"(unsigned char* result_key_begin, "
			"unsigned int* result_value_begin, unsigned char* key_begin, "
			"unsigned long long int *key_size, unsigned int* value_begin, " 
			"unsigned long long int *value_size)\n"
			"{\n"
			"    redfox::count_32(key_begin, "
			"    value_begin, result_key_begin, " 
			"    result_value_begin, key_size, value_size);\n"
			"}\n";
	else if(bytes(sourceA) >= 17 && bytes(sourceA) <= 32)
		return "#include <redfox/ra/interface/Reduce.h>\n"
			"extern \"C\" void set_count" + idstream.str() +"(unsigned char* result_key_begin, "
			"unsigned int* result_value_begin, unsigned char* key_begin, "
			"unsigned long long int *key_size, unsigned int* value_begin, " 
			"unsigned long long int *value_size)\n"
			"{\n"
			"    redfox::count_256(key_begin, "
			"    value_begin, result_key_begin, " 
			"    result_value_begin, key_size, value_size);\n"
			"}\n";

	return "";
}

std::string RelationalAlgebraKernel::_differenceGetResultSizeSource() const
{
	report("Getting CUDA source for the merge get size ptx kernel.");

	std::stringstream idstream;
	idstream << _id;

	return "#include <stdio.h>\n"
		"extern \"C\" __global__ void difference_get_result_size" + idstream.str() + "("
			"long long unsigned int* size, long long unsigned int* size_left)\n"
		"{\n"
		"    *size = *size_left;\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_unionGetResultSizeSource() const
{
	report("Getting CUDA source for the merge get size ptx kernel.");

	std::stringstream idstream;
	idstream << _id;

	return "#include <stdio.h>\n"
		"extern \"C\" __global__ void union_get_result_size" + idstream.str() + "("
			"long long unsigned int* size, long long unsigned int* size_left, long long unsigned int* size_right)\n"
		"{\n"
		"    *size = (*size_left) + (*size_right);\n"
		"}\n";
}

std::string RelationalAlgebraKernel::_unionSource() const
{
	report("Getting CUDA source for union ptx kernel.");

	std::string type = elementType(destination);

	std::stringstream idstream;
	idstream << _id;

	return "#include <stdio.h>\n"
		"#include <redfox/ra/interface/Tuple.h>\n"
		"#include <redfox/ra/interface/Union.h>\n"
		"typedef " + type    + " Tuple;\n"
		"typedef Tuple::BasicType TupleType;\n"
		"extern \"C\" void set_union" + idstream.str() + "(TupleType* result, long long unsigned int* size,"
		"TupleType* left_begin, const long long unsigned int left_size, "
		"TupleType* right_begin, const long long unsigned int right_size,"
		"unsigned long long int type)\n"
		"{\n"
			"    redfox::set_union(result, size, left_begin, left_begin + (left_size / sizeof(TupleType)),"
				"right_begin, right_begin + (right_size / sizeof(TupleType)), type);\n"

		"}\n";
}


//std::string RelationalAlgebraKernel::_reduceSource(Operator op) const
//{
//	report("Getting CUDA source for reduce ptx kernel.");
//	
//	std::string keytuple = elementType(sourceA);
//	std::string valuetype = mapValueType(sourceB[0].type);
//	std::string reducetype = mapValueType(destination_1[0].type);
//	std::string pred = "thrust::equal_to<KeyTuple::BasicType>";
//
//	std::string ope;
//
//	switch(op)
//	{
//	case Total:
//	{
//		ope = "thrust::plus<" + reducetype + ">";
//		break;
//	} 
//	case Max:
//	{
//		ope = "thrust::maximam<" + reducetype + ">";
//		break;
//	} 
//	case Min:
//	{
//		ope = "thrust::minimum<" + reducetype + ">";
//		break;
//	}
//	default: 
//	{
//		assertM(false, "Invalid reduction type.");
//	}
//	}
//
//	std::stringstream idstream;
//	idstream << _id;
//
//	return "#include <redfox/ra/interface/Reduce.h>\n"
//		"typedef " + keytuple     + " KeyTuple;\n"
//		"typedef KeyTuple::BasicType KeyType;\n"
//		"typedef " + valuetype    + " ValueType;\n"
//		"typedef redfox::WrappedType<ValueType> WrappedValueType;\n"
//		"typedef " + reducetype    + " ReduceType;\n"
//		"typedef redfox::WrappedType<ReduceType> WrappedReduceType;\n"
//		"extern \"C\" void set_reduce" + idstream.str() +"(KeyType* result_key_begin, "
//		"ReduceType* result_value_begin, const KeyType* key_begin, "
//		"unsigned long long int *key_size, const ReduceType* value_begin, " 
//		"unsigned long long int *value_size)\n"
//		"{\n"
//		"    redfox::reduce<KeyTuple, WrappedValueType, WrappedReduceType, "
//		+ pred + ", " + ope +" >(key_begin, "
//		"value_begin, result_key_begin, " 
//			"result_value_begin, key_size, value_size);\n"
//		"}\n";
//}

std::string RelationalAlgebraKernel::_fields() const
{
	std::stringstream stream;
	stream << _keyFields;
	
	return stream.str();
}

std::string RelationalAlgebraKernel::_threads() const
{
	std::stringstream stream;
	
	stream << _threadCount;
	
	return stream.str();
}

std::string RelationalAlgebraKernel::_ctas() const
{
	std::stringstream stream;
	
	stream << _ctaCount;
	
	return stream.str();
}

}

#endif

