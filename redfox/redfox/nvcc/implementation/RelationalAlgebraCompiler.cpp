/*! \file RelationalAlgebraCompiler.cpp
	\date Sunday October 31, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the RelationalAlgebraCompiler class.
*/

#ifndef RELATIONAL_ALGEBRA_COMPILER_CPP_INCLUDED
#define RELATIONAL_ALGEBRA_COMPILER_CPP_INCLUDED

// Red Fox Includes
#include <redfox/nvcc/interface/RelationalAlgebraCompiler.h>
#include <redfox/nvcc/interface/RelationalAlgebraKernel.h>
#include <redfox/nvcc/interface/CudaCompilerInterface.h>

#include <redfox/protocol/interface/RelationalAlgebra.pb.h>
#include <redfox/protocol/interface/HarmonyIR.pb.h>

// Hydrazine Includes
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/math.h>

// Standard Library Includes
#include <unordered_map>
#include <vector>
#include <set>
#include <algorithm>
#include <iostream>
#include <fstream>
// Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1 

#define TC_NUM (5)
//#define LINEITEM_NUM (41995307)
#define LINEITEM_NUM (6001215)
#define PARTSUPP_NUM (800000)
#define SUPPLIER_NUM (10000)
#define NATION_NUM (25)
#define REGION_NUM (5)
#define PART_NUM (200000)
#define ORDERS_NUM (1500000)
#define CUSTOMER_NUM (150000)

#define order_bits 23 
#define part_bits  18 
#define supp_bits 14 
#define cust_bits 18
#define line_bits 3
#define flag_bits 2
#define status_bits 1 
#define date_bits 64 
#define nation_bits 5
#define region_bits 3
#define size_bits 32 
#define float_bits 64
#define string_bits 64
#define partkey_bits 32
#define custkey_bits 32
#define priority_bits 32
#define quantity_bits 32
#define suppkey_bits 32
#define orderkey_bits 32

// Typedefs and Namespace Aliases
namespace common = blox::common::protocol;
namespace lb = blox::compiler::gpu;
//typedef google::protobuf::RepeatedPtrField<lb::GPUCompare> RepeatedGPUCompare;
typedef google::protobuf::RepeatedPtrField<lb::Exp> RepeatedExp;
typedef google::protobuf::RepeatedPtrField<lb::Agg> RepeatedAgg;
typedef google::protobuf::RepeatedPtrField<common::Constant>   RepeatedConstant;
typedef google::protobuf::RepeatedPtrField<common::Type>       RepeatedType;
typedef google::protobuf::RepeatedField<int>       RepeatedDomain;

namespace nvcc
{

typedef std::unordered_map<std::string, unsigned int> StringIdMap;
int blocknum = 0;
int maxblocknum = 1000; 
int kernel_id = 0;
int random_name = 0;
//int q22_join = 0;
//int q22_single = 0;

#if 0
static long long unsigned int byteReverse(long long unsigned int input)
{
	long long unsigned int output;
	
	char* in  = (char*)&input;
	char* out = (char*)&output;
	
	out[7] = in[0];
	out[6] = in[1];
	out[5] = in[2];
	out[4] = in[3];
	out[3] = in[4];
	out[2] = in[5];
	out[1] = in[6];
	out[0] = in[7];
	
	return output;
}

static unsigned int log2(unsigned int i)
{
	unsigned int targetlevel = 0;
	while (i >>= 1) ++targetlevel;
	
	return targetlevel;
}
#endif

static std::string compilePTXSource(const std::string& source)
{	
	report("Compiling source: " << source);
	nvcc::Compiler::compileToPTX(source);
	return nvcc::Compiler::compiledPTX();
}

static std::string compileBINSource(const std::string& source, 
	RelationalAlgebraKernel::Operator op)
{	
	report("Compiling source: " << source);
	std::string external = "redfox/ra/implementation/";

	switch(op)
	{
	case RelationalAlgebraKernel::Sort:
	{
		external += "Sort.cu";
		break;
	}
	case RelationalAlgebraKernel::Unique:
	{
		external += "Unique.cu";
		break;
	}
	case RelationalAlgebraKernel::Union:
	{
		external += "Union.cu";
		break;
	}
	case RelationalAlgebraKernel::Difference:
	{
		external += "Difference.cu";
		break;
	}
	case RelationalAlgebraKernel::Count:
	case RelationalAlgebraKernel::Total:
	case RelationalAlgebraKernel::Max:
	case RelationalAlgebraKernel::Min:
	{
		external += "Reduce.cu";
		break;
	}
	case RelationalAlgebraKernel::ModernGPUSortKey:
	case RelationalAlgebraKernel::ModernGPUSortPair:
	{
		external += "ModernGPUSort.cu";
		break;
	}
	case RelationalAlgebraKernel::RadixSortKey:
	case RelationalAlgebraKernel::RadixSortPair:
	{
		external += "b40cSort.cu";
		break;
	}
	case RelationalAlgebraKernel::ModernGPUJoinFindBounds:
	case RelationalAlgebraKernel::ModernGPUJoinMain:
	{
		external += "ModernGPUJoin.cu";
		break;
	}
	default:
	{
		assertM(false, "Invalid operation.");
		break;
	}
	}

	if(op == RelationalAlgebraKernel::ModernGPUSortKey || op == RelationalAlgebraKernel::ModernGPUSortPair)
		nvcc::Compiler::compileToModernBIN(source, external, "ModernGPUSort.o");
	else if(op == RelationalAlgebraKernel::RadixSortKey || op == RelationalAlgebraKernel::RadixSortPair)
		nvcc::Compiler::compileToB40CBIN(source, external, "b40cSort.o");
	else if(op == RelationalAlgebraKernel::ModernGPUJoinFindBounds || op == RelationalAlgebraKernel::ModernGPUJoinMain)
		nvcc::Compiler::compileToModernBIN(source, external, "ModernGPUJoin.o");
	else if(op == RelationalAlgebraKernel::Unique)
	{
		nvcc::Compiler::compileToBIN(source, external,  "Unique.o");
	}
	else if(op == RelationalAlgebraKernel::Union)
	{
		nvcc::Compiler::compileToBIN(source, external,  "Union.o");
	}
	else if(op == RelationalAlgebraKernel::Difference)
	{
		nvcc::Compiler::compileToBIN(source, external,  "Difference.o");
	}
	else if(op == RelationalAlgebraKernel::Count || op == RelationalAlgebraKernel::Total
		|| op == RelationalAlgebraKernel::Max || op == RelationalAlgebraKernel::Min)
	{
		nvcc::Compiler::compileToBIN(source, external,  "Reduce.o");
	}

	return nvcc::Compiler::compiledBIN();
}

static std::string getCompareEqualPTX()
{
	return compilePTXSource(
		"#include <redfox/ra/interface/ConditionalBranch.h>\n"
		"extern \"C\" __global__ void conditional_branch("
			"long long unsigned int* index, "
			"const char* a, const long long unsigned int* sizeA,"
			"const char* b, const long long unsigned int* sizeB)\n"
		"{\n"
		"    ra::cuda::conditionalBranch(index, "
			"a, a + *sizeA, b, b + *sizeB);\n"
		"}\n");
}

static std::string getMovePTX()
{
	return compilePTXSource(
		"#include <redfox/ra/interface/Copy.h>\n"
		"extern \"C\" __global__ void copy(char* out, "
			"const char* in, const long long unsigned int* size)\n"
		"{\n"
		"    ra::cuda::copy(out, in, size);\n"
		"}\n");
}

class VariableDescriptor
{
public:
	VariableDescriptor(unsigned int i) : id(i), unique_keys(0), isConstant(false), isSorted(0) {}

	VariableDescriptor(unsigned int i, unsigned int key_num) : id(i), unique_keys(key_num), isConstant(false), isSorted(0) {}

public:
	unsigned int                      id;
	unsigned int                      unique_keys;
	RelationalAlgebraKernel::Variable types;
	bool				  isConstant;
	unsigned int			  isSorted;
	std::vector<unsigned int>	  sorted_fields;
};

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

static void translateAppendCode(std::stringstream& stream,
        const lb::Exp& _exp, const StringIdMap& strings)
{
        if(_exp.tag() == lb::CALL)
        {
                translateAppendCode(stream, _exp.call().args(0), strings);
                translateAppendCode(stream, _exp.call().args(1), strings);
        }
        else if(_exp.tag() == lb::CONSTEXP)
        {
		std::string const_string =  _exp.constexp().literal().string_constant().value();
        	StringIdMap::const_iterator id = strings.find(const_string);
        	assert(id != strings.end());
		stream <<
			"{\n" 	
			"	char *const_string = StringTable + 128 * " << id->second << ";\n"	
			"	for(unsigned int i = 0; i < " << const_string.length() << "; ++i, ++tmp2)\n"
			"	{\n"	
			"		*tmp2 = const_string[i];\n"	
			"	}\n"	
			"}\n";	
	}
        else if(_exp.tag() == lb::INDEX)
        {
		unsigned int index = _exp.index().offset();
		stream <<
			"{\n" 	
			"	char *index_string = (char *)(ra::tuple::extract<unsigned long long int, " << index << ", SourceTuple>(begin[i]));\n"
			"	for(unsigned int i = 0; index_string[i] != \'\\0\'; ++i, ++tmp2)\n"
			"	{\n"	
			"		*tmp2 = index_string[i];\n"	
			"	}\n"	
			"}\n";	
	}
}

static std::string appendcode(const lb::Exp _exp, const StringIdMap& strings)
{
        std::stringstream stream;

        translateAppendCode(stream, _exp.call().args(0), strings);
        translateAppendCode(stream, _exp.call().args(1), strings);

	stream << "*tmp2 = \'\\0\';\n";
        return stream.str();
}

static std::string getAppendStringPTX(const lb::Exp& exp, const StringIdMap& strings, 
	const VariableDescriptor& source, const VariableDescriptor& destination)
{
	std::string resulttype = elementType(destination.types);
	std::string sourcetype = elementType(source.types);

        std::stringstream streampos;
        streampos << (destination.types.size() - 1);
        std::string pos = streampos.str();

	std::string append = appendcode(exp, strings);

	std::string code = 
			"#include <stdio.h>\n"
                        "#include <redfox/ra/interface/Tuple.h>\n"
                        "typedef " + resulttype    + " ResultTuple;\n"
                        "typedef " + sourcetype    + " SourceTuple;\n"
                        "typedef ResultTuple::BasicType ResultType;\n"
                        "typedef SourceTuple::BasicType SourceType;\n"
                        "extern \"C\" __global__ void append_string(ResultType* result, char * result_string, "
                        "const SourceType* begin, const long long unsigned int* size, char * StringTable)\n"
                        "{\n"
                        "       unsigned int step     = gridDim.x * blockDim.x;\n"
                        "       unsigned int start    = threadIdx.x + blockIdx.x * blockDim.x;\n"
                        "       unsigned int elements = (*size)/sizeof(ResultType);\n"
                        "       for(unsigned int i = start; i < elements; i += step)\n"
                        "       {\n"
                        "           if(i < elements)\n"
                        "           {\n"
                        "                       ResultType tmp1 = (ResultType)begin[i] << (unsigned int)64;\n"
                        "                       char *tmp2 = result_string + i * 128;\n"
                        "                       result[i] = ra::tuple::insert<unsigned long long int, " + pos + 
			",ResultTuple>(tmp1, (unsigned long long int)tmp2);\n" + append +
                        "           }\n"
                        "       }\n"
                        "}\n";

	return compilePTXSource(code);
}

typedef std::unordered_map<std::string, VariableDescriptor> VariableMap;

static bool checkSigned(const lb::Comp& compare, const VariableMap& variables)
{
	lb::Exp op1 = compare.op1();
	lb::Exp op2 = compare.op2();
	
	common::PrimitiveType_Kind kind;
report("checkSigned " << op1.tag() << " " << op2.tag());
	if(op1.tag() == lb::INDEX)
	{
		assert(op1.index().etyp().kind() == common::Type_Kind_PRIMITIVE);
		kind = op1.index().etyp().primitive().kind();
	}
	else 
	{
		assert(op2.tag() == lb::INDEX);
		assert(op2.index().etyp().kind() == common::Type_Kind_PRIMITIVE);
		kind = op2.index().etyp().primitive().kind();
	}

	switch(kind)
	{
	case common::PrimitiveType_Kind_INT:   /* fall through */
	{
		return true;
	}
	case common::PrimitiveType_Kind_FLOAT:
	case common::PrimitiveType_Kind_BOOL:     /* fall through */
	case common::PrimitiveType_Kind_COLOR:    /* fall through */
	case common::PrimitiveType_Kind_SHAPE:    /* fall through */
	case common::PrimitiveType_Kind_IMAGE:    /* fall through */
	case common::PrimitiveType_Kind_BLOB:     /* fall through */
	case common::PrimitiveType_Kind_DATETIME: /* fall through */
	case common::PrimitiveType_Kind_UINT:     /* fall through */
	case common::PrimitiveType_Kind_STRING:   /* fall through */
	case common::PrimitiveType_Kind_DECIMAL:
	{
		return false;
	}
	}
	
	return false;
}

static RelationalAlgebraKernel::Operator mapReduction(const lb::AggTag& tag)
{
	report("mapReduction.");
	switch(tag)
	{
		case lb::TOTAL: return RelationalAlgebraKernel::Total;
		case lb::MAX:   return RelationalAlgebraKernel::Max;
		case lb::MIN:   return RelationalAlgebraKernel::Min;
		default: 	return RelationalAlgebraKernel::InvalidOperator;
	}
}

static RelationalAlgebraKernel::Comparison translate(
	const lb::Comp& compare, const VariableMap& variables)
{
	bool isSigned = checkSigned(compare, variables);

	if(isSigned)
	{
		switch(compare.comparison())
		{
		case lb::Eq: return RelationalAlgebraKernel::Eq;
		case lb::Ne: return RelationalAlgebraKernel::Ne;
		case lb::Lt: return RelationalAlgebraKernel::Slt;
		case lb::Le: return RelationalAlgebraKernel::Sle;
		case lb::Gt: return RelationalAlgebraKernel::Sgt;
		case lb::Ge: return RelationalAlgebraKernel::Sge;
		}
	}
	else
	{
		switch(compare.comparison())
		{
		case lb::Eq: return RelationalAlgebraKernel::Eq;
		case lb::Ne: return RelationalAlgebraKernel::Ne;
		case lb::Lt: return RelationalAlgebraKernel::Lt;
		case lb::Le: return RelationalAlgebraKernel::Le;
		case lb::Gt: return RelationalAlgebraKernel::Gt;
		case lb::Ge: return RelationalAlgebraKernel::Ge;
		}
	}
	
	return RelationalAlgebraKernel::InvalidComparison;
}

static unsigned int bytes(const RelationalAlgebraKernel::Value& value)
{
	unsigned int count = 0;
	for(RelationalAlgebraKernel::Value::const_iterator
		constant = value.begin(); constant != value.end(); ++constant)
	{
		count += constant->limit;
	}
	
	return (count + 7) / 8;
}

static unsigned int bytes(const RelationalAlgebraKernel::Variable& type)
{
	unsigned int count = 0;
	for(RelationalAlgebraKernel::Variable::const_iterator
		t = type.begin(); t != type.end(); ++t)
	{
		count += t->limit;
	}
	
	return (count + 7) / 8;
}

static unsigned int bits(const RelationalAlgebraKernel::Variable& type)
{
	unsigned int count = 0;
	for(RelationalAlgebraKernel::Variable::const_iterator
		t = type.begin(); t != type.end(); ++t)
	{
		count += t->limit;
	}
	
	return count;
}

static unsigned int getTupleDataType(const RelationalAlgebraKernel::Variable& v)
{
	unsigned int bits = 0;
	for(RelationalAlgebraKernel::Variable::const_iterator
		i = v.begin(); i != v.end(); ++i)
	{
		bits += i->limit;
	}
	
	if(bits <= 8)  return RelationalAlgebraKernel::I8;
	if(bits <= 16) return RelationalAlgebraKernel::I16;
	if(bits <= 32) return RelationalAlgebraKernel::I32;
	if(bits <= 64) return RelationalAlgebraKernel::I64;
	if(bits <= 128) return RelationalAlgebraKernel::I128;
	if(bits <= 256) return RelationalAlgebraKernel::I256;
	if(bits <= 512) return RelationalAlgebraKernel::I512;
	
	return RelationalAlgebraKernel::I1024;	
}

static unsigned int getTupleDataTypeSize(const RelationalAlgebraKernel::Variable& v)
{
	unsigned int bits = 0;
	for(RelationalAlgebraKernel::Variable::const_iterator
		i = v.begin(); i != v.end(); ++i)
	{
		bits += i->limit;
	}
	
	if(bits <= 8)  return 1;
	if(bits <= 16) return 2;
	if(bits <= 32) return 4;
	if(bits <= 64) return 8;
	if(bits <= 128) return 16;
	if(bits <= 256) return 32;
	if(bits <= 512) return 64;
	
	return 128;	
}

static void mapArithOp(RelationalAlgebraKernel::ArithmeticOp &to, lb::ArithmeticOp from)
{
	switch(from)
	{
	case lb::Add:
	{
		report("+");
		to = RelationalAlgebraKernel::Add;
		break;
	}
	case lb::Subtract:
	{
		report("-");
		to = RelationalAlgebraKernel::Subtract;
		break;
	}
	case lb::Multiply:
	{
		report("*");
		to = RelationalAlgebraKernel::Multiply;
		break;
	}
	case lb::Divide:
	{
		report("/");
		to = RelationalAlgebraKernel::Divide;
		break;
	}
	case lb::Mod:
	{
		report("%");
		to = RelationalAlgebraKernel::Mod;
		break;
	}
	}
}

static void addArithExpNode(RelationalAlgebraKernel::ArithExpNode *parent,
	const lb::Exp& childN, VariableMap::const_iterator& source, bool isLeft)
{
	RelationalAlgebraKernel::ArithExpNode *childNode = 
		new RelationalAlgebraKernel::ArithExpNode;

	if(isLeft)
		parent->left = childNode;
	else
		parent->right = childNode;

	lb::Exp child = childN;

	if(childN.tag() == lb::MIXED) 
	{
		child = childN.mixed().exp1();
	}
	else if(childN.tag() == lb::CONVERT)
	{
		child = childN.convert().exp2().mixed().exp1();
	}

	if(child.tag() == lb::CONSTEXP)
	{
		childNode->kind = RelationalAlgebraKernel::ConstantNode;
		switch(child.constexp().literal().kind())
		{
		case common::Constant_Kind_BOOL:
		{
			childNode->type = RelationalAlgebraKernel::I8;
			childNode->i = child.constexp().literal().bool_constant().value();
			report("constant int " << childNode->i);
			break;
		}
		case common::Constant_Kind_INT:
		{
			childNode->type = RelationalAlgebraKernel::I64;
			childNode->i = child.constexp().literal().int_constant().value();
			report("constant int " << childNode->i);
			break;
		}
		case common::Constant_Kind_UINT:
		{
			childNode->type = RelationalAlgebraKernel::I64;
			childNode->i = child.constexp().literal().uint_constant().value();
			report("constant int " << childNode->i);
			break;
		}
		case common::Constant_Kind_FLOAT:
		{
			childNode->type = RelationalAlgebraKernel::F32;
			childNode->f= child.constexp().literal().float_constant().value();
			report("constant float " << childNode->f);
			break;
		}
		case common::Constant_Kind_STRING:
		{
			assertM(false, "String constants not supported");
			break;
		}
		case common::Constant_Kind_DATETIME:
		{
			childNode->type = RelationalAlgebraKernel::I64;
			childNode->i = child.constexp().literal().date_time_constant().value();
			report("constant date_time_int " << childNode->i);

			break;
		}
		}

		if(isLeft)
			parent->left = childNode;
		else
			parent->right = childNode;
	}
	else if(child.tag() == lb::INDEX)
	{
		childNode->kind = RelationalAlgebraKernel::IndexNode;
		int index = child.index().offset();
		childNode->type = source->second.types[index].type;
		childNode->i = index;
		report("index offset " << childNode->i);
	}
	else if(child.tag() == lb::ARITHEXP)
	{
		childNode->kind = RelationalAlgebraKernel::OperatorNode;
		mapArithOp(childNode->op, child.arithexp().op());

		addArithExpNode(childNode, child.arithexp().exp1(), source, true);
		addArithExpNode(childNode, child.arithexp().exp2(), source, false);
	}
}

static RelationalAlgebraKernel::ArithExpNode translateArith(const lb::Exp& arith,
	VariableMap::const_iterator& source)
{
	report("starting translating arith " << arith.tag());
	RelationalAlgebraKernel::ArithExpNode root;

	if(arith.tag() == lb::CONSTEXP)
	{
		root.kind = RelationalAlgebraKernel::ConstantNode;
	
		switch(arith.constexp().literal().kind())
		{
		case common::Constant_Kind_BOOL:
		{
			root.type = RelationalAlgebraKernel::I8;
			root.i = arith.constexp().literal().bool_constant().value();
			report("constant int " << root.i);
			break;
		}
		case common::Constant_Kind_INT:
		{
			root.type = RelationalAlgebraKernel::I64;
			root.i = arith.constexp().literal().int_constant().value();
			report("constant int " << root.i);
			break;
		}
		case common::Constant_Kind_UINT:
		{
			root.type = RelationalAlgebraKernel::I64;
			root.i = arith.constexp().literal().uint_constant().value();
			report("constant int " << root.i);
			break;
		}
		case common::Constant_Kind_FLOAT:
		{
			root.type = RelationalAlgebraKernel::F32;
			root.f= arith.constexp().literal().float_constant().value();
			report("constant float " << root.f);
			break;
		}
		case common::Constant_Kind_STRING:
		{
			assertM(false, "String constants not supported");
			break;
		}
		case common::Constant_Kind_DATETIME:
		{
			root.type = RelationalAlgebraKernel::I64;
			root.i = arith.constexp().literal().date_time_constant().value();
			report("constant date_time_int " << root.i);

			break;
		}
		}

		return root;
	}

	root.kind = RelationalAlgebraKernel::OperatorNode;
	if(arith.tag() == lb::ARITHEXP)
	{
		report("mapping the root operator");
		mapArithOp(root.op, arith.arithexp().op());
		addArithExpNode(&root, arith.arithexp().exp1(), source, true);
		addArithExpNode(&root, arith.arithexp().exp2(), source, false);
	}
	else if(arith.tag() == lb::CALL)
	{
		if(arith.call().calltype() == lb::FDatetime 
  		  && arith.call().callname().compare("subtract") == 0)
		{
			root.op = RelationalAlgebraKernel::Subtract;
			addArithExpNode(&root, arith.call().args(0), source, true);
			addArithExpNode(&root, arith.call().args(1), source, false);
		}
		else if(arith.call().calltype() == lb::FDatetime 
  		  && arith.call().callname().compare("part") == 0)
		{
			if(arith.call().args(1).constexp().literal().string_constant().value().compare("year") == 0)
			{
				root.op = RelationalAlgebraKernel::PartYear;
				addArithExpNode(&root, arith.call().args(0), source, true);
			}
		}
		else if(arith.call().calltype() == lb::FDatetime 
  		  && arith.call().callname().compare("add") == 0)
		{
			if(arith.call().args(2).constexp().literal().string_constant().value().compare("months") == 0)
				root.op = RelationalAlgebraKernel::AddMonth;
			else if(arith.call().args(2).constexp().literal().string_constant().value().compare("years") == 0)
				root.op = RelationalAlgebraKernel::AddYear;
			else
				root.op = RelationalAlgebraKernel::Add;

			addArithExpNode(&root, arith.call().args(0), source, true);
			addArithExpNode(&root, arith.call().args(1), source, false);
		}
	}

	return root;
}
static RelationalAlgebraKernel::ComparisonVector translateComparison(
	std::vector<lb::Comp>& compares, const VariableMap& variables, 
	const StringIdMap& strings, VariableMap::const_iterator& source)
{
	RelationalAlgebraKernel::ComparisonVector translatedComparisons;
	
	report("Translating comparison:");
	
	RelationalAlgebraKernel::ComparisonExpression expression;

	for(std::vector<lb::Comp>::const_iterator compare_i = compares.begin();
		compare_i != compares.end(); ++compare_i)
	{	
	const lb::Comp compare = *compare_i;
	
//	if(blocknum == 13)
//	expression.comparison = RelationalAlgebraKernel::Lt;
//	else
	if(source->second.types[0].type == RelationalAlgebraKernel::Pointer && compare.comparison() == lb::Eq)
		expression.comparison = RelationalAlgebraKernel::EqString;
	else
	expression.comparison = translate(compare, variables);
	
	if(compare.op1().tag() == lb::INDEX)
	{
		expression.a.type = RelationalAlgebraKernel::VariableIndex;
//		if(blocknum == 13)
//		expression.a.variableIndex = 3;
//		else
		expression.a.variableIndex = compare.op1().index().offset();
		report(" field " << compare.op1().index().offset());
	}
	else if(compare.op1().tag() == lb::CALL)
	{
		expression.a.type = RelationalAlgebraKernel::ConstantArith;
		expression.a.arith = translateArith(compare.op1(), source);
		report(" arith ");
	}
	else if(compare.op1().tag() == lb::ARITHEXP)
	{
		expression.a.type = RelationalAlgebraKernel::ConstantArith;
		expression.a.arith = translateArith(compare.op1(), source);
		report(" arith ");
	}
	else if(compare.op1().tag() == lb::CONSTEXP)
	{
		expression.a.type = RelationalAlgebraKernel::Constant;
		
		switch(compare.op1().constexp().literal().kind())
		{
		case common::Constant_Kind_BOOL:
		{
			expression.a.boolValue = 
				compare.op1().constexp().literal().bool_constant().value();
			report(" bool " << expression.a.boolValue);
			break;
		}
		case common::Constant_Kind_INT:
		{
			expression.a.intValue = 
				compare.op1().constexp().literal().int_constant().value();
			report(" int " << expression.a.intValue);
			break;
		}
		case common::Constant_Kind_UINT:
		{
			expression.a.intValue = 
				compare.op1().constexp().literal().uint_constant().value();
			report(" uint " << expression.a.intValue);
			break;
		}
		case common::Constant_Kind_FLOAT:
		{
			std::stringstream stream(
				compare.op1().constexp().literal().float_constant().value());
			stream >> expression.a.floatValue;
			report(" float " << expression.a.floatValue);
			break;
		}
		case common::Constant_Kind_STRING:
		{
			StringIdMap::const_iterator id = strings.find(
				compare.op1().constexp().literal().string_constant().value());
			assert(id != strings.end());
		
			if(expression.comparison == RelationalAlgebraKernel::Eq)	
				expression.comparison = RelationalAlgebraKernel::EqString;

			expression.a.stringId = id->second;
			report(" string " << id->first << " (" << id->second << ")");
			break;
		}
		case common::Constant_Kind_DATETIME:
		{
			expression.a.intValue =
				compare.op1().constexp().literal().date_time_constant().value();

			report(" date time " << expression.a.intValue);
			break;
		}
		}
	}

	if(compare.op2().tag() == lb::INDEX)
	{
		expression.b.type = RelationalAlgebraKernel::VariableIndex;
//		if(blocknum == 13)
//		expression.b.variableIndex = 5;
//		else
		expression.b.variableIndex = compare.op2().index().offset();
		report(" field " << compare.op2().index().offset());
	}
	else if(compare.op2().tag() == lb::CALL)
	{
		expression.b.type = RelationalAlgebraKernel::ConstantArith;
		expression.b.arith = translateArith(compare.op2(), source);
		report(" arith " << expression.b.arith.kind);
	}
	else if(compare.op2().tag() == lb::ARITHEXP)
	{
		expression.b.type = RelationalAlgebraKernel::ConstantArith;
		expression.b.arith = translateArith(compare.op2(), source);
		report(" arith " << expression.b.arith.op);
	}
	else if(compare.op2().tag() == lb::CONSTEXP)
	{
		expression.b.type = RelationalAlgebraKernel::Constant;
		
		switch(compare.op2().constexp().literal().kind())
		{
		case common::Constant_Kind_BOOL:
		{
			expression.b.boolValue = 
				compare.op2().constexp().literal().bool_constant().value();
			report(" bool " << expression.b.boolValue);
			break;
		}
		case common::Constant_Kind_INT:
		{
			expression.b.intValue = 
				compare.op2().constexp().literal().int_constant().value();
			report(" int " << expression.b.intValue);
			break;
		}
		case common::Constant_Kind_UINT:
		{
			expression.b.intValue = 
				compare.op2().constexp().literal().int_constant().value();
			report(" uint " << expression.b.intValue);
			break;
		}
		case common::Constant_Kind_FLOAT:
		{
			std::stringstream stream(
				compare.op2().constexp().literal().float_constant().value());
			stream >> expression.b.floatValue;
			report(" float " << expression.b.floatValue);
			break;
		}
		case common::Constant_Kind_STRING:
		{
			StringIdMap::const_iterator id = strings.find(
				compare.op2().constexp().literal().string_constant().value());
			assert(id != strings.end());
			
			if(expression.comparison == RelationalAlgebraKernel::Eq)	
				expression.comparison = RelationalAlgebraKernel::EqString;

			expression.b.stringId = id->second;
			report(" string " << id->first << " (" << id->second << ")");
			break;
		}
		case common::Constant_Kind_DATETIME:
		{
			expression.b.intValue =
				compare.op2().constexp().literal().date_time_constant().value();
			report(" date time " << expression.b.intValue);
			break;
		}
		}
	}
	
	translatedComparisons.push_back(expression);
	}

	return translatedComparisons;
}

static RelationalAlgebraKernel::ComparisonVector translateTest(
	const lb::Test& test, const VariableMap& variables, 
	const StringIdMap& strings)
{
	RelationalAlgebraKernel::ComparisonVector translatedComparisons;
	
	report("Translating test:");
	
	RelationalAlgebraKernel::ComparisonExpression expression;

	if(test.testname().compare("like") == 0)	
		expression.comparison = RelationalAlgebraKernel::Like;	
	else if(test.testname().compare("notlike") == 0)	
		expression.comparison = RelationalAlgebraKernel::NotLike;
	else	
		expression.comparison = RelationalAlgebraKernel::InvalidComparison;

	int j = 0;

	for(RepeatedExp::const_iterator 
		i = test.ops().begin();
		i != test.ops().end(); ++i, ++j)
	{
		if(j == 0)
		{
			if(i->tag() == lb::INDEX)
			{
				expression.a.type = RelationalAlgebraKernel::VariableIndex;
				expression.a.variableIndex = i->index().offset();
				report(" field " << i->index().offset());
			}
			else
			{
				expression.a.type = RelationalAlgebraKernel::Constant;
				
				switch(test.ops(0).constexp().literal().kind())
				{
				case common::Constant_Kind_BOOL:
				{
					expression.a.boolValue = 
						test.ops(0).constexp().literal().bool_constant().value();
					report(" bool " << expression.a.boolValue);
					break;
				}
				case common::Constant_Kind_INT:
				{
					expression.a.intValue = 
						test.ops(0).constexp().literal().int_constant().value();
					report(" int " << expression.a.intValue);
					break;
				}
				case common::Constant_Kind_UINT:
				{
					expression.a.intValue = 
						test.ops(0).constexp().literal().uint_constant().value();
					report(" uint " << expression.a.intValue);
					break;
				}
				case common::Constant_Kind_FLOAT:
				{
					std::stringstream stream(
						test.ops(0).constexp().literal().float_constant().value());
					stream >> expression.a.floatValue;
					report(" float " << expression.a.floatValue);
					break;
				}
				case common::Constant_Kind_STRING:
				{
					StringIdMap::const_iterator id = strings.find(
						test.ops(0).constexp().literal().string_constant().value());
					assert(id != strings.end());
				
					if(expression.comparison == RelationalAlgebraKernel::Eq)	
						expression.comparison = RelationalAlgebraKernel::EqString;
		
					expression.a.stringId = id->second;
					report(" string " << id->first << " (" << id->second << ")");
					break;
				}
				case common::Constant_Kind_DATETIME:
				{
					expression.a.intValue =
						test.ops(0).constexp().literal().date_time_constant().value();
		
					report(" date time " << expression.a.intValue);
					break;
				}
				}
			}
		}	
		else if(j == 1)
		{
			if(i->tag() == lb::INDEX)
			{
				expression.b.type = RelationalAlgebraKernel::VariableIndex;
				expression.b.variableIndex = i->index().offset();
				report(" field " << i->index().offset());
			}
			else
			{
				expression.b.type = RelationalAlgebraKernel::Constant;
				
				switch(test.ops(1).constexp().literal().kind())
				{
				case common::Constant_Kind_BOOL:
				{
					expression.b.boolValue = 
						test.ops(1).constexp().literal().bool_constant().value();
					report(" bool " << expression.b.boolValue);
					break;
				}
				case common::Constant_Kind_INT:
				{
					expression.b.intValue = 
						test.ops(1).constexp().literal().int_constant().value();
					report(" int " << expression.b.intValue);
					break;
				}
				case common::Constant_Kind_UINT:
				{
					expression.b.intValue = 
						test.ops(1).constexp().literal().int_constant().value();
					report(" uint " << expression.b.intValue);
					break;
				}
				case common::Constant_Kind_FLOAT:
				{
					std::stringstream stream(
						test.ops(1).constexp().literal().float_constant().value());
					stream >> expression.b.floatValue;
					report(" float " << expression.b.floatValue);
					break;
				}
				case common::Constant_Kind_STRING:
				{
					StringIdMap::const_iterator id = strings.find(
						test.ops(1).constexp().literal().string_constant().value());
					assert(id != strings.end());
					
					if(expression.comparison == RelationalAlgebraKernel::Eq)	
						expression.comparison = RelationalAlgebraKernel::EqString;
		
					expression.b.stringId = id->second;
					report(" string " << id->first << " (" << id->second << ")");
					break;
				}
				case common::Constant_Kind_DATETIME:
				{
					expression.b.intValue =
						test.ops(1).constexp().literal().date_time_constant().value();
					report(" date time " << expression.b.intValue);
					break;
				}
				}
			}
		}
	}
	
	translatedComparisons.push_back(expression);
	
	return translatedComparisons;
}

static VariableMap::iterator getTempBuffer(VariableMap& v,
	hir::pb::KernelControlFlowGraph& cfg, unsigned int bytes,
	const RelationalAlgebraKernel::Variable& types =
	RelationalAlgebraKernel::Variable())
{
	std::stringstream name;
	
	name << "_ZHarmonyTempVariable_" << v.size();
	
	VariableMap::iterator i = v.insert(std::make_pair(name.str(),
		VariableDescriptor(v.size()))).first;
	
	i->second.types = types;

	hir::pb::Variable& harmonyVariable = *cfg.add_variables();
	
	harmonyVariable.set_name(i->second.id);
	harmonyVariable.set_input(false);
	harmonyVariable.set_output(false);
	harmonyVariable.set_type(hir::pb::I8);
	harmonyVariable.set_size(bytes);
	harmonyVariable.mutable_data()->resize(bytes, '\0');
	
	return i;
}

static VariableMap::iterator getTempBufferByName(VariableMap& v,
	hir::pb::KernelControlFlowGraph& cfg, unsigned int bytes,
	unsigned int unique_key_num, unsigned int isSorted, std::vector<unsigned int> sorted_fields, std::string name,
	const RelationalAlgebraKernel::Variable& types =
	RelationalAlgebraKernel::Variable())
{
	VariableMap::iterator i = v.insert(std::make_pair(name,
		VariableDescriptor(v.size()))).first;
	
	i->second.types = types;
	i->second.unique_keys = unique_key_num;
	i->second.isSorted = isSorted;
	i->second.sorted_fields = sorted_fields;

	hir::pb::Variable& harmonyVariable = *cfg.add_variables();
	harmonyVariable.set_name(i->second.id);
	harmonyVariable.set_type(hir::pb::I8);
	harmonyVariable.set_size(bytes);
	harmonyVariable.set_input(false);
	harmonyVariable.set_output(false);

	harmonyVariable.mutable_data()->resize(bytes, '\0');
	
	return i;
}

static VariableMap::iterator getOutputStringBuffer(VariableMap& v,
	hir::pb::KernelControlFlowGraph& cfg, std::string name)
{
	VariableMap::iterator i = v.insert(std::make_pair(name,
		VariableDescriptor(v.size()))).first;
	
	i->second.types = RelationalAlgebraKernel::Variable();
	i->second.unique_keys = 0;

	hir::pb::Variable& harmonyVariable = *cfg.add_variables();
	
	harmonyVariable.set_name(i->second.id);
	harmonyVariable.set_type(hir::pb::I8);
	harmonyVariable.set_size(128*150000);
	harmonyVariable.set_input(true);
	harmonyVariable.set_output(true);

	harmonyVariable.mutable_data()->resize(128*150000, '\0');
	
	return i;
}

static VariableMap::iterator getTempIntVariable(VariableMap& v,
	hir::pb::KernelControlFlowGraph& cfg)
{
	std::stringstream name;
	
	name << "_ZHarmonyTempVariable_" << v.size();
	
	VariableMap::iterator i = v.insert(std::make_pair(name.str(),
		VariableDescriptor(v.size()))).first;
	
	i->second.types.push_back(
		RelationalAlgebraKernel::Element(RelationalAlgebraKernel::I64));

	hir::pb::Variable& harmonyVariable = *cfg.add_variables();
	
	harmonyVariable.set_name(i->second.id);
	harmonyVariable.set_type(hir::pb::I64);
	harmonyVariable.set_size(8);
	harmonyVariable.set_input(false);
	harmonyVariable.set_output(false);
	
	long long unsigned int value = 0;
	harmonyVariable.set_data(&value, sizeof(long long unsigned int));

	return i;
}

static VariableMap::iterator getConstantIntVariable(VariableMap& v,
	hir::pb::KernelControlFlowGraph& cfg, long long unsigned int value)
{
	typedef std::pair<VariableMap::iterator, bool> VariableInsertion;
	std::stringstream name;
	
	name << "_ZHarmony_Constant_U64_" << value;
	
	VariableInsertion i = v.insert(std::make_pair(name.str(),
		VariableDescriptor(v.size())));
	
	if(i.second)
	{
		i.first->second.types.push_back(RelationalAlgebraKernel::Element(
			RelationalAlgebraKernel::I32));

		hir::pb::Variable& harmonyVariable = *cfg.add_variables();
	
		harmonyVariable.set_name(i.first->second.id);
		harmonyVariable.set_input(true);
		harmonyVariable.set_output(false);
		harmonyVariable.set_type(hir::pb::I64);
		harmonyVariable.set_size(8);
	
		harmonyVariable.set_data(&value, sizeof(long long unsigned int));
	}
	
	return i.first;
}

/*! \brief A dummy class needed to represent a 128 bit int */
class Packed16Bytes
{
public:
	typedef long long unsigned int type;

public:
	type a;
	type b;

	const bool operator<(const Packed16Bytes& value) const
	{
		if(b < value.b)
		{
			return true;
		}
		else if(b == value.b && a < value.a)
		{
			return true;
		}
		
		return false;
	}
};

typedef std::vector<long long unsigned int> IntList;

static void appendData(IntList& output, 
	unsigned int& currentBits, long long unsigned int data, unsigned int bits)
{

	if(currentBits / 64 != (currentBits + bits) / 64)
	{
		output.push_back(0);
	}

	for(IntList::reverse_iterator word = output.rbegin(); 
		word != --output.rend(); ++word)
	{
		IntList::reverse_iterator next = word;
		++next;
		
		unsigned int shift = 64 - bits;
		
		*word = (*word << bits) | (*next >> shift);
	}
	
	output[0] = (output[0] << bits) | data;

	currentBits += bits;
}

static VariableMap::iterator getConstantVariable(VariableMap& v,
	hir::pb::KernelControlFlowGraph& cfg,
	const RelationalAlgebraKernel::Value& value)
{
	typedef std::pair<VariableMap::iterator, bool> VariableInsertion;
	//typedef std::vector<char> DataVector;
	std::stringstream name;
	
	name << "_ZHarmony_Constant_Value_" << v.size();
	
	VariableInsertion i = v.insert(std::make_pair(name.str(),
		VariableDescriptor(v.size())));
	
	if(i.second)
	{
		std::stringstream data;
		
		IntList tupleData(1, 0);
		unsigned int currentBits = 0;

		for(RelationalAlgebraKernel::Value::const_iterator 
			element = value.begin(); element != value.end(); ++element)
		{
			i.first->second.types.push_back(element->type);
			
			appendData(tupleData, currentBits,
				element->intValue, element->limit);
		}
		
		hir::pb::Variable& harmonyVariable = *cfg.add_variables();
	
		harmonyVariable.set_name(i.first->second.id);
		harmonyVariable.set_input(true);
		harmonyVariable.set_output(false);

		if(currentBits > 0)
		{
			unsigned int wordSize   = 0;
			unsigned int totalBytes = 0;
		
			if(currentBits <= 8)  wordSize = 1;
			else if(currentBits <= 16) wordSize = 2;
			else if(currentBits <= 32) wordSize = 4;
			else                       wordSize = 8;
			
			totalBytes = 0;

			for(IntList::const_iterator word = tupleData.begin();
				word != tupleData.end(); ++word, totalBytes += wordSize)
			{
				data.write((char*)&*word, wordSize);
			}
	
			harmonyVariable.set_type(hir::pb::I8);
			harmonyVariable.set_size(totalBytes);
		
			report("   Added constant tuple data '" << tupleData[0] << "' with "
				<< currentBits << " bits " << totalBytes
				<< " bytes and wordsize " << wordSize << ".");
		}
		
		harmonyVariable.set_data(data.str().data(), data.str().size());
	}
	
	return i.first;
}
#if 0
static void buildStringTable(StringIdMap& strings, const RepeatedType& key_types, const RepeatedType& types,
	const std::string& inputData)
{
	std::stringstream input(inputData);
	
	while(input.good())
	{
		for(RepeatedType::const_iterator type = key_types.begin();
			type != key_types.end(); ++type)
		{
			const common::PrimitiveType& primitive = type->primitive();
		
			switch(primitive.kind())
			{
			case common::PrimitiveType_Kind_BOOL:
			{
				input.seekg(1, std::ios::cur);
				break;
			}
			case common::PrimitiveType_Kind_COLOR:
			{
				assertM(false, "COLOR not supported");
				break;
			}
			case common::PrimitiveType_Kind_SHAPE:
			{
				assertM(false, "SHAPE not supported");
				break;
			}
			case common::PrimitiveType_Kind_IMAGE:
			{
				assertM(false, "IMAGE not supported");
				break;
			}
			case common::PrimitiveType_Kind_BLOB:
			{
				assertM(false, "BLOB not supported");
				break;
			}
			case common::PrimitiveType_Kind_DATETIME:
			{
//				assertM(false, "DATETIME not supported");
				input.seekg(8, std::ios::cur);
				report("Seeking ahead by 8");
				break;
			}
			case common::PrimitiveType_Kind_DECIMAL:
			{
				assertM(false, "DECIMAL not supported");
				break;
			}
			case common::PrimitiveType_Kind_FLOAT:
			{
				input.seekg(8, std::ios::cur);
				report("Seeking ahead by 8");
				break;
			}
			case common::PrimitiveType_Kind_UINT:
			case common::PrimitiveType_Kind_INT:
			{
				input.seekg(8, std::ios::cur);
				report("Seeking ahead by " << 8);
				break;
			}
			case common::PrimitiveType_Kind_STRING:
			{
				long long unsigned int length = 0;
				input.read((char*)&length, sizeof(long long unsigned int));
				
				length = byteReverse(length);
				
				if(length == 0) continue;

				std::string buffer(length, ' ');
				input.read((char*)buffer.data(), length);
								
				StringIdMap::const_iterator id = strings.insert(
					std::make_pair(buffer, strings.size())).first;

				report(" Mapping string '" << buffer
					<< "' to id " << id->second);
				break;
			}
			}
		}

		for(RepeatedType::const_iterator type = types.begin();
			type != types.end(); ++type)
		{
			const common::PrimitiveType& primitive = type->primitive();
		
			switch(primitive.kind())
			{
			case common::PrimitiveType_Kind_BOOL:
			{
				input.seekg(1, std::ios::cur);
				break;
			}
			case common::PrimitiveType_Kind_COLOR:
			{
				assertM(false, "COLOR not supported");
				break;
			}
			case common::PrimitiveType_Kind_SHAPE:
			{
				assertM(false, "SHAPE not supported");
				break;
			}
			case common::PrimitiveType_Kind_IMAGE:
			{
				assertM(false, "IMAGE not supported");
				break;
			}
			case common::PrimitiveType_Kind_BLOB:
			{
				assertM(false, "BLOB not supported");
				break;
			}
			case common::PrimitiveType_Kind_DATETIME:
			{
//				assertM(false, "DATETIME not supported");
				input.seekg(8, std::ios::cur);
				report("Seeking ahead by 8");
				break;
			}
			case common::PrimitiveType_Kind_DECIMAL:
			{
				assertM(false, "DECIMAL not supported");
				break;
			}
			case common::PrimitiveType_Kind_FLOAT:
			{
				input.seekg(8, std::ios::cur);
				report("Seeking ahead by 8");
				break;
			}
			case common::PrimitiveType_Kind_UINT:
			case common::PrimitiveType_Kind_INT:
			{
				input.seekg(8, std::ios::cur);
				report("Seeking ahead by " << 8);
				break;
			}
			case common::PrimitiveType_Kind_STRING:
			{
				long long unsigned int length = 0;
				input.read((char*)&length, sizeof(long long unsigned int));
				
				length = byteReverse(length);
				
				if(length == 0) continue;

				std::string buffer(length, ' ');
				input.read((char*)buffer.data(), length);
								
				StringIdMap::const_iterator id = strings.insert(
					std::make_pair(buffer, strings.size())).first;

				report(" Mapping string '" << buffer
					<< "' to id " << id->second);
				break;
			}
			}
		}
	}
}
#endif
static void buildStringTable(StringIdMap& strings, VariableMap& variables, 
	hir::pb::KernelControlFlowGraph& cfg)
{
	hir::pb::Variable& harmonyVariable = *cfg.add_variables();
	
	VariableMap::iterator v = variables.insert(std::make_pair(
		"StringTable", VariableDescriptor(variables.size(), 0))).first;

	harmonyVariable.set_name(v->second.id);
	harmonyVariable.set_input(true);
	harmonyVariable.set_output(false);
	harmonyVariable.set_size(strings.size() * 128);
	harmonyVariable.set_type(hir::pb::I8);

	char *data = (char *)malloc(128 * strings.size());

	for(StringIdMap::const_iterator i = strings.begin(); i != strings.end(); ++i)
	{
		strcpy(data + 128 * i->second, i->first.c_str());
	}

	harmonyVariable.set_data(data, strings.size() * 128);
}

static void addVariables(StringIdMap& stringIds, VariableMap& variables,
	hir::pb::KernelControlFlowGraph& cfg, const lb::GPUGraph& raGraph)
{
	// do a quick scan to add any variables with initial string data
//	for(int i = 0; i != raGraph.variables_size(); ++i)
//	{
//		const lb::GPUVariable& variable = raGraph.variables(i);
//
//		buildStringTable(stringIds, variable.keys(), variable.fields(), 
//			variable.initialdata());
//	}
	if(stringIds.size() > 0)
		buildStringTable(stringIds, variables, cfg);

	// Add the variables
	for(int i = 0; i != raGraph.variables_size(); ++i)
	{
		const lb::GPUVariable& variable = raGraph.variables(i);
	
		hir::pb::Variable& harmonyVariable = *cfg.add_variables();
		
		assert(variables.count(variable.varname()) == 0);
		
		report(" Adding variable " << variable.varname() << " " << variable.keys_size());
		
//		VariableMap::iterator v = variables.insert(std::make_pair(
//			variable.varname(), VariableDescriptor(i, variable.keys_size()))).first;
		VariableMap::iterator v = variables.insert(std::make_pair(
			variable.varname(), VariableDescriptor(variables.size(), variable.keys_size()))).first;


		for(int t = 0; t != variable.keys_size(); ++t)
		{
//			if(isupper(variable.varname().c_str()[0]) || variable.varname().c_str()[0] == '+')
			{
//				assert(variable.fields(t).kind() != common::Type_Kind_UNARY);
				if(variable.keys(t).kind() == common::Type_Kind_PRIMITIVE)
				{
					const common::PrimitiveType& primitive = variable.keys(t).primitive();
		
					switch(primitive.kind())
					{
					case common::PrimitiveType_Kind_BOOL:
					{
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I8, 8));
						break;
					}
					case common::PrimitiveType_Kind_COLOR:
					{
						assertM(false, "COLOR not supported");
						break;
					}
					case common::PrimitiveType_Kind_SHAPE:
					{
						assertM(false, "SHAPE not supported");
						break;
					}
					case common::PrimitiveType_Kind_IMAGE:
					{
						assertM(false, "IMAGE not supported");
						break;
					}
					case common::PrimitiveType_Kind_BLOB:
					{
						assertM(false, "BLOB not supported");
						break;
					}
					case common::PrimitiveType_Kind_DATETIME:
					{
		//				assertM(false, "DATETIME not supported");
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I64, 64));
		
						break;
					}
					case common::PrimitiveType_Kind_DECIMAL:
					{
						assertM(false, "DECIMAL not supported");
						break;
					}
					case common::PrimitiveType_Kind_UINT:
					case common::PrimitiveType_Kind_INT:
					{
						if(primitive.capacity() == 8)
						{
							v->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::I8, 8));
						}
						else if(primitive.capacity() == 16)
						{
							v->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::I16, 16));
						}
						else if(primitive.capacity() == 32)
						{
							v->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::I32, 32));
						}
						else if(primitive.capacity() == 64)
						{
							v->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::I64, 64));
						}
						else
						{
							assertM(false, "Invalid int size " << primitive.capacity());
						}
						break;
					}
					case common::PrimitiveType_Kind_FLOAT:
					{
						if(primitive.capacity() == 32)
						{
							v->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::F32, 32));
						}
						else if(primitive.capacity() == 64)
						{
							v->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::F64, 64));
						}
						else
						{
							assertM(false, "Invalid float size "
								<< primitive.capacity());
						}				
						break;
					}
					case common::PrimitiveType_Kind_STRING:
					{
		//				v->second.types.push_back(RelationalAlgebraKernel::Element(
		//					RelationalAlgebraKernel::I32,
		//					log2(hydrazine::nextPowerOfTwo(stringIds.size()))));
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::Pointer, 64));
		
						break;
					}
					}
				}
				else if(variable.keys(t).kind() == common::Type_Kind_UNARY)
				{
					if(variable.keys(t).unary().name().compare("RETURNFLAG") == 0)
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I8, flag_bits));
					else if(variable.keys(t).unary().name().compare("LINESTATUS") == 0)
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I8, status_bits));
					else if(variable.keys(t).unary().name().compare("LINENUMBER") == 0)
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I8, line_bits));
					else if(variable.keys(t).unary().name().compare("ORDER") == 0)
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I32, order_bits));
					else if(variable.keys(t).unary().name().compare("SUPPLIER") == 0)
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I16, supp_bits));
					else if(variable.keys(t).unary().name().compare("PART") == 0)
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I32, part_bits));
					else if(variable.keys(t).unary().name().compare("REGION") == 0)
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I8, region_bits));
					else if(variable.keys(t).unary().name().compare("NATION") == 0)
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I8, nation_bits));
					else if(variable.keys(t).unary().name().compare("CUSTOMER") == 0)
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I32, cust_bits));
				}
			}
		}

		for(int t = 0; t != variable.fields_size(); ++t)
		{
//			if(isupper(variable.varname().c_str()[0]) || variable.varname().c_str()[0] == '+')
			{
//				assert(variable.fields(t).kind() != common::Type_Kind_UNARY);
				if(variable.fields(t).kind() == common::Type_Kind_PRIMITIVE)
				{
					const common::PrimitiveType& primitive = variable.fields(t).primitive();
		
					switch(primitive.kind())
					{
					case common::PrimitiveType_Kind_BOOL:
					{
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I8, 8));
						break;
					}
					case common::PrimitiveType_Kind_COLOR:
					{
						assertM(false, "COLOR not supported");
						break;
					}
					case common::PrimitiveType_Kind_SHAPE:
					{
						assertM(false, "SHAPE not supported");
						break;
					}
					case common::PrimitiveType_Kind_IMAGE:
					{
						assertM(false, "IMAGE not supported");
						break;
					}
					case common::PrimitiveType_Kind_BLOB:
					{
						assertM(false, "BLOB not supported");
						break;
					}
					case common::PrimitiveType_Kind_DATETIME:
					{
		//				assertM(false, "DATETIME not supported");
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I64, 64));
		
						break;
					}
					case common::PrimitiveType_Kind_DECIMAL:
					{
						assertM(false, "DECIMAL not supported");
						break;
					}
					case common::PrimitiveType_Kind_UINT:
					case common::PrimitiveType_Kind_INT:
					{
						if(primitive.capacity() == 8)
						{
							v->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::I8, 8));
						}
						else if(primitive.capacity() == 16)
						{
							v->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::I16, 16));
						}
						else if(primitive.capacity() == 32)
						{
							v->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::I32, 32));
						}
						else if(primitive.capacity() == 64)
						{
							v->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::I64, 64));
						}
						else
						{
							assertM(false, "Invalid int size " << primitive.capacity());
						}
						break;
					}
					case common::PrimitiveType_Kind_FLOAT:
					{
						if(primitive.capacity() == 32)
						{
							v->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::F32, 32));
						}
						else if(primitive.capacity() == 64)
						{
							v->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::F64, 64));
						}
						else
						{
							assertM(false, "Invalid float size "
								<< primitive.capacity());
						}				
						break;
					}
					case common::PrimitiveType_Kind_STRING:
					{
		//				v->second.types.push_back(RelationalAlgebraKernel::Element(
		//					RelationalAlgebraKernel::I32,
		//					log2(hydrazine::nextPowerOfTwo(stringIds.size()))));
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::Pointer, 64));
		
						break;
					}
					}
				}
				else if(variable.fields(t).kind() == common::Type_Kind_UNARY)
				{
					if(variable.fields(t).unary().name().compare("RETURNFLAG") == 0)
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I8, 2));
					else if(variable.fields(t).unary().name().compare("LINESTATUS") == 0)
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I8, 1));
					else if(variable.fields(t).unary().name().compare("LINENUMBER") == 0)
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I8, 3));
					else if(variable.fields(t).unary().name().compare("ORDER") == 0)
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I32, 23));
					else if(variable.fields(t).unary().name().compare("SUPPLIER") == 0)
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I16, 14));
					else if(variable.fields(t).unary().name().compare("PART") == 0)
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I32, 18));
					else if(variable.fields(t).unary().name().compare("REGION") == 0)
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I8, 3));
					else if(variable.fields(t).unary().name().compare("NATION") == 0)
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I8, 5));
					else if(variable.fields(t).unary().name().compare("CUSTOMER") == 0)
						v->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I32, 18));
				}
			}
		}

		report(variable.varname() << " " << v->second.types.size());

//		harmonyVariable.set_name(i);
		harmonyVariable.set_name(v->second.id);

		if(isupper(variable.varname().c_str()[0]))
		{
			harmonyVariable.set_input(true);
			harmonyVariable.set_output(false);
			std::string filename = variable.varname();
			harmonyVariable.set_filename("../TPC-H/data/" + filename);
			harmonyVariable.set_type(hir::pb::I8);

			if(variable.varname().compare(0, 2, "L_") == 0)
			{
				v->second.isSorted = 2;
				v->second.sorted_fields.push_back(0);
				v->second.sorted_fields.push_back(1);
//				v->second.sorted_fields.push_back(2);
				harmonyVariable.set_size(LINEITEM_NUM * getTupleDataTypeSize(v->second.types));
			}
			else if(variable.varname().compare("A") == 0)
			{
				//v->second.isSorted = 1;
				//v->second.sorted_fields.push_back(0);
//				v->second.sorted_fields.push_back(1);
			        printf("[A] TC size %d size %d\n", TC_NUM, getTupleDataTypeSize(v->second.types));
				harmonyVariable.set_size(TC_NUM * getTupleDataTypeSize(v->second.types));
			}
			else if(variable.varname().compare("B") == 0)
			{
				//v->second.isSorted = 1;
				//v->second.sorted_fields.push_back(0);
//				v->second.sorted_fields.push_back(1);
				harmonyVariable.set_input(false);
				harmonyVariable.set_output(true);
			        printf("[B] TC size %d size %d\n", TC_NUM, getTupleDataTypeSize(v->second.types));
				harmonyVariable.set_size(TC_NUM * getTupleDataTypeSize(v->second.types));
			}
			else if(variable.varname().compare(0, 2, "N_") == 0)
			{
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);
//				v->second.sorted_fields.push_back(1);
				harmonyVariable.set_size(NATION_NUM * getTupleDataTypeSize(v->second.types));
			}
			else if(variable.varname().compare("PART_SUPPLIER") == 0)
			{
				v->second.isSorted = 2;
				v->second.sorted_fields.push_back(0);
				v->second.sorted_fields.push_back(1);
				harmonyVariable.set_size(PARTSUPP_NUM * getTupleDataTypeSize(v->second.types));
			}
			else if(variable.varname().compare(0, 3, "PS_") == 0)
			{
				v->second.isSorted = 2;
				v->second.sorted_fields.push_back(0);
				v->second.sorted_fields.push_back(1);
//				v->second.sorted_fields.push_back(2);
				harmonyVariable.set_size(PARTSUPP_NUM * getTupleDataTypeSize(v->second.types));
			}
			else if(variable.varname().compare(0, 2, "P_") == 0)
			{
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);
//				v->second.sorted_fields.push_back(1);
				harmonyVariable.set_size(PART_NUM * getTupleDataTypeSize(v->second.types));
			}
			else if(variable.varname().compare("PART") == 0)
			{
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);
				harmonyVariable.set_size(PART_NUM * getTupleDataTypeSize(v->second.types));
			}

			else if(variable.varname().compare(0, 2, "R_") == 0)
			{
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);
//				v->second.sorted_fields.push_back(1);
				harmonyVariable.set_size(REGION_NUM * getTupleDataTypeSize(v->second.types));
			}
			else if(variable.varname().compare(0, 2, "S_") == 0)
			{
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);
//				v->second.sorted_fields.push_back(1);
				harmonyVariable.set_size(SUPPLIER_NUM * getTupleDataTypeSize(v->second.types));
			}
			else if(variable.varname().compare(0, 2, "O_") == 0)
			{
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);
//				v->second.sorted_fields.push_back(1);
				harmonyVariable.set_size(ORDERS_NUM * getTupleDataTypeSize(v->second.types));
			}
			else if(variable.varname().compare("ORDER") == 0)
			{
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);
				harmonyVariable.set_size(ORDERS_NUM * getTupleDataTypeSize(v->second.types));
			}
			else if(variable.varname().compare(0, 2, "C_") == 0)
			{
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);
//				v->second.sorted_fields.push_back(1);
				harmonyVariable.set_size(CUSTOMER_NUM * getTupleDataTypeSize(v->second.types));
			}
			else if(variable.varname().compare(0, 3, "RF_") == 0)
			{
				v->second.isSorted = 1; 
				v->second.sorted_fields.push_back(0);
//				v->second.sorted_fields.push_back(1);
				harmonyVariable.set_size(3 * getTupleDataTypeSize(v->second.types));
			}

			if(variable.varname().compare("N_NAME") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(375);
			}
			if(variable.varname().compare("N_NAME_COPY") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(false);
				harmonyVariable_string.set_output(true);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(375);
			}
			else if(variable.varname().compare("P_MFGR") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(3000000);
			}
			else if(variable.varname().compare("P_TYPE") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(4800);
			}
			else if(variable.varname().compare("R_NAME") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(60);
			}
			else if(variable.varname().compare("S_ADDRESS") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(640000);
			}
			else if(variable.varname().compare("S_COMMENT") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(1280000);
			}
			else if(variable.varname().compare("S_NAME") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(190000);
			}
			else if(variable.varname().compare("S_PHONE") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(160000);
			}
			else if(variable.varname().compare("C_MKTSEGMENT") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(55);
			}
			else if(variable.varname().compare("O_ORDERPRIORITY") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(80);
			}
			else if(variable.varname().compare("P_NAME") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(12800000);
			}
			else if(variable.varname().compare("C_COMMENT") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(19200000);
			}
			else if(variable.varname().compare("C_PHONE") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(2400000);
			}
			else if(variable.varname().compare("C_ADDRESS") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(9600000);
			}
			else if(variable.varname().compare("RF_NAME") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(6);
			}
			else if(variable.varname().compare("C_NAME") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(2850000);
			}
			else if(variable.varname().compare("L_SHIPMODE") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(56);
			}
			else if(variable.varname().compare("O_COMMENT") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(192000000);
			}
			else if(variable.varname().compare("P_BRAND") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(405);
			}
			else if(variable.varname().compare("P_CONTAINER") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(3200000);
			}
			else if(variable.varname().compare("L_SHIPINSTRUCT") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(72);
			}
			else if(variable.varname().compare("O_ORDERSTATUS") == 0)
			{
				std::string filename_string = variable.varname() + "_STRING";
				VariableMap::iterator var = variables.insert
					(std::make_pair(filename_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_filename("../TPC-H/data/" + filename_string);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(4);
			}
		}
		else if(variable.varname().c_str()[0] == '+')
		{
			if(variable.varname().compare("+$logicQ1:_dateDelta") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 7776000;
				harmonyVariable.set_data(&data, sizeof(long long unsigned int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ2:_regionName") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[7] = "EUROPE";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 7);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(7);
			}
			else if(variable.varname().compare("+$logicQ2:_size") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned int data = 15;
				harmonyVariable.set_data(&data, sizeof(unsigned int));
				harmonyVariable.set_type(hir::pb::I32);
				harmonyVariable.set_size(4);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ2:_type") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, 8);
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[7] = "%BRASS";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 7);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(7);
			}
			else if(variable.varname().compare("+$logicQ3:_segment") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, 8);
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[9] = "BUILDING";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 9);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(9);
			}
			else if(variable.varname().compare("+$logicQ3:_date") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 795225600;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ4:_date") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 741484800;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ5:_date") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 757382400;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ5:_regionName") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[5] = "ASIA";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 5);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(5);
			}
			else if(variable.varname().compare("+$logicQ6:_date") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 757382400;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ6:_discount") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				double data = 0.06;
				harmonyVariable.set_data(&data, sizeof(double));
				harmonyVariable.set_type(hir::pb::F64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ6:_quantity") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				double data = 24;
				harmonyVariable.set_data(&data, sizeof(double));
				harmonyVariable.set_type(hir::pb::F64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ7:_nation1") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[8] = "GERMANY";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 8);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(8);
			}
			else if(variable.varname().compare("+$logicQ7:_nation2") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[7] = "FRANCE";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 7);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(7);
			}
			else if(variable.varname().compare("+$logicQ8:_type") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, 8);
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[23] = "ECONOMY ANODIZED STEEL";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 23);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(23);
			}
			else if(variable.varname().compare("+$logicQ8:_nation") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[7] = "BRAZIL";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 7);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(7);
			}
			else if(variable.varname().compare("+$logicQ8:_region") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[8] = "AMERICA";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 8);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(8);
			}
			else if(variable.varname().compare("+$logicQ9:_color") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[8] = "%green%";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 8);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(8);
			}
			else if(variable.varname().compare("+$logicQ10:_date") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 749433600;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ11:_nation") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[8] = "GERMANY";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 8);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(8);
			}
			else if(variable.varname().compare("+$logicQ11:_fraction") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				double data = 0.0001;
				harmonyVariable.set_data(&data, sizeof(double));
				harmonyVariable.set_type(hir::pb::F64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ12:_shipmodeIds") == 0)
			{
				v->second.isConstant = false;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data[2] = {0, 8};
				harmonyVariable.set_data(data, 2 * sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(16);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[16];
				string[0] = 'M';
				string[1] = 'A';
				string[2] = 'I';
				string[3] = 'L';
				string[4] = '\0';
				string[5] = '\0';
				string[6] = '\0';
				string[7] = '\0';
				string[8] = 'S';
				string[9] = 'H';
				string[10] = 'I';
				string[11] = 'P';
				string[12] = '\0';
				string[13] = '\0';
				string[14] = '\0';
				string[15] = '\0';

				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 16);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(16);
			}
			else if(variable.varname().compare("+$logicQ12:_date") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 757382400;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ13:_word1") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[8] = "special";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 8);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(8);
			}
			else if(variable.varname().compare("+$logicQ13:_word2") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[9] = "requests";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 9);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(9);
			}
			else if(variable.varname().compare("+$logicQ14:_date") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 809913600;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ15:_date") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 820454400;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ16:_brand") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[9] = "Brand#45";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 9);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(9);
			}
			else if(variable.varname().compare("+$logicQ16:_type") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[17] = "MEDIUM POLISHED%";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 17);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(17);
			}
			else if(variable.varname().compare("+$logicQ16:_size") == 0)
			{
				v->second.isConstant = false;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned int data[8] = {3, 9, 14, 19, 23, 36, 45, 49};

				harmonyVariable.set_data(&data, 32);
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(32);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ17:_brand") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[9] = "Brand#23";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 9);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(9);
			}
			else if(variable.varname().compare("+$logicQ17:_container") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[8] = "MED BOX";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 8);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(8);
			}
			else if(variable.varname().compare("+$logicQ18:_quantity") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				double data = 300.0f;

				harmonyVariable.set_data(&data, 8);
				harmonyVariable.set_type(hir::pb::F64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ19:_quantity1") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				double data = 1.0f;

				harmonyVariable.set_data(&data, 8);
				harmonyVariable.set_type(hir::pb::F64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ19:_quantity2") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				double data = 10.0f;

				harmonyVariable.set_data(&data, 8);
				harmonyVariable.set_type(hir::pb::F64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ19:_quantity3") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				double data = 20.0f;

				harmonyVariable.set_data(&data, 8);
				harmonyVariable.set_type(hir::pb::F64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ19:_brand1") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[9] = "Brand#12";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 9);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(9);
			}
			else if(variable.varname().compare("+$logicQ19:_brand2") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[9] = "Brand#23";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 9);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(9);
			}
			else if(variable.varname().compare("+$logicQ19:_brand3") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[9] = "Brand#34";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 9);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(9);
			}
			else if(variable.varname().compare("+$logicQ20:_color") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[8] = "forest%";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 8);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(8);
			}
			else if(variable.varname().compare("+$logicQ20:_date") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 757382400;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);
			}
			else if(variable.varname().compare("+$logicQ20:_nation") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[7] = "CANADA";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 7);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(7);
			}
			else if(variable.varname().compare("+$logicQ21:_nation") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);

				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[13] = "SAUDI ARABIA";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 13);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(13);
			}
			else if(variable.varname().compare("+$logicQ22:_I1") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);
	
				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[3] = "13";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 3);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(3);
			}
			else if(variable.varname().compare("+$logicQ22:_I2") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);
	
				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[3] = "31";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 3);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(3);
			}
			else if(variable.varname().compare("+$logicQ22:_I3") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);
	
				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[3] = "23";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 3);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(3);
			}
			else if(variable.varname().compare("+$logicQ22:_I4") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);
	
				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[3] = "29";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 3);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(3);
			}
			else if(variable.varname().compare("+$logicQ22:_I5") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);
	
				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[3] = "30";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 3);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(3);
			}
			else if(variable.varname().compare("+$logicQ22:_I6") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);
	
				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[3] = "18";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 3);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(3);
			}
			else if(variable.varname().compare("+$logicQ22:_I7") == 0)
			{
				v->second.isConstant = true;
				v->second.isSorted = 1;
				v->second.sorted_fields.push_back(0);
	
				unsigned long long int data = 0;
				harmonyVariable.set_data(&data, sizeof(unsigned long long int));
				harmonyVariable.set_type(hir::pb::I64);
				harmonyVariable.set_size(8);
				harmonyVariable.set_input(true);
				harmonyVariable.set_output(false);

				char string[3] = "17";
				std::string name_string = variable.varname() + "_string";
				VariableMap::iterator var = variables.insert
					(std::make_pair(name_string, VariableDescriptor(variables.size()))).first;

				hir::pb::Variable& harmonyVariable_string = *cfg.add_variables();
				harmonyVariable_string.set_name(var->second.id);
				harmonyVariable_string.set_input(true);
				harmonyVariable_string.set_output(false);
				harmonyVariable_string.set_data(string, 3);
				harmonyVariable_string.set_type(hir::pb::I8);
				harmonyVariable_string.set_size(3);
			}
			else
			{
				harmonyVariable.set_input(false);
				harmonyVariable.set_output(true);
				harmonyVariable.set_size(0);
				harmonyVariable.set_type(hir::pb::I8);
			}
		}
		else
		{
			harmonyVariable.set_input(false);
			harmonyVariable.set_output(false);
			harmonyVariable.set_size(0);
			harmonyVariable.set_type(hir::pb::I8);
		}

	}
}

static void addSortKey(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	VariableMap& variables, VariableMap::const_iterator dId,
	VariableMap::const_iterator sizeVariable)
{
	hir::pb::Kernel& sortKernel = *block.add_kernels();
	sortKernel.set_type(hir::pb::BinaryKernel);

	hir::pb::Operand& d = *sortKernel.add_operands();
	d.set_variable(dId->second.id);
	d.set_mode(hir::pb::InOut);

	hir::pb::Operand& dSize = *sortKernel.add_operands();
	dSize.set_variable(sizeVariable->second.id);
	dSize.set_mode(hir::pb::In);

	//radix sort max: 37(okay)	
	if(bits(dId->second.types) > 37)
	{
		VariableMap::iterator type = getConstantIntVariable(variables, 
			cfg, getTupleDataType(dId->second.types));
	
		RelationalAlgebraKernel binKernel(RelationalAlgebraKernel::ModernGPUSortKey,
			dId->second.types, type->second.types);
	
		hir::pb::Operand& a = *sortKernel.add_operands();
		a.set_variable(type->second.id);
		a.set_mode(hir::pb::In);
	
		binKernel.set_id(kernel_id++);
		sortKernel.set_name(binKernel.name());
		sortKernel.set_code(compileBINSource(binKernel.cudaSourceRepresentation(), 
			RelationalAlgebraKernel::ModernGPUSortKey));
	}
	else
	{
		VariableMap::iterator bit_size = getConstantIntVariable(variables, 
			cfg, bits(dId->second.types));
	
		RelationalAlgebraKernel binKernel(RelationalAlgebraKernel::RadixSortKey,
			dId->second.types, bit_size->second.types);
	
		hir::pb::Operand& a = *sortKernel.add_operands();
		a.set_variable(bit_size->second.id);
		a.set_mode(hir::pb::In);
	
		binKernel.set_id(kernel_id++);
		sortKernel.set_name(binKernel.name());
		sortKernel.set_code(compileBINSource(binKernel.cudaSourceRepresentation(), 
			RelationalAlgebraKernel::RadixSortKey));
	}
}

static void addSortPair(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	VariableMap& variables, VariableMap::const_iterator key,
	VariableMap::const_iterator value,
	VariableMap::const_iterator sizeVariable)
{
	//sort
	report("add sort.");
	hir::pb::Kernel& sortKernel = *block.add_kernels();
	sortKernel.set_type(hir::pb::BinaryKernel);

	hir::pb::Operand& d_key = *sortKernel.add_operands();
	d_key.set_variable(key->second.id);
	d_key.set_mode(hir::pb::InOut);

	hir::pb::Operand& d_value = *sortKernel.add_operands();
	d_value.set_variable(value->second.id);
	d_value.set_mode(hir::pb::InOut);

	hir::pb::Operand& d_key_Size = *sortKernel.add_operands();
	d_key_Size.set_variable(sizeVariable->second.id);
	d_key_Size.set_mode(hir::pb::In);

	//radix sort max: (23,128)x, (26,128)x, (26,64)okay,(32,32) okay
	if(bits(key->second.types) > 33 || bytes(value->second.types) > 16 || 
		(bits(key->second.types) >= 23 && bytes(value->second.types) > 8) || (bits(key->second.types) >= 32 && bytes(value->second.types) > 4))
	{

		VariableMap::iterator key_type = getConstantIntVariable(variables, 
			cfg, getTupleDataType(key->second.types));

		VariableMap::iterator value_type = getConstantIntVariable(variables, 
			cfg, getTupleDataType(value->second.types));
	
		RelationalAlgebraKernel binKernel(RelationalAlgebraKernel::ModernGPUSortPair,
			key->second.types, value->second.types, 
			key_type->second.types, value_type->second.types);
	
		hir::pb::Operand& a_key = *sortKernel.add_operands();
		a_key.set_variable(key_type->second.id);
		a_key.set_mode(hir::pb::In);
		
		hir::pb::Operand& a_value = *sortKernel.add_operands();
		a_value.set_variable(value_type->second.id);
		a_value.set_mode(hir::pb::In);
	
		binKernel.set_id(kernel_id++);
		sortKernel.set_name(binKernel.name());
		sortKernel.set_code(compileBINSource(binKernel.cudaSourceRepresentation(), 
			RelationalAlgebraKernel::ModernGPUSortPair));
	}
	else
	{
		VariableMap::iterator bit_size = getConstantIntVariable(variables, 
			cfg, bits(key->second.types));

		VariableMap::iterator value_type = getConstantIntVariable(variables, 
			cfg, getTupleDataType(value->second.types));

		RelationalAlgebraKernel binKernel(RelationalAlgebraKernel::RadixSortPair,
			key->second.types, value->second.types, 
			bit_size->second.types, value_type->second.types);
	
		hir::pb::Operand& a_key = *sortKernel.add_operands();
		a_key.set_variable(bit_size->second.id);
		a_key.set_mode(hir::pb::In);
		
		hir::pb::Operand& a_value = *sortKernel.add_operands();
		a_value.set_variable(value_type->second.id);
		a_value.set_mode(hir::pb::In);
	
		binKernel.set_id(kernel_id++);
		sortKernel.set_name(binKernel.name());
		sortKernel.set_code(compileBINSource(binKernel.cudaSourceRepresentation(), 
			RelationalAlgebraKernel::RadixSortPair));
	}
}

static void addUnion(hir::pb::BasicBlock& block, hir::pb::KernelControlFlowGraph& cfg,
	const lb::GPUAssign& assign, VariableMap& variables)
{
	const lb::GPUUnion& unionOp = assign.op().unionop();

	VariableMap::const_iterator sourceA = variables.find(unionOp.srca());
	assert(sourceA != variables.end());

	VariableMap::const_iterator sourceB = variables.find(unionOp.srcb());
	assert(sourceB != variables.end());

	VariableMap::iterator dId = variables.find(assign.dest());
	assert(dId != variables.end());

	dId->second.isSorted = dId->second.types.size();

	for(unsigned int i = 0; i < dId->second.types.size(); ++i)
		dId->second.sorted_fields.push_back(i);

	report("Merge");
	report("   destination: " 
		<< " (" << dId->second.id << ")");
	report("   srcA: " 
		<< " (" << sourceA->second.id << ")");
	report("   srcB: "  
		<< " (" << sourceB->second.id << ")");

	// Get the size of the input array
	hir::pb::Kernel& sizeKernel_0 = *block.add_kernels();
	
	VariableMap::const_iterator sizeVariable_0 = 
		getTempIntVariable(variables, cfg);
	sizeKernel_0.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& getSizeD_0  = *sizeKernel_0.add_operands();
	hir::pb::Operand& getSizeIn_0 = *sizeKernel_0.add_operands();
	
	getSizeIn_0.set_mode(hir::pb::In);
	getSizeIn_0.set_variable(sourceA->second.id);
	
	getSizeD_0.set_mode(hir::pb::Out);
	getSizeD_0.set_variable(sizeVariable_0->second.id);
	
	sizeKernel_0.set_name("get_size");
	sizeKernel_0.set_code("");

	hir::pb::Kernel& sizeKernel_1 = *block.add_kernels();
	
	VariableMap::const_iterator sizeVariable_1 = 
		getTempIntVariable(variables, cfg);
	sizeKernel_1.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& getSizeD_1  = *sizeKernel_1.add_operands();
	hir::pb::Operand& getSizeIn_1 = *sizeKernel_1.add_operands();
	
	getSizeIn_1.set_mode(hir::pb::In);
	getSizeIn_1.set_variable(sourceB->second.id);
	
	getSizeD_1.set_mode(hir::pb::Out);
	getSizeD_1.set_variable(sizeVariable_1->second.id);
	
	sizeKernel_1.set_name("get_size");
	sizeKernel_1.set_code("");

	if(sourceA->second.isSorted < sourceA->second.types.size()
		&& sourceA->second.unique_keys == 0)
		addSortKey(block, cfg, variables, sourceA, sizeVariable_0);

	if(sourceB->second.isSorted < sourceB->second.types.size()
		&& sourceB->second.unique_keys == 0)
		addSortKey(block, cfg, variables, sourceB, sizeVariable_1);

	// Get the size of the output array if the primitive size changed
	hir::pb::Kernel& adjustSizeKernel = *block.add_kernels();
	VariableMap::const_iterator sizeVariable = 
		getTempIntVariable(variables, cfg);

	adjustSizeKernel.set_type(hir::pb::ComputeKernel);
	
	VariableMap::iterator U32_1 = getConstantIntVariable(variables, cfg, 1);
	VariableMap::iterator U32_0 = getConstantIntVariable(variables, cfg, 0);
	
	hir::pb::Operand& ctas = *adjustSizeKernel.add_operands();
	ctas.set_mode(hir::pb::In);
	ctas.set_variable(U32_1->second.id);

	hir::pb::Operand& threads = *adjustSizeKernel.add_operands();
	threads.set_mode(hir::pb::In);
	threads.set_variable(U32_1->second.id);

	hir::pb::Operand& shared = *adjustSizeKernel.add_operands();
	shared.set_mode(hir::pb::In);
	shared.set_variable(U32_0->second.id);
	
	hir::pb::Operand& size_dest = *adjustSizeKernel.add_operands();
	size_dest.set_mode(hir::pb::Out);
	size_dest.set_variable(sizeVariable->second.id);
		
	hir::pb::Operand& size_0 = *adjustSizeKernel.add_operands();
	size_0.set_mode(hir::pb::In);
	size_0.set_variable(sizeVariable_0->second.id);

	hir::pb::Operand& size_1 = *adjustSizeKernel.add_operands();
	size_1.set_mode(hir::pb::In);
	size_1.set_variable(sizeVariable_1->second.id);

	RelationalAlgebraKernel ptxKernel(
		RelationalAlgebraKernel::UnionGetResultSize);

	ptxKernel.set_id(kernel_id++);
	adjustSizeKernel.set_name(ptxKernel.name());
	adjustSizeKernel.set_code(compilePTXSource(
		ptxKernel.cudaSourceRepresentation()));

	// Resize the output array
	hir::pb::Kernel& resizeResultKernel = *block.add_kernels();
	
	resizeResultKernel.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeResultD    = *resizeResultKernel.add_operands();
	hir::pb::Operand& resizeResultSize = *resizeResultKernel.add_operands();
	
	resizeResultD.set_mode(hir::pb::Out);
	resizeResultD.set_variable(dId->second.id);
	
	resizeResultSize.set_mode(hir::pb::InOut);
	resizeResultSize.set_variable(sizeVariable->second.id);

	resizeResultKernel.set_name("resize");
	resizeResultKernel.set_code("");

	// Add the union kernel
	hir::pb::Kernel& unionKernel = *block.add_kernels();

	unionKernel.set_type(hir::pb::BinaryKernel);
	hir::pb::Operand& d = *unionKernel.add_operands();

	d.set_variable(dId->second.id);
	d.set_mode(hir::pb::InOut);

	hir::pb::Operand& dSize = *unionKernel.add_operands();
	dSize.set_variable(sizeVariable->second.id);
	dSize.set_mode(hir::pb::InOut);

	VariableMap::iterator type = getConstantIntVariable(variables, 
		cfg, getTupleDataType(dId->second.types));

	RelationalAlgebraKernel binKernel(RelationalAlgebraKernel::Union,
		dId->second.types, type->second.types);

	hir::pb::Operand& a = *unionKernel.add_operands();
	a.set_variable(sourceA->second.id);
	a.set_mode(hir::pb::In);

	hir::pb::Operand& aSize = *unionKernel.add_operands();
	aSize.set_variable(sizeVariable_0->second.id);
	aSize.set_mode(hir::pb::In);

	hir::pb::Operand& b = *unionKernel.add_operands();
	b.set_variable(sourceB->second.id);
	b.set_mode(hir::pb::In);

	hir::pb::Operand& bSize = *unionKernel.add_operands();
	bSize.set_variable(sizeVariable_1->second.id);
	bSize.set_mode(hir::pb::In);

	hir::pb::Operand& c = *unionKernel.add_operands();
	c.set_variable(type->second.id);
	c.set_mode(hir::pb::In);

	binKernel.set_id(kernel_id++);
	unionKernel.set_name(binKernel.name());
	unionKernel.set_code(compileBINSource(binKernel.cudaSourceRepresentation(), 
		RelationalAlgebraKernel::Union));

	report("resize the output size of union.");
	hir::pb::Kernel& updateSizeKernel = *block.add_kernels();
	
	updateSizeKernel.set_type(hir::pb::UpdateSize);
	
	hir::pb::Operand& updateSizeD    
		= *updateSizeKernel.add_operands();
	hir::pb::Operand& updateSizeSize
		= *updateSizeKernel.add_operands();
	
	updateSizeD.set_mode(hir::pb::In);
	updateSizeD.set_variable(dId->second.id);
	
	updateSizeSize.set_mode(hir::pb::InOut);
	updateSizeSize.set_variable(sizeVariable->second.id);

	updateSizeKernel.set_name("updatesize");
	updateSizeKernel.set_code("");
}

static void addIntersection(hir::pb::BasicBlock& block,
	const lb::GPUAssign& assign, const VariableMap& variables)
{
	hir::pb::Kernel& kernel = *block.add_kernels();

	kernel.set_type(hir::pb::ComputeKernel);

	hir::pb::Operand& d = *kernel.add_operands();

	VariableMap::const_iterator dId = variables.find(assign.dest());
	assert(dId != variables.end());

	d.set_mode(hir::pb::Out);
	d.set_variable(dId->second.id);

	const lb::GPUIntersection& intersection = assign.op().intersection();
	
	VariableMap::const_iterator sourceA = variables.find(intersection.srca());
	assert(sourceA != variables.end());
	
	VariableMap::const_iterator sourceB = variables.find(intersection.srcb());
	assert(sourceB != variables.end());

	RelationalAlgebraKernel ptxKernel(RelationalAlgebraKernel::Intersection,
		dId->second.types, sourceA->second.types, sourceB->second.types);
	
	hir::pb::Operand& a = *kernel.add_operands();
	hir::pb::Operand& b = *kernel.add_operands();
	
	a.set_variable(sourceA->second.id);
	a.set_mode(hir::pb::In);
	b.set_variable(sourceB->second.id);
	b.set_mode(hir::pb::In);
	
	ptxKernel.set_id(kernel_id++);
	kernel.set_name(ptxKernel.name());
	kernel.set_code(compilePTXSource(ptxKernel.cudaSourceRepresentation()));
}

static void addProduct(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg,
	const lb::GPUAssign& assign, VariableMap& variables, std::string& dest)
{
	const lb::GPUJoin& product = assign.op().join();
	
	// Get reference to the kernel parameters
	VariableMap::const_iterator dId = variables.find(dest);
	assert(dId != variables.end());

	VariableMap::const_iterator sourceA = variables.find(product.srca());
	assert(sourceA != variables.end());

	VariableMap::const_iterator sourceB = variables.find(product.srcb());
	assert(sourceB != variables.end());

	report("   destination: " << assign.dest()
		<< " (" << dId->second.id << ")");
	report("   srcA: " << product.srca()
		<< " (" << sourceA->second.id << ")");
	report("   srcB: " << product.srcb()
		<< " (" << sourceB->second.id << ")");

	// Create scratch variables
	VariableMap::const_iterator resultSize = getTempIntVariable(variables, cfg);
	VariableMap::const_iterator leftSize   = getTempIntVariable(variables, cfg);
	VariableMap::const_iterator rightSize  = getTempIntVariable(variables, cfg);

	VariableMap::iterator U32_256 = getConstantIntVariable(variables, cfg, 256);
	VariableMap::iterator U32_350 = getConstantIntVariable(variables, cfg, 350);
	VariableMap::iterator U32_0   = getConstantIntVariable(variables, cfg, 0);
	VariableMap::iterator U32_1   = getConstantIntVariable(variables, cfg, 1);

	// Get the size of the left input
	hir::pb::Kernel& leftSizeKernel = *block.add_kernels();
	
	leftSizeKernel.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& leftSizeD    = *leftSizeKernel.add_operands();
	hir::pb::Operand& leftSizeLeft = *leftSizeKernel.add_operands();
	
	leftSizeD.set_mode(hir::pb::Out);
	leftSizeD.set_variable(leftSize->second.id);
	
	leftSizeLeft.set_mode(hir::pb::In);
	leftSizeLeft.set_variable(sourceA->second.id);
	
	leftSizeKernel.set_name("get_size");
	leftSizeKernel.set_code("");	

	// Get the size of the right input
	hir::pb::Kernel& rightSizeKernel = *block.add_kernels();
	
	rightSizeKernel.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& rightSizeD     = *rightSizeKernel.add_operands();
	hir::pb::Operand& rightSizeRight = *rightSizeKernel.add_operands();
	
	rightSizeD.set_mode(hir::pb::Out);
	rightSizeD.set_variable(rightSize->second.id);
	
	rightSizeRight.set_mode(hir::pb::In);
	rightSizeRight.set_variable(sourceB->second.id);
	
	rightSizeKernel.set_name("get_size");
	rightSizeKernel.set_code("");	

	// Determine the size of the output
	hir::pb::Kernel& resultSizeKernel = *block.add_kernels();
	
	resultSizeKernel.set_type(hir::pb::ComputeKernel);

	hir::pb::Operand& resultSizeCtas = *resultSizeKernel.add_operands();
	resultSizeCtas.set_mode(hir::pb::In);
	resultSizeCtas.set_variable(U32_1->second.id);

	hir::pb::Operand& resultSizeThreads = *resultSizeKernel.add_operands();
	resultSizeThreads.set_mode(hir::pb::In);
	resultSizeThreads.set_variable(U32_1->second.id);

	hir::pb::Operand& resultSizeShared = *resultSizeKernel.add_operands();
	resultSizeShared.set_mode(hir::pb::In);
	resultSizeShared.set_variable(U32_0->second.id);
	
	hir::pb::Operand& resultSizeD = *resultSizeKernel.add_operands();
	resultSizeD.set_mode(hir::pb::Out);
	resultSizeD.set_variable(resultSize->second.id);
		
	RelationalAlgebraKernel resultSizePtx(
		RelationalAlgebraKernel::ProductGetResultSize,
		dId->second.types, sourceA->second.types, sourceB->second.types);
	
	hir::pb::Operand& resultSizeA = *resultSizeKernel.add_operands();
	resultSizeA.set_variable(leftSize->second.id);
	resultSizeA.set_mode(hir::pb::In);
	
	hir::pb::Operand& resultSizeB = *resultSizeKernel.add_operands();
	resultSizeB.set_variable(rightSize->second.id);
	resultSizeB.set_mode(hir::pb::In);
	
	resultSizePtx.set_id(kernel_id++);
	resultSizeKernel.set_name(resultSizePtx.name());
	resultSizeKernel.set_code(compilePTXSource(
		resultSizePtx.cudaSourceRepresentation()));
	
	// Resize the output
	hir::pb::Kernel& resizeTempKernel = *block.add_kernels();
	
	resizeTempKernel.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeTempD    = *resizeTempKernel.add_operands();
	hir::pb::Operand& resizeTempSize = *resizeTempKernel.add_operands();
	
	resizeTempD.set_mode(hir::pb::Out);
	resizeTempD.set_variable(dId->second.id);
	
	resizeTempSize.set_mode(hir::pb::In);
	resizeTempSize.set_variable(resultSize->second.id);

	resizeTempKernel.set_name("resize");
	resizeTempKernel.set_code("");

	// Add the main product kernel
	hir::pb::Kernel& kernel = *block.add_kernels();

	kernel.set_type(hir::pb::ComputeKernel);

	hir::pb::Operand& ctas = *kernel.add_operands();
	ctas.set_mode(hir::pb::In);
//	ctas.set_variable(U32_128->second.id);
	ctas.set_variable(U32_350->second.id);

	hir::pb::Operand& threads = *kernel.add_operands();
	threads.set_mode(hir::pb::In);
	threads.set_variable(U32_256->second.id);

	hir::pb::Operand& shared = *kernel.add_operands();
	shared.set_mode(hir::pb::In);
	shared.set_variable(U32_0->second.id);

	hir::pb::Operand& d = *kernel.add_operands();
	d.set_mode(hir::pb::InOut);
	d.set_variable(dId->second.id);

	RelationalAlgebraKernel ptxKernel(RelationalAlgebraKernel::Product,
		dId->second.types, sourceA->second.types, sourceB->second.types);

	hir::pb::Operand& a = *kernel.add_operands();
	a.set_variable(sourceA->second.id);
	a.set_mode(hir::pb::In);

	hir::pb::Operand& aSize = *kernel.add_operands();
	aSize.set_variable(leftSize->second.id);
	aSize.set_mode(hir::pb::In);

	hir::pb::Operand& b = *kernel.add_operands();
	b.set_variable(sourceB->second.id);
	b.set_mode(hir::pb::In);

	hir::pb::Operand& bSize = *kernel.add_operands();
	bSize.set_variable(rightSize->second.id);
	bSize.set_mode(hir::pb::In);

	ptxKernel.set_id(kernel_id++);
	kernel.set_name(ptxKernel.name());
	kernel.set_code(compilePTXSource(ptxKernel.cudaSourceRepresentation()));
}

#if 0 
static void addMerge(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	VariableMap& variables, std::string srca, std::string srcb, std::string dest)
{
	VariableMap::const_iterator sourceA = variables.find(srca);
	assert(sourceA != variables.end());

	VariableMap::const_iterator sourceB = variables.find(srcb);
	assert(sourceB != variables.end());

	VariableMap::iterator dId = variables.find(dest);
	assert(dId != variables.end());

	report("Merge");
	report("   destination: " 
		<< " (" << dId->second.id << ")");
	report("   srcA: " 
		<< " (" << sourceA->second.id << ")");
	report("   srcB: "  
		<< " (" << sourceB->second.id << ")");

	// Get the size of the input array
	hir::pb::Kernel& sizeKernel_0 = *block.add_kernels();
	
	VariableMap::const_iterator sizeVariable_0 = 
		getTempIntVariable(variables, cfg);
	sizeKernel_0.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& getSizeD_0  = *sizeKernel_0.add_operands();
	hir::pb::Operand& getSizeIn_0 = *sizeKernel_0.add_operands();
	
	getSizeIn_0.set_mode(hir::pb::In);
	getSizeIn_0.set_variable(sourceA->second.id);
	
	getSizeD_0.set_mode(hir::pb::Out);
	getSizeD_0.set_variable(sizeVariable_0->second.id);
	
	sizeKernel_0.set_name("get_size");
	sizeKernel_0.set_code("");

	hir::pb::Kernel& sizeKernel_1 = *block.add_kernels();
	
	VariableMap::const_iterator sizeVariable_1 = 
		getTempIntVariable(variables, cfg);
	sizeKernel_1.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& getSizeD_1  = *sizeKernel_1.add_operands();
	hir::pb::Operand& getSizeIn_1 = *sizeKernel_1.add_operands();
	
	getSizeIn_1.set_mode(hir::pb::In);
	getSizeIn_1.set_variable(sourceB->second.id);
	
	getSizeD_1.set_mode(hir::pb::Out);
	getSizeD_1.set_variable(sizeVariable_1->second.id);
	
	sizeKernel_1.set_name("get_size");
	sizeKernel_1.set_code("");

	// Get the size of the output array if the primitive size changed
	hir::pb::Kernel& adjustSizeKernel = *block.add_kernels();
	VariableMap::const_iterator sizeVariable = 
		getTempIntVariable(variables, cfg);

	adjustSizeKernel.set_type(hir::pb::ComputeKernel);
	
	VariableMap::iterator U32_1 = getConstantIntVariable(variables, cfg, 1);
	VariableMap::iterator U32_0 = getConstantIntVariable(variables, cfg, 0);
	
	hir::pb::Operand& ctas = *adjustSizeKernel.add_operands();
	ctas.set_mode(hir::pb::In);
	ctas.set_variable(U32_1->second.id);

	hir::pb::Operand& threads = *adjustSizeKernel.add_operands();
	threads.set_mode(hir::pb::In);
	threads.set_variable(U32_1->second.id);

	hir::pb::Operand& shared = *adjustSizeKernel.add_operands();
	shared.set_mode(hir::pb::In);
	shared.set_variable(U32_0->second.id);
	
	hir::pb::Operand& size_dest = *adjustSizeKernel.add_operands();
	size_dest.set_mode(hir::pb::Out);
	size_dest.set_variable(sizeVariable->second.id);
		
	hir::pb::Operand& size_0 = *adjustSizeKernel.add_operands();
	size_0.set_mode(hir::pb::In);
	size_0.set_variable(sizeVariable_0->second.id);

	hir::pb::Operand& size_1 = *adjustSizeKernel.add_operands();
	size_1.set_mode(hir::pb::In);
	size_1.set_variable(sizeVariable_1->second.id);

	RelationalAlgebraKernel ptxKernel(
		RelationalAlgebraKernel::CombineGetResultSize);

	ptxKernel.set_id(kernel_id++);
	adjustSizeKernel.set_name(ptxKernel.name());
	adjustSizeKernel.set_code(compilePTXSource(
		ptxKernel.cudaSourceRepresentation()));

	// Resize the output array
	hir::pb::Kernel& resizeResultKernel = *block.add_kernels();
	
	resizeResultKernel.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeResultD    = *resizeResultKernel.add_operands();
	hir::pb::Operand& resizeResultSize = *resizeResultKernel.add_operands();
	
	resizeResultD.set_mode(hir::pb::Out);
	resizeResultD.set_variable(dId->second.id);
	
	resizeResultSize.set_mode(hir::pb::InOut);
	resizeResultSize.set_variable(sizeVariable->second.id);

	resizeResultKernel.set_name("resize");
	resizeResultKernel.set_code("");

	// Add the merge kernel
	hir::pb::Kernel& mergeKernel = *block.add_kernels();

	mergeKernel.set_type(hir::pb::BinaryKernel);
	hir::pb::Operand& d = *mergeKernel.add_operands();

	d.set_variable(dId->second.id);
	d.set_mode(hir::pb::InOut);

	hir::pb::Operand& dSize = *mergeKernel.add_operands();
	dSize.set_variable(sizeVariable->second.id);
	dSize.set_mode(hir::pb::InOut);

	VariableMap::iterator type = getConstantIntVariable(variables, 
		cfg, getTupleDataType(dId->second.types));

	RelationalAlgebraKernel binKernel(RelationalAlgebraKernel::Combine,
		dId->second.types, type->second.types);

	hir::pb::Operand& a = *mergeKernel.add_operands();
	a.set_variable(sourceA->second.id);
	a.set_mode(hir::pb::In);

	hir::pb::Operand& aSize = *mergeKernel.add_operands();
	aSize.set_variable(sizeVariable_0->second.id);
	aSize.set_mode(hir::pb::In);

	hir::pb::Operand& b = *mergeKernel.add_operands();
	b.set_variable(sourceB->second.id);
	b.set_mode(hir::pb::In);

	hir::pb::Operand& bSize = *mergeKernel.add_operands();
	bSize.set_variable(sizeVariable_1->second.id);
	bSize.set_mode(hir::pb::In);

	hir::pb::Operand& c = *mergeKernel.add_operands();
	c.set_variable(type->second.id);
	c.set_mode(hir::pb::In);

	binKernel.set_id(kernel_id++);
	mergeKernel.set_name(binKernel.name());
	mergeKernel.set_code(compileBINSource(binKernel.cudaSourceRepresentation(), 
		RelationalAlgebraKernel::Combine));

	report("resize the output size of unique.");
	hir::pb::Kernel& updateSizeKernel = *block.add_kernels();
	
	updateSizeKernel.set_type(hir::pb::UpdateSize);
	
	hir::pb::Operand& updateSizeD    
		= *updateSizeKernel.add_operands();
	hir::pb::Operand& updateSizeSize
		= *updateSizeKernel.add_operands();
	
	updateSizeD.set_mode(hir::pb::In);
	updateSizeD.set_variable(dId->second.id);
	
	updateSizeSize.set_mode(hir::pb::InOut);
	updateSizeSize.set_variable(sizeVariable->second.id);

	updateSizeKernel.set_name("updatesize");
	updateSizeKernel.set_code("");
}
#endif

static void addSingle(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, StringIdMap& strings,
	VariableMap& variables, const lb::GPUAssign& assign)
{
//	typedef std::pair<StringIdMap::iterator, bool> StringInsertion;
	std::string dest;

	VariableMap::const_iterator dId_merge = variables.find(assign.dest());
	assert(dId_merge != variables.end());

	VariableMap::iterator dId = getTempBufferByName(variables, cfg, 0, 
		0, dId_merge->second.isSorted, dId_merge->second.sorted_fields, dest, dId_merge->second.types);

	VariableMap::const_iterator StringTable = variables.find("StringTable");
	assert(StringTable != variables.end());

	// Get the current size of the input
	RelationalAlgebraKernel::Value value;
	
	const lb::GPUSingle& single = assign.op().single();
	
	for(RepeatedConstant::const_iterator 
		constant = single.element().begin();
		constant != single.element().end(); ++constant)
	{
		RelationalAlgebraKernel::ConstantElement c;

		switch(constant->kind())
		{
		case common::Constant_Kind_BOOL:
		{
			c.type      = RelationalAlgebraKernel::I8;
			c.boolValue = constant->bool_constant().value();
			c.limit     = 8;
			break;
		}
		case common::Constant_Kind_INT:
		{
			c.type     = RelationalAlgebraKernel::I64;
			c.intValue = constant->int_constant().value();
			c.limit    = 64;
			break;
		}
		case common::Constant_Kind_UINT:
		{
			c.type     = RelationalAlgebraKernel::I64;
			c.intValue = constant->uint_constant().value();
			c.limit    = 64;
			break;
		}
		case common::Constant_Kind_FLOAT:
		{
			std::stringstream stream(
				constant->float_constant().value());
			c.type = RelationalAlgebraKernel::F64;
			stream >> c.floatValue;
			c.limit = 64;
			break;
		}
		case common::Constant_Kind_STRING:
		{
//			c.type = RelationalAlgebraKernel::I32;
//			
//			StringInsertion insertion = strings.insert(
//				std::make_pair(constant->string_constant().value(),
//				strings.size()));
//			
//			c.stringId = insertion.first->second;
//			c.limit    = log2(hydrazine::nextPowerOfTwo(strings.size()));

			c.type = RelationalAlgebraKernel::Pointer;
			c.intValue = strings.find(constant->string_constant().value())->second;
//			c.stringId = strings.find(constant->string_constant().value())->second;
			c.limit = 64;
			
			break;
		}
		case common::Constant_Kind_DATETIME:
		{
			assertM(false, "Date constants not implemented.");
			break;
		}
		}
		
		value.push_back(c);
	}

	VariableMap::const_iterator sizeVariable = getConstantIntVariable(
		variables, cfg, bytes(value));
	VariableMap::iterator U32_0 = getConstantIntVariable(variables, cfg, 0);
	VariableMap::iterator U32_1 = getConstantIntVariable(variables, cfg, 1);
	
	// Resize the kernel to 1
	hir::pb::Kernel& resizeKernel = *block.add_kernels();
	
	resizeKernel.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeD    = *resizeKernel.add_operands();
	hir::pb::Operand& resizeSize = *resizeKernel.add_operands();
	
	resizeD.set_mode(hir::pb::Out);
	resizeD.set_variable(dId->second.id);
	
	resizeSize.set_mode(hir::pb::In);
	resizeSize.set_variable(sizeVariable->second.id);

	resizeKernel.set_name("resize");
	resizeKernel.set_code("");
	
	// Create the kernel to assign the value
	hir::pb::Kernel& kernel = *block.add_kernels();

	kernel.set_type(hir::pb::ComputeKernel);

	hir::pb::Operand& kernelCtas = *kernel.add_operands();
	kernelCtas.set_mode(hir::pb::In);
	kernelCtas.set_variable(U32_1->second.id);
	
	hir::pb::Operand& kernelThreads = *kernel.add_operands();
	kernelThreads.set_mode(hir::pb::In);
	kernelThreads.set_variable(U32_1->second.id);
	
	hir::pb::Operand& kernelShared = *kernel.add_operands();
	kernelShared.set_mode(hir::pb::In);
	kernelShared.set_variable(U32_0->second.id);
	
	hir::pb::Operand& d = *kernel.add_operands();
	d.set_mode(hir::pb::InOut);
	d.set_variable(dId->second.id);

	hir::pb::Operand& a = *kernel.add_operands(); // element index
	a.set_mode(hir::pb::In);
	a.set_variable(U32_0->second.id);
	
	VariableMap::iterator constant = getConstantVariable(variables, cfg, value);
	hir::pb::Operand& b = *kernel.add_operands(); // value
	b.set_mode(hir::pb::In);
	b.set_variable(constant->second.id);
	
	hir::pb::Operand& string = *kernel.add_operands(); // value
	string.set_mode(hir::pb::In);
	string.set_variable(StringTable->second.id);
	
	RelationalAlgebraKernel ptxKernel(dId->second.types, value);
	
	ptxKernel.set_id(kernel_id++);
	kernel.set_name(ptxKernel.name());
	kernel.set_code(compilePTXSource(ptxKernel.cudaSourceRepresentation()));

}

static void addDifference(hir::pb::BasicBlock& block, hir::pb::KernelControlFlowGraph& cfg, 
	const lb::GPUAssign& assign, VariableMap& variables)
{
	const lb::GPUDifference& difference = assign.op().difference();
	
	VariableMap::const_iterator sourceA = variables.find(difference.srca());
	assert(sourceA != variables.end());
	
	VariableMap::const_iterator sourceB = variables.find(difference.srcb());
	assert(sourceB != variables.end());

	VariableMap::iterator dId = variables.find(assign.dest());
	assert(dId != variables.end());

	dId->second.isSorted = dId->second.types.size();

	for(unsigned int i = 0; i < dId->second.types.size(); ++i)
		dId->second.sorted_fields.push_back(i);

	report("Difference");
	report("   destination: " 
		<< " (" << dId->second.id << ")");
	report("   srcA: " 
		<< " (" << sourceA->second.id << ")");
	report("   srcB: "  
		<< " (" << sourceB->second.id << ")");

	// Get the size of the input array
	hir::pb::Kernel& sizeKernel_0 = *block.add_kernels();
	
	VariableMap::const_iterator sizeVariable_0 = 
		getTempIntVariable(variables, cfg);
	sizeKernel_0.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& getSizeD_0  = *sizeKernel_0.add_operands();
	hir::pb::Operand& getSizeIn_0 = *sizeKernel_0.add_operands();
	
	getSizeIn_0.set_mode(hir::pb::In);
	getSizeIn_0.set_variable(sourceA->second.id);
	
	getSizeD_0.set_mode(hir::pb::Out);
	getSizeD_0.set_variable(sizeVariable_0->second.id);
	
	sizeKernel_0.set_name("get_size");
	sizeKernel_0.set_code("");

	hir::pb::Kernel& sizeKernel_1 = *block.add_kernels();
	
	VariableMap::const_iterator sizeVariable_1 = 
		getTempIntVariable(variables, cfg);
	sizeKernel_1.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& getSizeD_1  = *sizeKernel_1.add_operands();
	hir::pb::Operand& getSizeIn_1 = *sizeKernel_1.add_operands();
	
	getSizeIn_1.set_mode(hir::pb::In);
	getSizeIn_1.set_variable(sourceB->second.id);
	
	getSizeD_1.set_mode(hir::pb::Out);
	getSizeD_1.set_variable(sizeVariable_1->second.id);
	
	sizeKernel_1.set_name("get_size");
	sizeKernel_1.set_code("");

	if(sourceA->second.isSorted < sourceA->second.types.size())
		addSortKey(block, cfg, variables, sourceA, sizeVariable_0);

	if(sourceB->second.isSorted < sourceB->second.types.size())
		addSortKey(block, cfg, variables, sourceB, sizeVariable_1);

	//get size
	hir::pb::Kernel& adjustSizeKernel = *block.add_kernels();
	VariableMap::const_iterator sizeVariable = 
		getTempIntVariable(variables, cfg);

	adjustSizeKernel.set_type(hir::pb::ComputeKernel);
	
	VariableMap::iterator U32_1 = getConstantIntVariable(variables, cfg, 1);
	VariableMap::iterator U32_0 = getConstantIntVariable(variables, cfg, 0);
	
	hir::pb::Operand& ctas = *adjustSizeKernel.add_operands();
	ctas.set_mode(hir::pb::In);
	ctas.set_variable(U32_1->second.id);

	hir::pb::Operand& threads = *adjustSizeKernel.add_operands();
	threads.set_mode(hir::pb::In);
	threads.set_variable(U32_1->second.id);

	hir::pb::Operand& shared = *adjustSizeKernel.add_operands();
	shared.set_mode(hir::pb::In);
	shared.set_variable(U32_0->second.id);
	
	hir::pb::Operand& size_dest = *adjustSizeKernel.add_operands();
	size_dest.set_mode(hir::pb::Out);
	size_dest.set_variable(sizeVariable->second.id);
		
	hir::pb::Operand& size_0 = *adjustSizeKernel.add_operands();
	size_0.set_mode(hir::pb::In);
	size_0.set_variable(sizeVariable_0->second.id);

	RelationalAlgebraKernel ptxKernel(
		RelationalAlgebraKernel::DifferenceGetResultSize);

	ptxKernel.set_id(kernel_id++);
	adjustSizeKernel.set_name(ptxKernel.name());
	adjustSizeKernel.set_code(compilePTXSource(
		ptxKernel.cudaSourceRepresentation()));

	// Resize the output array
	hir::pb::Kernel& resizeResultKernel = *block.add_kernels();
	
	resizeResultKernel.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeResultD    = *resizeResultKernel.add_operands();
	hir::pb::Operand& resizeResultSize = *resizeResultKernel.add_operands();
	
	resizeResultD.set_mode(hir::pb::Out);
	resizeResultD.set_variable(dId->second.id);
	
	resizeResultSize.set_mode(hir::pb::InOut);
	resizeResultSize.set_variable(sizeVariable_0->second.id);

	resizeResultKernel.set_name("resize");
	resizeResultKernel.set_code("");

	// Add the difference kernel
	hir::pb::Kernel& DiffereneceKernel = *block.add_kernels();

	DiffereneceKernel.set_type(hir::pb::BinaryKernel);
	hir::pb::Operand& d = *DiffereneceKernel.add_operands();

	d.set_variable(dId->second.id);
	d.set_mode(hir::pb::InOut);

	hir::pb::Operand& dSize = *DiffereneceKernel.add_operands();
	dSize.set_variable(sizeVariable->second.id);
	dSize.set_mode(hir::pb::InOut);

	VariableMap::iterator type = getConstantIntVariable(variables, 
		cfg, getTupleDataType(dId->second.types));

	RelationalAlgebraKernel binKernel(RelationalAlgebraKernel::Difference,
		dId->second.types, type->second.types);

	hir::pb::Operand& a = *DiffereneceKernel.add_operands();
	a.set_variable(sourceA->second.id);
	a.set_mode(hir::pb::In);

	hir::pb::Operand& aSize = *DiffereneceKernel.add_operands();
	aSize.set_variable(sizeVariable_0->second.id);
	aSize.set_mode(hir::pb::In);

	hir::pb::Operand& b = *DiffereneceKernel.add_operands();
	b.set_variable(sourceB->second.id);
	b.set_mode(hir::pb::In);

	hir::pb::Operand& bSize = *DiffereneceKernel.add_operands();
	bSize.set_variable(sizeVariable_1->second.id);
	bSize.set_mode(hir::pb::In);

	hir::pb::Operand& c = *DiffereneceKernel.add_operands();
	c.set_variable(type->second.id);
	c.set_mode(hir::pb::In);

	binKernel.set_id(kernel_id++);
	DiffereneceKernel.set_name(binKernel.name());
	DiffereneceKernel.set_code(compileBINSource(binKernel.cudaSourceRepresentation(), 
		RelationalAlgebraKernel::Difference));

	report("resize the output size of unique.");
	hir::pb::Kernel& updateSizeKernel = *block.add_kernels();
	
	updateSizeKernel.set_type(hir::pb::UpdateSize);
	
	hir::pb::Operand& updateSizeD    
		= *updateSizeKernel.add_operands();
	hir::pb::Operand& updateSizeSize
		= *updateSizeKernel.add_operands();
	
	updateSizeD.set_mode(hir::pb::In);
	updateSizeD.set_variable(dId->second.id);
	
	updateSizeSize.set_mode(hir::pb::InOut);
	updateSizeSize.set_variable(sizeVariable->second.id);

	updateSizeKernel.set_name("updatesize");
	updateSizeKernel.set_code("");
}

static void addJoin(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	const lb::GPUAssign& assign, VariableMap& variables,
	std::string dest)
{
	const lb::GPUJoin& join = assign.op().join();

	VariableMap::const_iterator dId = variables.find(dest);
	assert(dId != variables.end());

	VariableMap::iterator sourceA = variables.find(join.srca());
	assert(sourceA != variables.end());
	
	VariableMap::iterator sourceB = variables.find(join.srcb());
	assert(sourceB != variables.end());

	report("   destination: " << dest
		<< " (" << dId->second.id << ")");
	report("   srcA: " << join.srca()
		<< " (" << sourceA->second.id << ")");
	report("   srcB: " << join.srcb()
		<< " (" << sourceB->second.id << ")");
		
	VariableMap::iterator U32_1 = getConstantIntVariable(variables, cfg, 1);
	VariableMap::iterator U32_0 = getConstantIntVariable(variables, cfg, 0);

	VariableMap::iterator U32_350 = getConstantIntVariable(variables, cfg, 350);
	VariableMap::iterator U32_256 = getConstantIntVariable(variables, cfg, 256);
	
	// Get the size of the left input
	VariableMap::const_iterator leftSize = getTempIntVariable(variables, cfg);
	
	hir::pb::Kernel& leftSizeKernel = *block.add_kernels();
	
	leftSizeKernel.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& leftSizeD    = *leftSizeKernel.add_operands();
	hir::pb::Operand& leftSizeLeft = *leftSizeKernel.add_operands();
	
	leftSizeD.set_mode(hir::pb::Out);
	leftSizeD.set_variable(leftSize->second.id);
	
	leftSizeLeft.set_mode(hir::pb::In);
	leftSizeLeft.set_variable(sourceA->second.id);
	
	leftSizeKernel.set_name("get_size");
	leftSizeKernel.set_code("");	
	
	// Get the size of the right input
	VariableMap::const_iterator rightSize = getTempIntVariable(variables, cfg);
	
	hir::pb::Kernel& rightSizeKernel = *block.add_kernels();
	
	rightSizeKernel.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& rightSizeD     = *rightSizeKernel.add_operands();
	hir::pb::Operand& rightSizeRight = *rightSizeKernel.add_operands();
	
	rightSizeD.set_mode(hir::pb::Out);
	rightSizeD.set_variable(rightSize->second.id);
	
	rightSizeRight.set_mode(hir::pb::In);
	rightSizeRight.set_variable(sourceB->second.id);
	
	rightSizeKernel.set_name("get_size");
	rightSizeKernel.set_code("");	

	bool skipSplitLeft = false;
	bool skipSplitRight = false;

	if((unsigned int)join.keycount() == (unsigned int)sourceA->second.types.size())
		skipSplitLeft = true;

	if((unsigned int)join.keycount() == (unsigned int)sourceB->second.types.size())
		skipSplitRight = true;

	// Start Adding Project
	VariableMap::iterator temp_split_left_key;
	VariableMap::iterator temp_split_right_key;

	VariableMap::iterator temp_split_left_value;
	VariableMap::iterator temp_split_right_value;

	VariableMap::const_iterator sizeVariable_leftkey;
	VariableMap::const_iterator sizeVariable_rightkey; 

	VariableMap::const_iterator sizeVariable_leftvalue;
	VariableMap::const_iterator sizeVariable_rightvalue; 

	if(!skipSplitLeft)
	{ 
		temp_split_left_key = getTempBuffer(variables, cfg, 0);
		temp_split_left_key->second.types.insert(temp_split_left_key->second.types.begin(), 
				sourceA->second.types.begin(), sourceA->second.types.begin() + join.keycount());

		temp_split_left_value = getTempBuffer(variables, cfg, 0);
		temp_split_left_value->second.types.insert(temp_split_left_value->second.types.begin(), 
				 sourceA->second.types.begin() + join.keycount(), sourceA->second.types.end());
	}
	else
	{
		temp_split_left_key = sourceA;
		sizeVariable_leftkey = leftSize;
	}

	if(!skipSplitRight) 
	{
		temp_split_right_key = getTempBuffer(variables, cfg, 0);
		temp_split_right_key->second.types.insert(temp_split_right_key->second.types.begin(), 
				sourceB->second.types.begin(), sourceB->second.types.begin() + join.keycount());

		temp_split_right_value = getTempBuffer(variables, cfg, 0);
		temp_split_right_value->second.types.insert(temp_split_right_value->second.types.begin(), 
				 sourceB->second.types.begin() + join.keycount(), sourceB->second.types.end());
	}
	else
	{
		temp_split_right_key = sourceB;
		sizeVariable_rightkey = rightSize;
	}

	if(!skipSplitLeft)
	{
		sizeVariable_leftkey = getTempIntVariable(variables, cfg);
		sizeVariable_leftvalue = getTempIntVariable(variables, cfg);

		// Get the size of the output array of split
		report("get the output size of split.");
		hir::pb::Kernel& adjustSizeKernel = *block.add_kernels();

		adjustSizeKernel.set_type(hir::pb::ComputeKernel);
		
		hir::pb::Operand& ctas = *adjustSizeKernel.add_operands();
		ctas.set_mode(hir::pb::In);
		ctas.set_variable(U32_1->second.id);
	
		hir::pb::Operand& threads = *adjustSizeKernel.add_operands();
		threads.set_mode(hir::pb::In);
		threads.set_variable(U32_1->second.id);
	
		hir::pb::Operand& shared = *adjustSizeKernel.add_operands();
		shared.set_mode(hir::pb::In);
		shared.set_variable(U32_0->second.id);
		
		hir::pb::Operand& size = *adjustSizeKernel.add_operands();
		hir::pb::Operand& size_key = *adjustSizeKernel.add_operands();
		hir::pb::Operand& size_value = *adjustSizeKernel.add_operands();

		size.set_mode(hir::pb::In);
		size.set_variable(leftSize->second.id);
		size_key.set_mode(hir::pb::Out);
		size_key.set_variable(sizeVariable_leftkey->second.id);
		size_value.set_mode(hir::pb::Out);
		size_value.set_variable(sizeVariable_leftvalue->second.id);

		RelationalAlgebraKernel ptxKernel(
			temp_split_left_key->second.types, temp_split_left_value->second.types, sourceA->second.types, 
			RelationalAlgebraKernel::SplitGetResultSize);

		ptxKernel.set_id(kernel_id++);
		adjustSizeKernel.set_name(ptxKernel.name());
		adjustSizeKernel.set_code(compilePTXSource(
			ptxKernel.cudaSourceRepresentation()));

		// Resize the output array of split
		report("resize the output size of split.");
		hir::pb::Kernel& resizeResultKernel_split_key = *block.add_kernels();
		
		resizeResultKernel_split_key.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeResultD_split_key    
			= *resizeResultKernel_split_key.add_operands();
		hir::pb::Operand& resizeResultSize_split_key 
			= *resizeResultKernel_split_key.add_operands();
		
		resizeResultD_split_key.set_mode(hir::pb::Out);
		resizeResultD_split_key.set_variable(temp_split_left_key->second.id);
		
		resizeResultSize_split_key.set_mode(hir::pb::In);
		resizeResultSize_split_key.set_variable(sizeVariable_leftkey->second.id);
	
		resizeResultKernel_split_key.set_name("resize");
		resizeResultKernel_split_key.set_code("");
	
		hir::pb::Kernel& resizeResultKernel_split_value = *block.add_kernels();
		
		resizeResultKernel_split_value.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeResultD_split_value    
			= *resizeResultKernel_split_value.add_operands();
		hir::pb::Operand& resizeResultSize_split_value 
			= *resizeResultKernel_split_value.add_operands();
		
		resizeResultD_split_value.set_mode(hir::pb::Out);
		resizeResultD_split_value.set_variable(temp_split_left_value->second.id);
		
		resizeResultSize_split_value.set_mode(hir::pb::In);
		resizeResultSize_split_value.set_variable(sizeVariable_leftvalue->second.id);
	
		resizeResultKernel_split_value.set_name("resize");
		resizeResultKernel_split_value.set_code("");

		report("add split.");
		hir::pb::Kernel& kernel_split = *block.add_kernels();
	
		kernel_split.set_type(hir::pb::ComputeKernel);
		
		hir::pb::Operand& ctas_split = *kernel_split.add_operands();
		ctas_split.set_mode(hir::pb::In);
		ctas_split.set_variable(U32_350->second.id);
		
		hir::pb::Operand& threads_split = *kernel_split.add_operands();
		threads_split.set_mode(hir::pb::In);
		threads_split.set_variable(U32_256->second.id);
		
		hir::pb::Operand& shared_split = *kernel_split.add_operands();
		shared_split.set_mode(hir::pb::In);
		shared_split.set_variable(U32_0->second.id);
	
		hir::pb::Operand& d_key = *kernel_split.add_operands();

		d_key.set_mode(hir::pb::InOut);
		d_key.set_variable(temp_split_left_key->second.id);
	
		hir::pb::Operand& d_value = *kernel_split.add_operands();

		d_value.set_mode(hir::pb::InOut);
		d_value.set_variable(temp_split_left_value->second.id);
	
		RelationalAlgebraKernel ptxKernel_split(RelationalAlgebraKernel::Split, 
			temp_split_left_key->second.types,
			temp_split_left_value->second.types, sourceA->second.types, 
			join.keycount());
	
		hir::pb::Operand& a_split = *kernel_split.add_operands();
		a_split.set_variable(sourceA->second.id);
		a_split.set_mode(hir::pb::In);

		ptxKernel_split.set_id(kernel_id++);
		kernel_split.set_name(ptxKernel_split.name());
		kernel_split.set_code(compilePTXSource(ptxKernel_split.cudaSourceRepresentation()));
		
		hir::pb::Operand& aSize_split = *kernel_split.add_operands();
		aSize_split.set_variable(leftSize->second.id);
		aSize_split.set_mode(hir::pb::In);
	}

	if(!skipSplitRight)
	{
		sizeVariable_rightkey = getTempIntVariable(variables, cfg);
		sizeVariable_rightvalue = getTempIntVariable(variables, cfg);

		// Get the size of the output array of split
		report("get the output size of split.");
		hir::pb::Kernel& adjustSizeKernel = *block.add_kernels();

		adjustSizeKernel.set_type(hir::pb::ComputeKernel);
		
		hir::pb::Operand& ctas = *adjustSizeKernel.add_operands();
		ctas.set_mode(hir::pb::In);
		ctas.set_variable(U32_1->second.id);
	
		hir::pb::Operand& threads = *adjustSizeKernel.add_operands();
		threads.set_mode(hir::pb::In);
		threads.set_variable(U32_1->second.id);
	
		hir::pb::Operand& shared = *adjustSizeKernel.add_operands();
		shared.set_mode(hir::pb::In);
		shared.set_variable(U32_0->second.id);
		
		hir::pb::Operand& size = *adjustSizeKernel.add_operands();
		hir::pb::Operand& size_key = *adjustSizeKernel.add_operands();
		hir::pb::Operand& size_value = *adjustSizeKernel.add_operands();

		size.set_mode(hir::pb::In);
		size.set_variable(rightSize->second.id);
		size_key.set_mode(hir::pb::Out);
		size_key.set_variable(sizeVariable_rightkey->second.id);
		size_value.set_mode(hir::pb::Out);
		size_value.set_variable(sizeVariable_rightvalue->second.id);

		RelationalAlgebraKernel ptxKernel(
			temp_split_right_key->second.types, temp_split_right_value->second.types, sourceB->second.types, 
			RelationalAlgebraKernel::SplitGetResultSize);

		ptxKernel.set_id(kernel_id++);
		adjustSizeKernel.set_name(ptxKernel.name());
		adjustSizeKernel.set_code(compilePTXSource(
			ptxKernel.cudaSourceRepresentation()));

		// Resize the output array of split
		report("resize the output size of split.");
		hir::pb::Kernel& resizeResultKernel_split_key = *block.add_kernels();
		
		resizeResultKernel_split_key.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeResultD_split_key    
			= *resizeResultKernel_split_key.add_operands();
		hir::pb::Operand& resizeResultSize_split_key 
			= *resizeResultKernel_split_key.add_operands();
		
		resizeResultD_split_key.set_mode(hir::pb::Out);
		resizeResultD_split_key.set_variable(temp_split_right_key->second.id);
		
		resizeResultSize_split_key.set_mode(hir::pb::In);
		resizeResultSize_split_key.set_variable(sizeVariable_rightkey->second.id);
	
		resizeResultKernel_split_key.set_name("resize");
		resizeResultKernel_split_key.set_code("");
	
		hir::pb::Kernel& resizeResultKernel_split_value = *block.add_kernels();
		
		resizeResultKernel_split_value.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeResultD_split_value    
			= *resizeResultKernel_split_value.add_operands();
		hir::pb::Operand& resizeResultSize_split_value 
			= *resizeResultKernel_split_value.add_operands();
		
		resizeResultD_split_value.set_mode(hir::pb::Out);
		resizeResultD_split_value.set_variable(temp_split_right_value->second.id);
		
		resizeResultSize_split_value.set_mode(hir::pb::In);
		resizeResultSize_split_value.set_variable(sizeVariable_rightvalue->second.id);
	
		resizeResultKernel_split_value.set_name("resize");
		resizeResultKernel_split_value.set_code("");

		report("add split.");
		hir::pb::Kernel& kernel_split = *block.add_kernels();
	
		kernel_split.set_type(hir::pb::ComputeKernel);
		
		hir::pb::Operand& ctas_split = *kernel_split.add_operands();
		ctas_split.set_mode(hir::pb::In);
		ctas_split.set_variable(U32_350->second.id);
		
		hir::pb::Operand& threads_split = *kernel_split.add_operands();
		threads_split.set_mode(hir::pb::In);
		threads_split.set_variable(U32_256->second.id);
		
		hir::pb::Operand& shared_split = *kernel_split.add_operands();
		shared_split.set_mode(hir::pb::In);
		shared_split.set_variable(U32_0->second.id);
	
		hir::pb::Operand& d_key = *kernel_split.add_operands();

		d_key.set_mode(hir::pb::InOut);
		d_key.set_variable(temp_split_right_key->second.id);
	
		hir::pb::Operand& d_value = *kernel_split.add_operands();

		d_value.set_mode(hir::pb::InOut);
		d_value.set_variable(temp_split_right_value->second.id);
	
		RelationalAlgebraKernel ptxKernel_split(RelationalAlgebraKernel::Split, 
			temp_split_right_key->second.types,
			temp_split_right_value->second.types, sourceB->second.types, 
			join.keycount());
	
		hir::pb::Operand& a_split = *kernel_split.add_operands();
		a_split.set_variable(sourceB->second.id);
		a_split.set_mode(hir::pb::In);

		ptxKernel_split.set_id(kernel_id++);
		kernel_split.set_name(ptxKernel_split.name());
		kernel_split.set_code(compilePTXSource(ptxKernel_split.cudaSourceRepresentation()));
		
		hir::pb::Operand& aSize_split = *kernel_split.add_operands();
		aSize_split.set_variable(rightSize->second.id);
		aSize_split.set_mode(hir::pb::In);
	}

	if(sourceA->second.isSorted < (unsigned int)join.keycount())
	{
		if(!skipSplitLeft)
			addSortPair(block, cfg, variables, temp_split_left_key, temp_split_left_value, sizeVariable_leftkey);
		else
			addSortKey(block, cfg, variables, temp_split_left_key, sizeVariable_leftkey);
	}

	if(sourceB->second.isSorted < (unsigned int)join.keycount())
	{
		if(!skipSplitRight)
			addSortPair(block, cfg, variables, temp_split_right_key, temp_split_right_value, sizeVariable_rightkey);
		else
			addSortKey(block, cfg, variables, temp_split_right_key, sizeVariable_rightkey);
	}

	VariableMap::iterator lowerBound  = getTempBuffer(variables, cfg, 0);
	VariableMap::iterator leftCount = getTempBuffer(variables, cfg, 0);

	VariableMap::const_iterator lowerBoundSize = getTempIntVariable(variables, cfg);
//	VariableMap::const_iterator leftCountSize = getTempIntVariable(variables, cfg);

	{
		// Add the kernel to get the size of the temp output
		hir::pb::Kernel& getTempSize = *block.add_kernels();
	
		getTempSize.set_type(hir::pb::ComputeKernel);
		
		hir::pb::Operand& getTempCtas = *getTempSize.add_operands();
		getTempCtas.set_mode(hir::pb::In);
		getTempCtas.set_variable(U32_1->second.id);
	
		hir::pb::Operand& getTempThreads = *getTempSize.add_operands();
		getTempThreads.set_mode(hir::pb::In);
		getTempThreads.set_variable(U32_1->second.id);
	
		hir::pb::Operand& getTempShared = *getTempSize.add_operands();
		getTempShared.set_mode(hir::pb::In);
		getTempShared.set_variable(U32_0->second.id);
	
		hir::pb::Operand& tempSizeOperand1 = *getTempSize.add_operands();
		tempSizeOperand1.set_variable(lowerBoundSize->second.id);
		tempSizeOperand1.set_mode(hir::pb::Out);
		
		hir::pb::Operand& getTempLeftSize = *getTempSize.add_operands();
		getTempLeftSize.set_variable(leftSize->second.id);
		getTempLeftSize.set_mode(hir::pb::In);
		
		RelationalAlgebraKernel getTempSizePtx(
			RelationalAlgebraKernel::ModernGPUJoinTempSize,	sourceA->second.types);
	
		getTempSizePtx.setCtaCount(1);
	
		getTempSizePtx.set_id(kernel_id++);
		getTempSize.set_name(getTempSizePtx.name());
		getTempSize.set_code(compilePTXSource(
			getTempSizePtx.cudaSourceRepresentation()));
	}
	{	
		// Resize the temp output to the appropriate size
		hir::pb::Kernel& resizeTempKernel = *block.add_kernels();
		
		resizeTempKernel.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeTempD    = *resizeTempKernel.add_operands();
		hir::pb::Operand& resizeTempSize = *resizeTempKernel.add_operands();
		
		resizeTempD.set_mode(hir::pb::Out);
		resizeTempD.set_variable(lowerBound->second.id);
		
		resizeTempSize.set_mode(hir::pb::In);
		resizeTempSize.set_variable(lowerBoundSize->second.id);
	
		resizeTempKernel.set_name("resize");
		resizeTempKernel.set_code("");
	}
	{	
		// Resize the temp output to the appropriate size
		hir::pb::Kernel& resizeTempKernel = *block.add_kernels();
		
		resizeTempKernel.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeTempD    = *resizeTempKernel.add_operands();
		hir::pb::Operand& resizeTempSize = *resizeTempKernel.add_operands();
		
		resizeTempD.set_mode(hir::pb::Out);
		resizeTempD.set_variable(leftCount->second.id);
		
		resizeTempSize.set_mode(hir::pb::In);
		resizeTempSize.set_variable(lowerBoundSize->second.id);
	
		resizeTempKernel.set_name("resize");
		resizeTempKernel.set_code("");
	}

        VariableMap::const_iterator sizeVariable = getTempIntVariable(variables, cfg);

	{
		report("add join find bounds.");

		hir::pb::Kernel& findBoundKernel = *block.add_kernels();
		findBoundKernel.set_type(hir::pb::BinaryKernel);

		hir::pb::Operand& d_lowerBound = *findBoundKernel.add_operands();
		d_lowerBound.set_variable(lowerBound->second.id);
		d_lowerBound.set_mode(hir::pb::InOut);
	
		hir::pb::Operand& d_leftCount = *findBoundKernel.add_operands();
		d_leftCount.set_variable(leftCount->second.id);
		d_leftCount.set_mode(hir::pb::InOut);

		hir::pb::Operand& d_size = *findBoundKernel.add_operands();
		d_size.set_variable(sizeVariable->second.id);
		d_size.set_mode(hir::pb::Out);

		hir::pb::Operand& a_left_key = *findBoundKernel.add_operands();
		a_left_key.set_variable(temp_split_left_key->second.id);
		a_left_key.set_mode(hir::pb::In);

		hir::pb::Operand& a_left_key_size = *findBoundKernel.add_operands();
		a_left_key_size.set_variable(sizeVariable_leftkey->second.id);
		a_left_key_size.set_mode(hir::pb::In);

		hir::pb::Operand& a_right_key = *findBoundKernel.add_operands();
		a_right_key.set_variable(temp_split_right_key->second.id);
		a_right_key.set_mode(hir::pb::In);

		hir::pb::Operand& a_right_key_size = *findBoundKernel.add_operands();
		a_right_key_size.set_variable(sizeVariable_rightkey->second.id);
		a_right_key_size.set_mode(hir::pb::In);

		RelationalAlgebraKernel binKernel(RelationalAlgebraKernel::ModernGPUJoinFindBounds,
			lowerBound->second.types, leftCount->second.types,
			temp_split_left_key->second.types, temp_split_right_key->second.types);
		findBoundKernel.set_type(hir::pb::BinaryKernel);

		binKernel.set_id(kernel_id++);
		findBoundKernel.set_name(binKernel.name());
		findBoundKernel.set_code(compileBINSource(binKernel.cudaSourceRepresentation(), 
			RelationalAlgebraKernel::ModernGPUJoinFindBounds));
	}

        VariableMap::const_iterator sizeVariable_index = getTempIntVariable(variables, cfg);

	{
		// Add the kernel to get the size of the temp output
		hir::pb::Kernel& getResultSize = *block.add_kernels();
	
		getResultSize.set_type(hir::pb::ComputeKernel);
		
		hir::pb::Operand& getTempCtas = *getResultSize.add_operands();
		getTempCtas.set_mode(hir::pb::In);
		getTempCtas.set_variable(U32_1->second.id);
	
		hir::pb::Operand& getTempThreads = *getResultSize.add_operands();
		getTempThreads.set_mode(hir::pb::In);
		getTempThreads.set_variable(U32_1->second.id);
	
		hir::pb::Operand& getTempShared = *getResultSize.add_operands();
		getTempShared.set_mode(hir::pb::In);
		getTempShared.set_variable(U32_0->second.id);
	
		hir::pb::Operand& tempSizeOperand1 = *getResultSize.add_operands();
		tempSizeOperand1.set_variable(sizeVariable_index->second.id);
		tempSizeOperand1.set_mode(hir::pb::Out);
		
		hir::pb::Operand& tempSizeOperand2 = *getResultSize.add_operands();
		tempSizeOperand2.set_variable(sizeVariable->second.id);
		tempSizeOperand2.set_mode(hir::pb::InOut);
		
		RelationalAlgebraKernel getResultSizePtx(
			RelationalAlgebraKernel::ModernGPUJoinResultSize, dId->second.types);
	
		getResultSizePtx.setCtaCount(1);
	
		getResultSizePtx.set_id(kernel_id++);
		getResultSize.set_name(getResultSizePtx.name());
		getResultSize.set_code(compilePTXSource(
			getResultSizePtx.cudaSourceRepresentation()));
	}

	VariableMap::iterator leftIndices  = getTempBuffer(variables, cfg, 0);
	VariableMap::iterator rightIndices = getTempBuffer(variables, cfg, 0);

	{	
		// Resize the temp output to the appropriate size
		hir::pb::Kernel& resizeTempKernel = *block.add_kernels();
		
		resizeTempKernel.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeTempD    = *resizeTempKernel.add_operands();
		hir::pb::Operand& resizeTempSize = *resizeTempKernel.add_operands();
		
		resizeTempD.set_mode(hir::pb::Out);
		resizeTempD.set_variable(leftIndices->second.id);
		
		resizeTempSize.set_mode(hir::pb::In);
		resizeTempSize.set_variable(sizeVariable_index->second.id);
	
		resizeTempKernel.set_name("resize");
		resizeTempKernel.set_code("");
	}
	{	
		// Resize the temp output to the appropriate size
		hir::pb::Kernel& resizeTempKernel = *block.add_kernels();
		
		resizeTempKernel.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeTempD    = *resizeTempKernel.add_operands();
		hir::pb::Operand& resizeTempSize = *resizeTempKernel.add_operands();
		
		resizeTempD.set_mode(hir::pb::Out);
		resizeTempD.set_variable(rightIndices->second.id);
		
		resizeTempSize.set_mode(hir::pb::In);
		resizeTempSize.set_variable(sizeVariable_index->second.id);
	
		resizeTempKernel.set_name("resize");
		resizeTempKernel.set_code("");
	}
	{
		report("add join main kernel");

		hir::pb::Kernel& joinMainKernel = *block.add_kernels();
		joinMainKernel.set_type(hir::pb::BinaryKernel);

		hir::pb::Operand& d_left = *joinMainKernel.add_operands();
		d_left.set_variable(leftIndices->second.id);
		d_left.set_mode(hir::pb::InOut);
	
		hir::pb::Operand& d_right = *joinMainKernel.add_operands();
		d_right.set_variable(rightIndices->second.id);
		d_right.set_mode(hir::pb::InOut);
	
		hir::pb::Operand& d_size = *joinMainKernel.add_operands();
		d_size.set_variable(sizeVariable_index->second.id);
		d_size.set_mode(hir::pb::In);

		hir::pb::Operand& a_lowerBound = *joinMainKernel.add_operands();
		a_lowerBound.set_variable(lowerBound->second.id);
		a_lowerBound.set_mode(hir::pb::In);

		hir::pb::Operand& a_count = *joinMainKernel.add_operands();
		a_count.set_variable(leftCount->second.id);
		a_count.set_mode(hir::pb::In);

		hir::pb::Operand& a_size = *joinMainKernel.add_operands();
		a_size.set_variable(lowerBoundSize->second.id);
		a_size.set_mode(hir::pb::In);

		hir::pb::Operand& a_right_key_size = *joinMainKernel.add_operands();
		a_right_key_size.set_variable(rightSize->second.id);
		a_right_key_size.set_mode(hir::pb::In);

		RelationalAlgebraKernel binKernel(RelationalAlgebraKernel::ModernGPUJoinMain,
			lowerBound->second.types, leftCount->second.types,
			temp_split_left_key->second.types, temp_split_right_key->second.types);
		joinMainKernel.set_type(hir::pb::BinaryKernel);

		binKernel.set_id(kernel_id++);
		joinMainKernel.set_name(binKernel.name());
		joinMainKernel.set_code(compileBINSource(binKernel.cudaSourceRepresentation(), 
			RelationalAlgebraKernel::ModernGPUJoinFindBounds));
	}
	{	
		// Resize the temp output to the appropriate size
		hir::pb::Kernel& resizeTempKernel = *block.add_kernels();
		
		resizeTempKernel.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeTempD    = *resizeTempKernel.add_operands();
		hir::pb::Operand& resizeTempSize = *resizeTempKernel.add_operands();
		
		resizeTempD.set_mode(hir::pb::Out);
		resizeTempD.set_variable(dId->second.id);
		
		resizeTempSize.set_mode(hir::pb::In);
		resizeTempSize.set_variable(sizeVariable->second.id);
	
		resizeTempKernel.set_name("resize");
		resizeTempKernel.set_code("");
	}
	{
		// Add the kernel to get the size of the temp output
		hir::pb::Kernel& joinGather = *block.add_kernels();
	
		joinGather.set_type(hir::pb::ComputeKernel);
		
		hir::pb::Operand& getTempCtas = *joinGather.add_operands();
		getTempCtas.set_mode(hir::pb::In);
		getTempCtas.set_variable(U32_350->second.id);
	
		hir::pb::Operand& getTempThreads = *joinGather.add_operands();
		getTempThreads.set_mode(hir::pb::In);
		getTempThreads.set_variable(U32_256->second.id);
	
		hir::pb::Operand& getTempShared = *joinGather.add_operands();
		getTempShared.set_mode(hir::pb::In);
		getTempShared.set_variable(U32_0->second.id);
	
		hir::pb::Operand& d = *joinGather.add_operands();
		d.set_variable(dId->second.id);
		d.set_mode(hir::pb::InOut);

		hir::pb::Operand& left_key = *joinGather.add_operands();
		left_key.set_variable(temp_split_left_key->second.id);
		left_key.set_mode(hir::pb::In);

		hir::pb::Operand& left_index = *joinGather.add_operands();
		left_index.set_variable(leftIndices->second.id);
		left_index.set_mode(hir::pb::In);

		hir::pb::Operand& right_key = *joinGather.add_operands();
		right_key.set_variable(temp_split_right_key->second.id);
		right_key.set_mode(hir::pb::In);

		hir::pb::Operand& right_index = *joinGather.add_operands();
		right_index.set_variable(rightIndices->second.id);
		right_index.set_mode(hir::pb::In);

		hir::pb::Operand& size = *joinGather.add_operands();
		size.set_variable(sizeVariable_index->second.id);
		size.set_mode(hir::pb::In);

		if(!skipSplitLeft)
		{
			hir::pb::Operand& left_value = *joinGather.add_operands();
			left_value.set_variable(temp_split_left_value->second.id);
			left_value.set_mode(hir::pb::In);
		}

		if(!skipSplitRight)
		{
			hir::pb::Operand& right_value = *joinGather.add_operands();
			right_value.set_variable(temp_split_right_value->second.id);
			right_value.set_mode(hir::pb::In);
		}

		if(skipSplitLeft && skipSplitRight)
		{
		RelationalAlgebraKernel joinGatherPtx(
			RelationalAlgebraKernel::ModernGPUJoinGather, dId->second.types,
			temp_split_left_key->second.types, temp_split_right_key->second.types, join.keycount(),
			RelationalAlgebraKernel::Variable(), RelationalAlgebraKernel::Variable());

		joinGatherPtx.setCtaCount(1);
		joinGatherPtx.set_id(kernel_id++);
		joinGather.set_name(joinGatherPtx.name());
		joinGather.set_code(compilePTXSource(
			joinGatherPtx.cudaSourceRepresentation()));
		}
		else if(skipSplitLeft && !skipSplitRight)
		{
		RelationalAlgebraKernel joinGatherPtx(
			RelationalAlgebraKernel::ModernGPUJoinGather, dId->second.types,
			temp_split_left_key->second.types, temp_split_right_key->second.types, join.keycount(),
			RelationalAlgebraKernel::Variable(), temp_split_right_value->second.types);
		joinGatherPtx.setCtaCount(1);
		joinGatherPtx.set_id(kernel_id++);
		joinGather.set_name(joinGatherPtx.name());
		joinGather.set_code(compilePTXSource(
			joinGatherPtx.cudaSourceRepresentation()));
		}
		else if(!skipSplitLeft && skipSplitRight)
		{
		RelationalAlgebraKernel joinGatherPtx(
			RelationalAlgebraKernel::ModernGPUJoinGather, dId->second.types,
			temp_split_left_key->second.types, temp_split_right_key->second.types, join.keycount(),
			temp_split_left_value->second.types, RelationalAlgebraKernel::Variable());
		joinGatherPtx.setCtaCount(1);
		joinGatherPtx.set_id(kernel_id++);
		joinGather.set_name(joinGatherPtx.name());
		joinGather.set_code(compilePTXSource(
			joinGatherPtx.cudaSourceRepresentation()));
		}
		else if(!skipSplitLeft && !skipSplitRight)
		{
		RelationalAlgebraKernel joinGatherPtx(
			RelationalAlgebraKernel::ModernGPUJoinGather, dId->second.types,
			temp_split_left_key->second.types, temp_split_right_key->second.types, join.keycount(),
			temp_split_left_value->second.types, temp_split_right_value->second.types);
		joinGatherPtx.setCtaCount(1);
		joinGatherPtx.set_id(kernel_id++);
		joinGather.set_name(joinGatherPtx.name());
		joinGather.set_code(compilePTXSource(
			joinGatherPtx.cudaSourceRepresentation()));
		}
	}
}

static void addString(hir::pb::BasicBlock& block, VariableMap& variables,
	hir::pb::KernelControlFlowGraph& cfg, 
	std::string source, std::string name, unsigned int index)
{
	VariableMap::const_iterator sourceA = variables.find(source);
	assert(sourceA != variables.end());

	VariableMap::iterator dId = variables.find(name);
	assert(dId != variables.end());

	report("Set String Address");
	report("   name: " << name
		<< " (" << dId->second.id << ")");
	report("   srcA: " << source 
		<< " (" << sourceA->second.id << ")");

	// Get the size of the input array
	hir::pb::Kernel& sizeKernel = *block.add_kernels();
	
	VariableMap::const_iterator sizeVariable = 
		getTempIntVariable(variables, cfg);
	sizeKernel.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& getSizeD  = *sizeKernel.add_operands();
	hir::pb::Operand& getSizeIn = *sizeKernel.add_operands();
	
	getSizeIn.set_mode(hir::pb::In);
	getSizeIn.set_variable(sourceA->second.id);
	
	getSizeD.set_mode(hir::pb::Out);
	getSizeD.set_variable(sizeVariable->second.id);
	
	sizeKernel.set_name("get_size");
	sizeKernel.set_code("");

	// Add the string address calculation kernel
	hir::pb::Kernel& kernel = *block.add_kernels();

	kernel.set_type(hir::pb::ComputeKernel);

	VariableMap::iterator U32_256 = getConstantIntVariable(variables, cfg, 256);
	VariableMap::iterator U32_350 = getConstantIntVariable(variables, cfg, 350);
//	VariableMap::iterator U32_32  = getConstantIntVariable(variables, cfg, 32);
	VariableMap::iterator U32_0   = getConstantIntVariable(variables, cfg, 0);
	
	hir::pb::Operand& ctas = *kernel.add_operands();
	ctas.set_mode(hir::pb::In);
	ctas.set_variable(U32_350->second.id);
//	ctas.set_variable(U32_256->second.id);
	
	hir::pb::Operand& threads = *kernel.add_operands();
	threads.set_mode(hir::pb::In);
	threads.set_variable(U32_256->second.id);
//	threads.set_variable(U32_32->second.id);
	
	hir::pb::Operand& shared = *kernel.add_operands();
	shared.set_mode(hir::pb::In);
	shared.set_variable(U32_0->second.id);

	hir::pb::Operand& d = *kernel.add_operands();
	d.set_mode(hir::pb::In);
	d.set_variable(dId->second.id);

	RelationalAlgebraKernel ptxKernel(sourceA->second.types, index);

	hir::pb::Operand& a = *kernel.add_operands();
	a.set_variable(sourceA->second.id);
	a.set_mode(hir::pb::In);

	ptxKernel.set_id(kernel_id++);
	kernel.set_name(ptxKernel.name());
	kernel.set_code(compilePTXSource(ptxKernel.cudaSourceRepresentation()));

	hir::pb::Operand& aSize = *kernel.add_operands();
	aSize.set_variable(sizeVariable->second.id);
	aSize.set_mode(hir::pb::In);
}



static void addUnique(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	VariableMap& variables, VariableMap::const_iterator dId,
	VariableMap::const_iterator sizeVariable)
{
	hir::pb::Kernel& uniqueKernel = *block.add_kernels();

	uniqueKernel.set_type(hir::pb::BinaryKernel);
	hir::pb::Operand& d = *uniqueKernel.add_operands();

	d.set_variable(dId->second.id);
	d.set_mode(hir::pb::InOut);

	hir::pb::Operand& dSize = *uniqueKernel.add_operands();
	dSize.set_variable(sizeVariable->second.id);
	dSize.set_mode(hir::pb::InOut);
	
	VariableMap::iterator type = getConstantIntVariable(variables, 
		cfg, getTupleDataType(dId->second.types));

	RelationalAlgebraKernel binKernel(RelationalAlgebraKernel::Unique,
		dId->second.types, type->second.types);

	hir::pb::Operand& a = *uniqueKernel.add_operands();
	a.set_variable(type->second.id);
	a.set_mode(hir::pb::In);

	binKernel.set_id(kernel_id++);
	uniqueKernel.set_name(binKernel.name());
	uniqueKernel.set_code(compileBINSource(binKernel.cudaSourceRepresentation(), 
		RelationalAlgebraKernel::Unique));

	report("resize the output size of unique.");
	hir::pb::Kernel& resizeResultKernel = *block.add_kernels();
	
	resizeResultKernel.set_type(hir::pb::UpdateSize);
	
	hir::pb::Operand& resizeResultD    
		= *resizeResultKernel.add_operands();
	hir::pb::Operand& resizeResultSize
		= *resizeResultKernel.add_operands();
	
	resizeResultD.set_mode(hir::pb::In);
	resizeResultD.set_variable(dId->second.id);
	
	resizeResultSize.set_mode(hir::pb::In);
	resizeResultSize.set_variable(sizeVariable->second.id);

	resizeResultKernel.set_name("updatesize");
	resizeResultKernel.set_code("");
}

static void addProjection(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	const RepeatedExp& reordering, VariableMap& variables,
	std::string& source, std::string& dest, int srcwidth)
{
	VariableMap::const_iterator sourceA = variables.find(source);
	assert(sourceA != variables.end());

	VariableMap::iterator dId = variables.find(dest);
	assert(dId != variables.end());

	report("Projection");
	report("   destination: " << dest
		<< " (" << dId->second.id << " " << dId->second.unique_keys << ")");
	report("   srcA: " << source 
		<< " (" << sourceA->second.id << " " << sourceA->second.unique_keys << ")");

	// get keep fields
	RelationalAlgebraKernel::IndexVector keepFields;
	int numArith = 0;

	for(RepeatedExp::const_iterator 
		i = reordering.begin();
		i != reordering.end(); ++i)
	{
		if(i->tag() == lb::INDEX)
		{	
			int j = i->index().offset();
			keepFields.push_back(j);
		}
		else if(i->tag() == lb::MIXED)
		{
			int j = srcwidth + numArith;
			keepFields.push_back(j);
			numArith++; 
		}
		else if(i->tag() == lb::CONVERT)
		{
			int j = srcwidth + numArith;
			keepFields.push_back(j);
			numArith++; 
		}
		else if(i->tag() == lb::ARITHEXP)
		{
			int j = srcwidth + numArith;
			keepFields.push_back(j);
			numArith++; 
		}
		else if(i->tag() == lb::CALL)
		{
			if(i->call().calltype() == lb::FDatetime 
  			  && i->call().callname().compare("subtract") == 0)
			{
				int j = srcwidth + numArith;
				keepFields.push_back(j);
				numArith++;
			}
			else if(i->call().calltype() == lb::FDatetime 
  			  && i->call().callname().compare("part") == 0)
			{
				int j = srcwidth + numArith;
				keepFields.push_back(j);
				numArith++;
			}
			else if(i->call().calltype() == lb::FDatetime 
  			  && i->call().callname().compare("add") == 0)
			{
				int j = srcwidth + numArith;
				keepFields.push_back(j);
				numArith++;
			}
			else if(i->call().calltype() == lb::FString 
  			  && i->call().callname().compare("add") == 0)
			{
				int j = srcwidth + numArith;
				keepFields.push_back(j);
				numArith++;
			}
			else if(i->call().calltype() == lb::FString 
  			  && i->call().callname().compare("substring") == 0)
			{
				int j = srcwidth + numArith;
				keepFields.push_back(j);
				numArith++;
			}
		}
	}

	assert(!keepFields.empty());
	report("Keep fields " 
		<< hydrazine::toString(keepFields.begin(), keepFields.end())
		<< " out of " << sourceA->second.types.size());
//	assert(keepFields.size() <= sourceA->second.types.size());

	bool foundAllKeys = true;

	for(unsigned int i = 0; i < sourceA->second.sorted_fields.size(); i++)
	{
		if(std::find(keepFields.begin(), keepFields.end(), sourceA->second.sorted_fields[i]) == keepFields.end())
		{
			foundAllKeys = false;
			break;
		}
	}

	if(foundAllKeys)
	{
		dId->second.sorted_fields = sourceA->second.sorted_fields;

		for(unsigned int i = 0; i < sourceA->second.sorted_fields.size(); i++)
			for(unsigned int j = 0; j < keepFields.size(); ++j)
				if(keepFields[j] == sourceA->second.sorted_fields[i])
				{
					dId->second.sorted_fields[i] = j;
					break;
				}
	}
	else
		dId->second.sorted_fields.clear();

	RelationalAlgebraKernel::IndexVector::const_iterator
		keep = keepFields.begin();

	{
		unsigned int i = 0;
		for(; keep != keepFields.end() && i < sourceA->second.isSorted; ++i, ++keep)
		{
			if(*keep != i)
			{
				dId->second.isSorted = i;
				break;
			}
		}

		if(keep == keepFields.end() || i == sourceA->second.isSorted)
		{ 
			dId->second.isSorted = sourceA->second.isSorted;
		}

		if(dId->second.isSorted == 0)
		{
			for(unsigned int i = 0; i < dId->second.sorted_fields.size(); ++i)
				if(dId->second.sorted_fields[i] == i)
					dId->second.isSorted = i + 1;
				else
					break;
		}
	}

	// Get the size of the input array
	hir::pb::Kernel& sizeKernel = *block.add_kernels();
	
	VariableMap::const_iterator sizeVariable = 
		getTempIntVariable(variables, cfg);
	sizeKernel.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& getSizeD  = *sizeKernel.add_operands();
	hir::pb::Operand& getSizeIn = *sizeKernel.add_operands();
	
	getSizeIn.set_mode(hir::pb::In);
	getSizeIn.set_variable(sourceA->second.id);
	
	getSizeD.set_mode(hir::pb::Out);
	getSizeD.set_variable(sizeVariable->second.id);
	
	sizeKernel.set_name("get_size");
	sizeKernel.set_code("");

	VariableMap::iterator U32_1 = getConstantIntVariable(variables, cfg, 1);
	VariableMap::iterator U32_0 = getConstantIntVariable(variables, cfg, 0);
	
	// Get the size of the output array if the primitive size changed
	if(bytes(dId->second.types) != bytes(sourceA->second.types))
	{
		hir::pb::Kernel& adjustSizeKernel = *block.add_kernels();

		adjustSizeKernel.set_type(hir::pb::ComputeKernel);
	
		hir::pb::Operand& ctas = *adjustSizeKernel.add_operands();
		ctas.set_mode(hir::pb::In);
		ctas.set_variable(U32_1->second.id);
	
		hir::pb::Operand& threads = *adjustSizeKernel.add_operands();
		threads.set_mode(hir::pb::In);
		threads.set_variable(U32_1->second.id);
	
		hir::pb::Operand& shared = *adjustSizeKernel.add_operands();
		shared.set_mode(hir::pb::In);
		shared.set_variable(U32_0->second.id);
		
		hir::pb::Operand& size = *adjustSizeKernel.add_operands();

		size.set_mode(hir::pb::InOut);
		size.set_variable(sizeVariable->second.id);
		
		RelationalAlgebraKernel ptxKernel(
			RelationalAlgebraKernel::ProjectGetResultSize, 
			dId->second.types, sourceA->second.types);

		ptxKernel.set_id(kernel_id++);
		adjustSizeKernel.set_name(ptxKernel.name());
		adjustSizeKernel.set_code(compilePTXSource(
			ptxKernel.cudaSourceRepresentation()));
	}

	// Resize the output array
	hir::pb::Kernel& resizeResultKernel = *block.add_kernels();
	
	resizeResultKernel.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeResultD    = *resizeResultKernel.add_operands();
	hir::pb::Operand& resizeResultSize = *resizeResultKernel.add_operands();
	
	resizeResultD.set_mode(hir::pb::Out);
	resizeResultD.set_variable(dId->second.id);
	resizeResultSize.set_mode(hir::pb::InOut);
	resizeResultSize.set_variable(sizeVariable->second.id);

	resizeResultKernel.set_name("resize");
	resizeResultKernel.set_code("");

	// Add the projection kernel
	hir::pb::Kernel& kernel = *block.add_kernels();

	kernel.set_type(hir::pb::ComputeKernel);

	VariableMap::iterator U32_256 = getConstantIntVariable(variables, cfg, 256);
	VariableMap::iterator U32_350 = getConstantIntVariable(variables, cfg, 350);
	
	hir::pb::Operand& ctas = *kernel.add_operands();
	ctas.set_mode(hir::pb::In);
//	ctas.set_variable(U32_256->second.id);
	ctas.set_variable(U32_350->second.id);
	
	hir::pb::Operand& threads = *kernel.add_operands();
	threads.set_mode(hir::pb::In);
	threads.set_variable(U32_256->second.id);
	
	hir::pb::Operand& shared = *kernel.add_operands();
	shared.set_mode(hir::pb::In);
	shared.set_variable(U32_0->second.id);

	hir::pb::Operand& d = *kernel.add_operands();
	d.set_mode(hir::pb::InOut);
	d.set_variable(dId->second.id);

	RelationalAlgebraKernel ptxKernel(dId->second.types,
		sourceA->second.types, keepFields);

	hir::pb::Operand& a = *kernel.add_operands();
	a.set_variable(sourceA->second.id);
	a.set_mode(hir::pb::In);

	ptxKernel.set_id(kernel_id++);
	kernel.set_name(ptxKernel.name());
	kernel.set_code(compilePTXSource(ptxKernel.cudaSourceRepresentation()));
	
	hir::pb::Operand& aSize = *kernel.add_operands();
	aSize.set_variable(sizeVariable->second.id);
	aSize.set_mode(hir::pb::InOut);
	
	bool needUnique = false;

	if(dId->second.unique_keys > 0)
	{
		if(dId->second.unique_keys > sourceA->second.unique_keys && (sourceA->second.sorted_fields.size() > 0 && keepFields[0] != sourceA->second.sorted_fields[0]))
		{
			bool secondcheck = true;

			if(sourceA->second.sorted_fields.size() == dId->second.unique_keys)	
			{
				for(unsigned int i = 0; i < sourceA->second.sorted_fields.size(); ++i)
				{
					if(find(keepFields.begin(), keepFields.end(), sourceA->second.sorted_fields[i]) == keepFields.end())
					{
						secondcheck = false;
						break;
					}
				}
			}
				
				if(!secondcheck) needUnique = true;
		}
		else if(dId->second.unique_keys == sourceA->second.unique_keys)
		{
			needUnique = false;
		}
		else
		{
			unsigned int i = 0;
			for(; i < dId->second.unique_keys; ++i)
				if(keepFields[i] != i)
					break;
			if(i == dId->second.unique_keys)
				needUnique = true;
		}			
	}

	// If sorting was required, it is also necessary to eliminate duplcate
	if(needUnique)
	{	
		if(keepFields.size() == 1 && sourceA->second.types.size()> 1)
		{
//			if(sourceA->second.isSorted < keepFields.size())
			if(sourceA->second.sorted_fields.size() == 0 || (sourceA->second.sorted_fields.size() > 0 && keepFields[0] != sourceA->second.sorted_fields[0]))
				addSortKey(block, cfg, variables, dId, sizeVariable);

			addUnique(block, cfg, variables, dId, sizeVariable);
		}
	}
}

static void addAppendString(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	const lb::Exp& arith,
	VariableMap& variables,
	std::string& source, std::string& dest, const StringIdMap& strings)
{
	VariableMap::const_iterator sourceA = variables.find(source);
	assert(sourceA != variables.end());

	VariableMap::const_iterator dId = variables.find(dest);
	assert(dId != variables.end());

	VariableMap::const_iterator StringTable = variables.find("StringTable");
	assert(StringTable != variables.end());

	VariableMap::iterator dString_Id = getTempBuffer(variables, cfg, 0);

	VariableMap::const_iterator stringSizeVariable = getTempIntVariable(variables, cfg);
	
	// Get the size of the input array
	hir::pb::Kernel& sizeKernel = *block.add_kernels();
	
	VariableMap::const_iterator sizeVariable = 
		getTempIntVariable(variables, cfg);
	sizeKernel.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& getSizeD  = *sizeKernel.add_operands();
	hir::pb::Operand& getSizeIn = *sizeKernel.add_operands();
	
	getSizeIn.set_mode(hir::pb::In);
	getSizeIn.set_variable(sourceA->second.id);
	
	getSizeD.set_mode(hir::pb::Out);
	getSizeD.set_variable(sizeVariable->second.id);
	
	sizeKernel.set_name("get_size");
	sizeKernel.set_code("");

	// Get the size of the output array
	hir::pb::Kernel& adjustSizeKernel = *block.add_kernels();

	adjustSizeKernel.set_type(hir::pb::ComputeKernel);
	
	VariableMap::iterator U32_1_size = getConstantIntVariable(variables, cfg, 1);
	VariableMap::iterator U32_0_size = getConstantIntVariable(variables, cfg, 0);
	
	hir::pb::Operand& ctas_size = *adjustSizeKernel.add_operands();
	ctas_size.set_mode(hir::pb::In);
	ctas_size.set_variable(U32_1_size->second.id);

	hir::pb::Operand& threads_size = *adjustSizeKernel.add_operands();
	threads_size.set_mode(hir::pb::In);
	threads_size.set_variable(U32_1_size->second.id);

	hir::pb::Operand& shared_size = *adjustSizeKernel.add_operands();
	shared_size.set_mode(hir::pb::In);
	shared_size.set_variable(U32_0_size->second.id);
	
	hir::pb::Operand& size = *adjustSizeKernel.add_operands();
	size.set_mode(hir::pb::InOut);
	size.set_variable(sizeVariable->second.id);
		
	hir::pb::Operand& stringSize = *adjustSizeKernel.add_operands();
	stringSize.set_mode(hir::pb::Out);
	stringSize.set_variable(stringSizeVariable->second.id);

	RelationalAlgebraKernel ptxKernel_size(
		RelationalAlgebraKernel::AppendStringGetResultSize, 
		dId->second.types, sourceA->second.types);

	ptxKernel_size.set_id(kernel_id++);
	adjustSizeKernel.set_name(ptxKernel_size.name());
	adjustSizeKernel.set_code(compilePTXSource(
		ptxKernel_size.cudaSourceRepresentation()));

	// Resize the output array
	hir::pb::Kernel& resizeResultKernel = *block.add_kernels();
	
	resizeResultKernel.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeResultD    = *resizeResultKernel.add_operands();
	hir::pb::Operand& resizeResultSize = *resizeResultKernel.add_operands();
	
	resizeResultD.set_mode(hir::pb::Out);
	resizeResultD.set_variable(dId->second.id);
	
	resizeResultSize.set_mode(hir::pb::In);
	resizeResultSize.set_variable(sizeVariable->second.id);

	resizeResultKernel.set_name("resize");
	resizeResultKernel.set_code("");

	hir::pb::Kernel& resizeStringResultKernel = *block.add_kernels();
	
	resizeStringResultKernel.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeStringResultD    = *resizeStringResultKernel.add_operands();
	hir::pb::Operand& resizeStringResultSize = *resizeStringResultKernel.add_operands();
	
	resizeStringResultD.set_mode(hir::pb::Out);
	resizeStringResultD.set_variable(dString_Id->second.id);
	
	resizeStringResultSize.set_mode(hir::pb::In);
	resizeStringResultSize.set_variable(stringSizeVariable->second.id);

	resizeStringResultKernel.set_name("resize");
	resizeStringResultKernel.set_code("");

	// Add the arithmetic kernel
	hir::pb::Kernel& kernel = *block.add_kernels();

	kernel.set_type(hir::pb::ComputeKernel);

	VariableMap::iterator U32_256 = getConstantIntVariable(variables, cfg, 256);
	VariableMap::iterator U32_350 = getConstantIntVariable(variables, cfg, 350);
//	VariableMap::iterator U32_32  = getConstantIntVariable(variables, cfg, 32);
	VariableMap::iterator U32_0   = getConstantIntVariable(variables, cfg, 0);
	
	hir::pb::Operand& ctas = *kernel.add_operands();
	ctas.set_mode(hir::pb::In);
	ctas.set_variable(U32_350->second.id);
	
	hir::pb::Operand& threads = *kernel.add_operands();
	threads.set_mode(hir::pb::In);
	threads.set_variable(U32_256->second.id);
	
	hir::pb::Operand& shared = *kernel.add_operands();
	shared.set_mode(hir::pb::In);
	shared.set_variable(U32_0->second.id);

	hir::pb::Operand& d = *kernel.add_operands();
	d.set_mode(hir::pb::InOut);
	d.set_variable(dId->second.id);
	
	hir::pb::Operand& dString = *kernel.add_operands();
	dString.set_mode(hir::pb::InOut);
	dString.set_variable(dString_Id->second.id);
	
	hir::pb::Operand& a = *kernel.add_operands();
	a.set_variable(sourceA->second.id);
	a.set_mode(hir::pb::In);

	kernel_id++;
	kernel.set_name("append_string");
	kernel.set_code(getAppendStringPTX(arith, strings, sourceA->second, dId->second));
	
	hir::pb::Operand& aSize = *kernel.add_operands();
	aSize.set_variable(sizeVariable->second.id);
	aSize.set_mode(hir::pb::In);

	hir::pb::Operand& string = *kernel.add_operands();
	string.set_variable(StringTable->second.id);
	string.set_mode(hir::pb::In);
}

static void addSubString(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	const lb::Exp& arith,
	VariableMap& variables,
	std::string& source, std::string& dest, const StringIdMap& strings)
{
	VariableMap::const_iterator sourceA = variables.find(source);
	assert(sourceA != variables.end());

	VariableMap::const_iterator dId = variables.find(dest);
	assert(dId != variables.end());

//	VariableMap::const_iterator StringTable = variables.find("StringTable");
//	assert(StringTable != variables.end());

	VariableMap::iterator StringTable = getOutputStringBuffer(
		variables, cfg, "OutputStringTable");
	assert(StringTable != variables.end());
	
	// Get the size of the input array
	hir::pb::Kernel& sizeKernel = *block.add_kernels();
	
	VariableMap::const_iterator sizeVariable = 
		getTempIntVariable(variables, cfg);
	sizeKernel.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& getSizeD  = *sizeKernel.add_operands();
	hir::pb::Operand& getSizeIn = *sizeKernel.add_operands();
	
	getSizeIn.set_mode(hir::pb::In);
	getSizeIn.set_variable(sourceA->second.id);
	
	getSizeD.set_mode(hir::pb::Out);
	getSizeD.set_variable(sizeVariable->second.id);
	
	sizeKernel.set_name("get_size");
	sizeKernel.set_code("");

	hir::pb::Kernel& sizeKernel2 = *block.add_kernels();
	VariableMap::const_iterator StringTableSizeVariable = 
		getTempIntVariable(variables, cfg);
	sizeKernel2.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& getSizeD2  = *sizeKernel2.add_operands();
	hir::pb::Operand& getSizeIn2 = *sizeKernel2.add_operands();
	
	getSizeIn2.set_mode(hir::pb::InOut);
	getSizeIn2.set_variable(StringTable->second.id);
	
	getSizeD2.set_mode(hir::pb::Out);
	getSizeD2.set_variable(StringTableSizeVariable->second.id);
	
	sizeKernel2.set_name("get_size");
	sizeKernel2.set_code("");

	// Get the size of the output array
	hir::pb::Kernel& adjustSizeKernel = *block.add_kernels();

	adjustSizeKernel.set_type(hir::pb::ComputeKernel);
	
	VariableMap::iterator U32_1_size = getConstantIntVariable(variables, cfg, 1);
	VariableMap::iterator U32_0_size = getConstantIntVariable(variables, cfg, 0);
	
	hir::pb::Operand& ctas_size = *adjustSizeKernel.add_operands();
	ctas_size.set_mode(hir::pb::In);
	ctas_size.set_variable(U32_1_size->second.id);

	hir::pb::Operand& threads_size = *adjustSizeKernel.add_operands();
	threads_size.set_mode(hir::pb::In);
	threads_size.set_variable(U32_1_size->second.id);

	hir::pb::Operand& shared_size = *adjustSizeKernel.add_operands();
	shared_size.set_mode(hir::pb::In);
	shared_size.set_variable(U32_0_size->second.id);
	
	hir::pb::Operand& size = *adjustSizeKernel.add_operands();
	size.set_mode(hir::pb::InOut);
	size.set_variable(sizeVariable->second.id);

	RelationalAlgebraKernel ptxKernel_size(
		RelationalAlgebraKernel::SubStringGetResultSize, 
		dId->second.types, sourceA->second.types);

	ptxKernel_size.set_id(kernel_id++);
	adjustSizeKernel.set_name(ptxKernel_size.name());
	adjustSizeKernel.set_code(compilePTXSource(
		ptxKernel_size.cudaSourceRepresentation()));

	// Resize the output array
	hir::pb::Kernel& resizeResultKernel = *block.add_kernels();
	
	resizeResultKernel.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeResultD    = *resizeResultKernel.add_operands();
	hir::pb::Operand& resizeResultSize = *resizeResultKernel.add_operands();
	
	resizeResultD.set_mode(hir::pb::Out);
	resizeResultD.set_variable(dId->second.id);
	
	resizeResultSize.set_mode(hir::pb::In);
	resizeResultSize.set_variable(sizeVariable->second.id);

	resizeResultKernel.set_name("resize");
	resizeResultKernel.set_code("");

	// Add the arithmetic kernel
	hir::pb::Kernel& kernel = *block.add_kernels();

	kernel.set_type(hir::pb::ComputeKernel);

	VariableMap::iterator U32_256 = getConstantIntVariable(variables, cfg, 256);
	VariableMap::iterator U32_350  = getConstantIntVariable(variables, cfg, 350);
//	VariableMap::iterator U32_32  = getConstantIntVariable(variables, cfg, 32);
	VariableMap::iterator U32_0   = getConstantIntVariable(variables, cfg, 0);
	
	hir::pb::Operand& ctas = *kernel.add_operands();
	ctas.set_mode(hir::pb::In);
	ctas.set_variable(U32_350->second.id);
	
	hir::pb::Operand& threads = *kernel.add_operands();
	threads.set_mode(hir::pb::In);
	threads.set_variable(U32_256->second.id);
	
	hir::pb::Operand& shared = *kernel.add_operands();
	shared.set_mode(hir::pb::In);
	shared.set_variable(U32_0->second.id);

	hir::pb::Operand& d = *kernel.add_operands();
	d.set_mode(hir::pb::InOut);
	d.set_variable(dId->second.id);

	hir::pb::Operand& a = *kernel.add_operands();
	a.set_variable(sourceA->second.id);
	a.set_mode(hir::pb::In);
	
	RelationalAlgebraKernel ptxKernel(dId->second.types, sourceA->second.types, /*(blocknum == 39) ? 3 :*/ arith.call().args(0).index().offset());

	ptxKernel.set_id(kernel_id++);
	kernel.set_name(ptxKernel.name());
	kernel.set_code(compilePTXSource(ptxKernel.cudaSourceRepresentation()));

	hir::pb::Operand& aSize = *kernel.add_operands();
	aSize.set_variable(sizeVariable->second.id);
	aSize.set_mode(hir::pb::In);

	hir::pb::Operand& string = *kernel.add_operands();
	string.set_variable(StringTable->second.id);
	string.set_mode(hir::pb::In);

	hir::pb::Operand& stringSize = *kernel.add_operands();
	stringSize.set_variable(StringTableSizeVariable->second.id);
	stringSize.set_mode(hir::pb::In);
}

static void addArith(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	const lb::Exp& arith,
	VariableMap& variables,
	std::string& source, std::string& dest)
{
	VariableMap::const_iterator sourceA = variables.find(source);
	assert(sourceA != variables.end());

	VariableMap::const_iterator dId = variables.find(dest);
	assert(dId != variables.end());

	// Get the size of the input array
	hir::pb::Kernel& sizeKernel = *block.add_kernels();
	
	VariableMap::const_iterator sizeVariable = 
		getTempIntVariable(variables, cfg);
	sizeKernel.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& getSizeD  = *sizeKernel.add_operands();
	hir::pb::Operand& getSizeIn = *sizeKernel.add_operands();
	
	getSizeIn.set_mode(hir::pb::In);
	getSizeIn.set_variable(sourceA->second.id);
	
	getSizeD.set_mode(hir::pb::Out);
	getSizeD.set_variable(sizeVariable->second.id);
	
	sizeKernel.set_name("get_size");
	sizeKernel.set_code("");

	// Get the size of the output array
	hir::pb::Kernel& adjustSizeKernel = *block.add_kernels();

	adjustSizeKernel.set_type(hir::pb::ComputeKernel);
	
	VariableMap::iterator U32_1_size = getConstantIntVariable(variables, cfg, 1);
	VariableMap::iterator U32_0_size = getConstantIntVariable(variables, cfg, 0);
	
	hir::pb::Operand& ctas_size = *adjustSizeKernel.add_operands();
	ctas_size.set_mode(hir::pb::In);
	ctas_size.set_variable(U32_1_size->second.id);

	hir::pb::Operand& threads_size = *adjustSizeKernel.add_operands();
	threads_size.set_mode(hir::pb::In);
	threads_size.set_variable(U32_1_size->second.id);

	hir::pb::Operand& shared_size = *adjustSizeKernel.add_operands();
	shared_size.set_mode(hir::pb::In);
	shared_size.set_variable(U32_0_size->second.id);
	
	hir::pb::Operand& size = *adjustSizeKernel.add_operands();

	size.set_mode(hir::pb::InOut);
	size.set_variable(sizeVariable->second.id);
	
	RelationalAlgebraKernel ptxKernel_size(
		RelationalAlgebraKernel::ArithGetResultSize, 
		dId->second.types, sourceA->second.types);

	ptxKernel_size.set_id(kernel_id++);
	adjustSizeKernel.set_name(ptxKernel_size.name());
	adjustSizeKernel.set_code(compilePTXSource(
		ptxKernel_size.cudaSourceRepresentation()));

	// Resize the output array
	hir::pb::Kernel& resizeResultKernel = *block.add_kernels();
	
	resizeResultKernel.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeResultD    = *resizeResultKernel.add_operands();
	hir::pb::Operand& resizeResultSize = *resizeResultKernel.add_operands();
	
	resizeResultD.set_mode(hir::pb::Out);
	resizeResultD.set_variable(dId->second.id);
	
	resizeResultSize.set_mode(hir::pb::In);
	resizeResultSize.set_variable(sizeVariable->second.id);

	resizeResultKernel.set_name("resize");
	resizeResultKernel.set_code("");

	// Add the arithmetic kernel
	hir::pb::Kernel& kernel = *block.add_kernels();

	kernel.set_type(hir::pb::ComputeKernel);

	VariableMap::iterator U32_350 = getConstantIntVariable(variables, cfg, 350);
	VariableMap::iterator U32_512  = getConstantIntVariable(variables, cfg, 512);
	VariableMap::iterator U32_0   = getConstantIntVariable(variables, cfg, 0);
	
	hir::pb::Operand& ctas = *kernel.add_operands();
	ctas.set_mode(hir::pb::In);
	ctas.set_variable(U32_350->second.id);
	
	hir::pb::Operand& threads = *kernel.add_operands();
	threads.set_mode(hir::pb::In);
	threads.set_variable(U32_512->second.id);
	
	hir::pb::Operand& shared = *kernel.add_operands();
	shared.set_mode(hir::pb::In);
	shared.set_variable(U32_0->second.id);

	hir::pb::Operand& d = *kernel.add_operands();

	d.set_mode(hir::pb::InOut);
	d.set_variable(dId->second.id);
	
	RelationalAlgebraKernel ptxKernel(translateArith(arith, sourceA),
		dId->second.types, sourceA->second.types);

	hir::pb::Operand& a = *kernel.add_operands();

	a.set_variable(sourceA->second.id);
	a.set_mode(hir::pb::In);

	ptxKernel.set_id(kernel_id++);
	kernel.set_name(ptxKernel.name());
	kernel.set_code(compilePTXSource(ptxKernel.cudaSourceRepresentation()));
	
	hir::pb::Operand& aSize = *kernel.add_operands();
	
	aSize.set_variable(sizeVariable->second.id);
	aSize.set_mode(hir::pb::In);
}

static void addConv(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	RelationalAlgebraKernel::DataType& type,
	unsigned int& offset,
	VariableMap& variables,
	std::string& source, std::string& dest)
{
	VariableMap::const_iterator sourceA = variables.find(source);
	assert(sourceA != variables.end());

	VariableMap::const_iterator dId = variables.find(dest);
	assert(dId != variables.end());

	// Get the size of the input array
	hir::pb::Kernel& sizeKernel = *block.add_kernels();
	
	VariableMap::const_iterator sizeVariable = 
		getTempIntVariable(variables, cfg);
	sizeKernel.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& getSizeD  = *sizeKernel.add_operands();
	hir::pb::Operand& getSizeIn = *sizeKernel.add_operands();
	
	getSizeIn.set_mode(hir::pb::In);
	getSizeIn.set_variable(sourceA->second.id);
	
	getSizeD.set_mode(hir::pb::Out);
	getSizeD.set_variable(sizeVariable->second.id);
	
	sizeKernel.set_name("get_size");
	sizeKernel.set_code("");

	// Get the size of the output array
	hir::pb::Kernel& adjustSizeKernel = *block.add_kernels();

	adjustSizeKernel.set_type(hir::pb::ComputeKernel);
	
	VariableMap::iterator U32_1_size = getConstantIntVariable(variables, cfg, 1);
	VariableMap::iterator U32_0_size = getConstantIntVariable(variables, cfg, 0);
	
	hir::pb::Operand& ctas_size = *adjustSizeKernel.add_operands();
	ctas_size.set_mode(hir::pb::In);
	ctas_size.set_variable(U32_1_size->second.id);

	hir::pb::Operand& threads_size = *adjustSizeKernel.add_operands();
	threads_size.set_mode(hir::pb::In);
	threads_size.set_variable(U32_1_size->second.id);

	hir::pb::Operand& shared_size = *adjustSizeKernel.add_operands();
	shared_size.set_mode(hir::pb::In);
	shared_size.set_variable(U32_0_size->second.id);
	
	hir::pb::Operand& size = *adjustSizeKernel.add_operands();

	size.set_mode(hir::pb::InOut);
	size.set_variable(sizeVariable->second.id);
	
	RelationalAlgebraKernel ptxKernel_size(
		RelationalAlgebraKernel::ConvGetResultSize, 
		dId->second.types, sourceA->second.types);

	ptxKernel_size.set_id(kernel_id++);
	adjustSizeKernel.set_name(ptxKernel_size.name());
	adjustSizeKernel.set_code(compilePTXSource(
		ptxKernel_size.cudaSourceRepresentation()));

	// Resize the output array
	hir::pb::Kernel& resizeResultKernel = *block.add_kernels();
	
	resizeResultKernel.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeResultD    = *resizeResultKernel.add_operands();
	hir::pb::Operand& resizeResultSize = *resizeResultKernel.add_operands();
	
	resizeResultD.set_mode(hir::pb::Out);
	resizeResultD.set_variable(dId->second.id);
	
	resizeResultSize.set_mode(hir::pb::In);
	resizeResultSize.set_variable(sizeVariable->second.id);

	resizeResultKernel.set_name("resize");
	resizeResultKernel.set_code("");

	// Add the conv kernel
	hir::pb::Kernel& kernel = *block.add_kernels();

	kernel.set_type(hir::pb::ComputeKernel);

	VariableMap::iterator U32_256 = getConstantIntVariable(variables, cfg, 256);
	VariableMap::iterator U32_0   = getConstantIntVariable(variables, cfg, 0);
	
	hir::pb::Operand& ctas = *kernel.add_operands();
	ctas.set_mode(hir::pb::In);
	ctas.set_variable(U32_256->second.id);
	
	hir::pb::Operand& threads = *kernel.add_operands();
	threads.set_mode(hir::pb::In);
	threads.set_variable(U32_256->second.id);
	
	hir::pb::Operand& shared = *kernel.add_operands();
	shared.set_mode(hir::pb::In);
	shared.set_variable(U32_0->second.id);

	hir::pb::Operand& d = *kernel.add_operands();

	d.set_mode(hir::pb::InOut);
	d.set_variable(dId->second.id);

	RelationalAlgebraKernel ptxKernel(type, offset,
		dId->second.types, sourceA->second.types);

	hir::pb::Operand& a = *kernel.add_operands();

	a.set_variable(sourceA->second.id);
	a.set_mode(hir::pb::In);

	ptxKernel.set_id(kernel_id++);
	kernel.set_name(ptxKernel.name());
	kernel.set_code(compilePTXSource(ptxKernel.cudaSourceRepresentation()));
	
	hir::pb::Operand& aSize = *kernel.add_operands();
	
	aSize.set_variable(sizeVariable->second.id);
	aSize.set_mode(hir::pb::In);
}

static void addSelectField(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	const lb::GPUMapFilter& mapFilter, std::vector<lb::Comp>& comp,
	const lb::Test& test, bool isComp, 
	VariableMap& variables,
	const StringIdMap& stringIds, std::string& dest)
{
	const unsigned int selectCtaCount = 128;

	// Get references to inputs/outputs
	VariableMap::const_iterator sourceA = variables.find(mapFilter.srca());
	assert(sourceA != variables.end());

	VariableMap::const_iterator dId = variables.find(dest);
	assert(dId != variables.end());

	report("   destination: " << dest
		<< " (" << dId->second.id << ")");
	report("   srcA: " << mapFilter.srca()
		<< " (" << sourceA->second.id << ")");

	// Allocate scratch variables
	VariableMap::const_iterator outputSize = getTempIntVariable(variables, cfg);
	VariableMap::const_iterator inputSize  = getTempIntVariable(variables, cfg);

	VariableMap::iterator U32_selectCtas 
		= getConstantIntVariable(variables, cfg, selectCtaCount);
	VariableMap::iterator U32_scannedValues
		= getConstantIntVariable(variables, cfg, selectCtaCount + 1);
	
	VariableMap::iterator U32_256 = getConstantIntVariable(variables, cfg, 256);
	VariableMap::iterator U32_128 = getConstantIntVariable(variables, cfg, 128);
	VariableMap::iterator U32_0   = getConstantIntVariable(variables, cfg, 0);
	VariableMap::iterator U32_1   = getConstantIntVariable(variables, cfg, 1);

	// Create a temporary array to facilitate the select
	VariableMap::iterator temp = getTempBuffer(variables, cfg, 0);
	temp->second.types = dId->second.types;

	// Create a histogram to track the elements per cta
	VariableMap::iterator histogram  = getTempBuffer(variables,
		cfg, /*(selectCtaCount + 1)*/ 512 * sizeof(unsigned int));

	// Get the size of the input	
	hir::pb::Kernel& inputSizeKernel = *block.add_kernels();
	
	inputSizeKernel.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& inputSizeD    = *inputSizeKernel.add_operands();
	hir::pb::Operand& inputSizeLeft = *inputSizeKernel.add_operands();
	
	inputSizeD.set_mode(hir::pb::Out);
	inputSizeD.set_variable(inputSize->second.id);
	
	inputSizeLeft.set_mode(hir::pb::In);
	inputSizeLeft.set_variable(sourceA->second.id);
	
	inputSizeKernel.set_name("get_size");
	inputSizeKernel.set_code("");

	// Resize the temp output to the appropriate size
	hir::pb::Kernel& resizeTempKernel = *block.add_kernels();
	
	resizeTempKernel.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeTempD    = *resizeTempKernel.add_operands();
	hir::pb::Operand& resizeTempSize = *resizeTempKernel.add_operands();
	
	resizeTempD.set_mode(hir::pb::Out);
	resizeTempD.set_variable(temp->second.id);
	
	resizeTempSize.set_mode(hir::pb::In);
	resizeTempSize.set_variable(inputSize->second.id);

	resizeTempKernel.set_name("resize");
	resizeTempKernel.set_code("");

	// Add the main select kernel
	hir::pb::Kernel& kernel = *block.add_kernels();

	kernel.set_type(hir::pb::ComputeKernel);

	hir::pb::Operand& selectCtas = *kernel.add_operands();
	selectCtas.set_mode(hir::pb::In);
	selectCtas.set_variable(U32_selectCtas->second.id);

	hir::pb::Operand& selectThreads = *kernel.add_operands();
	selectThreads.set_mode(hir::pb::In);
	selectThreads.set_variable(U32_128->second.id);

	hir::pb::Operand& selectShared = *kernel.add_operands();
	selectShared.set_mode(hir::pb::In);
	selectShared.set_variable(U32_0->second.id);

	hir::pb::Operand& d = *kernel.add_operands();
	d.set_mode(hir::pb::InOut);
	d.set_variable(temp->second.id);

	hir::pb::Operand& selectHistogram = *kernel.add_operands();
	selectHistogram.set_mode(hir::pb::Out);
	selectHistogram.set_variable(histogram->second.id);

	hir::pb::Operand& a = *kernel.add_operands();
	a.set_variable(sourceA->second.id);
	a.set_mode(hir::pb::In);

	hir::pb::Operand& aSize = *kernel.add_operands();
	aSize.set_variable(inputSize->second.id);
	aSize.set_mode(hir::pb::In);

	if(isComp)	
	{
		if((comp[0].op1().tag() == lb::CONSTEXP && comp[0].op1().constexp().literal().kind() == common::Constant_Kind_STRING)
			|| (comp[0].op2().tag() == lb::CONSTEXP && comp[0].op2().constexp().literal().kind() == common::Constant_Kind_STRING))
		{
			VariableMap::const_iterator stringTable = variables.find("StringTable");
			assert(stringTable != variables.end());

			hir::pb::Operand& strings = *kernel.add_operands();
			strings.set_variable(stringTable->second.id);
			strings.set_mode(hir::pb::In);
		}

		RelationalAlgebraKernel ptxKernel(
			translateComparison(comp, variables, stringIds, sourceA),
			dId->second.types, sourceA->second.types);

		ptxKernel.setCtaCount(selectCtaCount);
		ptxKernel.setThreadCount(128);
		ptxKernel.set_id(kernel_id++);
		kernel.set_name(ptxKernel.name());
		kernel.set_code(compilePTXSource(
			ptxKernel.cudaSourceRepresentation()));
	}
	else
	{
		if((test.ops(0).tag() == lb::CONSTEXP && test.ops(0).constexp().literal().kind() == common::Constant_Kind_STRING) 
			|| (test.ops(1).tag() == lb::CONSTEXP && test.ops(1).constexp().literal().kind() == common::Constant_Kind_STRING))
		{
			VariableMap::const_iterator stringTable = variables.find("StringTable");
			assert(stringTable != variables.end());

			hir::pb::Operand& strings = *kernel.add_operands();
			strings.set_variable(stringTable->second.id);
			strings.set_mode(hir::pb::In);
		}

		RelationalAlgebraKernel ptxKernel(
			translateTest(test, variables, stringIds),
			dId->second.types, sourceA->second.types);

		ptxKernel.setCtaCount(selectCtaCount);
		ptxKernel.setThreadCount(128);
	
		ptxKernel.set_id(kernel_id++);
		kernel.set_name(ptxKernel.name());
		kernel.set_code(compilePTXSource(
			ptxKernel.cudaSourceRepresentation()));
	}
	
	// Add a scan kernel to determine the size of the gather
	hir::pb::Kernel& scan = *block.add_kernels();

	hir::pb::Operand& scanCtas = *scan.add_operands();
	scanCtas.set_mode(hir::pb::In);
	scanCtas.set_variable(U32_1->second.id);

	hir::pb::Operand& scanThreads = *scan.add_operands();
	scanThreads.set_mode(hir::pb::In);
	scanThreads.set_variable(U32_scannedValues->second.id);

	hir::pb::Operand& scanShared = *scan.add_operands();
	scanShared.set_mode(hir::pb::In);
	scanShared.set_variable(U32_0->second.id);

	scan.set_type(hir::pb::ComputeKernel);
	
	hir::pb::Operand& scanA = *scan.add_operands();

	scanA.set_variable(histogram->second.id);
	scanA.set_mode(hir::pb::InOut);

	RelationalAlgebraKernel scanPtx(
		RelationalAlgebraKernel::Scan, histogram->second.types);

	scanPtx.set_id(kernel_id++);
	scan.set_name(scanPtx.name());
	scan.set_code(compilePTXSource(scanPtx.cudaSourceRepresentation()));

	// Get the size of the final result
	hir::pb::Kernel& resultSize = *block.add_kernels();

	hir::pb::Operand& resultSizeCtas = *resultSize.add_operands();
	resultSizeCtas.set_mode(hir::pb::In);
	resultSizeCtas.set_variable(U32_1->second.id);

	hir::pb::Operand& resultSizeThreads = *resultSize.add_operands();
	resultSizeThreads.set_mode(hir::pb::In);
	resultSizeThreads.set_variable(U32_1->second.id);

	hir::pb::Operand& resultSizeShared = *resultSize.add_operands();
	resultSizeShared.set_mode(hir::pb::In);
	resultSizeShared.set_variable(U32_0->second.id);

	resultSize.set_type(hir::pb::ComputeKernel);

	hir::pb::Operand& resultSizeD = *resultSize.add_operands();

	resultSizeD.set_variable(outputSize->second.id);
	resultSizeD.set_mode(hir::pb::Out);
	
	hir::pb::Operand& resultSizeHistogram = *resultSize.add_operands();

	resultSizeHistogram.set_variable(histogram->second.id);
	resultSizeHistogram.set_mode(hir::pb::In);

	RelationalAlgebraKernel resultSizePtx(
		RelationalAlgebraKernel::SelectGetResultSize,
		dId->second.types, histogram->second.types);

	resultSizePtx.setCtaCount(selectCtaCount);

	resultSizePtx.set_id(kernel_id++);
	resultSize.set_name(resultSizePtx.name());
	resultSize.set_code(compilePTXSource(
		resultSizePtx.cudaSourceRepresentation()));
	
	// Resize the result
	hir::pb::Kernel& resizeResultKernel = *block.add_kernels();
	
	resizeResultKernel.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeResultD    = *resizeResultKernel.add_operands();
	hir::pb::Operand& resizeResultSize = *resizeResultKernel.add_operands();
	
	resizeResultD.set_mode(hir::pb::Out);
	resizeResultD.set_variable(dId->second.id);
	
	resizeResultSize.set_mode(hir::pb::In);
	resizeResultSize.set_variable(outputSize->second.id);

	resizeResultKernel.set_name("resize");
	resizeResultKernel.set_code("");

	// Gather the intermediate results together
	hir::pb::Kernel& gather = *block.add_kernels();

	hir::pb::Operand& gatherCtas = *gather.add_operands();
	gatherCtas.set_mode(hir::pb::In);
	gatherCtas.set_variable(U32_selectCtas->second.id);

	hir::pb::Operand& gatherThreads = *gather.add_operands();
	gatherThreads.set_mode(hir::pb::In);
	gatherThreads.set_variable(U32_256->second.id);

	hir::pb::Operand& gatherShared = *gather.add_operands();
	gatherShared.set_mode(hir::pb::In);
	gatherShared.set_variable(U32_0->second.id);

	gather.set_type(hir::pb::ComputeKernel);
	
	hir::pb::Operand& gatherD = *gather.add_operands();
	gatherD.set_variable(dId->second.id);
	gatherD.set_mode(hir::pb::InOut);

	hir::pb::Operand& gatherA = *gather.add_operands();
	gatherA.set_variable(temp->second.id);
	gatherA.set_mode(hir::pb::In);

	hir::pb::Operand& gatherASize = *gather.add_operands();
	gatherASize.set_variable(inputSize->second.id);
	gatherASize.set_mode(hir::pb::In);

	hir::pb::Operand& gatherHistogram = *gather.add_operands();
	gatherHistogram.set_variable(histogram->second.id);
	gatherHistogram.set_mode(hir::pb::In);
	
	RelationalAlgebraKernel gatherPtx(
		RelationalAlgebraKernel::SelectGather, dId->second.types, 
		temp->second.types, histogram->second.types);

	gatherPtx.setCtaCount(selectCtaCount);
	gatherPtx.setThreadCount(128);

	gatherPtx.set_id(kernel_id++);
	gather.set_name(gatherPtx.name());
	gather.set_code(compilePTXSource(gatherPtx.cudaSourceRepresentation()));	
}

static void addSelectConstant(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	const lb::GPUAssign& assign, VariableMap& variables,
	std::string dest)
{
	const unsigned int selectCtaCount = 128;

	// Get references to inputs/outputs
	VariableMap::const_iterator source, sourceConstant;

	VariableMap::const_iterator sourceA = variables.find(assign.op().join().srca());
	assert(sourceA != variables.end());

	VariableMap::const_iterator sourceB = variables.find(assign.op().join().srcb());
	assert(sourceB != variables.end());

	if(sourceA->second.isConstant)
	{
		source = sourceB;
		sourceConstant = sourceA;
	}
	else if(sourceB->second.isConstant)
	{
		source = sourceA;
		sourceConstant = sourceB;
	}

	VariableMap::const_iterator dId = variables.find(dest);
	assert(dId != variables.end());

	report("   destination: " << dest
		<< " (" << dId->second.id << ")");
	report("   src: " << assign.op().join().srca()
		<< " (" << source->second.id << ")");
	report("   srcConst: " << assign.op().join().srcb()
		<< " (" << sourceConstant->second.id << ")");

	// Allocate scratch variables
	VariableMap::const_iterator outputSize = getTempIntVariable(variables, cfg);
	VariableMap::const_iterator inputSize  = getTempIntVariable(variables, cfg);

	VariableMap::iterator U32_selectCtas 
		= getConstantIntVariable(variables, cfg, selectCtaCount);
	VariableMap::iterator U32_scannedValues
		= getConstantIntVariable(variables, cfg, selectCtaCount + 1);
	
	VariableMap::iterator U32_256 = getConstantIntVariable(variables, cfg, 256);
	VariableMap::iterator U32_128 = getConstantIntVariable(variables, cfg, 128);
	VariableMap::iterator U32_0   = getConstantIntVariable(variables, cfg, 0);
	VariableMap::iterator U32_1   = getConstantIntVariable(variables, cfg, 1);

	// Create a temporary array to facilitate the select
	VariableMap::iterator temp = getTempBuffer(variables, cfg, 0);
	temp->second.types = dId->second.types;

	// Create a histogram to track the elements per cta
	VariableMap::iterator histogram  = getTempBuffer(variables,
		cfg, /*(selectCtaCount + 1)*/ 512 * sizeof(unsigned int));

	// Get the size of the input	
	hir::pb::Kernel& inputSizeKernel = *block.add_kernels();
	
	inputSizeKernel.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& inputSizeD    = *inputSizeKernel.add_operands();
	hir::pb::Operand& inputSizeLeft = *inputSizeKernel.add_operands();
	
	inputSizeD.set_mode(hir::pb::Out);
	inputSizeD.set_variable(inputSize->second.id);
	
	inputSizeLeft.set_mode(hir::pb::In);
	inputSizeLeft.set_variable(source->second.id);
	
	inputSizeKernel.set_name("get_size");
	inputSizeKernel.set_code("");

	// Resize the temp output to the appropriate size
	hir::pb::Kernel& resizeTempKernel = *block.add_kernels();
	
	resizeTempKernel.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeTempD    = *resizeTempKernel.add_operands();
	hir::pb::Operand& resizeTempSize = *resizeTempKernel.add_operands();
	
	resizeTempD.set_mode(hir::pb::Out);
	resizeTempD.set_variable(temp->second.id);
	
	resizeTempSize.set_mode(hir::pb::In);
	resizeTempSize.set_variable(inputSize->second.id);

	resizeTempKernel.set_name("resize");
	resizeTempKernel.set_code("");

	// Add the main select kernel
	hir::pb::Kernel& kernel = *block.add_kernels();

	kernel.set_type(hir::pb::ComputeKernel);

	hir::pb::Operand& selectCtas = *kernel.add_operands();
	selectCtas.set_mode(hir::pb::In);
	selectCtas.set_variable(U32_selectCtas->second.id);

	hir::pb::Operand& selectThreads = *kernel.add_operands();
	selectThreads.set_mode(hir::pb::In);
	selectThreads.set_variable(U32_128->second.id);

	hir::pb::Operand& selectShared = *kernel.add_operands();
	selectShared.set_mode(hir::pb::In);
	selectShared.set_variable(U32_0->second.id);

	hir::pb::Operand& d = *kernel.add_operands();
	d.set_mode(hir::pb::InOut);
	d.set_variable(temp->second.id);

	hir::pb::Operand& selectHistogram = *kernel.add_operands();
	selectHistogram.set_mode(hir::pb::Out);
	selectHistogram.set_variable(histogram->second.id);

	hir::pb::Operand& a = *kernel.add_operands();
	a.set_variable(source->second.id);
	a.set_mode(hir::pb::In);

	hir::pb::Operand& aSize = *kernel.add_operands();
	aSize.set_variable(inputSize->second.id);
	aSize.set_mode(hir::pb::In);
	
	hir::pb::Operand& b = *kernel.add_operands();
	b.set_variable(sourceConstant->second.id);
	b.set_mode(hir::pb::In);

	RelationalAlgebraKernel::ComparisonVector comparisons;
	RelationalAlgebraKernel::ComparisonExpression expression;

	if(sourceConstant->second.types[0].type == RelationalAlgebraKernel::Pointer)
		expression.comparison = RelationalAlgebraKernel::EqString;
	else
		expression.comparison = RelationalAlgebraKernel::Eq;

	expression.a.type = RelationalAlgebraKernel::VariableIndex;
	expression.a.variableIndex = 0;
	expression.b.type = RelationalAlgebraKernel::ConstantVariable; 
	comparisons.push_back(expression);

	RelationalAlgebraKernel ptxKernel(
		comparisons, dId->second.types, 
		source->second.types, sourceConstant->second.types);

	ptxKernel.setCtaCount(selectCtaCount);
	ptxKernel.setThreadCount(128);

	ptxKernel.set_id(kernel_id++);
	kernel.set_name(ptxKernel.name());
	kernel.set_code(compilePTXSource(
		ptxKernel.cudaSourceRepresentation()));
		
	// Add a scan kernel to determine the size of the gather
	hir::pb::Kernel& scan = *block.add_kernels();

	hir::pb::Operand& scanCtas = *scan.add_operands();
	scanCtas.set_mode(hir::pb::In);
	scanCtas.set_variable(U32_1->second.id);

	hir::pb::Operand& scanThreads = *scan.add_operands();
	scanThreads.set_mode(hir::pb::In);
	scanThreads.set_variable(U32_scannedValues->second.id);

	hir::pb::Operand& scanShared = *scan.add_operands();
	scanShared.set_mode(hir::pb::In);
	scanShared.set_variable(U32_0->second.id);

	scan.set_type(hir::pb::ComputeKernel);
	
	hir::pb::Operand& scanA = *scan.add_operands();

	scanA.set_variable(histogram->second.id);
	scanA.set_mode(hir::pb::InOut);

	RelationalAlgebraKernel scanPtx(
		RelationalAlgebraKernel::Scan, histogram->second.types);

	scanPtx.set_id(kernel_id++);
	scan.set_name(scanPtx.name());
	scan.set_code(compilePTXSource(scanPtx.cudaSourceRepresentation()));

	// Get the size of the final result
	hir::pb::Kernel& resultSize = *block.add_kernels();

	hir::pb::Operand& resultSizeCtas = *resultSize.add_operands();
	resultSizeCtas.set_mode(hir::pb::In);
	resultSizeCtas.set_variable(U32_1->second.id);

	hir::pb::Operand& resultSizeThreads = *resultSize.add_operands();
	resultSizeThreads.set_mode(hir::pb::In);
	resultSizeThreads.set_variable(U32_1->second.id);

	hir::pb::Operand& resultSizeShared = *resultSize.add_operands();
	resultSizeShared.set_mode(hir::pb::In);
	resultSizeShared.set_variable(U32_0->second.id);

	resultSize.set_type(hir::pb::ComputeKernel);

	hir::pb::Operand& resultSizeD = *resultSize.add_operands();

	resultSizeD.set_variable(outputSize->second.id);
	resultSizeD.set_mode(hir::pb::Out);
	
	hir::pb::Operand& resultSizeHistogram = *resultSize.add_operands();

	resultSizeHistogram.set_variable(histogram->second.id);
	resultSizeHistogram.set_mode(hir::pb::In);

	RelationalAlgebraKernel resultSizePtx(
		RelationalAlgebraKernel::SelectGetResultSize,
		dId->second.types, histogram->second.types);

	resultSizePtx.setCtaCount(selectCtaCount);

	resultSizePtx.set_id(kernel_id++);
	resultSize.set_name(resultSizePtx.name());
	resultSize.set_code(compilePTXSource(
		resultSizePtx.cudaSourceRepresentation()));
	
	// Resize the result
	hir::pb::Kernel& resizeResultKernel = *block.add_kernels();
	
	resizeResultKernel.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeResultD    = *resizeResultKernel.add_operands();
	hir::pb::Operand& resizeResultSize = *resizeResultKernel.add_operands();
	
	resizeResultD.set_mode(hir::pb::Out);
	resizeResultD.set_variable(dId->second.id);
	
	resizeResultSize.set_mode(hir::pb::In);
	resizeResultSize.set_variable(outputSize->second.id);

	resizeResultKernel.set_name("resize");
	resizeResultKernel.set_code("");

	// Gather the intermediate results together
	hir::pb::Kernel& gather = *block.add_kernels();

	hir::pb::Operand& gatherCtas = *gather.add_operands();
	gatherCtas.set_mode(hir::pb::In);
	gatherCtas.set_variable(U32_selectCtas->second.id);

	hir::pb::Operand& gatherThreads = *gather.add_operands();
	gatherThreads.set_mode(hir::pb::In);
	gatherThreads.set_variable(U32_256->second.id);

	hir::pb::Operand& gatherShared = *gather.add_operands();
	gatherShared.set_mode(hir::pb::In);
	gatherShared.set_variable(U32_0->second.id);

	gather.set_type(hir::pb::ComputeKernel);
	
	hir::pb::Operand& gatherD = *gather.add_operands();
	gatherD.set_variable(dId->second.id);
	gatherD.set_mode(hir::pb::InOut);

	hir::pb::Operand& gatherA = *gather.add_operands();
	gatherA.set_variable(temp->second.id);
	gatherA.set_mode(hir::pb::In);

	hir::pb::Operand& gatherASize = *gather.add_operands();
	gatherASize.set_variable(inputSize->second.id);
	gatherASize.set_mode(hir::pb::In);

	hir::pb::Operand& gatherHistogram = *gather.add_operands();
	gatherHistogram.set_variable(histogram->second.id);
	gatherHistogram.set_mode(hir::pb::In);
	
	RelationalAlgebraKernel gatherPtx(
		RelationalAlgebraKernel::SelectGather, dId->second.types, 
		temp->second.types, histogram->second.types);

	gatherPtx.setCtaCount(selectCtaCount);
	gatherPtx.setThreadCount(128);

	gatherPtx.set_id(kernel_id++);
	gather.set_name(gatherPtx.name());
	gather.set_code(compilePTXSource(gatherPtx.cudaSourceRepresentation()));	
}

static void checkReordering(const RepeatedExp& reordering, unsigned int requiredSize,
	bool& hasProject, int& numArith, int& numConv)
{
	hasProject = false;

	std::vector<unsigned int> index;

	bool arithAtFront = false;

	if(reordering.begin()->tag() == lb::CALL)
		arithAtFront = true;

	for(RepeatedExp::const_iterator i = reordering.begin(); i != reordering.end(); ++i)
	{
		if(i->tag() == lb::INDEX)
			index.push_back(i->index().offset());
	}

	if(arithAtFront)
		hasProject = true;
	else if(index.size() != requiredSize)
		hasProject = true;
	else
	{
		for(unsigned int i = 0; i < requiredSize; ++i)
		{
			if(index[i] != i)
			{
				hasProject = true;
				break;
			}		
		}	
	}

	for(RepeatedExp::const_iterator 
		i = reordering.begin();
		i != reordering.end(); ++i)
	{
		report("check reordering " << i->tag())
		if(i->tag() == lb::ARITHEXP)
			numArith++;
		else if(i->tag() == lb::CONSTEXP)
			numArith++;
		else if(i->tag() == lb::CALL)
		{
			if(i->call().calltype() == lb::FDatetime 
				&& i->call().callname().compare("subtract") == 0)
				numArith++;
			else if(i->call().calltype() == lb::FDatetime 
				&& i->call().callname().compare("part") == 0)
				numArith++;
			else if(i->call().calltype() == lb::FDatetime 
				&& i->call().callname().compare("add") == 0)
				numArith++;
			else if(i->call().calltype() == lb::FString 
				&& i->call().callname().compare("add") == 0)
				numArith++;
			else if(i->call().calltype() == lb::FString 
				&& i->call().callname().compare("substring") == 0)
				numArith++;
		}
		else if(i->tag() == lb::MIXED)
		{
			numConv++;
		}
		else if(i->tag() == lb::CONVERT)
		{
			numConv++;
		}
	}
}

static void addReordering(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	const lb::GPUAssign& assign, const RepeatedExp& reordering, 
	VariableMap& variables,
	std::string& source, bool hasProject, int numArith, int numConv, int srcwidth,
	const StringIdMap& stringIds)
{
	std::string convSource;
	std::string convDest;

	if(numConv > 0)
	{
		convSource = source;
		RepeatedExp::const_iterator j = reordering.begin();

		for(int i = 0; i < numConv; ++i)
		{
			report("      Conv");

			RelationalAlgebraKernel::DataType type = RelationalAlgebraKernel::InvalidDataType;
			unsigned int size = 0;
			unsigned int offset = 0;

			while(j != reordering.end())
			{
				if(j->tag() == lb::MIXED) 
				{
					report("	mixed " << j->mixed().range().typ() << " " << j->mixed().range().size());
					if(j->mixed().range().typ() == lb::NInt || j->mixed().range().typ() == lb::NUInt)
					{
						if(j->mixed().range().size() == 64)
							type = RelationalAlgebraKernel::I64;
					}
					else if(j->mixed().range().typ() == lb::NFloat)
					{
						if(j->mixed().range().size() == 64)
							type = RelationalAlgebraKernel::F64;
					}
					size = j->mixed().range().size();
					offset = j->mixed().exp1().index().offset();

					break;
				}
				else if(j->tag() == lb::CONVERT)
				{
					report("	convert")
					if(j->convert().convrng().n().typ() == lb::NFloat)
						if(j->convert().convrng().n().size() == 64)
							type = RelationalAlgebraKernel::F64;

					size = j->convert().convrng().n().size();
					offset = j->convert().exp2().mixed().exp1().index().offset();

					break;
				}

				j++;
			}

			j++;
	
			VariableMap::const_iterator sourceA = variables.find(convSource);
			assert(sourceA != variables.end());

			report("source " << sourceA->second.types.size());		

			if(i == numConv - 1 && !hasProject && numArith == 0)
			{
				convDest = assign.dest();

				if(sourceA->second.isSorted == sourceA->second.types.size())
				{
					variables.find(convDest)->second.isSorted = sourceA->second.isSorted + 1;
					variables.find(convDest)->second.sorted_fields = sourceA->second.sorted_fields;
					variables.find(convDest)->second.sorted_fields.push_back(sourceA->second.types.size());
				}
				else
				{
					variables.find(convDest)->second.isSorted = sourceA->second.isSorted;
					variables.find(convDest)->second.sorted_fields = sourceA->second.sorted_fields;
				}
			}
			else
			{
				std::stringstream stream;
				stream << source << "_conv" << i;
				convDest = stream.str();

				unsigned int isSorted = sourceA->second.isSorted;
				std::vector<unsigned int> sorted_fields = sourceA->second.sorted_fields;

				if(sourceA->second.isSorted == sourceA->second.types.size())
				{
					isSorted++;
					sorted_fields.push_back(sourceA->second.types.size());
				}

				VariableMap::iterator dId = getTempBufferByName(
					variables, cfg, 0, sourceA->second.unique_keys, isSorted, sorted_fields, convDest, sourceA->second.types);
				
				report("dest after initial " << dId->second.types.size());	
		
				dId->second.types.push_back(RelationalAlgebraKernel::Element(
					type, size));
	
				report("dest after push back " << dId->second.types.size());
			}

			addConv(block, cfg, type, offset, variables, convSource, convDest);

			convSource = convDest;
		}
	}

	std::string arithSource;
	std::string arithDest;

	if(numArith > 0)
	{
		if(numConv == 0)
			arithSource = source;
		else
			arithSource = convDest;

		RepeatedExp::const_iterator j = reordering.begin();

		for(int i = 0; i < numArith; ++i)
		{
			report("      Arith");

			lb::Exp arithExp;

			while(j != reordering.end())
			{
				if(j->tag() == lb::ARITHEXP) 
				{
					arithExp = *j;
					break;
				}
				else if(j->tag() == lb::CONSTEXP)
				{
					arithExp = *j;
					break;
				}
				else if(j->tag() == lb::CALL)
				{
					if(j->call().calltype() == lb::FDatetime 
						&& j->call().callname().compare("subtract") == 0)
					{
						arithExp = *j;
						break;
					}
					else if(j->call().calltype() == lb::FDatetime 
						&& j->call().callname().compare("part") == 0)
					{
						arithExp = *j;
						break;
					}
					else if(j->call().calltype() == lb::FDatetime 
						&& j->call().callname().compare("add") == 0)
					{
						arithExp = *j;
						break;
					}
					else if(j->call().calltype() == lb::FString
						&& j->call().callname().compare("add") == 0)
					{
						arithExp = *j;
						break;
					}
					else if(j->call().calltype() == lb::FString
						&& j->call().callname().compare("substring") == 0)
					{
						arithExp = *j;
						break;
					}
				}

				j++;
			}

			j++;
	
			VariableMap::const_iterator sourceA = variables.find(arithSource);
			assert(sourceA != variables.end());

			report("source " << arithSource << " " << sourceA->second.types.size());		

			if(i == numArith - 1 && !hasProject)
			{
				arithDest = assign.dest();

				if(sourceA->second.isSorted == sourceA->second.types.size())
				{
					variables.find(arithDest)->second.isSorted = sourceA->second.isSorted + 1;
					variables.find(arithDest)->second.sorted_fields = sourceA->second.sorted_fields;
					variables.find(arithDest)->second.sorted_fields.push_back(sourceA->second.types.size());
				}
				else
				{
					variables.find(arithDest)->second.isSorted = sourceA->second.isSorted;
					variables.find(arithDest)->second.sorted_fields = sourceA->second.sorted_fields;
				}
			}
			else
			{
				std::stringstream stream;
				stream << source << "_arith" << i << random_name++;

				arithDest = stream.str();

				unsigned int isSorted = sourceA->second.isSorted;
				std::vector<unsigned int> sorted_fields = sourceA->second.sorted_fields;

				if(sourceA->second.isSorted == sourceA->second.types.size())
				{
					isSorted++;
					sorted_fields.push_back(sourceA->second.types.size());
				}

				VariableMap::iterator dId = getTempBufferByName(
					variables, cfg, 0, sourceA->second.unique_keys, isSorted, sorted_fields, 
					arithDest, sourceA->second.types);
			
				report("dest after initial " << dId->second.types.size() << " " << dId->second.id << " " << arithDest);	

				if(arithExp.tag() == lb::ARITHEXP || arithExp.tag() == lb::CONSTEXP)
				{
					switch(arithExp.arithexp().domain().typ())
					{
					case lb::NInt:
					case lb::NUInt:
					{
						if(arithExp.arithexp().domain().size() == 8)
							dId->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::I8, 8));
						else if(arithExp.arithexp().domain().size() == 16)
							dId->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::I16, 16));
						else if(arithExp.arithexp().domain().size() == 32)
							dId->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::I32, 32));
						else if(arithExp.arithexp().domain().size() == 64)
							dId->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::I64, 64));
						else if(arithExp.arithexp().domain().size() == 128)
							dId->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::I128, 128));
			
						break;
					}
					case lb::NFloat:
					{
						if(arithExp.arithexp().domain().size() == 32)
						{
							dId->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::F32, 32));
						}
						else if(arithExp.arithexp().domain().size() == 64)
						{
							dId->second.types.push_back(RelationalAlgebraKernel::Element(
								RelationalAlgebraKernel::F64, 64));
						}
						
						break;
					}
					case lb::NDecimal:
					{
						break;
					} 
					}
				}
				else if(arithExp.tag() == lb::CALL)
				{
					if(arithExp.call().calltype() == lb::FDatetime 
						&& arithExp.call().callname().compare("subtract") == 0)
					{
						dId->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I64, 64));
					}
					else if(arithExp.call().calltype() == lb::FDatetime 
						&& arithExp.call().callname().compare("part") == 0)
					{
						dId->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I64, 64));
					}
					else if(arithExp.call().calltype() == lb::FDatetime 
						&& arithExp.call().callname().compare("add") == 0)
					{
						dId->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::I64, 64));
					}
					else if(arithExp.call().calltype() == lb::FString 
						&& arithExp.call().callname().compare("add") == 0)
					{
						dId->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::Pointer, 64));
					}
					else if(arithExp.call().calltype() == lb::FString 
						&& arithExp.call().callname().compare("substring") == 0)
					{
						dId->second.types.push_back(RelationalAlgebraKernel::Element(
							RelationalAlgebraKernel::Pointer, 64));
					}
	
				}
	
				report("dest after push back " << dId->second.types.size());
			}

			if(arithExp.tag() == lb::CALL && arithExp.call().calltype() == lb::FString 
				&& arithExp.call().callname().compare("add") == 0)
				addAppendString(block, cfg, arithExp, variables, arithSource, arithDest, stringIds);
			else if(arithExp.tag() == lb::CALL && arithExp.call().calltype() == lb::FString 
				&& arithExp.call().callname().compare("substring") == 0)
				addSubString(block, cfg, arithExp, variables, arithSource, arithDest, stringIds);
			else	
			addArith(block, cfg, arithExp, variables, arithSource, arithDest);

			arithSource = arithDest;
		}
	}

//	if(blocknum == 5)
//	{
//		addMerge(block, cfg, variables, "+$logicQ14:_prodMult_0", "+$logicQ14:_prodMult_1", "+$logicQ14:_prodMult");
//	}
	if(hasProject)
	{
		std::string projectSource;
		std::string projectDest = assign.dest();

		if(numArith == 0)
			projectSource = source;
		else
			projectSource = arithDest;		

		addProjection(block, cfg, reordering, variables, projectSource, projectDest, srcwidth);
	}
}

static void addMapFilter(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	const lb::GPUAssign& assign, VariableMap& variables,
	const StringIdMap& stringIds)
{
	const lb::GPUMapFilter& mapFilter = assign.op().mapfilter();
	VariableMap::const_iterator sourceA = variables.find(mapFilter.srca());
	assert(sourceA != variables.end());

	bool hasSelect = false;
	bool hasProject = false;
	int numArith = 0;
	int numConv = 0;

	std::vector<lb::Comp> comp;
	lb::Test test;
	const lb::Compare* compare = &(mapFilter.predicate());
	bool isComp = true;

	report("start finding compare...");

	while(compare->tag() != lb::ALWAYS)
	{
		report("start one iteration " << compare->tag());

		if(compare->tag() == lb::COMP)
		{
			report("find one compare");
			comp.push_back(compare->comp());
			hasSelect = true;
			isComp = true;
			break;
		}
		else if(compare->tag() == lb::TEST)
		{
			report("find one test");
			assert(compare->test().testtype() == lb::FString);
			assert(compare->test().ops().size() == 2);
			test = compare->test();
			hasSelect = true;
			isComp = false;
			break;
		}
		else if(compare->tag() == lb::AND)
		{
			report("find an and!!!");
			if(compare->andcomp().and1().tag() == lb::ALWAYS 
				&& compare->andcomp().and2().tag() != lb::ALWAYS)
			{
				report("always and !always " << compare->andcomp().and2().tag());
				compare = &(compare->andcomp().and2());
				report("always and !always " << compare->tag());
			}
			else if(compare->andcomp().and2().tag() == lb::ALWAYS 
				&& compare->andcomp().and1().tag() != lb::ALWAYS)
			{
				report("!always and always");
				compare = &(compare->andcomp().and1());
			}
			else if(compare->andcomp().and1().tag() == lb::ALWAYS 
				&& compare->andcomp().and2().tag() == lb::ALWAYS)
			{
				report("always and always");
				break;
			}
			else if(compare->andcomp().and1().tag() != lb::ALWAYS 
				&& compare->andcomp().and2().tag() != lb::ALWAYS)
			{
				report("!always and !always ");
				comp.push_back(compare->andcomp().and1().comp());

				if(compare->andcomp().and2().tag() == lb::COMP)
					comp.push_back(compare->andcomp().and2().comp());
				else if (compare->andcomp().and2().andcomp().and2().tag() == lb::COMP)
					comp.push_back(compare->andcomp().and2().andcomp().and2().comp());
				else
					comp.push_back(compare->andcomp().and2().andcomp().and2().andcomp().and2().andcomp().and2().comp());

				hasSelect = true;
				isComp = true;

				break;
			}
			else
			{
				assertM(false, "unknown compare.");
			}
		}
		else if(compare->tag() == lb::ALWAYS)
		{
			report("find one always");
			break;
		}

		report("end one iteration");
	}

	report("hasSelect? " << hasSelect);

	checkReordering(mapFilter.reordering(), mapFilter.srcwidth(), hasProject, numArith, numConv);

	std::string selectDest;
	std::string reorderingSource;

	if(hasSelect)
	{
		if(!hasProject && numArith == 0 && numConv == 0)
		{
			selectDest = assign.dest();
			variables.find(selectDest)->second.isSorted = sourceA->second.isSorted;
			variables.find(selectDest)->second.sorted_fields = sourceA->second.sorted_fields;

		}
		else
		{
			selectDest = assign.dest() + "_select";		
			getTempBufferByName(variables, cfg, 0, sourceA->second.unique_keys, sourceA->second.isSorted, sourceA->second.sorted_fields, 
				selectDest, sourceA->second.types);
		}

		addSelectField(block, cfg, mapFilter, comp, test, isComp, variables, stringIds, selectDest);
		reorderingSource = selectDest;
	}
	else
		reorderingSource = mapFilter.srca();

	if(hasProject || numArith > 0 || numConv > 0)
		addReordering(block, cfg, assign, mapFilter.reordering(), 
			variables, reorderingSource, hasProject, numArith, numConv, mapFilter.srcwidth(), stringIds);
}

static void addMapJoin(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	const lb::GPUAssign& assign, VariableMap& variables,
	const StringIdMap& stringIds)
{
	const lb::GPUJoin& join = assign.op().join();

	VariableMap::const_iterator sourceA = variables.find(join.srca());
	assert(sourceA != variables.end());
	VariableMap::const_iterator sourceB = variables.find(join.srcb());
	assert(sourceB != variables.end());

	bool hasProject = false;
	int numArith = 0;
	int numConv = 0;
	int requiredSize = sourceA->second.types.size() + 
		sourceB->second.types.size() - join.keycount(); 

	checkReordering(join.args(), requiredSize, hasProject, numArith, numConv);

	std::string joinDest;

	unsigned int isSorted;
	std::vector<unsigned int> sorted_fields;

	if(join.keycount() > 0)
	{
		if(join.keycount() == 1 && sourceA->second.isConstant)
		{
			isSorted = sourceB->second.isSorted;
			sorted_fields = sourceB->second.sorted_fields;
		}
		else if(join.keycount() == 1 && sourceB->second.isConstant)
		{
			isSorted = sourceA->second.isSorted;
			sorted_fields = sourceA->second.sorted_fields;
		}
		else
		{
			unsigned int fromA = MAX(sourceA->second.isSorted, (unsigned int)join.keycount());

			unsigned int fromB = (sourceB->second.isSorted > (unsigned int)join.keycount())
				? sourceB->second.isSorted - join.keycount() : 0;

			if(sourceA->second.isSorted == sourceA->second.types.size())
				isSorted = fromA + fromB; 
			else
				isSorted = fromA;
			
			for(unsigned int i = 0; i < isSorted; ++i)
				sorted_fields.push_back(i);
		}
	}
	else
	{
		isSorted = (sourceA->second.isSorted == sourceA->second.types.size()) 
			? (sourceA->second.isSorted + sourceB->second.isSorted) : sourceA->second.isSorted;
		
		for(unsigned int i = 0; i < isSorted; ++i)
			sorted_fields.push_back(i);
	}

	if(!hasProject && numArith == 0 && numConv == 0)
	{
		joinDest = assign.dest();
		variables.find(joinDest)->second.isSorted = isSorted;
		variables.find(joinDest)->second.sorted_fields = sorted_fields;
	}
	else
	{
		joinDest = assign.dest() + "_join";		

		unsigned int unique_key_num = 0;
	
		if(join.keycount() > 0)
		{
			if(join.keycount() != 1 || (!sourceA->second.isConstant && !sourceB->second.isConstant))
				unique_key_num = join.keycount();
		}
		else
			unique_key_num = MAX(sourceA->second.unique_keys, sourceB->second.unique_keys);	
	
		VariableMap::iterator dId = getTempBufferByName(variables, cfg, 0, 
			unique_key_num, isSorted, sorted_fields, joinDest, sourceA->second.types);
	
		dId->second.types.insert(dId->second.types.end(), 
			sourceB->second.types.begin() + join.keycount(), sourceB->second.types.end());
	}

	if(join.keycount() > 0)
	{
		if(join.keycount() == 1 && (sourceA->second.isConstant || sourceB->second.isConstant))
			addSelectConstant(block, cfg, assign, variables, joinDest);
		else
			addJoin(block, cfg, assign, variables, joinDest);
	}
	else
		addProduct(block, cfg, assign, variables, joinDest);

	if(hasProject || numArith > 0 || numConv > 0)
		addReordering(block, cfg, assign, join.args(), 
			variables, joinDest, hasProject, numArith, numConv, requiredSize, stringIds);
}

static void addSingleAgg(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	const lb::GPUAssign& assign, VariableMap& variables)
{
	const lb::GPUAgg& agg = assign.op().agg();

	VariableMap::iterator sourceA = variables.find(agg.srca());
	assert(sourceA != variables.end());

	VariableMap::const_iterator dId = variables.find(assign.dest());
	assert(dId != variables.end());

	// Get the size of the input array
	hir::pb::Kernel& sizeKernel = *block.add_kernels();
	
	VariableMap::const_iterator sizeVariable = 
		getTempIntVariable(variables, cfg);
	sizeKernel.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& getSizeD  = *sizeKernel.add_operands();
	hir::pb::Operand& getSizeIn = *sizeKernel.add_operands();
	
	getSizeIn.set_mode(hir::pb::In);
	getSizeIn.set_variable(sourceA->second.id);
	
	getSizeD.set_mode(hir::pb::Out);
	getSizeD.set_variable(sizeVariable->second.id);
	
	sizeKernel.set_name("get_size");
	sizeKernel.set_code("");

	assertM(agg.range_size() == 1,
                "Only one reduction supported per aggregation so far..");

	RepeatedAgg::const_iterator aggregation = agg.range().begin();

	if(aggregation->tag() != lb::COUNT)
	{
		// Start Adding Project
		VariableMap::iterator temp_project = getTempBuffer(variables, cfg, 0);
	
		for(RepeatedDomain::const_iterator i = agg.domains().begin();
			i != agg.domains().end(); ++i)
			temp_project->second.types.push_back(sourceA->second.types[*i]);
	
		RelationalAlgebraKernel::IndexVector keepFields(agg.domains().begin(), agg.domains().end());
	
		switch(aggregation->tag())
		{
		case lb::COUNT:
		{
	//		temp_project->second.types.push_back(RelationalAlgebraKernel::I32);
			report("count");
			break;
		}
		case lb::TOTAL:
		{
			int dom = aggregation->total().aggdom();
			temp_project->second.types.push_back(sourceA->second.types[dom]);
			keepFields.push_back(dom);
			report("total");
			break;
		}
		case lb::MIN:
		{
			int dom = aggregation->min().aggdom();
			temp_project->second.types.push_back(sourceA->second.types[dom]);
			keepFields.push_back(dom);
			report("min");
			break;
		}
		case lb::MAX:
		{
			int dom = aggregation->max().aggdom();
			temp_project->second.types.push_back(sourceA->second.types[dom]);
			keepFields.push_back(dom);
			report("max");
			break;
		}
		}

		bool skipProject = false;

		if(keepFields.size() == 1 && keepFields[0] == 0 && sourceA->second.types.size() == 1)
			skipProject = true;

		if(!skipProject)
		{	
		// Get the size of the output array if the primitive size changed
		if(bytes(temp_project->second.types) != bytes(sourceA->second.types))
		{
			report("get the size of the output");
			hir::pb::Kernel& adjustSizeKernel = *block.add_kernels();
	
			adjustSizeKernel.set_type(hir::pb::ComputeKernel);
			
			VariableMap::iterator U32_1 = getConstantIntVariable(variables, cfg, 1);
			VariableMap::iterator U32_0 = getConstantIntVariable(variables, cfg, 0);
			
			hir::pb::Operand& ctas = *adjustSizeKernel.add_operands();
			ctas.set_mode(hir::pb::In);
			ctas.set_variable(U32_1->second.id);
		
			hir::pb::Operand& threads = *adjustSizeKernel.add_operands();
			threads.set_mode(hir::pb::In);
			threads.set_variable(U32_1->second.id);
		
			hir::pb::Operand& shared = *adjustSizeKernel.add_operands();
			shared.set_mode(hir::pb::In);
			shared.set_variable(U32_0->second.id);
			
			hir::pb::Operand& size = *adjustSizeKernel.add_operands();
	
			size.set_mode(hir::pb::InOut);
			size.set_variable(sizeVariable->second.id);
			
			RelationalAlgebraKernel ptxKernel(
				RelationalAlgebraKernel::ProjectGetResultSize, 
				temp_project->second.types, sourceA->second.types);
	
			ptxKernel.set_id(kernel_id++);
			adjustSizeKernel.set_name(ptxKernel.name());
			adjustSizeKernel.set_code(compilePTXSource(
				ptxKernel.cudaSourceRepresentation()));
		}
	
		// Resize the output array
		report("resize the output");
		hir::pb::Kernel& resizeResultKernel = *block.add_kernels();
		
		resizeResultKernel.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeResultD    = *resizeResultKernel.add_operands();
		hir::pb::Operand& resizeResultSize = *resizeResultKernel.add_operands();
		
		resizeResultD.set_mode(hir::pb::Out);
		resizeResultD.set_variable(temp_project->second.id);
		
		resizeResultSize.set_mode(hir::pb::In);
		resizeResultSize.set_variable(sizeVariable->second.id);
	
		resizeResultKernel.set_name("resize");
		resizeResultKernel.set_code("");
	
		// Add the projection kernel
		report("add the projection");
		hir::pb::Kernel& kernel_project = *block.add_kernels();
	
		kernel_project.set_type(hir::pb::ComputeKernel);
	
		VariableMap::iterator U32_256_project = getConstantIntVariable(variables, cfg, 256);
		VariableMap::iterator U32_32_project  = getConstantIntVariable(variables, cfg, 32);
		VariableMap::iterator U32_0_project   = getConstantIntVariable(variables, cfg, 0);
		
		hir::pb::Operand& ctas_project = *kernel_project.add_operands();
		ctas_project.set_mode(hir::pb::In);
		ctas_project.set_variable(U32_256_project->second.id);
		
		hir::pb::Operand& threads_project = *kernel_project.add_operands();
		threads_project.set_mode(hir::pb::In);
		threads_project.set_variable(U32_32_project->second.id);
		
		hir::pb::Operand& shared_project = *kernel_project.add_operands();
		shared_project.set_mode(hir::pb::In);
		shared_project.set_variable(U32_0_project->second.id);
		
		hir::pb::Operand& d_project = *kernel_project.add_operands();
	
		d_project.set_mode(hir::pb::InOut);
		d_project.set_variable(temp_project->second.id);
	
		RelationalAlgebraKernel ptxKernel_project(temp_project->second.types,
			sourceA->second.types, keepFields);
	
		hir::pb::Operand& a_project = *kernel_project.add_operands();
	
		a_project.set_variable(sourceA->second.id);
		a_project.set_mode(hir::pb::In);
	
		ptxKernel_project.set_id(kernel_id++);
		kernel_project.set_name(ptxKernel_project.name());
		kernel_project.set_code(compilePTXSource(ptxKernel_project.cudaSourceRepresentation()));
		
		hir::pb::Operand& aSize_project = *kernel_project.add_operands();
		
		aSize_project.set_variable(sizeVariable->second.id);
		aSize_project.set_mode(hir::pb::In);
		
		assert(!keepFields.empty());
		report("Keep fields " 
			<< hydrazine::toString(keepFields.begin(), keepFields.end())
			<< " out of " << sourceA->second.types.size());
		assert(keepFields.size() <= sourceA->second.types.size());
		}
		else
			temp_project = sourceA;	
	
		// reduce by key
		report(" Reduce by Key.");
	
		// Resize the output array of reduce 
		report("resize the output size of reduce.");
		hir::pb::Kernel& resizeResultKernel_reduce = *block.add_kernels();
		
		VariableMap::iterator U32_8 = getConstantIntVariable(variables, cfg, 8);
		resizeResultKernel_reduce.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeResultD_reduce    
			= *resizeResultKernel_reduce.add_operands();
		hir::pb::Operand& resizeResultSize_reduce 
			= *resizeResultKernel_reduce.add_operands();
		
		resizeResultD_reduce.set_mode(hir::pb::Out);
		resizeResultD_reduce.set_variable(dId->second.id);
		
		resizeResultSize_reduce.set_mode(hir::pb::In);
		resizeResultSize_reduce.set_variable(U32_8->second.id);
	
		resizeResultKernel_reduce.set_name("resize");
		resizeResultKernel_reduce.set_code("");
	
		hir::pb::Kernel& reduceKernel = *block.add_kernels();
		reduceKernel.set_type(hir::pb::BinaryKernel);
	
		hir::pb::Operand& d_reduce = *reduceKernel.add_operands();
		d_reduce.set_variable(dId->second.id);
		d_reduce.set_mode(hir::pb::InOut);
	
		hir::pb::Operand& a_reduce = *reduceKernel.add_operands();
		a_reduce.set_variable(temp_project->second.id);
		a_reduce.set_mode(hir::pb::In);
	
		hir::pb::Operand& a_reduce_size = *reduceKernel.add_operands();
		a_reduce_size.set_variable(sizeVariable->second.id);
		a_reduce_size.set_mode(hir::pb::In);
	
		RelationalAlgebraKernel binKernel(temp_project->second.types, 
			mapReduction(aggregation->tag()));
	
		binKernel.set_id(kernel_id++);
		reduceKernel.set_name(binKernel.name());
		reduceKernel.set_code(compileBINSource(binKernel.cudaSourceRepresentation(), 
			mapReduction(aggregation->tag())));
	}
	else
	{
		// Starting add the split kernel
		VariableMap::iterator temp_generate = getTempBuffer(variables, cfg, 0);
	
		temp_generate->second.types.push_back(RelationalAlgebraKernel::Element(RelationalAlgebraKernel::I32, 32));
		VariableMap::const_iterator sizeVariable_value = 
			getTempIntVariable(variables, cfg);

		// Get the size of the output array of split
		report("get the output size of generate.");
		hir::pb::Kernel& adjustSizeKernel = *block.add_kernels();

		adjustSizeKernel.set_type(hir::pb::ComputeKernel);
		
		VariableMap::iterator U32_1 = getConstantIntVariable(variables, cfg, 1);
		VariableMap::iterator U32_0 = getConstantIntVariable(variables, cfg, 0);
		
		hir::pb::Operand& ctas = *adjustSizeKernel.add_operands();
		ctas.set_mode(hir::pb::In);
		ctas.set_variable(U32_1->second.id);
	
		hir::pb::Operand& threads = *adjustSizeKernel.add_operands();
		threads.set_mode(hir::pb::In);
		threads.set_variable(U32_1->second.id);
	
		hir::pb::Operand& shared = *adjustSizeKernel.add_operands();
		shared.set_mode(hir::pb::In);
		shared.set_variable(U32_0->second.id);

		hir::pb::Operand& size_value = *adjustSizeKernel.add_operands();
		hir::pb::Operand& size = *adjustSizeKernel.add_operands();

		size_value.set_mode(hir::pb::Out);
		size_value.set_variable(sizeVariable_value->second.id);

		size.set_mode(hir::pb::In);
		size.set_variable(sizeVariable->second.id);

		RelationalAlgebraKernel ptxKernel(
			temp_generate->second.types, sourceA->second.types, 
			RelationalAlgebraKernel::GenerateGetResultSize);

		ptxKernel.set_id(kernel_id++);
		adjustSizeKernel.set_name(ptxKernel.name());
		adjustSizeKernel.set_code(compilePTXSource(
			ptxKernel.cudaSourceRepresentation()));
	
		// Resize the output array of generate
		hir::pb::Kernel& resizeResultKernel_generate_value = *block.add_kernels();
		
		resizeResultKernel_generate_value.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeResultD_generate_value    
			= *resizeResultKernel_generate_value.add_operands();
		hir::pb::Operand& resizeResultSize_generate_value 
			= *resizeResultKernel_generate_value.add_operands();
		
		resizeResultD_generate_value.set_mode(hir::pb::Out);
		resizeResultD_generate_value.set_variable(temp_generate->second.id);
		
		resizeResultSize_generate_value.set_mode(hir::pb::In);
		resizeResultSize_generate_value.set_variable(sizeVariable_value->second.id);
	
		resizeResultKernel_generate_value.set_name("resize");
		resizeResultKernel_generate_value.set_code("");

		//Add Generate
		report("add generate.");
		hir::pb::Kernel& kernel_generate = *block.add_kernels();
	
		kernel_generate.set_type(hir::pb::ComputeKernel);

		VariableMap::iterator U32_256_generate = getConstantIntVariable(variables, cfg, 256);
		VariableMap::iterator U32_32_generate  = getConstantIntVariable(variables, cfg, 32);
		VariableMap::iterator U32_0_generate   = getConstantIntVariable(variables, cfg, 0);
		
		hir::pb::Operand& ctas_generate = *kernel_generate.add_operands();
		ctas_generate.set_mode(hir::pb::In);
		ctas_generate.set_variable(U32_256_generate->second.id);
		
		hir::pb::Operand& threads_generate = *kernel_generate.add_operands();
		threads_generate.set_mode(hir::pb::In);
		threads_generate.set_variable(U32_32_generate->second.id);
		
		hir::pb::Operand& shared_generate = *kernel_generate.add_operands();
		shared_generate.set_mode(hir::pb::In);
		shared_generate.set_variable(U32_0_generate->second.id);
	
		hir::pb::Operand& d_value = *kernel_generate.add_operands();
		d_value.set_mode(hir::pb::InOut);
		d_value.set_variable(temp_generate->second.id);
	
		hir::pb::Operand& d_value_size = *kernel_generate.add_operands();
		d_value_size.set_mode(hir::pb::In);
		d_value_size.set_variable(sizeVariable_value->second.id);
	
		RelationalAlgebraKernel ptxKernel_generate(RelationalAlgebraKernel::Generate, 
			temp_generate->second.types); 

		ptxKernel_generate.set_id(kernel_id++);
		kernel_generate.set_name(ptxKernel_generate.name());
		kernel_generate.set_code(compilePTXSource(ptxKernel_generate.cudaSourceRepresentation()));

		// reduce by key
		report(" Reduce by Key.");
	
		// Resize the output array of reduce 
		report("resize the output size of reduce.");
		hir::pb::Kernel& resizeResultKernel_reduce = *block.add_kernels();
		
		VariableMap::iterator U32_4 = getConstantIntVariable(variables, cfg, 4);
		resizeResultKernel_reduce.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeResultD_reduce    
			= *resizeResultKernel_reduce.add_operands();
		hir::pb::Operand& resizeResultSize_reduce 
			= *resizeResultKernel_reduce.add_operands();
		
		resizeResultD_reduce.set_mode(hir::pb::Out);
		resizeResultD_reduce.set_variable(dId->second.id);
		
		resizeResultSize_reduce.set_mode(hir::pb::In);
		resizeResultSize_reduce.set_variable(U32_4->second.id);
	
		resizeResultKernel_reduce.set_name("resize");
		resizeResultKernel_reduce.set_code("");
	
		hir::pb::Kernel& reduceKernel = *block.add_kernels();
		reduceKernel.set_type(hir::pb::BinaryKernel);
	
		hir::pb::Operand& d_reduce = *reduceKernel.add_operands();
		d_reduce.set_variable(dId->second.id);
		d_reduce.set_mode(hir::pb::InOut);
	
		hir::pb::Operand& a_reduce = *reduceKernel.add_operands();
		a_reduce.set_variable(temp_generate->second.id);
		a_reduce.set_mode(hir::pb::In);
	
		hir::pb::Operand& a_reduce_size = *reduceKernel.add_operands();
		a_reduce_size.set_variable(sizeVariable_value->second.id);
		a_reduce_size.set_mode(hir::pb::In);
	
		RelationalAlgebraKernel binKernel(temp_generate->second.types, 
			RelationalAlgebraKernel::Count);
	
		binKernel.set_id(kernel_id++);
		reduceKernel.set_name(binKernel.name());
		reduceKernel.set_code(compileBINSource(binKernel.cudaSourceRepresentation(), 
			RelationalAlgebraKernel::Count));
	}
}

static void addAgg(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	const lb::GPUAssign& assign, VariableMap& variables)
{
	const lb::GPUAgg& agg = assign.op().agg();

	VariableMap::iterator sourceA = variables.find(agg.srca());
	assert(sourceA != variables.end());

	VariableMap::iterator dId = variables.find(assign.dest());
	assert(dId != variables.end());

	if(dId->second.types.size() == 1)
	{
		addSingleAgg(block, cfg, assign, variables);

		dId->second.isSorted = 1;
		dId->second.sorted_fields.push_back(0);
		return;
	}

	dId->second.isSorted = dId->second.types.size();

	for(unsigned int i = 0; i < dId->second.isSorted; ++i)
		dId->second.sorted_fields.push_back(i);

	// Get the size of the input array
	hir::pb::Kernel& sizeKernel = *block.add_kernels();
	
	VariableMap::const_iterator sizeVariable = 
		getTempIntVariable(variables, cfg);
	sizeKernel.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& getSizeD  = *sizeKernel.add_operands();
	hir::pb::Operand& getSizeIn = *sizeKernel.add_operands();
	
	getSizeIn.set_mode(hir::pb::In);
	getSizeIn.set_variable(sourceA->second.id);
	
	getSizeD.set_mode(hir::pb::Out);
	getSizeD.set_variable(sizeVariable->second.id);
	
	sizeKernel.set_name("get_size");
	sizeKernel.set_code("");

	// Start Adding Project
	VariableMap::iterator temp_project = getTempBuffer(variables, cfg, 0);

    assertM(agg.range_size() == 1,
    	"Only one reduction supported per aggregation so far..");

	RepeatedAgg::const_iterator aggregation = agg.range().begin();

	for(RepeatedDomain::const_iterator i = agg.domains().begin();
		i != agg.domains().end(); ++i)
		temp_project->second.types.push_back(sourceA->second.types[*i]);

	RelationalAlgebraKernel::IndexVector keepFields(agg.domains().begin(), agg.domains().end());

	switch(aggregation->tag())
	{
	case lb::COUNT:
	{
//		temp_project->second.types.push_back(RelationalAlgebraKernel::I32);
		report("count");
		break;
	}
	case lb::TOTAL:
	{
		int dom = aggregation->total().aggdom();
		temp_project->second.types.push_back(sourceA->second.types[dom]);
		keepFields.push_back(dom);
		report("total");
		break;
	}
	case lb::MIN:
	{
		int dom = aggregation->min().aggdom();
		temp_project->second.types.push_back(sourceA->second.types[dom]);
		keepFields.push_back(dom);
		report("min");
		break;
	}
	case lb::MAX:
	{
		int dom = aggregation->max().aggdom();
		temp_project->second.types.push_back(sourceA->second.types[dom]);
		keepFields.push_back(dom);
		report("max");
		break;
	}
	}

	bool skipProject = true;

	if(keepFields.size() == sourceA->second.types.size())
	{
		for(unsigned int i = 0; i < keepFields.size(); ++i)
		{
			if(keepFields[i] != i)
			{
				skipProject = false;
				break;
			}		
		}	
	}
	else
		skipProject = false;

	if(!skipProject)
	{
	// Get the size of the output array if the primitive size changed
	if(bytes(temp_project->second.types) != bytes(sourceA->second.types))
	{
		report("get the size of the output");
		hir::pb::Kernel& adjustSizeKernel = *block.add_kernels();

		adjustSizeKernel.set_type(hir::pb::ComputeKernel);
		
		VariableMap::iterator U32_1 = getConstantIntVariable(variables, cfg, 1);
		VariableMap::iterator U32_0 = getConstantIntVariable(variables, cfg, 0);
		
		hir::pb::Operand& ctas = *adjustSizeKernel.add_operands();
		ctas.set_mode(hir::pb::In);
		ctas.set_variable(U32_1->second.id);
	
		hir::pb::Operand& threads = *adjustSizeKernel.add_operands();
		threads.set_mode(hir::pb::In);
		threads.set_variable(U32_1->second.id);
	
		hir::pb::Operand& shared = *adjustSizeKernel.add_operands();
		shared.set_mode(hir::pb::In);
		shared.set_variable(U32_0->second.id);
		
		hir::pb::Operand& size = *adjustSizeKernel.add_operands();

		size.set_mode(hir::pb::InOut);
		size.set_variable(sizeVariable->second.id);
		
		RelationalAlgebraKernel ptxKernel(
			RelationalAlgebraKernel::ProjectGetResultSize, 
			temp_project->second.types, sourceA->second.types);

		ptxKernel.set_id(kernel_id++);
		adjustSizeKernel.set_name(ptxKernel.name());
		adjustSizeKernel.set_code(compilePTXSource(
			ptxKernel.cudaSourceRepresentation()));
	}

	// Resize the output array
	report("resize the output");
	hir::pb::Kernel& resizeResultKernel = *block.add_kernels();
	
	resizeResultKernel.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeResultD    = *resizeResultKernel.add_operands();
	hir::pb::Operand& resizeResultSize = *resizeResultKernel.add_operands();
	
	resizeResultD.set_mode(hir::pb::Out);
	resizeResultD.set_variable(temp_project->second.id);
	
	resizeResultSize.set_mode(hir::pb::In);
	resizeResultSize.set_variable(sizeVariable->second.id);

	resizeResultKernel.set_name("resize");
	resizeResultKernel.set_code("");

	// Add the projection kernel
	report("add the projection");
	hir::pb::Kernel& kernel_project = *block.add_kernels();

	kernel_project.set_type(hir::pb::ComputeKernel);

	VariableMap::iterator U32_256_project = getConstantIntVariable(variables, cfg, 256);
	VariableMap::iterator U32_350_project  = getConstantIntVariable(variables, cfg, 350);
	VariableMap::iterator U32_0_project   = getConstantIntVariable(variables, cfg, 0);
	
	hir::pb::Operand& ctas_project = *kernel_project.add_operands();
	ctas_project.set_mode(hir::pb::In);
	ctas_project.set_variable(U32_350_project->second.id);
//	ctas_project.set_variable(U32_256_project->second.id);
	
	hir::pb::Operand& threads_project = *kernel_project.add_operands();
	threads_project.set_mode(hir::pb::In);
	threads_project.set_variable(U32_256_project->second.id);
	
	hir::pb::Operand& shared_project = *kernel_project.add_operands();
	shared_project.set_mode(hir::pb::In);
	shared_project.set_variable(U32_0_project->second.id);
	
	hir::pb::Operand& d_project = *kernel_project.add_operands();

	d_project.set_mode(hir::pb::InOut);
	d_project.set_variable(temp_project->second.id);

	RelationalAlgebraKernel ptxKernel_project(temp_project->second.types,
		sourceA->second.types, keepFields);

	hir::pb::Operand& a_project = *kernel_project.add_operands();

	a_project.set_variable(sourceA->second.id);
	a_project.set_mode(hir::pb::In);

	ptxKernel_project.set_id(kernel_id++);
	kernel_project.set_name(ptxKernel_project.name());
	kernel_project.set_code(compilePTXSource(ptxKernel_project.cudaSourceRepresentation()));
	
	hir::pb::Operand& aSize_project = *kernel_project.add_operands();
	
	aSize_project.set_variable(sizeVariable->second.id);
	aSize_project.set_mode(hir::pb::In);
	
	assert(!keepFields.empty());
	report("Keep fields " 
		<< hydrazine::toString(keepFields.begin(), keepFields.end())
		<< " out of " << sourceA->second.types.size());
	assert(keepFields.size() <= sourceA->second.types.size());
	}
	else
	{
		temp_project = sourceA;	
	}

	bool skipSort = true;

	if(sourceA->second.sorted_fields.size() == 0)
	{
		skipSort = false;
	}
	else
	{
		for(unsigned int i = 0; i < sourceA->second.sorted_fields.size() && i < keepFields.size(); ++i)
		{
			std::cout << "agg " << sourceA->second.sorted_fields[i] << "\n";
			if(sourceA->second.sorted_fields[i] != keepFields[i])
			{
				skipSort = false;
				break;
			}
		}
	}

	VariableMap::iterator temp_merge_key;
	VariableMap::iterator temp_merge_value;
	VariableMap::const_iterator sizeVariable_key = 
		getTempIntVariable(variables, cfg);
	VariableMap::const_iterator sizeVariable_value = 
		getTempIntVariable(variables, cfg);

	if(aggregation->tag() != lb::COUNT)
	{
		// Starting add the split kernel
		VariableMap::iterator temp_split_key = getTempBuffer(variables, cfg, 0);
		VariableMap::iterator temp_split_value = getTempBuffer(variables, cfg, 0);

		for(RepeatedDomain::const_iterator i = agg.domains().begin();
			i != agg.domains().end(); ++i)
			temp_split_key->second.types.push_back(sourceA->second.types[*i]);

		temp_split_value->second.types.push_back(*(temp_project->second.types.rbegin()));

		// Get the size of the output array of split
		report("get the output size of split.");
		hir::pb::Kernel& adjustSizeKernel = *block.add_kernels();

		adjustSizeKernel.set_type(hir::pb::ComputeKernel);
		
		VariableMap::iterator U32_1 = getConstantIntVariable(variables, cfg, 1);
		VariableMap::iterator U32_0 = getConstantIntVariable(variables, cfg, 0);
		
		hir::pb::Operand& ctas = *adjustSizeKernel.add_operands();
		ctas.set_mode(hir::pb::In);
		ctas.set_variable(U32_1->second.id);
	
		hir::pb::Operand& threads = *adjustSizeKernel.add_operands();
		threads.set_mode(hir::pb::In);
		threads.set_variable(U32_1->second.id);
	
		hir::pb::Operand& shared = *adjustSizeKernel.add_operands();
		shared.set_mode(hir::pb::In);
		shared.set_variable(U32_0->second.id);
		
		hir::pb::Operand& size = *adjustSizeKernel.add_operands();
		hir::pb::Operand& size_key = *adjustSizeKernel.add_operands();
		hir::pb::Operand& size_value = *adjustSizeKernel.add_operands();

		size.set_mode(hir::pb::In);
		size.set_variable(sizeVariable->second.id);
		size_key.set_mode(hir::pb::Out);
		size_key.set_variable(sizeVariable_key->second.id);
		size_value.set_mode(hir::pb::Out);
		size_value.set_variable(sizeVariable_value->second.id);

		RelationalAlgebraKernel ptxKernel(
			temp_split_key->second.types, temp_split_value->second.types, temp_project->second.types, 
			RelationalAlgebraKernel::SplitGetResultSize);

		ptxKernel.set_id(kernel_id++);
		adjustSizeKernel.set_name(ptxKernel.name());
		adjustSizeKernel.set_code(compilePTXSource(
			ptxKernel.cudaSourceRepresentation()));
	
		// Resize the output array of split
		report("resize the output size of split.");
		hir::pb::Kernel& resizeResultKernel_split_key = *block.add_kernels();
		
		resizeResultKernel_split_key.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeResultD_split_key    
			= *resizeResultKernel_split_key.add_operands();
		hir::pb::Operand& resizeResultSize_split_key 
			= *resizeResultKernel_split_key.add_operands();
		
		resizeResultD_split_key.set_mode(hir::pb::Out);
		resizeResultD_split_key.set_variable(temp_split_key->second.id);
		
		resizeResultSize_split_key.set_mode(hir::pb::In);
		resizeResultSize_split_key.set_variable(sizeVariable_key->second.id);
	
		resizeResultKernel_split_key.set_name("resize");
		resizeResultKernel_split_key.set_code("");
	
		hir::pb::Kernel& resizeResultKernel_split_value = *block.add_kernels();
		
		resizeResultKernel_split_value.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeResultD_split_value    
			= *resizeResultKernel_split_value.add_operands();
		hir::pb::Operand& resizeResultSize_split_value 
			= *resizeResultKernel_split_value.add_operands();
		
		resizeResultD_split_value.set_mode(hir::pb::Out);
		resizeResultD_split_value.set_variable(temp_split_value->second.id);
		
		resizeResultSize_split_value.set_mode(hir::pb::In);
		resizeResultSize_split_value.set_variable(sizeVariable_value->second.id);
	
		resizeResultKernel_split_value.set_name("resize");
		resizeResultKernel_split_value.set_code("");
	
		//Add Split
		report("add split.");
		hir::pb::Kernel& kernel_split = *block.add_kernels();
	
		kernel_split.set_type(hir::pb::ComputeKernel);

		VariableMap::iterator U32_256_split = getConstantIntVariable(variables, cfg, 256);
		VariableMap::iterator U32_350_split  = getConstantIntVariable(variables, cfg, 350);
		VariableMap::iterator U32_0_split   = getConstantIntVariable(variables, cfg, 0);
		
		hir::pb::Operand& ctas_split = *kernel_split.add_operands();
		ctas_split.set_mode(hir::pb::In);
		ctas_split.set_variable(U32_350_split->second.id);
//		ctas_split.set_variable(U32_256_split->second.id);
		
		hir::pb::Operand& threads_split = *kernel_split.add_operands();
		threads_split.set_mode(hir::pb::In);
		threads_split.set_variable(U32_256_split->second.id);
//		threads_split.set_variable(U32_32_split->second.id);
		
		hir::pb::Operand& shared_split = *kernel_split.add_operands();
		shared_split.set_mode(hir::pb::In);
		shared_split.set_variable(U32_0_split->second.id);
	
		hir::pb::Operand& d_key = *kernel_split.add_operands();

		d_key.set_mode(hir::pb::InOut);
		d_key.set_variable(temp_split_key->second.id);
	
		hir::pb::Operand& d_value = *kernel_split.add_operands();

		d_value.set_mode(hir::pb::InOut);
		d_value.set_variable(temp_split_value->second.id);
	
		RelationalAlgebraKernel ptxKernel_split(RelationalAlgebraKernel::Split, 
			temp_split_key->second.types,
			temp_split_value->second.types, temp_project->second.types, 
			agg.domains_size());
	
		hir::pb::Operand& a_split = *kernel_split.add_operands();
		a_split.set_variable(temp_project->second.id);
		a_split.set_mode(hir::pb::In);

		ptxKernel_split.set_id(kernel_id++);
		kernel_split.set_name(ptxKernel_split.name());
		kernel_split.set_code(compilePTXSource(ptxKernel_split.cudaSourceRepresentation()));
		
		hir::pb::Operand& aSize_split = *kernel_split.add_operands();
		aSize_split.set_variable(sizeVariable->second.id);
		aSize_split.set_mode(hir::pb::In);
	
		// skip sorting if only the lower dimensions were removed
		if(!skipSort)
		{
			//sort
			addSortPair(block, cfg, variables, temp_split_key, temp_split_value, sizeVariable_key);
		}
	
		// reduce by key
		report(" Reduce by Key.");

		VariableMap::iterator temp_reduce_key = getTempBuffer(variables, cfg, 0, 
			temp_split_key->second.types);
		VariableMap::iterator temp_reduce_value = getTempBuffer(variables, cfg, 0,
			temp_split_value->second.types);

		// Resize the output array of reduce 
		report("resize the output size of reduce.");
		hir::pb::Kernel& resizeResultKernel_reduce_key = *block.add_kernels();
		
		resizeResultKernel_reduce_key.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeResultD_reduce_key    
			= *resizeResultKernel_reduce_key.add_operands();
		hir::pb::Operand& resizeResultSize_reduce_key 
			= *resizeResultKernel_reduce_key.add_operands();
		
		resizeResultD_reduce_key.set_mode(hir::pb::Out);
		resizeResultD_reduce_key.set_variable(temp_reduce_key->second.id);
		
		resizeResultSize_reduce_key.set_mode(hir::pb::In);
		resizeResultSize_reduce_key.set_variable(sizeVariable_key->second.id);
	
		resizeResultKernel_reduce_key.set_name("resize");
		resizeResultKernel_reduce_key.set_code("");
	
		hir::pb::Kernel& resizeResultKernel_reduce_value = *block.add_kernels();
		
		resizeResultKernel_reduce_value.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeResultD_reduce_value    
			= *resizeResultKernel_reduce_value.add_operands();
		hir::pb::Operand& resizeResultSize_reduce_value 
			= *resizeResultKernel_reduce_value.add_operands();
		
		resizeResultD_reduce_value.set_mode(hir::pb::Out);
		resizeResultD_reduce_value.set_variable(temp_reduce_value->second.id);
		
		resizeResultSize_reduce_value.set_mode(hir::pb::In);
		resizeResultSize_reduce_value.set_variable(sizeVariable_value->second.id);
	
		resizeResultKernel_reduce_value.set_name("resize");
		resizeResultKernel_reduce_value.set_code("");
	
		hir::pb::Kernel& reduceKernel = *block.add_kernels();
		reduceKernel.set_type(hir::pb::BinaryKernel);

		hir::pb::Operand& d_reduce_key = *reduceKernel.add_operands();
		d_reduce_key.set_variable(temp_reduce_key->second.id);
		d_reduce_key.set_mode(hir::pb::InOut);
	
		hir::pb::Operand& d_reduce_value = *reduceKernel.add_operands();
		d_reduce_value.set_variable(temp_reduce_value->second.id);
		d_reduce_value.set_mode(hir::pb::InOut);

		hir::pb::Operand& a_reduce_key = *reduceKernel.add_operands();
		a_reduce_key.set_variable(temp_split_key->second.id);
		a_reduce_key.set_mode(hir::pb::In);

		hir::pb::Operand& a_reduce_key_size = *reduceKernel.add_operands();
		a_reduce_key_size.set_variable(sizeVariable_key->second.id);
		a_reduce_key_size.set_mode(hir::pb::InOut);

		hir::pb::Operand& a_reduce_value = *reduceKernel.add_operands();
		a_reduce_value.set_variable(temp_split_value->second.id);
		a_reduce_value.set_mode(hir::pb::In);

		hir::pb::Operand& a_reduce_value_size = *reduceKernel.add_operands();
		a_reduce_value_size.set_variable(sizeVariable_value->second.id);
		a_reduce_value_size.set_mode(hir::pb::InOut);

		RelationalAlgebraKernel binKernel(mapReduction(aggregation->tag()),
			temp_split_key->second.types, temp_split_value->second.types,
			temp_split_key->second.types, temp_split_value->second.types);

		binKernel.set_id(kernel_id++);
		reduceKernel.set_name(binKernel.name());
		reduceKernel.set_code(compileBINSource(binKernel.cudaSourceRepresentation(), 
			mapReduction(aggregation->tag())));
		
		temp_merge_key = temp_reduce_key;
		temp_merge_value = temp_reduce_value;
	}
	else
	{
		// skip sorting if only the lower dimensions were removed
		if(!skipSort)
		{
			addSortKey(block, cfg, variables, temp_project, sizeVariable);
		}

		// Starting add the split kernel
		VariableMap::iterator temp_generate = getTempBuffer(variables, cfg, 0);
	
		temp_generate->second.types.push_back(RelationalAlgebraKernel::Element(RelationalAlgebraKernel::I32, 32));

		// Get the size of the output array of split
		report("get the output size of generate.");
		hir::pb::Kernel& adjustSizeKernel = *block.add_kernels();

		adjustSizeKernel.set_type(hir::pb::ComputeKernel);
		
		VariableMap::iterator U32_1 = getConstantIntVariable(variables, cfg, 1);
		VariableMap::iterator U32_0 = getConstantIntVariable(variables, cfg, 0);
		
		hir::pb::Operand& ctas = *adjustSizeKernel.add_operands();
		ctas.set_mode(hir::pb::In);
		ctas.set_variable(U32_1->second.id);
	
		hir::pb::Operand& threads = *adjustSizeKernel.add_operands();
		threads.set_mode(hir::pb::In);
		threads.set_variable(U32_1->second.id);
	
		hir::pb::Operand& shared = *adjustSizeKernel.add_operands();
		shared.set_mode(hir::pb::In);
		shared.set_variable(U32_0->second.id);

		hir::pb::Operand& size_value = *adjustSizeKernel.add_operands();
		hir::pb::Operand& size = *adjustSizeKernel.add_operands();

		size_value.set_mode(hir::pb::Out);
		size_value.set_variable(sizeVariable_value->second.id);

		size.set_mode(hir::pb::In);
		size.set_variable(sizeVariable->second.id);

		RelationalAlgebraKernel ptxKernel(
			temp_generate->second.types, temp_project->second.types, 
			RelationalAlgebraKernel::GenerateGetResultSize);

		ptxKernel.set_id(kernel_id++);
		adjustSizeKernel.set_name(ptxKernel.name());
		adjustSizeKernel.set_code(compilePTXSource(
			ptxKernel.cudaSourceRepresentation()));
	
		// Resize the output array of generate
		hir::pb::Kernel& resizeResultKernel_generate_value = *block.add_kernels();
		
		resizeResultKernel_generate_value.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeResultD_generate_value    
			= *resizeResultKernel_generate_value.add_operands();
		hir::pb::Operand& resizeResultSize_generate_value 
			= *resizeResultKernel_generate_value.add_operands();
		
		resizeResultD_generate_value.set_mode(hir::pb::Out);
		resizeResultD_generate_value.set_variable(temp_generate->second.id);
		
		resizeResultSize_generate_value.set_mode(hir::pb::In);
		resizeResultSize_generate_value.set_variable(sizeVariable_value->second.id);
	
		resizeResultKernel_generate_value.set_name("resize");
		resizeResultKernel_generate_value.set_code("");

		//Add Generate
		report("add generate.");
		hir::pb::Kernel& kernel_generate = *block.add_kernels();
	
		kernel_generate.set_type(hir::pb::ComputeKernel);

		VariableMap::iterator U32_256_generate = getConstantIntVariable(variables, cfg, 256);
		VariableMap::iterator U32_350_generate  = getConstantIntVariable(variables, cfg, 350);
		VariableMap::iterator U32_0_generate   = getConstantIntVariable(variables, cfg, 0);
		
		hir::pb::Operand& ctas_generate = *kernel_generate.add_operands();
		ctas_generate.set_mode(hir::pb::In);
		ctas_generate.set_variable(U32_350_generate->second.id);
		
		hir::pb::Operand& threads_generate = *kernel_generate.add_operands();
		threads_generate.set_mode(hir::pb::In);
		threads_generate.set_variable(U32_256_generate->second.id);
		
		hir::pb::Operand& shared_generate = *kernel_generate.add_operands();
		shared_generate.set_mode(hir::pb::In);
		shared_generate.set_variable(U32_0_generate->second.id);
	
		hir::pb::Operand& d_value = *kernel_generate.add_operands();
		d_value.set_mode(hir::pb::InOut);
		d_value.set_variable(temp_generate->second.id);
	
		hir::pb::Operand& d_value_size = *kernel_generate.add_operands();
		d_value_size.set_mode(hir::pb::In);
		d_value_size.set_variable(sizeVariable_value->second.id);
	
		RelationalAlgebraKernel ptxKernel_generate(RelationalAlgebraKernel::Generate, 
			temp_generate->second.types); 

		ptxKernel_generate.set_id(kernel_id++);
		kernel_generate.set_name(ptxKernel_generate.name());
		kernel_generate.set_code(compilePTXSource(ptxKernel_generate.cudaSourceRepresentation()));

		// reduce by key
		report(" Reduce by Key.");

		VariableMap::iterator temp_reduce_key = getTempBuffer(variables, cfg, 0, 
			temp_project->second.types);
		VariableMap::iterator temp_reduce_value = getTempBuffer(variables, cfg, 0);
		temp_reduce_value->second.types.push_back(RelationalAlgebraKernel::Element(RelationalAlgebraKernel::I32, 32));

		// Resize the output array of reduce 
		report("resize the output size of reduce.");
		hir::pb::Kernel& resizeResultKernel_reduce_key = *block.add_kernels();
		
		resizeResultKernel_reduce_key.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeResultD_reduce_key    
			= *resizeResultKernel_reduce_key.add_operands();
		hir::pb::Operand& resizeResultSize_reduce_key 
			= *resizeResultKernel_reduce_key.add_operands();
		
		resizeResultD_reduce_key.set_mode(hir::pb::Out);
		resizeResultD_reduce_key.set_variable(temp_reduce_key->second.id);
		
		resizeResultSize_reduce_key.set_mode(hir::pb::In);
		resizeResultSize_reduce_key.set_variable(sizeVariable->second.id);
	
		resizeResultKernel_reduce_key.set_name("resize");
		resizeResultKernel_reduce_key.set_code("");
	
		hir::pb::Kernel& resizeResultKernel_reduce_value = *block.add_kernels();
		
		resizeResultKernel_reduce_value.set_type(hir::pb::Resize);
		
		hir::pb::Operand& resizeResultD_reduce_value    
			= *resizeResultKernel_reduce_value.add_operands();
		hir::pb::Operand& resizeResultSize_reduce_value 
			= *resizeResultKernel_reduce_value.add_operands();
		
		resizeResultD_reduce_value.set_mode(hir::pb::Out);
		resizeResultD_reduce_value.set_variable(temp_reduce_value->second.id);
		
		resizeResultSize_reduce_value.set_mode(hir::pb::In);
		resizeResultSize_reduce_value.set_variable(sizeVariable_value->second.id);
	
		resizeResultKernel_reduce_value.set_name("resize");
		resizeResultKernel_reduce_value.set_code("");
	
		hir::pb::Kernel& reduceKernel = *block.add_kernels();
		reduceKernel.set_type(hir::pb::BinaryKernel);

		hir::pb::Operand& d_reduce_key = *reduceKernel.add_operands();
		d_reduce_key.set_variable(temp_reduce_key->second.id);
		d_reduce_key.set_mode(hir::pb::InOut);
	
		hir::pb::Operand& d_reduce_value = *reduceKernel.add_operands();
		d_reduce_value.set_variable(temp_reduce_value->second.id);
		d_reduce_value.set_mode(hir::pb::InOut);

		hir::pb::Operand& a_reduce_key = *reduceKernel.add_operands();
		a_reduce_key.set_variable(temp_project->second.id);
		a_reduce_key.set_mode(hir::pb::In);

		hir::pb::Operand& a_reduce_key_size = *reduceKernel.add_operands();
		a_reduce_key_size.set_variable(sizeVariable->second.id);
		a_reduce_key_size.set_mode(hir::pb::InOut);

		hir::pb::Operand& a_reduce_value = *reduceKernel.add_operands();
		a_reduce_value.set_variable(temp_generate->second.id);
		a_reduce_value.set_mode(hir::pb::In);

		hir::pb::Operand& a_reduce_value_size = *reduceKernel.add_operands();
		a_reduce_value_size.set_variable(sizeVariable_value->second.id);
		a_reduce_value_size.set_mode(hir::pb::InOut);

		RelationalAlgebraKernel binKernel(RelationalAlgebraKernel::Count,
			temp_reduce_key->second.types, temp_reduce_value->second.types,
			temp_project->second.types, temp_generate->second.types);
		reduceKernel.set_type(hir::pb::BinaryKernel);

		binKernel.set_id(kernel_id++);
		reduceKernel.set_name(binKernel.name());
		reduceKernel.set_code(compileBINSource(binKernel.cudaSourceRepresentation(), 
			RelationalAlgebraKernel::Count));

		temp_merge_key = temp_reduce_key;
		temp_merge_value = temp_reduce_value;
		sizeVariable_key = sizeVariable;
	}

	// Get the size of the output array of merge

	report("get the output size of merge.\n");
	hir::pb::Kernel& adjustSizeKernel = *block.add_kernels();

	adjustSizeKernel.set_type(hir::pb::ComputeKernel);
	
	VariableMap::iterator U32_1 = getConstantIntVariable(variables, cfg, 1);
	VariableMap::iterator U32_0 = getConstantIntVariable(variables, cfg, 0);
	
	hir::pb::Operand& ctas = *adjustSizeKernel.add_operands();
	ctas.set_mode(hir::pb::In);
	ctas.set_variable(U32_1->second.id);

	hir::pb::Operand& threads = *adjustSizeKernel.add_operands();
	threads.set_mode(hir::pb::In);
	threads.set_variable(U32_1->second.id);

	hir::pb::Operand& shared = *adjustSizeKernel.add_operands();
	shared.set_mode(hir::pb::In);
	shared.set_variable(U32_0->second.id);
	
	hir::pb::Operand& size = *adjustSizeKernel.add_operands();

	size.set_mode(hir::pb::InOut);
	size.set_variable(sizeVariable->second.id);
	
	hir::pb::Operand& size_value = *adjustSizeKernel.add_operands();

	size_value.set_mode(hir::pb::In);
	size_value.set_variable(sizeVariable_value->second.id);

	RelationalAlgebraKernel ptxKernel(
		dId->second.types, 
		temp_merge_value->second.types, 
		RelationalAlgebraKernel::MergeGetResultSize);

	ptxKernel.set_id(kernel_id++);
	adjustSizeKernel.set_name(ptxKernel.name());
	adjustSizeKernel.set_code(compilePTXSource(
		ptxKernel.cudaSourceRepresentation()));

	// Resize the output array of merge

	report("resize the output size of merge.");
	hir::pb::Kernel& resizeResultKernel_merge = *block.add_kernels();
	
	resizeResultKernel_merge.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeResultD_merge    = *resizeResultKernel_merge.add_operands();
	hir::pb::Operand& resizeResultSize_merge = *resizeResultKernel_merge.add_operands();
	
	resizeResultD_merge.set_mode(hir::pb::Out);
	resizeResultD_merge.set_variable(dId->second.id);
	
	resizeResultSize_merge.set_mode(hir::pb::In);
	resizeResultSize_merge.set_variable(sizeVariable->second.id);

	resizeResultKernel_merge.set_name("resize");
	resizeResultKernel_merge.set_code("");

	// Add the merge kernel
	report("merge.");
	hir::pb::Kernel& kernel_merge = *block.add_kernels();
	kernel_merge.set_type(hir::pb::ComputeKernel);

	VariableMap::iterator U32_256_merge = getConstantIntVariable(variables, cfg, 256);
	VariableMap::iterator U32_350_merge  = getConstantIntVariable(variables, cfg, 350);
	VariableMap::iterator U32_0_merge   = getConstantIntVariable(variables, cfg, 0);
	
	hir::pb::Operand& ctas_merge = *kernel_merge.add_operands();
	ctas_merge.set_mode(hir::pb::In);
	ctas_merge.set_variable(U32_350_merge->second.id);
	
	hir::pb::Operand& threads_merge = *kernel_merge.add_operands();
	threads_merge.set_mode(hir::pb::In);
	threads_merge.set_variable(U32_256_merge->second.id);
	
	hir::pb::Operand& shared_merge = *kernel_merge.add_operands();
	shared_merge.set_mode(hir::pb::In);
	shared_merge.set_variable(U32_0_merge->second.id);

	RelationalAlgebraKernel ptxKernel_merge(RelationalAlgebraKernel::Merge,
		dId->second.types,
		temp_merge_key->second.types, temp_merge_value->second.types,
		agg.domains_size());

	hir::pb::Operand& d = *kernel_merge.add_operands();
	d.set_mode(hir::pb::InOut);
	d.set_variable(dId->second.id);

	hir::pb::Operand& a_merge_key = *kernel_merge.add_operands();
	a_merge_key.set_variable(temp_merge_key->second.id);
	a_merge_key.set_mode(hir::pb::In);

	hir::pb::Operand& aSize_merge_key = *kernel_merge.add_operands();
	aSize_merge_key.set_variable(sizeVariable_key->second.id);
	aSize_merge_key.set_mode(hir::pb::In);
	
	hir::pb::Operand& a_merge_value = *kernel_merge.add_operands();
	a_merge_value.set_variable(temp_merge_value->second.id);
	a_merge_value.set_mode(hir::pb::In);

	hir::pb::Operand& aSize_merge_value = *kernel_merge.add_operands();
	aSize_merge_value.set_variable(sizeVariable_value->second.id);
	aSize_merge_value.set_mode(hir::pb::In);

	ptxKernel_merge.set_id(kernel_id++);
	kernel_merge.set_name(ptxKernel_merge.name());
	kernel_merge.set_code(compilePTXSource(ptxKernel_merge.cudaSourceRepresentation()));
}

void addConditionalBranch(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	const lb::GPUCond& cond, VariableMap& variables)
{
	VariableMap::const_iterator aId = variables.find(cond.src1());
	assert(aId != variables.end());
	
	VariableMap::const_iterator bId = variables.find(cond.src2());
	assert(bId != variables.end());

	// Get the size of the first input
	hir::pb::Kernel& sizeKernelA = *block.add_kernels();
	
	VariableMap::const_iterator sizeVariableA = 
		getTempIntVariable(variables, cfg);
	sizeKernelA.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& sizeA            = *sizeKernelA.add_operands();
	hir::pb::Operand& sizeKernelAInput = *sizeKernelA.add_operands();
	
	sizeKernelAInput.set_mode(hir::pb::In);
	sizeKernelAInput.set_variable(aId->second.id);
	
	sizeA.set_mode(hir::pb::Out);
	sizeA.set_variable(sizeVariableA->second.id);
	
	sizeKernelA.set_name("get_size");
	sizeKernelA.set_code("");

	// Get the size of the second input
	hir::pb::Kernel& sizeKernelB = *block.add_kernels();
	
	VariableMap::const_iterator sizeVariableB = 
		getTempIntVariable(variables, cfg);
	sizeKernelB.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& sizeB            = *sizeKernelB.add_operands();
	hir::pb::Operand& sizeKernelBInput = *sizeKernelB.add_operands();
	
	sizeKernelBInput.set_mode(hir::pb::In);
	sizeKernelBInput.set_variable(bId->second.id);
	
	sizeB.set_mode(hir::pb::Out);
	sizeB.set_variable(sizeVariableB->second.id);
	
	sizeKernelB.set_name("get_size");
	sizeKernelB.set_code("");

	// Add the control
	block.mutable_control()->set_name("ConditionalBranch");

	VariableMap::iterator U32_32  = getConstantIntVariable(variables, cfg, 32);
	VariableMap::iterator U32_128 = getConstantIntVariable(variables, cfg, 128);
	VariableMap::iterator U32_0   = getConstantIntVariable(variables, cfg, 0);

	hir::pb::Operand& ctas = *block.mutable_control()->add_operands();
	ctas.set_mode(hir::pb::In);
	ctas.set_variable(U32_128->second.id);

	hir::pb::Operand& threads = *block.mutable_control()->add_operands();
	threads.set_mode(hir::pb::In);
	threads.set_variable(U32_32->second.id);

	hir::pb::Operand& shared = *block.mutable_control()->add_operands();
	shared.set_mode(hir::pb::In);
	shared.set_variable(U32_0->second.id);

	hir::pb::Operand& a = *block.mutable_control()->add_operands();
	a.set_mode(hir::pb::In);
	a.set_variable(aId->second.id);

	hir::pb::Operand& aSize = *block.mutable_control()->add_operands();
	aSize.set_mode(hir::pb::In);
	aSize.set_variable(sizeVariableA->second.id);

	hir::pb::Operand& b = *block.mutable_control()->add_operands();
	b.set_mode(hir::pb::In);
	b.set_variable(bId->second.id);

	hir::pb::Operand& bSize = *block.mutable_control()->add_operands();
	bSize.set_mode(hir::pb::In);
	bSize.set_variable(sizeVariableB->second.id);

	block.mutable_control()->set_name("conditional_branch");
	block.mutable_control()->set_type(hir::pb::ControlDecision);
	block.mutable_control()->add_targets(cond.yes());
	block.mutable_control()->add_targets(cond.no());
	block.mutable_control()->set_code(getCompareEqualPTX());
}

static void addUnconditionalBranch(hir::pb::BasicBlock& block,
	const lb::GPUGoto& jump)
{
	block.mutable_control()->set_name("UnconditionalBranch");
	block.mutable_control()->set_type(hir::pb::UnconditionalBranch);
	block.mutable_control()->add_targets(jump.target());
	block.mutable_control()->set_code("");
}

static void addMove(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg, 
	const lb::GPUMove& move, VariableMap& variables)
{
	std::string src = move.src1();
//	if(move.src1().compare("$logicQ13:_customerOrderCount") == 0)
//		src = "^$logicQ13:_customerOrderCount";
	VariableMap::const_iterator aId = variables.find(src);
	assert(aId != variables.end());
	VariableMap::const_iterator dId = variables.find(move.dest());
	assert(dId != variables.end());

	hir::pb::Kernel& sizeKernel = *block.add_kernels();
	
	VariableMap::const_iterator sizeVariable = 
		getTempIntVariable(variables, cfg);
	sizeKernel.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& sizeD = *sizeKernel.add_operands();
	hir::pb::Operand& input = *sizeKernel.add_operands();
	
	input.set_mode(hir::pb::In);
	input.set_variable(aId->second.id);
	
	sizeD.set_mode(hir::pb::Out);
	sizeD.set_variable(sizeVariable->second.id);
	
	sizeKernel.set_name("get_size");
	sizeKernel.set_code("");

	// Resize the output to the input's size
	hir::pb::Kernel& resizeKernel = *block.add_kernels();
	
	resizeKernel.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeD    = *resizeKernel.add_operands();
	hir::pb::Operand& resizeSize = *resizeKernel.add_operands();
	
	resizeD.set_mode(hir::pb::InOut);
	resizeD.set_variable(dId->second.id);
	
	resizeSize.set_mode(hir::pb::In);
	resizeSize.set_variable(sizeVariable->second.id);

	resizeKernel.set_name("resize");
	resizeKernel.set_code("");

	// Copy the input into the output
	hir::pb::Kernel& kernel = *block.add_kernels();
	
	kernel.set_name("copy");
	kernel.set_type(hir::pb::ComputeKernel);
	kernel.set_code(getMovePTX());
	kernel_id++;

	VariableMap::iterator U32_350  = getConstantIntVariable(variables, cfg, 350);
	VariableMap::iterator U32_256 = getConstantIntVariable(variables, cfg, 256);
//	VariableMap::iterator U32_128 = getConstantIntVariable(variables, cfg, 128);
	VariableMap::iterator U32_0   = getConstantIntVariable(variables, cfg, 0);

	hir::pb::Operand& ctas = *kernel.add_operands();
	ctas.set_mode(hir::pb::In);
//	ctas.set_variable(U32_128->second.id);
	ctas.set_variable(U32_350->second.id);

	hir::pb::Operand& threads = *kernel.add_operands();
	threads.set_mode(hir::pb::In);
//	threads.set_variable(U32_32->second.id);
	threads.set_variable(U32_256->second.id);

	hir::pb::Operand& shared = *kernel.add_operands();
	shared.set_mode(hir::pb::In);
	shared.set_variable(U32_0->second.id);

	hir::pb::Operand& d = *kernel.add_operands();
	d.set_mode(hir::pb::InOut);
	d.set_variable(dId->second.id);

	hir::pb::Operand& a = *kernel.add_operands();
	a.set_mode(hir::pb::In);
	a.set_variable(aId->second.id);

	hir::pb::Operand& b = *kernel.add_operands();
	b.set_mode(hir::pb::In);
	b.set_variable(sizeVariable->second.id);
}

static void addExit(hir::pb::BasicBlock& block)
{
	block.mutable_control()->set_name("Exit");
	block.mutable_control()->set_type(hir::pb::Exit);
	block.mutable_control()->set_code("");
}

#if 0
static VariableMap::iterator addCopy(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg,
	const lb::GPUAssign& assign, VariableMap& variables)
{
	VariableMap::const_iterator dId = variables.find(assign.dest());
	assert(dId != variables.end());

	hir::pb::Kernel& sizeKernel = *block.add_kernels();
	
	VariableMap::const_iterator sizeVariable = 
		getTempIntVariable(variables, cfg);
	sizeKernel.set_type(hir::pb::GetSize);
	
	hir::pb::Operand& sizeD = *sizeKernel.add_operands();
	hir::pb::Operand& input = *sizeKernel.add_operands();
	
	input.set_mode(hir::pb::In);
	input.set_variable(dId->second.id);
	
	sizeD.set_mode(hir::pb::Out);
	sizeD.set_variable(sizeVariable->second.id);
	
	sizeKernel.set_name("get_size");
	sizeKernel.set_code("");

	// Create a temporay buffer for the variable
	VariableMap::iterator temp = getTempBuffer(variables, cfg, 0);
	temp->second.types = dId->second.types;

	// Resize the temp output to the real output's size
	hir::pb::Kernel& resizeKernel = *block.add_kernels();
	
	resizeKernel.set_type(hir::pb::Resize);
	
	hir::pb::Operand& resizeD    = *resizeKernel.add_operands();
	hir::pb::Operand& resizeSize = *resizeKernel.add_operands();
	
	resizeD.set_mode(hir::pb::InOut);
	resizeD.set_variable(temp->second.id);
	
	resizeSize.set_mode(hir::pb::In);
	resizeSize.set_variable(sizeVariable->second.id);

	resizeKernel.set_name("resize");
	resizeKernel.set_code("");

	// Copy the output into the output temp
	hir::pb::Kernel& kernel = *block.add_kernels();
	
	kernel.set_name("copy");
	kernel.set_type(hir::pb::ComputeKernel);
	kernel.set_code(getMovePTX());
	
	hir::pb::Operand& d = *kernel.add_operands();
	
	VariableMap::iterator U32_32  = getConstantIntVariable(variables, cfg, 32);
	VariableMap::iterator U32_128 = getConstantIntVariable(variables, cfg, 128);
	VariableMap::iterator U32_0   = getConstantIntVariable(variables, cfg, 0);

	hir::pb::Operand& ctas = *kernel.add_operands();
	ctas.set_mode(hir::pb::In);
	ctas.set_variable(U32_128->second.id);

	hir::pb::Operand& threads = *kernel.add_operands();
	threads.set_mode(hir::pb::In);
	threads.set_variable(U32_32->second.id);

	hir::pb::Operand& shared = *kernel.add_operands();
	shared.set_mode(hir::pb::In);
	shared.set_variable(U32_0->second.id);

	d.set_mode(hir::pb::Out);
	d.set_variable(temp->second.id);

	hir::pb::Operand& a = *kernel.add_operands();
	
	a.set_mode(hir::pb::In);
	a.set_variable(dId->second.id);

	hir::pb::Operand& b = *kernel.add_operands();
	
	b.set_mode(hir::pb::In);
	b.set_variable(sizeVariable->second.id);

	return temp;
}

static void addMerge(hir::pb::BasicBlock& block,
	hir::pb::KernelControlFlowGraph& cfg,
	const lb::GPUAssign& assign, VariableMap& variables,
	VariableMap::iterator previousCopy)
{
	VariableMap::const_iterator dId = variables.find(assign.dest());
	assert(dId != variables.end());

	// Add the merge kernel
	hir::pb::Kernel& mergeKernel = *block.add_kernels();
	
	mergeKernel.set_type(hir::pb::Merge);
		
	hir::pb::Operand& mergeD = *mergeKernel.add_operands();
	mergeD.set_mode(hir::pb::InOut);
	mergeD.set_variable(dId->second.id);
	
	hir::pb::Operand& mergeA = *mergeKernel.add_operands();
	mergeA.set_mode(hir::pb::In);
	mergeA.set_variable(previousCopy->second.id);

	VariableMap::iterator type = getConstantIntVariable(variables, 
		cfg, getTupleDataType(dId->second.types));
	
	hir::pb::Operand& mergeType = *mergeKernel.add_operands();

	mergeType.set_variable(type->second.id);
	mergeType.set_mode(hir::pb::In);

	mergeKernel.set_name("merge");
	mergeKernel.set_code("");

	hir::pb::Kernel& uniqueKernel = *block.add_kernels();

	uniqueKernel.set_type(hir::pb::Unique);
	hir::pb::Operand& d = *uniqueKernel.add_operands();

	d.set_variable(dId->second.id);
	d.set_mode(hir::pb::InOut);
	
	hir::pb::Operand& a = *uniqueKernel.add_operands();

	a.set_variable(type->second.id);
	a.set_mode(hir::pb::In);

	uniqueKernel.set_name("unique");
	uniqueKernel.set_code("");
}
#endif

static void addBlock(hir::pb::KernelControlFlowGraph& cfg, StringIdMap& strings,
	VariableMap& variables, const lb::GPUGraph& raGraph, int blockNumber)
{
	const lb::GPUSequence& sequence = raGraph.sequences(blockNumber);
	
	hir::pb::BasicBlock& block = *cfg.add_blocks();
	
	block.set_id(sequence.uniqueidentifier());

	report("Adding block " << sequence.uniqueidentifier());

	if(sequence.uniqueidentifier() == 0)
	{
		std::vector<std::string> string0;
		std::vector<std::string> string1;
		std::vector<std::string> string2;

		for(VariableMap::iterator v = variables.begin(); v != variables.end(); ++v)
		{
			report("variable " << v->first << " " << v->second.id << " " << v->second.types.size() << " " << v->second.unique_keys);
			if(v->first.compare("N_NAME_STRING") == 0)
				string1.push_back("N_NAME");
			else if(v->first.compare("N_NAME_COPY_STRING") == 0)
				string1.push_back("N_NAME_COPY");
			else if(v->first.compare("P_MFGR_STRING") == 0)
				string1.push_back("P_MFGR");
			else if(v->first.compare("P_TYPE_STRING") == 0)
				string1.push_back("P_TYPE");
			else if(v->first.compare("R_NAME_STRING") == 0)
				string1.push_back("R_NAME");
			else if(v->first.compare("S_ADDRESS_STRING") == 0)
				string1.push_back("S_ADDRESS");
			else if(v->first.compare("S_COMMENT_STRING") == 0)
				string1.push_back("S_COMMENT");
			else if(v->first.compare("S_NAME_STRING") == 0)
				string1.push_back("S_NAME");
			else if(v->first.compare("S_PHONE_STRING") == 0)
				string1.push_back("S_PHONE");
			else if(v->first.compare("C_MKTSEGMENT_STRING") == 0)
				string1.push_back("C_MKTSEGMENT");
			else if(v->first.compare("P_NAME_STRING") == 0)
				string1.push_back("P_NAME");
			else if(v->first.compare("O_ORDERPRIORITY_STRING") == 0)
				string1.push_back("O_ORDERPRIORITY"); 
			else if(v->first.compare("C_COMMENT_STRING") == 0)
				string1.push_back("C_COMMENT"); 
			else if(v->first.compare("C_PHONE_STRING") == 0)
				string1.push_back("C_PHONE"); 
			else if(v->first.compare("C_ADDRESS_STRING") == 0)
				string1.push_back("C_ADDRESS"); 
			else if(v->first.compare("C_NAME_STRING") == 0)
				string1.push_back("C_NAME"); 
			else if(v->first.compare("RF_NAME_STRING") == 0)
				string1.push_back("RF_NAME"); 
			else if(v->first.compare("O_COMMENT_STRING") == 0)
				string1.push_back("O_COMMENT"); 
			else if(v->first.compare("P_BRAND_STRING") == 0)
				string1.push_back("P_BRAND"); 
			else if(v->first.compare("P_CONTAINER_STRING") == 0)
				string1.push_back("P_CONTAINER"); 
			else if(v->first.compare("O_ORDERSTATUS_STRING") == 0)
				string1.push_back("O_ORDERSTATUS"); 
			else if(v->first.compare("L_SHIPMODE_STRING") == 0)
				string2.push_back("L_SHIPMODE"); 
			else if(v->first.compare("L_SHIPINSTRUCT_STRING") == 0)
				string2.push_back("L_SHIPINSTRUCT"); 
			else if(v->first.compare("+$logicQ2:_regionName_string") == 0)
				string0.push_back("+$logicQ2:_regionName");
			else if(v->first.compare("+$logicQ2:_type_string") == 0)
				string0.push_back("+$logicQ2:_type");
			else if(v->first.compare("+$logicQ3:_segment_string") == 0)
				string0.push_back("+$logicQ3:_segment");
			else if(v->first.compare("+$logicQ5:_regionName_string") == 0)
				string0.push_back("+$logicQ5:_regionName");
			else if(v->first.compare("+$logicQ7:_nation1_string") == 0)
				string0.push_back("+$logicQ7:_nation1");
			else if(v->first.compare("+$logicQ7:_nation2_string") == 0)
				string0.push_back("+$logicQ7:_nation2");
			else if(v->first.compare("+$logicQ8:_nation_string") == 0)
				string0.push_back("+$logicQ8:_nation");
			else if(v->first.compare("+$logicQ8:_region_string") == 0)
				string0.push_back("+$logicQ8:_region");
			else if(v->first.compare("+$logicQ8:_type_string") == 0)
				string0.push_back("+$logicQ8:_type");
			else if(v->first.compare("+$logicQ9:_color_string") == 0)
				string0.push_back("+$logicQ9:_color");
			else if(v->first.compare("+$logicQ11:_nation_string") == 0)
				string0.push_back("+$logicQ11:_nation");
			else if(v->first.compare("+$logicQ12:_shipmodeIds_string") == 0)
				string0.push_back("+$logicQ12:_shipmodeIds");
			else if(v->first.compare("+$logicQ13:_word1_string") == 0)
				string0.push_back("+$logicQ13:_word1");
			else if(v->first.compare("+$logicQ13:_word2_string") == 0)
				string0.push_back("+$logicQ13:_word2");
			else if(v->first.compare("+$logicQ16:_brand_string") == 0)
				string0.push_back("+$logicQ16:_brand");
			else if(v->first.compare("+$logicQ16:_type_string") == 0)
				string0.push_back("+$logicQ16:_type");
			else if(v->first.compare("+$logicQ17:_brand_string") == 0)
				string0.push_back("+$logicQ17:_brand");
			else if(v->first.compare("+$logicQ17:_container_string") == 0)
				string0.push_back("+$logicQ17:_container");
			else if(v->first.compare("+$logicQ19:_brand1_string") == 0)
				string0.push_back("+$logicQ19:_brand1");
			else if(v->first.compare("+$logicQ19:_brand2_string") == 0)
				string0.push_back("+$logicQ19:_brand2");
			else if(v->first.compare("+$logicQ19:_brand3_string") == 0)
				string0.push_back("+$logicQ19:_brand3");
			else if(v->first.compare("+$logicQ20:_color_string") == 0)
				string0.push_back("+$logicQ20:_color");
			else if(v->first.compare("+$logicQ20:_nation_string") == 0)
				string0.push_back("+$logicQ20:_nation");
			else if(v->first.compare("+$logicQ21:_nation_string") == 0)
				string0.push_back("+$logicQ21:_nation");
			else if(v->first.compare("+$logicQ22:_I1_string") == 0)
				string0.push_back("+$logicQ22:_I1");
			else if(v->first.compare("+$logicQ22:_I2_string") == 0)
				string0.push_back("+$logicQ22:_I2");
			else if(v->first.compare("+$logicQ22:_I3_string") == 0)
				string0.push_back("+$logicQ22:_I3");
			else if(v->first.compare("+$logicQ22:_I4_string") == 0)
				string0.push_back("+$logicQ22:_I4");
			else if(v->first.compare("+$logicQ22:_I5_string") == 0)
				string0.push_back("+$logicQ22:_I5");
			else if(v->first.compare("+$logicQ22:_I6_string") == 0)
				string0.push_back("+$logicQ22:_I6");
			else if(v->first.compare("+$logicQ22:_I7_string") == 0)
				string0.push_back("+$logicQ22:_I7");
		}

		for(std::vector<std::string>::iterator i = string0.begin(); i != string0.end(); ++i)
		{
			std::string tmp = *i + "_string";
			addString(block, variables, cfg, *i, tmp, 0);
		}

		for(std::vector<std::string>::iterator i = string1.begin(); i != string1.end(); ++i)
		{
			std::string tmp = *i + "_STRING";
			addString(block, variables, cfg, *i, tmp, 1);
		}

		for(std::vector<std::string>::iterator i = string2.begin(); i != string2.end(); ++i)
		{
			std::string tmp = *i + "_STRING";
			addString(block, variables, cfg, *i, tmp, 2);
		}
	}

	for(int c = 0; c != sequence.operators_size(); ++c)
	{
		const lb::GPUCommand& command = sequence.operators(c);

		switch(command.tag())
		{
		case lb::ASSIGN:
		{
//			VariableMap::iterator previousValue = addCopy(block, cfg,
//				command.assign(), variables);
			report(" Assign");
			switch(command.assign().op().tag())
			{
			case lb::UNION:
			{
				report("  Union");
				if(blocknum++ < maxblocknum)
				addUnion(block, cfg, command.assign(), variables);
				break;
			}
			case lb::INTERSECTION:
			{
				report("  Intersection");
				if(blocknum++ < maxblocknum)
				addIntersection(block, command.assign(), variables);
				break;
			}
			case lb::PRODUCT:
			{
				report("  Product");
//				addProduct(block, cfg, command.assign(), variables);
				break;
			}
			case lb::SINGLE:
			{
				report("  Single");
				if(blocknum++ < maxblocknum) break;
				addSingle(block, cfg, strings, variables, command.assign());
				break;
			}
			case lb::DIFFERENCE:
			{
				report("  Difference");
				if(blocknum++ < maxblocknum)
				addDifference(block, cfg, command.assign(), variables);
				break;
			}
			case lb::JOIN:
			{
				report("  Join");
				if(blocknum++ < maxblocknum)
//				if(blocknum != 38) 
				addMapJoin(block, cfg, command.assign(), variables, strings);
				break;
			}
//			case lb::PROJECTION:
//			{
//				report("  Projection");
//				addProjection(block, cfg, command.assign(), variables);
//				break;
//			}
//			case lb::SELECT:
//			{
//				report("  Select");
//				addSelect(block, cfg, command.assign(), variables, strings);
//				break;
//			}
			case lb::MAPFILTER:
			{
				report("  MapFilter");
				if(blocknum++ < maxblocknum)
				addMapFilter(block, cfg, command.assign(), variables, strings);
				break;
			}
			case lb::AGG:
			{
				report("  Agg");
				if(blocknum++ < maxblocknum)
				addAgg(block, cfg, command.assign(), variables);
			}
			}
			
//			addMerge(block, cfg, command.assign(), variables, previousValue);
			break;
		}
		case lb::COND:
		{
			report(" Conditional");
			addConditionalBranch(block, cfg, command.cond(), variables);
			break;
		}
		case lb::GOTO:
		{
			report(" Goto");
			addUnconditionalBranch(block, command.jump());
			break;
		}
		case lb::MOVE:
		{
			report(" Move");
			if(blocknum++ < maxblocknum)
			addMove(block, cfg, command.move(), variables);			
			break;
		}
		case lb::HALT:
		{
			report(" Exit");
			addExit(block);
			break;
		}
		}
	}

//	blocknum++;
}

static void setupStringsFromBlock(StringIdMap& stringIds,
	const lb::GPUGraph& raGraph, int blockNumber)
{
	const lb::GPUSequence& sequence = raGraph.sequences(blockNumber);

	for(int c = 0; c != sequence.operators_size(); ++c)
	{
		const lb::GPUCommand& command = sequence.operators(c);
	
		if(command.tag() == lb::ASSIGN)
		{
			if(lb::SINGLE == command.assign().op().tag())
			{
				const lb::GPUSingle& single = command.assign().op().single();
	
				for(RepeatedConstant::const_iterator 
					constant = single.element().begin();
					constant != single.element().end(); ++constant)
				{
					if(constant->kind() == common::Constant_Kind_STRING)
					{
						stringIds.insert(
							std::make_pair(constant->string_constant().value(),
							stringIds.size()));
					}
				}
			}
			else if(command.assign().op().tag() == lb::MAPFILTER)
			{
				const lb::GPUMapFilter& mapFilter = command.assign().op().mapfilter();

				if(mapFilter.predicate().tag() == lb::COMP)
				{
					const lb::Comp& comp = mapFilter.predicate().comp();

					if(comp.op1().tag() == lb::CONSTEXP && comp.op1().constexp().literal().kind() == common::Constant_Kind_STRING)
					{
						stringIds.insert(std::make_pair(comp.op1().constexp().literal().string_constant().value(), 
							stringIds.size()));
					}
					
					if(comp.op2().tag() == lb::CONSTEXP && 
						comp.op2().constexp().literal().kind() == common::Constant_Kind_STRING)
					{
						stringIds.insert(std::make_pair(comp.op2().constexp().literal().string_constant().value(), 
							stringIds.size()));
					}
				}
				else if(mapFilter.predicate().tag() == lb::TEST)
				{
					const lb::Test& test = mapFilter.predicate().test();

					if(test.testtype() == lb::FString && 
						(test.testname().compare("like") == 0 || test.testname().compare("notlike") == 0))
					{
						for(int i = 0; i < test.ops_size(); ++i)
						{
							if(test.ops(i).tag() == lb::CONSTEXP && 
								test.ops(i).constexp().literal().kind() == common::Constant_Kind_STRING)
								stringIds.insert(std::make_pair
									(test.ops(i).constexp().literal().string_constant().value(), 
									stringIds.size()));
						}
					}
				}
				else if(mapFilter.predicate().tag() == lb::AND && mapFilter.predicate().andcomp().and2().tag() == lb::TEST)
				{
					const lb::Test& test = mapFilter.predicate().andcomp().and2().test();

					if(test.testtype() == lb::FString && 
						(test.testname().compare("like") == 0 || test.testname().compare("notlike") == 0))
					{
						for(int i = 0; i < test.ops_size(); ++i)
						{
							if(test.ops(i).tag() == lb::CONSTEXP && 
								test.ops(i).constexp().literal().kind() == common::Constant_Kind_STRING)
								stringIds.insert(std::make_pair
									(test.ops(i).constexp().literal().string_constant().value(), 
									stringIds.size()));
						}
					}
				}	

			}
			else if(command.assign().op().tag() == lb::JOIN)
			{
				const lb::GPUJoin& join = command.assign().op().join();

				for(int i = 0; i < join.args_size(); ++i)
				{
					const lb::Exp& exp = join.args(i);

					if(exp.tag() == lb::CALL && exp.call().calltype() == lb::FString && exp.call().callname().compare("add") == 0)
					{
						stringIds.insert(std::make_pair("%", stringIds.size()));
					}
				}
			}
		}
	}
//	stringIds.insert(std::make_pair("R", stringIds.size()));
//	stringIds.insert(std::make_pair("1-URGENT", stringIds.size()));
//	stringIds.insert(std::make_pair("2-HIGH", stringIds.size()));
//	stringIds.insert(std::make_pair("%", stringIds.size()));
//	stringIds.insert(std::make_pair("PROMO%", stringIds.size()));
//	stringIds.insert(std::make_pair("%Customer%Complaints%", stringIds.size()));
//	stringIds.insert(std::make_pair("SM CASE", stringIds.size()));
//	stringIds.insert(std::make_pair("SM BOX", stringIds.size()));
//	stringIds.insert(std::make_pair("SM PACK", stringIds.size()));
//	stringIds.insert(std::make_pair("SM PKG", stringIds.size()));
//	stringIds.insert(std::make_pair("MED BAG", stringIds.size()));
//	stringIds.insert(std::make_pair("MED BOX", stringIds.size()));
//	stringIds.insert(std::make_pair("MED PKG", stringIds.size()));
//	stringIds.insert(std::make_pair("MED PACK", stringIds.size()));
//	stringIds.insert(std::make_pair("LG CASE", stringIds.size()));
//	stringIds.insert(std::make_pair("LG BOX", stringIds.size()));
//	stringIds.insert(std::make_pair("LG PACK", stringIds.size()));
//	stringIds.insert(std::make_pair("LG PKG", stringIds.size()));
//	stringIds.insert(std::make_pair("AIR", stringIds.size()));
//	stringIds.insert(std::make_pair("DELIVER IN PERSON", stringIds.size()));
//	stringIds.insert(std::make_pair("AIR REG", stringIds.size()));
//	stringIds.insert(std::make_pair("F", stringIds.size()));

//	for(unsigned int i = 0; i < 10; ++i)
//		for(unsigned int j = 0; j < 10; ++j)
//		{
//			std::stringstream stream;
//			stream << i << j;
//			stringIds.insert(std::make_pair(stream.str(), stringIds.size()));
//		}

//	for(unsigned int i = 10; i < 35; ++i)
//	{
//		std::stringstream stream;
//		stream << i;
//		stringIds.insert(std::make_pair(stream.str(), stringIds.size()));
//	}
}

#if 0
static void addTest(std::ostream& hir,
	const lb::GPUTest& test, const VariableMap& variables,
	const StringIdMap& stringIds)
{
	report("Adding test " << test.name());

	hir::pb::Test harmonyTest;

	harmonyTest.set_name(test.name());
	harmonyTest.set_programname(test.programname());
	
	for(int i = 0; i != test.inputs_size(); ++i)
	{
		const lb::GPUVariable& variable = test.inputs(i);
	
		report(" Adding input " << variable.varname());

		hir::pb::Variable& harmonyVariable = *harmonyTest.add_inputs();

		VariableMap::const_iterator v = variables.find(variable.varname());
		assert(v != variables.end());

		harmonyVariable.set_name(v->second.id);
		harmonyVariable.set_data(formatData(stringIds, variable.fields(), 
			variable.initialdata()));
	}

	for(int o = 0; o != test.outputs_size(); ++o)
	{
		const lb::GPUVariable& variable = test.outputs(o);

		report(" Adding output " << variable.varname());

		hir::pb::Variable& harmonyVariable = *harmonyTest.add_outputs();

		VariableMap::const_iterator v = variables.find(variable.varname());
		assert(v != variables.end());

		harmonyVariable.set_name(v->second.id);
		harmonyVariable.set_data(formatData(stringIds, variable.fields(), 
			variable.initialdata()));
	}	

	for(int f = 0; f != test.features_size(); ++f)
	{
		harmonyTest.add_features(test.features(f));
	}
	
	std::stringstream tempBuffer;
	
	if(!harmonyTest.SerializeToOstream(&tempBuffer))
	{
		throw hydrazine::Exception("Failed to serialize protocol buffer "
			"containing Harmony IR Test.");
	}
	
	long long unsigned int bytes = tempBuffer.str().size();
	hir.write((const char*)&bytes, sizeof(long long unsigned int));
	hir.write(tempBuffer.str().c_str(), bytes);
}

static void setupTestVariables(StringIdMap& stringIds, const lb::GPUTest& test)
{
	for(int i = 0; i != test.inputs_size(); ++i)
	{
		const lb::GPUVariable& variable = test.inputs(i);

		report("Setting up intput test variable " << variable.varname());
		buildStringTable(stringIds, variable.fields(), 
			variable.initialdata());
	}

	for(int o = 0; o != test.outputs_size(); ++o)
	{
		const lb::GPUVariable& variable = test.outputs(o);

		report("Setting up output test variable " << variable.varname());
		buildStringTable(stringIds, variable.fields(), 
			variable.initialdata());
	}
}
#endif

void RelationalAlgebraCompiler::compile(
	std::ostream& hir, std::istream& ra) const
{
	lb::GPUGraph raGraph;
	
	if(!raGraph.ParseFromIstream(&ra))
	{
		throw hydrazine::Exception("Failed to parse protocol buffer "
			"containing Relational Algebra Graph.");
	}

	hir::pb::KernelControlFlowGraph cfg;
	
	cfg.set_name(raGraph.graphname());
	cfg.set_entry(raGraph.entry());
	cfg.set_exit(raGraph.exit());
//	cfg.set_testcount(raGraph.tests_size());
	cfg.set_testcount(0);
	
	VariableMap variables;
	StringIdMap stringIds;

	// Building a table of possible strings is necessary to compute
	// the required bits per string
//	for(int t = 0; t != raGraph.tests_size(); ++t)
//	{
//		setupTestVariables(stringIds, raGraph.tests(t));
//	}

	for(int b = 0; b != raGraph.sequences_size(); ++b)
	{
      		setupStringsFromBlock(stringIds, raGraph, b);		
	}

	addVariables(stringIds, variables, cfg, raGraph);
	
	// Generate PTX for each block
	for(int b = 0; b != raGraph.sequences_size(); ++b)
	{
		addBlock(cfg, stringIds, variables, raGraph, b);		
	}
	
	std::stringstream hirTempBuffer;
	
	if(!cfg.SerializeToOstream(&hirTempBuffer))
	{
		throw hydrazine::Exception("Failed to serialize protocol buffer "
			"containing Harmony IR.");
	}
	
	long long unsigned int bytes = hirTempBuffer.str().size();
	hir.write((char*)&bytes, sizeof(long long unsigned int));
	hir.write(hirTempBuffer.str().c_str(), bytes);
	
//	for(int t = 0; t != raGraph.tests_size(); ++t)
//	{
//		addTest(hir, raGraph.tests(t), variables, stringIds);
//	}
}

/*! \brief Read in a stream containing a RA pb, get a readable string */
std::string RelationalAlgebraCompiler::getRAString(std::istream& ra) const
{
	lb::GPUGraph raGraph;
	
	if(!raGraph.ParseFromIstream(&ra))
	{
		throw hydrazine::Exception("Failed to parse protocol buffer "
			"containing Relational Algebra Graph.");
	}

	return raGraph.DebugString();
}

/*! \brief Read in a stream containing a HIR pb, get a readable string */
std::string RelationalAlgebraCompiler::getHIRString(std::istream& hir) const
{
	hir::pb::KernelControlFlowGraph cfg;

	long long unsigned int bytes = 0;
	hir.read((char*)&bytes, sizeof(long long unsigned int));

	std::string message(bytes, ' ');

	hir.read((char*)message.c_str(), bytes);
	
	std::stringstream stream(message);
	
	if(!cfg.ParseFromIstream(&stream))
	{
		throw hydrazine::Exception("Failed to parse protocol buffer "
			"containing Harmony IR.");
	}
	
	return cfg.DebugString();
}

}

#endif

