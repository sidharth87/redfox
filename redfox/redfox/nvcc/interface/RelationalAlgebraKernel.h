/*! \file RelationalAlgebraKernel.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Sunday October 31, 2010
	\brief The header file for the RelationalAlgebraKernel class.
*/

#ifndef RELATIONAL_ALGEBRA_KERNEL_H_INCLUDED
#define RELATIONAL_ALGEBRA_KERNEL_H_INCLUDED

// Standard Library Includes
#include <vector>
#include <string>

namespace nvcc
{

/*! \brief A class for representing a single RA kernel */
class RelationalAlgebraKernel
{
public:
	/*! \brief All possible relational algebra operators */
	enum Operator
	{
		InvalidOperator,
		Union,
		Intersection,
		JoinFindBounds,
		ModernGPUJoinFindBounds,
		JoinTempSize,
		ModernGPUJoinTempSize,
		ModernGPUJoinResultSize,
		ModernGPUJoinMain,
		ModernGPUJoinGather,
		JoinMain,
		Scan,
		JoinGetSize,
		JoinGather,
		ProductGetResultSize,
		Product,
		Difference,
		DifferenceGetResultSize,
		Project,
		ProjectGetResultSize,
		SelectMain,
		SelectGetResultSize,
		SelectGather,
		AssignValue,
		Arithmetic,
		ArithGetResultSize,
		AppendStringGetResultSize,
		SubStringGetResultSize,
		SubString,
		Split,
		SplitKey,
		SplitGetResultSize,
		SplitKeyGetResultSize,
		Merge,
		MergeGetResultSize,
		Sort,
		ModernGPUSortPair,
		ModernGPUSortKey,
		RadixSortPair,
		RadixSortKey,
		Unique,
		Total,
		SingleTotal,
		SingleCount,
		SingleMax,
		Count,	
		Max,
		Min,
		Generate,
		GenerateGetResultSize,
		ConvGetResultSize,
		Convert,
		SetString,
		UnionGetResultSize
	};

	/*! \brief All possible basic data types */
	enum DataType
	{
		InvalidDataType,
		I8,
		I16,
		I32,
		I64,
		I128,
		I256,
		I512,
		I1024,
		F32,
		F64,
		Pointer
	};

	class ConstantElement
	{
	public:
		DataType type;
		union
		{
			unsigned int           stringId;
			long long unsigned int intValue;
			bool                   boolValue;
			double                 floatValue;
		};
		long long unsigned int limit; // Max value
	};
	
	class Element
	{
	public:
		Element(DataType type, long long unsigned int limit = 8192);
	
	public:
		DataType     type;
		long long unsigned int limit; // Max value
	};

	enum ArithmeticOp 
	{
		Add,        
		Multiply,   
		Subtract,   
		Divide,     
		Mod,
		AddMonth,
		AddYear,
		PartYear	
	};

	enum ArithExpNodeKind
	{
		OperatorNode,
		ConstantNode,
		IndexNode
	};

	class ArithExpNode
	{
	public:
		ArithExpNodeKind kind;
		ArithmeticOp op;
		DataType type;
		unsigned int long long i;
		std::string f;
		ArithExpNode* left;
		ArithExpNode* right;
	};
		
	/*! \brief All possible comparison operators */
	enum Comparison
	{
		InvalidComparison,
		Eq,
		Ne,
		Slt,
		Lt,
		Sle,
		Le,
		Sgt,
		Gt,
		Sge,
		Ge,
		Like,
		NotLike,
		EqString
	};
	
	enum ComparisonOperandType
	{
		Constant,
		ConstantVariable,
		VariableIndex,
		ConstantArith	
	};

	class ComparisonOperand
	{
	public:
		ComparisonOperandType type;
		union
		{
			unsigned int           variableIndex;
			unsigned int           stringId;
			long long unsigned int intValue;
			bool                   boolValue;
			double                 floatValue;
		};

		ArithExpNode	       arith;
	};
	
	class ComparisonExpression
	{
	public:
		Comparison        comparison;
		ComparisonOperand a;
		ComparisonOperand b;
	};


	/*! \brief A set of comparisons that need to be anded together */
	typedef std::vector<ComparisonExpression> ComparisonVector;
	
	/*! \brief A Variable is an n-ary tuple of primitives */
	typedef std::vector<Element> Variable;
	
	/*! \brief A Value is an n-ary tuple of constants */
	typedef std::vector<ConstantElement> Value;
	
	/*! \brief A list of fields to keep */
	typedef std::vector<unsigned int> IndexVector;

public:
	/*! \brief Get a string representation of an Operator */
	static std::string toString(Operator op, unsigned int id);

	/*! \brief Get a string representation of a DataType */
	static std::string mapValueType(DataType type);
	
	/*! \brief Get a string representation of an arithmetic expression */
	static void translateArithNode(std::stringstream& stream, ArithExpNode* node, 
		bool isLeft, std::string tupleName, std::string sourceName);
	static std::string arithToString(ArithExpNode _arithExp, 
		std::string tupleName, std::string sourceName);
	static std::string arithOpToString(ArithmeticOp op);

public:
	/*! \brief Create a RA kernel with specific types and op */
	RelationalAlgebraKernel(Operator op = InvalidOperator,
		const Variable& d = Variable(), const Variable& a = Variable(),
		const Variable& b = Variable(), const Variable& c = Variable());
	/*! \brief Create a RA kernel with specific types and op */
	RelationalAlgebraKernel(Operator op, const Variable& d, const Variable& a,
		const Variable& b, unsigned int keyFields);
	/*! \brief Create a RA kernel with specific types and op */
	RelationalAlgebraKernel(Operator op, const Variable& d,
		const Variable& a, unsigned int threads);
	/*! \brief Create a RA kernel with specific types and op */
	RelationalAlgebraKernel(Operator op, const Variable& d,
		unsigned int keyFields);
	/*! \brief Create a RA kernel with specific types and op */
	RelationalAlgebraKernel(Operator op, const Variable& d, const Variable& a,
		const Variable& b, const Variable& c, unsigned int keyFields,
		unsigned int threads);
	/*! \brief Create a RA kernel that requires a comparison */
//	RelationalAlgebraKernel(const ComparisonVector& c, 
//		const Variable& d = Variable(), const Variable& a = Variable());
	/*! \brief Create a RA kernel that requires a comparison */
	RelationalAlgebraKernel(const ComparisonVector& c, 
		const Variable& d = Variable(), const Variable& a = Variable(), 
		const Variable& b = Variable());
	/*! \brief Create a RA kernel that requires an initial value */
	RelationalAlgebraKernel(const Variable& d, const Value& v);
	/*! \brief Create a RA kernel that requires an index list */
	RelationalAlgebraKernel(const Variable& d,
		const Variable& a, const IndexVector& i);
	/*! \brief Create a RA kernel that requires an arithmetic expression tree */
	RelationalAlgebraKernel(const ArithExpNode& arith, 
		const Variable& d, const Variable& a);
	/*! \brief Create a RA kernel that requires an index list */
	RelationalAlgebraKernel(Operator op, const Variable& a,
		const Variable& b, const Variable& c, const int& domains);

	RelationalAlgebraKernel(Operator op, const Variable& a,
		const Variable& b, const Variable& c, const int& domains, const Variable& d, const Variable& e);

	RelationalAlgebraKernel(Operator op, const Variable& d,
		const Variable& a, const int& domains);
	/*! \brief Create a RA kernel that requires an index list */
//	RelationalAlgebraKernel(Operator op, const Variable& a,
//		const Variable& b);
	RelationalAlgebraKernel(const Variable& d,
		const Variable& a, const Variable& b, Operator op);

	RelationalAlgebraKernel(const Variable& d,
		const Variable& a, Operator op);

	RelationalAlgebraKernel(const Variable& a, Operator op);
	
	RelationalAlgebraKernel(const Variable& a, unsigned int index);

	RelationalAlgebraKernel(const Variable&d, const Variable& a, unsigned int index);

	RelationalAlgebraKernel(const DataType type, const unsigned int offset, const Variable& d,
		const Variable& a);

	/*! \brief Set the cta count */
	void setCtaCount(unsigned int ctas);
	
	/*! \brief Set the thread count */
	void setThreadCount(unsigned int threads);

	/*! \brief Set the thread count */
	void set_id(unsigned int id);

	/*! \brief Get the operator */
	Operator op() const;
	
	/*! \brief Get the operator */
	unsigned int id() const;

	/*! \brief Get the comparison operator */
	const ComparisonVector& comparisons() const;
	
	/*! \brief Get the value if this is an assignment kernel */
	const Value& value() const;

public:
	/*! \brief Get a cuda source representation of the kernel */
	std::string cudaSourceRepresentation() const;
	/*! \brief Get a canonical name for the kernel */
	std::string name() const;
	
public:
	/*! \brief The output type */
	Variable destination;
	/*! \brief The 2nd output type */
	Variable destination_1;
	/*! \brief The first input type */
	Variable sourceA;
	/*! \brief The second input type */
	Variable sourceB;
	/*! \brief The third input type */
	Variable sourceC;
	/*! \brief The first input type */
	Variable sourceD;

private:
	/*! \brief Get a cuda source representation of the union */
	std::string _unionSource() const;
	/*! \brief Get a cuda source representation of the intersection */
	std::string _intersectionSource() const;
	/*! \brief Get a cuda source representation of the product result size */
	std::string _productGetResultSizeSource() const;
	/*! \brief Get a cuda source representation of the product */
	std::string _productSource() const;
	/*! \brief Get a cuda source representation of the difference */
	std::string _differenceSource() const;
	/*! \brief Get cuda source of the join lowerbound function */
	std::string _joinFindBoundsSource() const;
	/*! \brief Get cuda source of the join lowerbound function */
	std::string _moderngpuJoinFindBoundsSource() const;
	/*! \brief Get cuda source of the join lowerbound function */
	std::string _moderngpuJoinGatherSource() const;
	/*! \brief Get a cuda source representation of the join tempsize function */
	std::string _joinTempSizeSource() const;
	/*! \brief Get a cuda source representation of the join tempsize function */
	std::string _moderngpuJoinTempSizeSource() const;
	/*! \brief Get a cuda source representation of the join tempsize function */
	std::string _moderngpuJoinResultSizeSource() const;
	/*! \brief Get a cuda source representation of the join main function */
	std::string _joinMainSource() const;
	/*! \brief Get a cuda source representation of the join main function */
	std::string _moderngpuJoinMainSource() const;
	/*! \brief Get a cuda source representation of the join scan function */
	std::string _scanSource() const;
	/*! \brief Get a cuda source representation of the join getsize function */
	std::string _joinGetSizeSource() const;
	/*! \brief Get a cuda source representation of the join gather function */
	std::string _joinGatherSource() const;
	/*! \brief Get a cuda source representation of the projection */
	std::string _projectSource() const;
	/*! \brief Get a cuda source representation of the project size function */
	std::string _projectGetResultSizeSource() const;
	/*! \brief Get a cuda source representation of the selection */
	std::string _selectMainSource() const;
	/*! \brief Get a cuda source representation of the select size function */
	std::string _selectGetResultSizeSource() const;
	/*! \brief Get a cuda source representation of the select gather */
	std::string _selectGatherSource() const;
	/*! \brief Get a cuda source representation of the assignment */
	std::string _assignSource() const;
	/*! \brief Get a cuda source representation of the arithmetic expression */
	std::string _arithSource() const;
	/*! \brief Get a cuda source representation of the arithmetic size function */
	std::string _arithGetResultSizeSource() const;
	/*! \brief Get a cuda source representation of the arithmetic size function */
	std::string _appendStringGetResultSizeSource() const;
	/*! \brief Get a cuda source representation of the arithmetic expression */
	std::string _splitSource() const;
	/*! \brief Get a cuda source representation of the arithmetic expression */
	std::string _splitkeySource() const;
	/*! \brief Get a cuda source representation of the arithmetic expression */
	std::string _splitKeyGetResultSizeSource() const;
	/*! \brief Get a cuda source representation of the arithmetic expression */
	std::string _splitGetResultSizeSource() const;
	/*! \brief Get a cuda source representation of the arithmetic expression */
	std::string _mergeSource() const;
	/*! \brief Get a cuda source representation of the arithmetic expression */
	std::string _mergeGetResultSizeSource() const;
	/*! \brief Get a cuda source representation of the arithmetic expression */
	std::string _sortSource() const;
	/*! \brief Get a cuda source representation of the arithmetic expression */
	std::string _moderngpuSortPairSource() const;
	/*! \brief Get a cuda source representation of the arithmetic expression */
	std::string _moderngpuSortKeySource() const;
	/*! \brief Get a cuda source representation of the arithmetic expression */
	std::string _b40cSortPairSource() const;
	/*! \brief Get a cuda source representation of the arithmetic expression */
	std::string _b40cSortKeySource() const;
	/*! \brief Get a cuda source representation of the arithmetic expression */
	std::string _uniqueSource() const;
	/*! \brief Get a cuda source representation of the arithmetic expression */
	std::string _countSource() const;
	/*! \brief Get a cuda source representation of the arithmetic expression */
	std::string _reduceSource(Operator op) const;

	std::string _singleReduceSource() const;
	/*! \brief Get a cuda source representation of the arithmetic expression */
	std::string _generateSource() const;

	std::string _generateGetResultSizeSource() const;

	std::string _convGetResultSizeSource() const;
	
	std::string _substringGetResultSizeSource() const;

	std::string _substringSource() const;

	std::string _convertSource() const;
	
	std::string _setStringSource() const;

	std::string _unionGetResultSizeSource() const;

	std::string _differenceGetResultSizeSource() const;

	/*! \brief Get a string representation of the number of key fields */
	std::string _fields() const;
	/*! \brief Get a string representation of the number of threads */
	std::string _threads() const;
	/*! \brief Get a string representation of the number of ctas */
	std::string _ctas() const;

private:
	Operator         _operator;    //! The type of operator
	ComparisonVector _comparisons; //! The comparisons being performed
	Value            _value;       //! Initial value to assign to the dest
	IndexVector      _indicies;    //! Inidicies for kernel fields
	unsigned int     _keyFields;   //! Number of fields to use for keys
	ArithExpNode	 _arithExp;    //! Arithmetic expression tree
	int		 _domains;     //! The key fields of split
	unsigned int     _threadCount; //! Number of threads to pass to the kernel
	unsigned int     _ctaCount;    //! Number of ctas to pass to the kernel
	unsigned int     _id;	       //! Number of ctas to pass to the kernel
	DataType	 _type;
	unsigned int	 _offset;
	unsigned int	 _index;
};

}

#endif


