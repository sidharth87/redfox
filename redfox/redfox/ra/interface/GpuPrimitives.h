/*!
	\file GpuPrimitives.h
	\date Friday May 29, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the set of GPU primitives needed to implement
		arbitrary datalog programs.
*/

#ifndef GPU_PRIMITIVES_H_INCLUDED
#define GPU_PRIMITIVES_H_INCLUDED

namespace gpu
{
	/*!
		\brief The set of low level operations that can be performed on gpu
			data structures.
			
		These are the C++ interfaces to corresponding GPU operations, they
		should be called from the host program.
	*/
	namespace algorithms
	{
		/*!
			\brief Compute the intersection between two data structures, storing
				the result in the third.  This version uses key decompression.
				
			\tparam DataStructure The type of data structure being joined.
			\tparam Operator The operator used to combine two tuples.  
				Should be a functor.
			\tparam Compare The comparison operator used to order the left and
				right data structures.
			\tparam LeftDecompressor The decompressor for the left array.
			\tparam RightDecompressor The decompressor for the right array.
			
			\param result The data structure to store the result in
			\param left The left data structure to join with the right
			\param right The right data structure to join with the left
			
			Note that this takes the key from the left array for the result.
		*/
		template< typename Operator, typename DataStructure, typename Compare, 
			typename LeftDecompressor, typename RightDecompressor >
		void set_intersection( DataStructure& result, const DataStructure& left,
			const DataStructure& right, Operator op, Compare comp, 
			LeftDecompressor leftDecompressor, 
			RightDecompressor rightDecompressor );

		/*!
			\brief Compute the intersection between two data structures, storing
				the result in the third.
				
			\tparam DataStructure The type of data structure being joined.
			\tparam Operator The operator used to combine two tuples.  
				Should be a functor.
			\tparam Compare The comparison operator used to order the left and
				right data structures.
			
			\param result The data structure to store the result in
			\param left The left data structure to join with the right
			\param right The right data structure to join with the left
			
		*/
		template< typename Operator, typename DataStructure, typename Compare >
		void set_intersection( DataStructure& result, const DataStructure& left,
			const DataStructure& right, Operator op, Compare comp );

		/*!
			\brief Compute the union between two data structures, storing
				the result in the third.
				
			\tparam DataStructure The type of data structure being joined.
			\tparam Operator The operator used to combine two tuples.  
				Should be a functor.
			\param result The data structure to store the result in
			\param left The left data structure to join with the right
			\param right The right data structure to join with the left
		*/
		template< typename Operator, typename DataStructure, typename Compare >
		void set_union( DataStructure& result, const DataStructure& left, 
			const DataStructure& right, Operator op, Compare comp );

		/*!
			\brief Compute the set difference between two data structures, 
				storing the result in the third.
				
			\tparam DataStructure The type of data structure being joined.
			\tparam Operator The operator used to combine two tuples.  
				Should be a functor.
			\param result The data structure to store the result in
			\param left The left data structure to join with the right
			\param right The right data structure to join with the left
		*/
		template< typename Operator, typename DataStructure, typename Compare >
		void set_difference( DataStructure& result, const DataStructure& left, 
			const DataStructure& right, Operator op, Compare comp );
	  
		/*!
			\brief Aggregate all of the tuples in a data structure that
				that compare equal into an output data structure
			
			\tparam DataStructure The type of data structure being aggregated
				across.
			\tparam Compare A functor used to compare elements.  Equal elements
				are aggregated.
			\param result The data structure to store the result in.
			\param input The input data structure to aggregate over.
		*/
		template< typename Operator, typename DataStructure, typename Compare >
		void aggregate( DataStructure& result, const DataStructure& input, 
			Operator op, Compare comp );
			
		/*!
			\brief Select all elements from a data structure that satisfy a
				binary predicate.
			
			\tparam DataStructure The type of data structure being aggregated
				across.
			\tparam Compare A unary predicate.
			\param result The data structure to store the result in.
			\param input The input data structure to select from.
		*/
		template< typename DataStructure, typename Compare >
		void select( DataStructure& result, const DataStructure& input, 
			Compare pred );

		/*!
			\brief Indirect lookup from the key of one data structure into
				the value of the second so:
				
			Key1,Key2 + Key2,Value2 -> Key1,Value2
			
			\tparam DataStructure The type of data structure being aggregated
				across.
			\tparam Compare The comparison for Key2
			\param result The data structure to store the result in.
			\param left The mapping input data structure
			\param right The mapped input data structure
		*/
		template< typename Result, typename Left, 
			typename Right, typename Compare >
		void map( Result& result, const Left& left, 
			const Right& right, Compare comp );

		/*!
			\brief Apply an operator to every element within a datastructure
			
			\tparam DataStructure The type of data structure being produced
			\tparam The operator being applied
			
			\param result The datastructure to store the result in
			\param in The input data structure
			\param op The operator being applied.
		*/
		template< typename DataStructure, typename Operator, typename Compare >
		void transform( DataStructure& result, 
			const DataStructure& in, Operator op, Compare comp );
			
			
		/*! \brief Extend the keyspace of a datastructure
			
			\tparam DataStructure The type of data structure being produced
			\tparam The operator being applied, each operator will be 
				applied to the cross product of left and right
			
			\param result The result is left projected onto the keyspace 
				of right
			\param left The original data structure being expanded
			\param right A data structure with a single dimension key space
			\param op An operator that compresses right's key and appends 
				it to left
			\param comp The comparison function used to sort the arrays
		*/
		template< typename DataStructure, typename Operator, typename Compare >
		void project( DataStructure& result, const DataStructure& left, 
			const DataStructure& right, Operator op, Compare comp );
	}
}

#endif

