/*!
	\file GpuPrimitives.inl
	\date Friday May 29, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the set of GPU primitives needed to implement
		arbitrary datalog programs.
*/

#ifndef GPU_PRIMITIVES_INL_INCLUDED
#define GPU_PRIMITIVES_INL_INCLUDED

#include <redfox/ra/interface/DetailedImplementations.h>
#include <redfox/ra/interface/Operators.h>

namespace gpu
{
	namespace algorithms
	{
	
		template< typename Operator, typename DataStructure, typename Compare,
			typename LeftDecompressor, typename RightDecompressor >
		void set_intersection( DataStructure& result, const DataStructure& left, 
			const DataStructure& right, Operator _op, Compare comp, 
			LeftDecompressor leftDecompressor, 
			RightDecompressor rightDecompressor )
		{
			typedef typename DataStructure::key_type Key;
			typedef typename DataStructure::mapped_type Value;
			gpu::operators::UnaryToBinary< Operator, Key > op( _op );
			gpu::algorithms::detail::set_intersection( result, left, right, 
				op, comp, leftDecompressor, rightDecompressor );
		}

		template< typename Operator, typename DataStructure, typename Compare >
		void set_intersection( DataStructure& result, const DataStructure& left, 
			const DataStructure& right, Operator _op, Compare comp )
		{
			typedef typename DataStructure::key_type Key;
			typedef typename DataStructure::mapped_type Value;
			gpu::operators::UnaryToBinary< Operator, Key > op( _op );
			gpu::algorithms::detail::set_intersection( result, left, right, 
				op, comp );
		}

		template< typename _Operator, typename DataStructure, typename Compare >
		void set_union( DataStructure& result, const DataStructure& left, 
			const DataStructure& right, _Operator _op, Compare comp )
		{
			typedef typename DataStructure::key_type Key;
			typedef typename DataStructure::mapped_type Value;
			typedef gpu::operators::UnaryToBinary< _Operator, Key > Operator;
			Operator op( _op );
			gpu::algorithms::detail::set_union<Key, Value, Operator, 
				Compare>( result, left, right, op, comp );
		}

		template< typename _Operator, typename DataStructure, typename Compare >
		void set_difference( DataStructure& result, const DataStructure& left, 
			const DataStructure& right, _Operator _op, Compare comp )
		{
			typedef typename DataStructure::key_type Key;
			typedef typename DataStructure::mapped_type Value;
			typedef gpu::operators::UnaryToBinary< _Operator, Key > Operator;
			Operator op( _op );
			gpu::algorithms::detail::set_difference<Key, Value, Operator, 
				Compare>( result, left, right, op, comp );
		}
	  		
		template< typename Operator, typename DataStructure, typename Compare >
		void aggregate( DataStructure& result, const DataStructure& input, 
			Operator op, Compare comp )
		{
			typedef typename DataStructure::key_type Key;
			typedef typename DataStructure::mapped_type Value;
			gpu::algorithms::detail::aggregate<Key, Value, Operator, Compare>( 
				result, input, op, comp );
		}

		template< typename DataStructure, typename Compare >
		void select( DataStructure& result, const DataStructure& input, 
			Compare pred )
		{
			typedef typename DataStructure::key_type Key;
			typedef typename DataStructure::mapped_type Value;
			gpu::algorithms::detail::select< Key, Value, Compare >( 
				result, input, pred );
		}

		template< typename Result, typename Left, 
			typename Right, typename Compare >
		void map( Result& result, const Left& left, 
			const Right& right, Compare comp )
		{
			typedef typename Left::key_type Key1;
			typedef typename Left::mapped_type Key2;
			typedef typename Right::mapped_type Value2;
			gpu::algorithms::detail::map< Key1, Key2, Value2, Compare >( 
				result, left, right, comp );
		}
		
		template< typename DataStructure, typename Operator, typename Compare >
		void transform( DataStructure& result, 
			const DataStructure& in, Operator op, Compare comp )
		{
			typedef typename DataStructure::key_type Key;
			typedef typename DataStructure::mapped_type Value;
			gpu::algorithms::detail::transform< Key, Value, Operator, Compare >( 
				result, in, op, comp );
		}
		
		template< typename DataStructure, typename Operator, typename Compare >
		void project( DataStructure& result, const DataStructure& left, 
			const DataStructure& right, Operator op, Compare comp )
		{
			typedef typename DataStructure::key_type Key;
			typedef typename DataStructure::mapped_type Value;
			gpu::algorithms::detail::project< Key, Value, Operator, Compare >( 
				result, left, right, op, comp );
		}
		
	}
}

#endif

