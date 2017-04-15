/*!
	\file Operators.h
	\date Friday June 5, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for Datalog operator functors.
*/

#ifndef OPERATORS_H_INCLUDED
#define OPERATORS_H_INCLUDED

#include <cmath>
#include <lb/lb/interface/KeyCompressor.h>

namespace gpu
{
	/*! \brief A namespace for GPU datalog operators */
	namespace operators
	{
		/*!	\brief The add operator */
		template< typename T >
		class add
		{
			public:
				typedef T type;
				
			public:
				type leftFactor;
				type rightFactor;
				type constant;

			public:
				add( const type& l = 1, const type& r = 1, const type& c = 0 ) 
					: leftFactor( l ), rightFactor( r ), constant( c )
				{
				
				}
				
				T operator()( const T& one, const T& two )
				{
					return leftFactor * one + rightFactor * two + constant;
				}
		};
		
		/*!	\brief The subtract operator */
		template< typename T >
		class subtract
		{
			public:
				typedef T type;

			public:
				T operator()( const T& one, const T& two )
				{
					return one - two;
				}
		};
		
		/*!	\brief The multiply operator */
		template< typename T >
		class multiply
		{
			public:
				typedef T type;

			private:
				type constant;

			public:
				multiply( const type& c = 1 ) : constant( c )
				{
				
				}
			
				T operator()( const type& one, const type& two )
				{
					return constant * one * two;
				}
		};
		
		/*!	\brief The divide operator */
		template< typename T >
		class divide
		{
			public:
				typedef T type;
			
			public:
				type _constant;

			public:
				divide( const type& c ) : _constant( c )
				{
				
				} 
			
				T operator()( const T& one, const T& two )
				{
					return _constant * ( one / two );
				}
		};
		
		/*!	\brief The log operator	*/
		template< typename T >
		class log
		{
			public:
				typedef T type;

			public:
				T operator()( const T& one )
				{
					return std::log( one );
				}
		};
		
		/*!	\brief The log10 operator */
		template< typename T >
		class log10
		{
			public:
				typedef T type;

			public:
				T operator()( const T& one )
				{
					return std::log10( one );
				}
		};
		
		/*!	\brief The exp operator	*/
		template< typename T >
		class exp
		{
			public:
				typedef T type;

			public:
				T operator()( const T& one )
				{
					return std::exp( one );
				}
		};
		
		/*!	\brief The sqrt operator */
		template< typename T >
		class sqrt
		{
			public:
				typedef T type;

			public:
				T operator()( const T& one )
				{
					return std::sqrt( one );
				}
		};
		
		/*!	\brief The tan operator	*/
		template< typename T >
		class tan
		{
			public:
				typedef T type;

			public:
				T operator()( const T& one )
				{
					return std::tan( one );
				}
		};
		
		/*!	\brief The cos operator */
		template< typename T >
		class cos
		{
			public:
				typedef T type;

			public:
				T operator()( const T& one )
				{
					return std::cos( one );
				}
		};
		
		/*!	\brief The sin operator */
		template< typename T >
		class sin
		{
			public:
				typedef T type;

			public:
				T operator()( const T& one )
				{
					return std::sin( one );
				}
		};
		
		/*!	\brief The ceil operator */
		template< typename T >
		class ceil
		{
			public:
				typedef T type;

			public:
				T operator()( const T& one )
				{
					return std::ceil( one );
				}
		};
		
		/*!	\brief The floor operator */
		template< typename T >
		class floor
		{
			public:
				typedef T type;

			public:
				T operator()( const T& one )
				{
					return std::floor( one );
				}
		};
		
		/*!	\brief The abs operator	*/
		template< typename T >
		class abs
		{
			public:
				typedef T type;

			public:
				T operator()( const T& one )
				{
					return std::abs( one );
				}
		};
		
		/*!	\brief The pow operator */
		template< typename T >
		class pow
		{
			public:
				typedef T type;

			public:
				T operator()( const T& one, const T& two )
				{
					return std::pow( one, two );
				}
		};
		
		/*!	\brief The isNan operator */
		template< typename T >
		class isNan
		{
			public:
				typedef T type;

			public:
				T operator()( const T& one )
				{
					return isnan( one );
				}
		};
		
		/*!	\brief The notNan operator */
		template< typename T >
		class notNan
		{
			public:
				typedef T type;

			public:
				T operator()( const T& one )
				{
					return !isnan( one );
				}
		};
		
		/*!	\brief The isFinite operator */
		template< typename T >
		class isFinite
		{
			public:
				typedef T type;

			public:
				T operator()( const T& one )
				{
					return !isinf( one );
				}
		};
		
		/*!	\brief The notFinite operator */
		template< typename T >
		class notFinite
		{
			public:
				typedef T type;

			public:
				T operator()( const T& one )
				{
					return isinf( one );
				}
		};
		
		/*! \brief Turn a unary operator as above to a binary operator */
		template< typename Operator, typename Key >
		class UnaryToBinary
		{
			public:
				typedef typename Operator::type type;
				typedef std::pair< Key, type > Pair;
			
			private:
				Operator _op;

			public:
				UnaryToBinary( const Operator& op ) : _op( op ) {}

				Pair operator()( const Pair& one, const Pair& two )
				{
					return Pair( one.first, _op( one.second, two.second ) );
				}
		};
		
		/*! \brief Decompress the key */
		template< typename T >
		class Decompressor : public std::unary_function< 
			lb::KeyCompressor::compressed_key, 
			lb::KeyCompressor::compressed_key >
		{
			public:
				typedef T value_type;
			
			public:
				lb::KeyCompressor::Decompressor _decompressor;
				
			public:
				Decompressor( const lb::KeyCompressor::Decompressor& d ) : 
					_decompressor( d )
				{
				}
			
				value_type operator()( const value_type& one )
				{
					return std::make_pair( _decompressor.decompress( 
						one.first ), one.second );
				}
		};
		
		/*! \brief Compress the key	*/
		template< typename T >
		class Compressor : public std::unary_function< 
			lb::KeyCompressor::compressed_key, 
			lb::KeyCompressor::compressed_key >
		{
			public:
				typedef T value_type;
			
			public:
				lb::KeyCompressor::Decompressor _decompressor;
				
			public:
				Compressor( const lb::KeyCompressor::Decompressor& d ) : 
					_decompressor( d )
				{
				}
			
				value_type operator()( const value_type& one )
				{
					return std::make_pair( _decompressor.compress( 
						one.first ), one.second );
				}
		};
		
		/*! Add an additional dimension to a compressed key using the value */
		template< typename T >
		class AppendValue : public std::unary_function< 
			lb::KeyCompressor::compressed_key, 
			lb::KeyCompressor::compressed_key >
		{
			public:
				typedef T value_type;
			
			public:
				lb::KeyCompressor::Decompressor _decompressor;
				
			public:
				AppendValue( const lb::KeyCompressor::Decompressor& d ) : 
					_decompressor( d )
				{
				}
			
				value_type operator()( const value_type& one )
				{
					return std::make_pair( _decompressor.compress( 
						one.first, one.second ), one.second );
				}
		};
		
		/*! Expand the dimension of the left by the right */
		template< typename T >
		class ProjectDimension : public std::binary_function< T, T, T >
		{
			public:
				typedef T value_type;
			
			public:
				lb::KeyCompressor::Decompressor _decompressor;
				
			public:
				ProjectDimension( const lb::KeyCompressor::Decompressor& d ) : 
					_decompressor( d )
				{
				}
			
				value_type operator()( const value_type& one, 
					const value_type& two )
				{
					return std::make_pair( _decompressor.compress( 
						two.first, one.first ), one.second );
				}
		};
		
		/*! Compare a key dimension to the value, set if true, unset if false */
		template< typename T, typename Compare >
		class CompareDimensionToValue : public std::unary_function< T, T >
		{
			public:
				typedef T value_type;
				typedef Compare compare_type;
			
			public:
				lb::KeyCompressor::Decompressor _decompressor;
				compare_type _comp;
				
			public:
				CompareDimensionToValue( 
					const lb::KeyCompressor::Decompressor& d, 
					const compare_type& c = compare_type() ) : 
					_decompressor( d ), _comp( c )
				{
				}
			
				value_type operator()( const value_type& one )
				{
					return std::make_pair( one.first, 
						_comp( _decompressor.decompress( 
						one.first ), one.second ) );
				}
		};
		
		/*! Select the left element if the corresponding right is true, 
			otherwise set to the default value */
		template< typename T >
		class SelectIf : public std::binary_function< T, T, T >
		{
			public:
				typedef T type;
			
			public:
				type defaultValue;
				bool flip;
			
			public:
				SelectIf( const type& d = 0, bool f = false ) 
					: defaultValue( d ), flip( f )
				{
					
				}
			
				type operator()( const type& one, const type& two )
				{
					return ( ( two != 0 ) ^ flip ) ? one : defaultValue;
				}
			
		};
		
		/*! \brief Remove a dimension */
		template< typename T >
		class RemoveDimension : public std::unary_function< T, T >
		{
			public:
				typedef T value_type;
			
			public:
				lb::KeyCompressor::Decompressor _decompressor;
				
			public:
				RemoveDimension( const lb::KeyCompressor::Decompressor& d ) : 
					_decompressor( d )
				{
				}
			
				value_type operator()( const value_type& one )
				{
					return std::make_pair( _decompressor.remove( one.first ), 
						one.second );
				}
		};
		
		/*! \brief Add a constant */
		template< typename T >
		class AddConstant : public std::unary_function< T, T >
		{
			public:
				typedef T value_type;
				typedef typename value_type::second_type second_type;
			
			public:
				second_type _constant;
				second_type _factor;
				
			public:
				AddConstant( const second_type& c, const second_type& f = 1 ) 
					: _constant( c ), _factor( f )
				{
				}
			
				value_type operator()( const value_type& one )
				{
					return std::make_pair( one.first, 
						_factor * one.second + _constant );
				}
		};
		
		/*!	\brief The min operator */
		template< typename T >
		class Min
		{
			public:
				typedef T type;

			public:
				T operator()( const T& one, const T& two )
				{
					return one < two ? one : two;
				}
		};
		
		/*!	\brief The min operator */
		template< typename T >
		class Max
		{
			public:
				typedef T type;

			public:
				T operator()( const T& one, const T& two )
				{
					return one < two ? two : one;
				}
		};
	}

}

#endif

