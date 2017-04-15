/*!
	\file Map.h
	\date Wednesday June 3, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the Map class.
*/

#ifndef GPU_MAP_H_INCLUDED
#define GPU_MAP_H_INCLUDED

#include <hydrazine/interface/ValueCompare.h>
#include <hydrazine/cuda/DevicePointerIterator.h>
#include <hydrazine/implementation/debug.h>

#include <hydrazine/implementation/debug.h>
#include <redfox/ra/interface/DeviceVectorWrapper.h>
#include <algorithm>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace gpu
{

	namespace types
	{
		
		template< typename Key, typename Value, 
			typename Compare = std::less< Key > >
		class Map
		{
			public:
				typedef Map type;
				typedef size_t size_type;
				typedef Key key_type;
				typedef Value mapped_type;
				typedef std::pair< const key_type, mapped_type > value_type;
				typedef ptrdiff_t difference_type;
				typedef value_type* pointer;
				typedef const value_type* const_pointer;
				
				typedef hydrazine::cuda::DevicePointerIterator< pointer, type >
					iterator;
				typedef hydrazine::cuda::DevicePointerIterator< const_pointer, 
					type > const_iterator;
				typedef std::reverse_iterator< iterator > reverse_iterator;
				typedef std::reverse_iterator< const_iterator >
					const_reverse_iterator;
				typedef std::pair< iterator, bool > insertion;
				typedef hydrazine::ValueCompare< Compare, type > value_compare;
				typedef DeviceVector< Key, Value, type > _dv;
				typedef Compare key_compare;
				
				typedef typename iterator::reference reference;
			
			private:
				static iterator _convert( typename _dv::iterator i )
				{
					return iterator( reinterpret_cast< pointer >( i.base() ) );
				}
				
				static const_iterator _convert( typename _dv::const_iterator i )
				{
					return const_iterator( reinterpret_cast< const_pointer >( 
						i.base() ) );
				}
				
				static typename _dv::iterator _convert( iterator i )
				{
					return typename _dv::iterator( reinterpret_cast< 
						typename _dv::pointer >( i.base() ) );
				}
				
				static typename _dv::const_iterator _convert( const_iterator i )
				{
					return typename _dv::const_iterator( reinterpret_cast< 
						typename _dv::const_pointer >( i.base() ) );				
				}
				
			private:
				value_compare _compare;
				void* _vector;
			
			public:
				explicit Map( const Compare& comp = Compare() ) : 
					_compare( comp ), _vector( _dv::newVector() )
				{
				
				}
			
				Map( const Map& map ) : _compare( map._compare ), 
					_vector( _dv::newVector( map._vector ) )
				{
				}
			
				template< typename ForwardIterator >
				Map( ForwardIterator first, ForwardIterator last, 
					const Compare& comp = Compare() ) : 
					_compare( comp ), _vector( _dv::newVector( first, last ) )
				{
					
				}
			
				~Map()
				{
					_dv::destroyVector(_vector);
				}
			
				Map& operator=( const Map& map )
				{
					_dv::copyVector( _vector, map._vector );
					_compare = map._compare;
					return *this;
				}
				
			public:
				void clear()
				{
					_dv::clearVector( _vector );
				}
			
			public:
				size_type size() const
				{
					return _dv::vectorSize( _vector );
				}
			
				size_type max_size() const
				{
					return _dv::vectorMaxSize( _vector );
				}
			
				bool empty() const
				{
					return _dv::vectorEmpty( _vector );
				}
			
			public:
				iterator begin()
				{
					return _convert( _dv::vectorBegin( _vector ) );
				}
			
				const_iterator begin() const
				{
					return _convert( _dv::vectorBegin( _vector ) );
				}

				iterator end()
				{
					return _convert( _dv::vectorEnd( _vector ) );
				}
			
				const_iterator end() const
				{
					return _convert( _dv::vectorEnd( _vector ) );
				}
			
				reverse_iterator rbegin()
				{
					return reverse_iterator( end() );
				}
			
				const_reverse_iterator rbegin() const
				{
					return const_reverse_iterator( end() );
				}
			
				reverse_iterator rend()
				{
					return reverse_iterator( begin() );
				}
			
				const_reverse_iterator rend() const
				{
					return const_reverse_iterator( begin() );
				}
		
			public:
				reference operator[]( const key_type& key )
				{
					iterator fi = find( key );
					if( fi == end() )
					{
						fi = insert( std::make_pair( key, 
							mapped_type() ) ).first;
					}
					return *fi;
				}
		
			public:
				insertion insert( const value_type& value )
				{
					report( "Inserting " << value.first << "," 
						<< value.second );
					iterator fi = lower_bound( value.first );
					if( fi != end() )
					{
						if( !_compare( value, *fi ) )
						{
							report( " Not inserting duplicate key " 
								<< (*fi).first 
								<< ", current value is " << (*fi).second );
							return insertion( fi, false );
						}
					}
					return insertion( _convert(_dv::vectorInsert( _vector, 
						_convert( fi ), value ) ), true );
				}
			
				iterator insert( iterator fi, const value_type& value )
				{
					if( fi != end() )
					{
						if( !_compare( *fi, value ) )
						{
							if( !_compare( value, *fi ) )
							{
								return fi;
							}
						}
						else
						{
							fi = lower_bound( value.first );
							if( !_compare( value, *fi ) )
							{
								return fi;
							}
						}
					}
					else
					{
						fi = lower_bound( value.first );
						if( fi != end() )
						{					
							if( !_compare( value, *fi ) )
							{
								return fi;
							}
						}
					}
					return _convert( _dv::vectorInsert( _vector, 
						_convert( fi ), value ) );	
				}
			
				template< typename ForwardIterator >
				void insert( ForwardIterator first, ForwardIterator end )
				{
					for( ForwardIterator fi = first; fi != end; ++fi )
					{
						insert( *fi );
					}
				}

				void erase( iterator fi )
				{
					_dv::vectorErase( _vector, _convert( fi ) );
				}
			
				size_type erase( const key_type& x )
				{
					iterator fi = find( x );
					if( fi != end() )
					{
						erase( fi );
						return 1;
					}
					return 0;
				}

				void erase( iterator first, iterator last )
				{
					_dv::vectorErase( _vector, _convert( first ), 
						_convert( last ) );
				}
			
				void swap( Map& map )
				{
					std::swap( map._compare, _compare );
					std::swap( map._vector, _vector );
				}
			
			public:
				iterator find( const key_type& key )
				{
					report( "Searching for key " << key );
					iterator fi = lower_bound( key );
					if( fi != end() )
					{
						if( !_compare( key, *fi ) )
						{
							report( " Found " << (*fi).first << "," 
								<< (*fi).second );
							return fi;
						}
					}
					return end();
				}

				const_iterator find( const key_type& key ) const
				{
					report( "Searching for key " << key );
					const_iterator fi = lower_bound( key );
					if( fi != end() )
					{
						if( !_compare( key, *fi ) )
						{
							report( " Found " << (*fi).first << "," 
								<< (*fi).second );
							return fi;
						}
					}
					return end();
				}

				size_type count( const key_type& key ) const
				{
					return find( key ) != end();
				}

				iterator lower_bound( const key_type& key )
				{
					return std::lower_bound( begin(), end(), key, _compare );
				}

				const_iterator lower_bound( const key_type& key ) const
				{
					return std::lower_bound( begin(), end(), key, _compare );
				}

				iterator upper_bound( const key_type& key )
				{
					return std::upper_bound( begin(), end(), key, _compare );
				}

				const_iterator upper_bound( const key_type& key ) const
				{
					return std::upper_bound( begin(), end(), key, _compare );
				}

				std::pair< iterator, iterator > equal_range( 
					const key_type& key )
				{
					return std::equal_range( begin(), end(), key, _compare );
				}

				std::pair< const_iterator, const_iterator > 
					equal_range( const key_type& key ) const
				{
					return std::equal_range( begin(), end(), key, _compare );
				}
				
			public:
				void reserve( size_type size )
				{
					_dv::reserve( _vector, size );
				}
				
				void resize( size_type size )
				{
					_dv::resize( _vector, size );
				}
			
			public:
				value_compare value_comp() const
				{
					return _compare;
				}
				
				key_compare key_comp() const
				{
					return _compare.compare();
				}
		
		};
		
		template < typename K, typename V, typename C >
		bool operator==(const Map< K, V, C >& x, const Map< K, V, C >& y)
		{
			if( x.size() != y.size() )
			{
				return false;
			}
			return std::equal( x.begin(), x.end(), y.begin() );
		}

		template < typename K, typename V, typename C >
		bool operator< (const Map< K, V, C >& x, const Map< K, V, C >& y)
		{
			return std::lexicographical_compare( x.begin(), x.end(), y.begin(), 
				y.end() );
		}

		template < typename K, typename V, typename C >
		bool operator!=(const Map< K, V, C >& x, const Map< K, V, C >& y)
		{
			return !( x == y );
		}
	
		template < typename K, typename V, typename C >
		bool operator> (const Map< K, V, C >& x, const Map< K, V, C >& y)
		{
			return y < x;
		}
	
		template < typename K, typename V, typename C >
		bool operator>=(const Map< K, V, C >& x, const Map< K, V, C >& y)
		{
			return !( x < y );
		}
	
		template < typename K, typename V, typename C >
		bool operator<=(const Map< K, V, C >& x, const Map< K, V, C >& y )
		{
			return !( x > y );
		}
	
		// specialized algorithms:
		template < typename K, typename V, typename C >
		void swap( Map< K, V, C >& x, Map< K, V, C >& y )
		{
			x.swap( y );
		}
		
		// To stream
		template < typename Key, typename Value, typename Compare >
		std::ostream& operator<<( std::ostream& out, 
			Map< Key, Value, Compare >& map )
		{
			out << "digraph DeviceMap\n{\n";
			out << "\tnode [ shape = record ];\n\n";
			out << "\tnode_" << &map << "[ label = \"";
			for( typename Map< Key, Value, Compare >::const_iterator 
				fi = map.begin(); fi != map.end(); ++fi )
			{
				if( fi != map.begin() )
				{
					out << "| ";
				}
				out << "{ " << (*fi).first << " | " << (*fi).second << " } ";
			}
			out << "\" ];\n";
			out << "}";
			return out;
		}

	}

}

#endif

