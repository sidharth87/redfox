/*!
	\file TestMap.cpp
	\date Wednesday May 27, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the TestMap class.
*/

#ifndef TEST_MAP_CPP_INCLUDED
#define TEST_MAP_CPP_INCLUDED

#include <redfox/ra/test/TestMap.h>
#include <hydrazine/implementation/Exception.h>
#include <hydrazine/implementation/ArgumentParser.h>
#include <fstream>

namespace test
{

	void TestMap::_init( Vector& v )
	{
		for( Vector::iterator fi = v.begin(); fi != v.end(); ++fi )
		{
			*fi = random() % elements;
		}
	}
	
	static void dumpMap( TestMap::LocalMap& localMap, const std::string& path, 
		const std::string& message = "failed" )
	{
		std::stringstream stream;
		stream << path << "/map_" + message + ".dot";
		std::ofstream out( stream.str().c_str() );
		if( !out.is_open() )
		{
			throw hydrazine::Exception( 
				"Could not open file " + stream.str() );
		}
		out << localMap;
	}
	
	static void dumpMap( TestMap::Map& localMap, const std::string& path, 
		const std::string& message = "reference" )
	{
		std::stringstream stream;
		stream << path << "/map_" + message + ".dot";
		std::ofstream out( stream.str().c_str() );
		if( !out.is_open() )
		{
			throw hydrazine::Exception( 
				"Could not open file " + stream.str() );
		}
		out << localMap;
	}

	bool TestMap::testRandom()
	{
		typedef std::pair< Map::iterator, bool > MapInsertion;
		typedef std::pair< LocalMap::iterator, bool > LocalMapInsertion;

		status << "Running Test Random\n";

		Map map;
		LocalMap localMap;
		Vector vector( elements );
		_init( vector );
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 3 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					MapInsertion mapInsertion = map.insert( 
						std::make_pair( vector[ index ], i ) );
					LocalMapInsertion localMapInsertion = localMap.insert( 
						std::make_pair( vector[ index ], i ) );
					if( mapInsertion.second != localMapInsertion.second )
					{
						status << "At index " << i << std::boolalpha 
							<< " map insertion " 
							<< mapInsertion.second 
							<< " did not match localMap insertion " 
							<< localMapInsertion.second << "\n";
						dumpMap( localMap, path );
						return false;
					}
					
					if( mapInsertion.first->first 
						!= (*localMapInsertion.first).first )
					{
						status << "At index " << i << " map key " 
							<< mapInsertion.first->first 
							<< " did not match localMap key " 
							<< (*localMapInsertion.first).first << "\n";
						dumpMap( localMap, path );
						return false;
					}

					if( mapInsertion.first->second 
						!= (*localMapInsertion.first).second )
					{
						status << "At index " << i << " map value " 
							<< mapInsertion.first->second 
							<< " did not match localMap value " 
							<< (*localMapInsertion.first).second << "\n";
						dumpMap( localMap, path );
						return false;
					}
					
					break;
				}
				
				case 1:
				{
					size_t index = random() % vector.size();
					Map::const_iterator mi = map.find( vector[ index ] );
					LocalMap::const_iterator ti = 
						localMap.find( vector[ index ] );
					bool mapEnd = mi == map.end();
					bool localMapEnd = ti == localMap.end();
										
					if( mapEnd || localMapEnd )
					{
						if( mapEnd != localMapEnd )
						{
							status << "At index " << i << std::boolalpha 
								<< " for key " << vector[ index ] << " map end "
								<< mapEnd << " did not match localMap end " 
								<< localMapEnd << "\n";
							dumpMap( localMap, path );
							return false;						
						}
					}
					else
					{
						if( mi->first != (*ti).first )
						{
							status << "At index " << i << " map key " 
								<< mi->first << " did not match localMap key " 
								<< (*ti).first << "\n";
							dumpMap( localMap, path );
							return false;
						}

						if( mi->second != (*ti).second )
						{
							status << "At index " << i << " map value " 
								<< mi->second << " did not match localMap value " 
								<< (*ti).second << "\n";
							dumpMap( localMap, path );
							return false;
						}
					}
					break;
				}
				
				case 2:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					LocalMap::iterator ti = localMap.find( vector[ index ] );
					bool mapEnd = mi == map.end();
					bool localMapEnd = ti == localMap.end();
										
					if( mapEnd || localMapEnd )
					{
						if( mapEnd != localMapEnd )
						{
							status << "At index " << i << std::boolalpha 
								<< " for key " << vector[ index ] << " map end "
								<< mapEnd << " did not match localMap end " 
								<< localMapEnd << "\n";
							dumpMap( localMap, path );
							return false;	
						}
					}
					else
					{
						map.erase( mi );
						localMap.erase( ti );
					}
					break;
				}
			}
		}
		
		status << " Test Random Passed\n";
		return true;
	}

	bool TestMap::testClear()
	{
		status << "Running Test Clear\n";

		Map map;
		LocalMap localMap;
		Vector vector( elements );
		_init( vector );
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map.insert( std::make_pair( vector[ index ], i ) );
					localMap.insert( std::make_pair( vector[ index ], i ) );
					
					if( map.size() != localMap.size() )
					{
						status << "Map size " << map.size() 
							<< " does not match localMap size " 
							<< localMap.size() << "\n";
						dumpMap( localMap, path );
						return false;
					}
					
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					LocalMap::iterator ti = localMap.find( vector[ index ] );
					
					if( mi != map.end() )
					{
						map.erase( mi );
						localMap.erase( ti );
						
						if( map.size() != localMap.size() )
						{
							status << "Map size " << map.size() 
								<< " does not match localMap size " 
								<< localMap.size() << "\n";
							dumpMap( localMap, path );
							return false;
						}
					}
					break;
				}
			}
		}
		
		map.clear();
		localMap.clear();
		
		if( map.size() != localMap.size() )
		{
			status << "Ater clear, map size " << map.size() 
				<< " does not match localMap size " 
				<< localMap.size() << "\n";
			dumpMap( localMap, path );
			return false;
		}
		
		if( !localMap.empty() )
		{
			status << "Ater clear, localMap reported not empty\n";
			dumpMap( localMap, path );
			return false;
		}
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map.insert( std::make_pair( vector[ index ], i ) );
					localMap.insert( std::make_pair( vector[ index ], i ) );
					
					if( map.size() != localMap.size() )
					{
						status << "Map size " << map.size() 
							<< " does not match localMap size " 
							<< localMap.size() << "\n";
						dumpMap( localMap, path );
						return false;
					}
					
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					LocalMap::iterator ti = localMap.find( vector[ index ] );
					
					if( mi != map.end() )
					{
						map.erase( mi );
						localMap.erase( ti );
						
						if( map.size() != localMap.size() )
						{
							status << "Map size " << map.size() 
								<< " does not match localMap size " 
								<< localMap.size() << "\n";
							dumpMap( localMap, path );
							return false;
						}
					}
					break;
				}
			}
		}
	
		status << " Test Clear Passed\n";
		return true;
	}
	
	bool TestMap::testIteration()
	{
		status << "Running Test Forward Iteration\n";

		Map map;
		LocalMap localMap;
		Vector vector( elements );
		_init( vector );
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map.insert( std::make_pair( vector[ index ], i ) );
					localMap.insert( std::make_pair( vector[ index ], i ) );					
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					LocalMap::iterator ti = localMap.find( vector[ index ] );
					
					if( mi != map.end() )
					{
						map.erase( mi );
						localMap.erase( ti );
					}
					break;
				}
			}
		}
		
		{
			Map::iterator mi = map.begin();
			LocalMap::iterator ti = localMap.begin();
	
			for( ; mi != map.end() && ti != localMap.end(); ++mi, ++ti )
			{
				if( mi->first != (*ti).first )
				{
					status << "Forward iteration failed, map key " << mi->first
						<< " does not match localMap key " << (*ti).first << "\n";
					dumpMap( localMap, path );
					return false;
				}
				if( mi->second != (*ti).second )
				{
					status << "Forward iteration failed, map value " 
						<< mi->second
						<< " does not match localMap value " << (*ti).second << "\n";
					dumpMap( localMap, path );
					return false;
				}
			}
			
			if( mi != map.end() )
			{
				status << "Forward iteration failed, map did not hit the end\n";
				dumpMap( localMap, path );
				return false;
			}
		
			if( ti != localMap.end() )
			{
				status << "Forward iteration failed, localMap did not hit end\n";
				dumpMap( localMap, path );
				return false;
			}		
		}
		
		{
			Map::const_iterator mi = map.begin();
			LocalMap::const_iterator ti = localMap.begin();
	
			for( ; mi != map.end() && ti != localMap.end(); ++mi, ++ti )
			{
				if( mi->first != (*ti).first )
				{
					status << "Forward const iteration failed, map key " 
						<< mi->first
						<< " does not match localMap key " << (*ti).first << "\n";
					dumpMap( localMap, path );
					return false;
				}
				if( mi->second != (*ti).second )
				{
					status << "Forward const iteration failed, map value " 
						<< mi->second
						<< " does not match localMap value " << (*ti).second << "\n";
					dumpMap( localMap, path );
					return false;
				}
			}
			
			if( mi != map.end() )
			{
				status << "Forward const iteration failed, map did " 
					<< "not hit the end\n";
				dumpMap( localMap, path );
				return false;
			}
		
			if( ti != localMap.end() )
			{
				status << "Forward const iteration failed, localMap" 
					<< " did not hit end\n";
				dumpMap( localMap, path );
				return false;
			}		
		}
		
		{
			Map::reverse_iterator mi = map.rbegin();
			LocalMap::reverse_iterator ti = localMap.rbegin();
	
			for( ; mi != map.rend() && ti != localMap.rend(); ++mi, ++ti )
			{
				if( mi->first != (*ti).first )
				{
					status << "Reverse iteration failed, map key " 
						<< mi->first
						<< " does not match localMap key " << (*ti).first << "\n";
					dumpMap( localMap, path );
					return false;
				}
				if( mi->second != (*ti).second )
				{
					status << "Reverse iteration failed, map value " 
						<< mi->second
						<< " does not match localMap value " << (*ti).second << "\n";
					dumpMap( localMap, path );
					return false;
				}
			}
			
			if( mi != map.rend() )
			{
				status << "Reverse iteration failed, map did " 
					<< "not hit the end\n";
				dumpMap( localMap, path );
				return false;
			}
		
			if( ti != localMap.rend() )
			{
				status << "Reverse iteration failed, localMap" 
					<< " did not hit end\n";
				dumpMap( localMap, path );
				return false;
			}		
		}
		
		{
			Map::const_reverse_iterator mi = map.rbegin();
			LocalMap::const_reverse_iterator ti = localMap.rbegin();
	
			for( ; mi != map.rend() && ti != localMap.rend(); ++mi, ++ti )
			{
				if( mi->first != (*ti).first )
				{
					status << "Reverse const iteration failed, map key " 
						<< mi->first
						<< " does not match localMap key " << (*ti).first << "\n";
					dumpMap( localMap, path );
					return false;
				}
				if( mi->second != (*ti).second )
				{
					status << "Reverse const iteration failed, map value " 
						<< mi->second
						<< " does not match localMap value " << (*ti).second << "\n";
					dumpMap( localMap, path );
					return false;
				}
			}
			
			if( mi != map.rend() )
			{
				status << "Reverse const iteration failed, map did " 
					<< "not hit the end\n";
				dumpMap( localMap, path );
				return false;
			}
		
			if( ti != localMap.rend() )
			{
				status << "Reverse const iteration failed, localMap" 
					<< " did not hit end\n";
				dumpMap( localMap, path );
				return false;
			}		
		}
		
		status << " Test Iteration Passed\n";
		return true;
			
	}
	
	bool TestMap::testComparisons()
	{
		status << "Running Test Comparisons\n";

		Map map, map1;
		LocalMap localMap, localMap1;
		Vector vector( elements );
		_init( vector );
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map.insert( std::make_pair( vector[ index ], i ) );
					localMap.insert( std::make_pair( vector[ index ], i ) );					
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					LocalMap::iterator ti = localMap.find( vector[ index ] );
					
					if( mi != map.end() )
					{
						map.erase( mi );
						localMap.erase( ti );
					}
					break;
				}
			}
		}
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map1.insert( std::make_pair( vector[ index ], i ) );
					localMap1.insert( std::make_pair( vector[ index ], i ) );					
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map1.find( vector[ index ] );
					LocalMap::iterator ti = localMap1.find( vector[ index ] );
					
					if( mi != map1.end() )
					{
						map1.erase( mi );
						localMap1.erase( ti );
					}
					break;
				}
			}
		}
		
		bool mapResult;
		bool localMapResult;
		
		mapResult = map == map1;
		localMapResult = localMap == localMap1;
		
		if( mapResult != localMapResult )
		{
			status << "For comparison ==, map result " << mapResult
				<< " does not match localMap result " << localMapResult << "\n";
			dumpMap( localMap, path, "failed" );
			dumpMap( localMap1, path, "failed1" );
			return false;
		}

		mapResult = map != map1;
		localMapResult = localMap != localMap1;

		if( mapResult != localMapResult )
		{
			status << "For comparison !=, map result " << mapResult
				<< " does not match localMap result " << localMapResult << "\n";
			dumpMap( localMap, path, "failed" );
			dumpMap( localMap1, path, "failed1" );
			return false;
		}

		mapResult = map < map1;
		localMapResult = localMap < localMap1;

		if( mapResult != localMapResult )
		{
			status << "For comparison <, map result " << mapResult
				<< " does not match localMap result " << localMapResult << "\n";
			dumpMap( localMap, path, "failed" );
			dumpMap( localMap1, path, "failed1" );
			dumpMap( map, path, "reference" );
			dumpMap( map1, path, "reference1" );
			return false;
		}

		mapResult = map <= map1;
		localMapResult = localMap <= localMap1;

		if( mapResult != localMapResult )
		{
			status << "For comparison <=, map result " << mapResult
				<< " does not match localMap result " << localMapResult << "\n";
			dumpMap( localMap, path, "failed" );
			dumpMap( localMap1, path, "failed1" );
			return false;
		}

		mapResult = map > map1;
		localMapResult = localMap > localMap1;

		if( mapResult != localMapResult )
		{
			status << "For comparison >, map result " << mapResult
				<< " does not match localMap result " << localMapResult << "\n";
			dumpMap( localMap, path, "failed" );
			dumpMap( localMap1, path, "failed1" );
			return false;
		}

		mapResult = map >= map1;
		localMapResult = localMap >= localMap1;

		if( mapResult != localMapResult )
		{
			status << "For comparison >=, map result " << mapResult
				<< " does not match localMap result " << localMapResult << "\n";
			dumpMap( localMap, path, "failed" );
			dumpMap( localMap1, path, "failed1" );
			return false;
		}
	
		status << " Test Comparison Passed\n";
		return true;
		
	}

	bool TestMap::testSearching()
	{
		status << "Running Test Searching\n";

		Map map, map1;
		LocalMap localMap, localMap1;
		Vector vector( elements );
		_init( vector );
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 3 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map.insert( std::make_pair( vector[ index ], i ) );
					localMap.insert( std::make_pair( vector[ index ], i ) );					
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					LocalMap::iterator ti = localMap.find( vector[ index ] );
					
					if( mi != map.end() )
					{
						map.erase( mi );
						localMap.erase( ti );
					}
					break;
				}
				
				case 2:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					LocalMap::iterator ti = localMap.find( vector[ index ] );
					
					if( mi != map.end() )
					{
						if( ti == localMap.end() )
						{
							status << "Map found key " << mi->first
								<< ", but it was not found in the localMap.\n";
							dumpMap( localMap, path );
							return false;
						}
						if( mi->first != (*ti).first )
						{
							status << "Find failed, map key " << mi->first
								<< " does not match localMap key " << (*ti).first 
								<< "\n";
							dumpMap( localMap, path );
							return false;
						}
						if( mi->second != (*ti).second )
						{
							status << "Find failed, map value " << mi->second
								<< " does not match localMap value " 
								<< (*ti).second << "\n";
							dumpMap( localMap, path );
							return false;
						}
					}
					
					mi = map.lower_bound( vector[ index ] );
					ti = localMap.lower_bound( vector[ index ] );
					
					if( mi != map.end() )
					{
						if( ti == localMap.end() )
						{
							status << "Map lower_bound found key " << mi->first
								<< ", but it was not found in the localMap.\n";
							dumpMap( localMap, path );
							return false;
						}
						if( mi->first != (*ti).first )
						{
							status << "Lower_bound failed, map key " 
								<< mi->first
								<< " does not match localMap key " << (*ti).first 
								<< "\n";
							dumpMap( localMap, path );
							return false;
						}
						if( mi->second != (*ti).second )
						{
							status << "Lower_bound failed, map value " 
								<< mi->second
								<< " does not match localMap value " 
								<< (*ti).second << "\n";
							dumpMap( localMap, path );
							return false;
						}
					}
					
					mi = map.upper_bound( vector[ index ] );
					ti = localMap.upper_bound( vector[ index ] );
					
					if( mi != map.end() )
					{
						if( ti == localMap.end() )
						{
							status << "Map upper_bound found key " << mi->first
								<< ", but it was not found in the localMap.\n";
							dumpMap( localMap, path );
							return false;
						}
						if( mi->first != (*ti).first )
						{
							status << "Upper_bound failed, map key " 
								<< mi->first
								<< " does not match localMap key " << (*ti).first 
								<< "\n";
							dumpMap( localMap, path );
							return false;
						}
						if( mi->second != (*ti).second )
						{
							status << "Upper_bound failed, map value " 
								<< mi->second
								<< " does not match localMap value " 
								<< (*ti).second << "\n";
							dumpMap( localMap, path );
							return false;
						}
					}
					
					std::pair< Map::iterator, Map::iterator > 
						mp = map.equal_range( vector[ index ] );
					std::pair< LocalMap::iterator, LocalMap::iterator > 
						tp = localMap.equal_range( vector[ index ] );
					
					if( mp.first != map.end() )
					{
						if( tp.first == localMap.end() )
						{
							status << "Map equal range lower key " 
								<< mp.first->first
								<< ", but it was not found in the localMap.\n";
							dumpMap( localMap, path );
							return false;
						}
						if( mp.first->first != (*tp.first).first )
						{
							status << "Equal_range failed, lower map key " 
								<< mp.first->first
								<< " does not match localMap key " 
								<< (*tp.first).first 
								<< "\n";
							dumpMap( localMap, path );
							return false;
						}
						if( mp.first->second != (*tp.first).second )
						{
							status << "Equal_range failed, lower map value " 
								<< mp.first->second
								<< " does not match localMap value " 
								<< (*tp.first).second << "\n";
							dumpMap( localMap, path );
							return false;
						}
					}
					
					if( mp.second != map.end() )
					{
						if( tp.second == localMap.end() )
						{			dumpMap( localMap, path, "failed" );

							status << "Map equal range upper key " 
								<< mp.second->first
								<< ", but it was not found in the localMap.\n";
							dumpMap( localMap, path );
							return false;
						}
						if( mp.second->first != (*tp.second).first )
						{
							status << "Equal_range failed, upper map key " 
								<< mp.second->first
								<< " does not match localMap key " 
								<< (*tp.second).first 
								<< "\n";
							dumpMap( localMap, path );
							return false;
						}
						if( mp.second->second != (*tp.second).second )
						{
							status << "Equal_range failed, upper map value " 
								<< mp.second->second
								<< " does not match localMap value " 
								<< (*tp.second).second << "\n";
							dumpMap( localMap, path );
							return false;
						}
					}
					
					break;
				}
			}
		}
		
		status << " Test Searching Passed.\n";
		return true;
	}

	bool TestMap::testSwap()
	{
		status << "Running Test Swap\n";

		Map map, map1;
		LocalMap localMap, localMap1;
		Vector vector( elements );
		_init( vector );
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map.insert( std::make_pair( vector[ index ], i ) );
					localMap.insert( std::make_pair( vector[ index ], i ) );					
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					LocalMap::iterator ti = localMap.find( vector[ index ] );
					
					if( mi != map.end() )
					{
						map.erase( mi );
						localMap.erase( ti );
					}
					break;
				}
			}
		}
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map1.insert( std::make_pair( vector[ index ], i ) );
					localMap1.insert( std::make_pair( vector[ index ], i ) );					
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map1.find( vector[ index ] );
					LocalMap::iterator ti = localMap1.find( vector[ index ] );
					
					if( mi != map1.end() )
					{
						map1.erase( mi );
						localMap1.erase( ti );
					}
					break;
				}
			}
		}
		
		map.swap( map1 );
		localMap.swap( localMap1 );
		
		{
			Map::iterator mi = map.begin();
			LocalMap::iterator ti = localMap.begin();

			for( ; mi != map.end() && ti != localMap.end(); ++mi, ++ti )
			{
				if( mi->first != (*ti).first )
				{
					status << "Swap failed, map key " << mi->first
						<< " does not match localMap key " << (*ti).first << "\n";
					dumpMap( localMap, path );
					return false;
				}
				if( mi->second != (*ti).second )
				{
					status << "Swap failed, map value " 
						<< mi->second
						<< " does not match localMap value " << (*ti).second 
						<< "\n";
					dumpMap( localMap, path );
					return false;
				}
			}
			
		}
		
		{
			Map::iterator mi = map1.begin();
			LocalMap::iterator ti = localMap1.begin();

			for( ; mi != map1.end() && ti != localMap1.end(); ++mi, ++ti )
			{
				if( mi->first != (*ti).first )
				{
					status << "Swap failed, map key " << mi->first
						<< " does not match localMap key " << (*ti).first << "\n";
					dumpMap( localMap1, path );
					return false;
				}
				if( mi->second != (*ti).second )
				{
					status << "Swap failed, map value " 
						<< mi->second
						<< " does not match localMap value " << (*ti).second 
						<< "\n";
					dumpMap( localMap1, path );
					return false;
				}
			}
		}
		
		status << "  Test Swap Passed.\n";
		return true;		
	}
	
	bool TestMap::testInsert()
	{
		status << "Running Test Insert\n";

		Map map;
		LocalMap localMap;
		Vector vector( elements );
		_init( vector );
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map.insert( std::make_pair( vector[ index ], i ) );
					LocalMap::iterator fi = 
						localMap.lower_bound( vector[ index ] );
					fi = localMap.insert( fi, 
						std::make_pair( vector[ index ], i ) );
					if( (*fi).first != vector[ index ] )
					{
						status << "Insert failed, returned iterator with key " 
						<< (*fi).first << " does not match inserted value " 
						<< vector[ index ] << "\n";
						dumpMap( localMap, path );
						return false;
					}		
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					LocalMap::iterator ti = localMap.find( vector[ index ] );
					
					if( mi != map.end() )
					{
						map.erase( mi );
						localMap.erase( ti );
					}
					break;
				}
			}
		}
		
		localMap.clear();
		localMap.insert( map.begin(), map.end() );
		
		{
			Map::iterator mi = map.begin();
			LocalMap::iterator ti = localMap.begin();

			for( ; mi != map.end() && ti != localMap.end(); ++mi, ++ti )
			{
				if( mi->first != (*ti).first )
				{
					status << "Insert failed, map key " << mi->first
						<< " does not match localMap key " << (*ti).first << "\n";
					dumpMap( localMap, path );
					return false;
				}
				if( mi->second != (*ti).second )
				{
					status << "Insert failed, map value " 
						<< mi->second
						<< " does not match localMap value " << (*ti).second 
						<< "\n";
					dumpMap( localMap, path );
					return false;
				}
			}
		}
		
		status << "  Test Insert Passed.\n";
		return true;
	}

	bool TestMap::testErase()
	{
		status << "Running Test Insert\n";

		Map map;
		LocalMap localMap;
		Vector vector( elements );
		_init( vector );
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map.insert( std::make_pair( vector[ index ], i ) );
					LocalMap::iterator fi = 
						localMap.lower_bound( vector[ index ] );
					fi = localMap.insert( fi, 
						std::make_pair( vector[ index ], i ) );
					if( (*fi).first != vector[ index ] )
					{
						status << "Insert failed, returned iterator with key " 
						<< (*fi).first << " does not match inserted value " 
						<< vector[ index ] << "\n";
						dumpMap( localMap, path );
						return false;
					}		
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					size_t index1 = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					Map::iterator mi1 = map.find( vector[ index1 ] );
					LocalMap::iterator ti = localMap.find( vector[ index ] );
					LocalMap::iterator ti1 = localMap.find( vector[ index1 ] );
					
					if( mi == map.end() && mi1 == map.end() )
					{
						break;
					}
					
					if( mi == map.end() && mi1 != map.end() )
					{
						std::swap( mi, mi1 );
						std::swap( ti, ti1 );
					}
					else if( mi != map.end() && mi1 != map.end() )
					{
						if( mi1->first < mi->first )
						{
							std::swap( mi, mi1 );
							std::swap( ti, ti1 );
						}
					}
					
					map.erase( mi, mi1 );
					localMap.erase( ti, ti1 );
					break;
				}
			}
		}
			
		{
			Map::iterator mi = map.begin();
			LocalMap::iterator ti = localMap.begin();

			for( ; mi != map.end() && ti != localMap.end(); ++mi, ++ti )
			{
				if( mi->first != (*ti).first )
				{
					status << "Erase failed, map key " << mi->first
						<< " does not match localMap key " << (*ti).first 
						<< "\n";
					dumpMap( localMap, path );
					return false;
				}
				if( mi->second != (*ti).second )
				{
					status << "Erase failed, map value " 
						<< mi->second
						<< " does not match localMap value " << (*ti).second 
						<< "\n";
					dumpMap( localMap, path );
					return false;
				}
			}
		}
		
		status << "  Test Erase Passed.\n";
		return true;
	}

	bool TestMap::testCopy()
	{
		status << "Running Test Copy\n";

		Map map;
		LocalMap localMap;
		Vector vector( elements );
		_init( vector );
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map.insert( std::make_pair( vector[ index ], i ) );
					localMap.insert( std::make_pair( vector[ index ], i ) );						
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					LocalMap::iterator ti = localMap.find( vector[ index ] );
					
					if( mi != map.end() )
					{
						map.erase( mi );
						localMap.erase( ti );
					}
					break;
				}
			}
		}
		
		LocalMap copy( localMap );
		
		{
			Map::iterator mi = map.begin();
			LocalMap::iterator ti = copy.begin();

			for( ; mi != map.end() && ti != copy.end(); ++mi, ++ti )
			{
				if( mi->first != (*ti).first )
				{
					status << "Copy failed, map key " << mi->first
						<< " does not match localMap key " << (*ti).first 
						<< "\n";
					dumpMap( copy, path );
					return false;
				}
				if( mi->second != (*ti).second )
				{
					status << "Copy failed, map value " 
						<< mi->second
						<< " does not match localMap value " << (*ti).second 
						<< "\n";
					dumpMap( copy, path );
					return false;
				}
			}
		}
		
		copy.clear();
		copy = localMap;

		{
			Map::iterator mi = map.begin();
			LocalMap::iterator ti = copy.begin();

			for( ; mi != map.end() && ti != copy.end(); ++mi, ++ti )
			{
				if( mi->first != (*ti).first )
				{
					status << "Assign failed, map key " << mi->first
						<< " does not match localMap key " << (*ti).first << "\n";
					dumpMap( copy, path );
					return false;
				}
				if( mi->second != (*ti).second )
				{
					status << "Assign failed, map value " 
						<< mi->second
						<< " does not match localMap value " << (*ti).second << "\n";
					dumpMap( copy, path );
					return false;
				}
			}
		}
	
		status << "  Test Copy Passed.\n";
		return true;
	}
	
	void TestMap::doBenchmark()
	{
		LocalMap localMap;
		Vector vector( elements );
		_init( vector );
				
		for( Vector::iterator fi = vector.begin(); fi != vector.end(); ++fi )
		{
			std::pair< LocalMap::iterator, bool > insertion 
				= localMap.insert( std::make_pair( *fi, fi - vector.begin() ) );
			if( insertion.second )
			{
				std::stringstream stream;
				stream << (fi - vector.begin());
				dumpMap( localMap, path, stream.str() );
			}
		}
		
		for( Vector::iterator fi = vector.begin(); fi != vector.end(); ++fi )
		{
			LocalMap::iterator ti = localMap.find( *fi );
			if( ti != localMap.end() )
			{
				localMap.erase( ti );
				std::stringstream stream;
				stream << (fi - vector.begin() + elements);
				dumpMap( localMap, path, stream.str() );
			}
		}	
	}

	bool TestMap::doTest()
	{
		if( benchmark )
		{
			doBenchmark();
			return true;
		}
		else
		{
			return testRandom() && testClear() && testIteration() 
				&& testComparisons() && testSearching() && testSwap() 
				&& testInsert() && testErase() && testCopy();
		}
	}

	TestMap::TestMap()
	{
		name = "TestMap";
		description = "A unit test for the LocalMap mapping data structure. ";
		description += "Test Points: 1) Randomly insert and remove elements ";
		description += "from a std::map and a Map assert that the final ";
		description += "versions have the exact same elements stored in the ";
		description += "same order. 2) Add elements and then clear the Map. ";
		description += " Assert that there are no elements after the clear and";
		description += " that the correct number is reported by size after ";
		description += "each insertion. 3) Test iteraion through the Map.";
		description += "4) Test each of the comparison ";
		description += "operators. 5) Test searching functions. 6) Test ";
		description += "swapping with another map 7) Test all of the insert ";
		description += "functions. 8) Test all of the erase functions. 9) ";
		description += "Test assignment and copy constructors. 10) Do not ";
		description += "run any tests, simply add a ";
		description += "sequence to the Map and write it out to graph viz ";
		description += "files after each operaton.";
	}

}

int main( int argc, char** argv )
{
	hydrazine::ArgumentParser parser( argc, argv );
	test::TestMap test;
	parser.description( test.testDescription() );

	parser.parse( "-s", "--seed", test.seed, 0, 
		"Seed for random tests, 0 implies seed with time." );
	parser.parse( "-v", "--verbose", test.verbose, false, 
		"Print out info after the test." );
	parser.parse( "-b", "--benchmark", test.benchmark, false,
		"Rather than testing, print out the localMap after several changes." );
	parser.parse( "-e", "--elements", test.elements, 10,
		"The number of elements to add to the localMap." );
	parser.parse( "-i", "--iterations", test.iterations, 1000,
		"The number of iterations to perform tests." );
	parser.parse( "-p", "--path", test.path, "temp",
		"Relative path to store graphviz representations of the localMap." );
	parser.parse();
	
	test.test();
	
	return test.passed();
}

#endif

