/*!
	\file TestGpuPrimitives.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Friday June 5, 2009
	\brief The source file for the TestGpuPrimitives class.
*/

#ifndef TEST_GPU_PRIMITIVES_CPP_INCLUDED
#define TEST_GPU_PRIMITIVES_CPP_INCLUDED

#include <redfox/ra/test/TestGpuPrimitives.h>
#include <redfox/ra/interface/GpuPrimitives.h>
#include <hydrazine/implementation/ArgumentParser.h>
#include <hydrazine/implementation/Exception.h>
#include <redfox/ra/interface/Operators.h>
#include <redfox/ra/interface/Comparisons.h>

namespace test
{

	typedef gpu::operators::add< unsigned int > add;
	typedef gpu::comparisons::lt< unsigned int > less;
	typedef std::pair< unsigned int, unsigned int > Tuple;
		
	void TestGpuPrimitives::_init( HostMap& h )
	{
		for( unsigned int i = 0; i < elements; ++i )
		{
			h.insert( std::make_pair( random() % keyLimit, 
				random() % valueLimit ) );
		}
	}
	
	bool TestGpuPrimitives::_compare( const HostMap& h , const DeviceMap& d )
	{
		if( h.size() != d.size() )
		{
			status << " Gpu map size " << d.size() 
				<< " does not match host size " << h.size() << "\n";
			return false;
		}

		size_t count = 0;
		HostMap::const_iterator hi = h.begin();
		DeviceMap::const_iterator di = d.begin();
		
		for( ; di != d.end() && hi != h.end(); ++hi, ++di, ++count )
		{
			if( (*di).first != hi->first )
			{
				status << " at element " << count << " gpu key " << (*di).first
					<< " does not match host key " << (*hi).first << "\n";
				return false;
			}
			
			if( (*di).second != hi->second )
			{
				status << " at element " << count << " gpu value " 
					<< (*di).second << " does not match host value " 
					<< (*hi).second << "\n";
				return false;
			}			
		}
	
		return true;
	}

	bool TestGpuPrimitives::testSetIntersection()
	{
		status << "Set Intersection Test\n";
			
		HostMap hostLeft;
		HostMap hostRight;
		HostMap hostResult;
		
		_init( hostLeft );
		_init( hostRight );
		
		DeviceMap deviceLeft( hostLeft.begin(), hostLeft.end() );
		DeviceMap deviceRight( hostRight.begin(), hostRight.end() );
		DeviceMap deviceResult;
		
		gpu::algorithms::set_intersection( hostResult, 
			hostRight, hostLeft, add(), less() );
		gpu::algorithms::set_intersection( deviceResult, 
			deviceRight, deviceLeft, add(), less() );
		
		if( !_compare( hostResult, deviceResult ) )
		{
			status << " Set Intersection Test Failed\n";
			return false;
		}
		
		status << " Set Intersection Test Passed\n";
		return true;	
	}
	
	bool TestGpuPrimitives::testSetUnion()
	{
		status << "Set Union Test\n";
			
		HostMap hostLeft;
		HostMap hostRight;
		HostMap hostResult;
		
		_init( hostLeft );
		_init( hostRight );
		
		DeviceMap deviceLeft( hostLeft.begin(), hostLeft.end() );
		DeviceMap deviceRight( hostRight.begin(), hostRight.end() );
		DeviceMap deviceResult;
		
		gpu::algorithms::set_union( hostResult, 
			hostRight, hostLeft, add(), less() );
		gpu::algorithms::set_union( deviceResult, 
			deviceRight, deviceLeft, add(), less() );
		
		if( !_compare( hostResult, deviceResult ) )
		{
			status << " Set Union Test Failed\n";
			return false;
		}
		
		status << " Set Union Test Passed\n";
		return true;
	}
	
	bool TestGpuPrimitives::testSetDifference()
	{
		status << "Set Difference Test\n";
			
		HostMap hostLeft;
		HostMap hostRight;
		HostMap hostResult;
		
		_init( hostLeft );
		_init( hostRight );
		
		DeviceMap deviceLeft( hostLeft.begin(), hostLeft.end() );
		DeviceMap deviceRight( hostRight.begin(), hostRight.end() );
		DeviceMap deviceResult;
		
		gpu::algorithms::set_difference( hostResult, 
			hostRight, hostLeft, add(), less() );
		gpu::algorithms::set_difference( deviceResult, 
			deviceRight, deviceLeft, add(), less() );
		
		if( !_compare( hostResult, deviceResult ) )
		{
			status << " Set Difference Test Failed\n";
			return false;
		}
		
		status << " Set Difference Test Passed\n";
		return true;
	}
	
	bool TestGpuPrimitives::testAggregation()
	{
		status << "Aggregation Test\n";
			
		HostMap host;
		HostMap hostResult;
		
		_init( host );
		
		DeviceMap device( host.begin(), host.end() );
		DeviceMap deviceResult;
		
		gpu::algorithms::aggregate( hostResult, host, add(), 
			less() );
		gpu::algorithms::aggregate( deviceResult, device, add(), 
			less() );
		
		if( !_compare( hostResult, deviceResult ) )
		{
			status << " Aggregation Test Failed\n";
			return false;
		}
		
		status << " Aggregation Test Passed\n";
		return true;
	}

	bool TestGpuPrimitives::testSelect()
	{
		status << "Selection Test\n";
			
		HostMap host;
		HostMap hostResult;
		
		_init( host );
		
		DeviceMap device( host.begin(), host.end() );
		DeviceMap deviceResult;
		
		gpu::comparisons::unary< less > unary;
		unary.value = random() % keyLimit;
		
		gpu::algorithms::select( hostResult, host, unary );
		gpu::algorithms::select( deviceResult, device, unary );
		
		if( !_compare( hostResult, deviceResult ) )
		{
			status << " Selection Test Failed\n";
			return false;
		}
		
		status << " Selection Test Passed\n";
		return true;
	}
	
	bool TestGpuPrimitives::testMap()
	{
		status << "Map Test\n";
			
		HostMap hostLeft;
		HostMap hostRight;
		HostMap hostResult;
		
		_init( hostLeft );
		_init( hostRight );
		
		DeviceMap deviceLeft( hostLeft.begin(), hostLeft.end() );
		DeviceMap deviceRight( hostRight.begin(), hostRight.end() );
		DeviceMap deviceResult;
		
		gpu::algorithms::map( hostResult, hostRight, hostLeft, less() );
		gpu::algorithms::map( deviceResult, deviceRight, deviceLeft, less() );
		
		if( !_compare( hostResult, deviceResult ) )
		{
			status << " Map Test Failed\n";
			return false;
		}
		
		status << " Map Test Passed\n";
		return true;
	}

		
	bool TestGpuPrimitives::doTest()
	{
		try
		{
			return testSetIntersection() && testSetUnion() 
				&& testSetDifference() && testAggregation() && testSelect() 
				&& testMap();
		}
		catch( hydrazine::Exception& e )
		{
			status << "Test threw an exception: " << e.message << "\n";
			return false;
		}
	}

	TestGpuPrimitives::TestGpuPrimitives()
	{
		name = "TestGpuPrimitives";
		description = "A test for the logicblox gpu primitives for ";
		description += "implementing datalog. Test Points: 1) Create two ";
		description += "random arrays in host and gpu data structures, ";
		description += "compute the intersections and make sure that they ";
		description += "match. 2) Create two random arrays in host and gpu ";
		description += "data structures, compute the unions and make sure ";
		description += "that they match. 3) Create two random arrays in host";
		description += " and gpu data structures, compute the differences ";
		description += "and make sure that they match. 4) Create a random ";
		description += "array in host and gpu data structures, compute the ";
		description += "aggregates and make sure that they match. 5) Create";
		description += " a random array in host and gpu data structures, ";
		description += "select using a random predicate, test both version";
		description += " to make sure they match. 6) Create two random arrays";
		description += " in host and gpu data structures, map them together";
		description += " and make sure that they match.";
	}
}

int main( int argc, char** argv )
{
	hydrazine::ArgumentParser parser( argc, argv );
	test::TestGpuPrimitives test;
	parser.description( test.testDescription() );

	parser.parse( "-s", "--seed", test.seed, 0, 
		"Seed for random tests, 0 implies seed with time." );
	parser.parse( "-v", "--verbose", test.verbose, false, 
		"Print out info after the test." );
	parser.parse( "-e", "--elements", test.elements, 10,
		"The number of elements add to each data structure." );
	parser.parse( "-k", "--key-limit", test.keyLimit, 
		test.elements, "The max value for a randomly generated key." );
	parser.parse( "-a", "--value-limit", test.valueLimit, 1000,
		"The max value for a randomly generated value." );
	parser.parse();
	
	test.test();
	
	return test.passed();
}

#endif

