/*!
	\brief TestIntersectionPerformance.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Thursday June 11, 2009
	\brief The source file for the TestIntersectionPerformance class.
*/

#ifndef TEST_INTERSECTION_PERFORMANCE_CPP_INCLUDED
#define TEST_INTERSECTION_PERFORMANCE_CPP_INCLUDED

#include <redfox/ra/test/TestIntersectionPerformance.h>
#include <redfox/ra/interface/GpuPrimitives.h>
#include <redfox/ra/interface/Operators.h>
#include <hydrazine/implementation/ArgumentParser.h>
#include <redfox/ra/interface/Comparisons.h>

namespace test
{
	
	typedef gpu::operators::add< unsigned int > add;
	typedef gpu::comparisons::lt< unsigned int > less;
	
	void TestIntersectionPerformance::_init( HostMap& h )
	{
		for( unsigned int i = 0; i < elements; ++i )
		{
			h.insert( std::make_pair( random() % keyLimit, 
				random() % valueLimit ) );
		}
	}
	
	bool TestIntersectionPerformance::_compare( const HostMap& h , 
		const DeviceMap& d )
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

	bool TestIntersectionPerformance::benchmark()
	{
		status << "Running Intersection Benchmark\n";
		
		hydrazine::Timer timer;
		
		timer.start();
		
		HostMap host;
		HostMap hostLeft;
		HostMap hostRight;
		
		_init( hostLeft );
		_init( hostRight );
		
		timer.stop();
		status << " Host initialization time " << timer.toString() << "\n";
		
		timer.start();
		
		DeviceMap device;
		DeviceMap deviceLeft( hostLeft.begin(), hostLeft.end() );
		DeviceMap deviceRight( hostRight.begin(), hostRight.end() );
		
		timer.stop();
		status << " Device initialization time " << timer.toString() << "\n";

		status << " Running host intersection.\n";
		timer.start();
		gpu::algorithms::set_intersection( host, hostLeft, hostRight, add(), 
			less() );
		timer.stop();
		status << "  Intersection time was " << timer.toString() << ".\n";
		
		status << " Running device intersection.\n";
		timer.start();
		gpu::algorithms::set_intersection( device, deviceLeft, 
			deviceRight, add(), less() );
		timer.stop();
		status << "  Device intersection time was " << timer.toString() 
			<< ".\n";
		
		status << " Intersection Benchmark Passed.\n";

		return true;
		
	}

	bool TestIntersectionPerformance::doTest()
	{
		return benchmark();
	}
	
	TestIntersectionPerformance::TestIntersectionPerformance()
	{
		name = "TestIntersectionPerformance";
		
		description = "A performance test for the GPU primitive set "; 
		description += "intersection operator. 1) Create two random cpu sets, ";
		description += "copy into gpu equivalents, do a set intersection, ";
		description += "copy the results back.  Measure the time of each ";
		description += "operation.";
	}

}

int main( int argc, char** argv )
{
	hydrazine::ArgumentParser parser( argc, argv );
	test::TestIntersectionPerformance test;
	parser.description( test.testDescription() );

	parser.parse( "-s", "--seed", test.seed, 0, 
		"Seed for random tests, 0 implies seed with time." );
	parser.parse( "-v", "--verbose", test.verbose, false, 
		"Print out info after the test." );
	parser.parse( "-e", "--elements", test.elements, 10,
		"The number of elements add to each data structure." );
	parser.parse( "-k", "--key-limit", test.keyLimit, 
		test.elements, "The max value for a randomly generated key." );
	parser.parse( "-v", "--value-limit", test.valueLimit, 1000,
		"The max value for a randomly generated value." );
	parser.parse();
	
	test.test();
	
	return test.passed();
}

#endif

