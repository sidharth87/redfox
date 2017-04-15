/*!
	\file TestGpuPrimitives.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Friday June 5, 2009
	\brief The header file for the TestGpuPrimitives class.
*/

#ifndef TEST_GPU_PRIMITIVES_H_INCLUDED
#define TEST_GPU_PRIMITIVES_H_INCLUDED

#include <hydrazine/interface/Test.h>
#include <redfox/ra/interface/Map.h>
#include <lb/host/interface/Map.h>

namespace test
{

	/*!
		\brief A test for the logicblox gpu primitives for implementing datalog.
		
		Test Points:
			1) Create two random arrays in host and gpu data structures, compute
				the intersections and make sure that they match.
				
			2) Create two random arrays in host and gpu data structures, compute
				the unions and make sure that they match.
				
			3) Create two random arrays in host and gpu data structures, compute
				the differences and make sure that they match.
			
			4) Create a random array in host and gpu data structures, compute
				the aggregates and make sure that they match.
				
			5) Create a random array in host and gpu data structures, select
				using a random predicate, test both version to make sure they 
				match.
				
			6) Create two random arrays in host and gpu data structures, map
				them together and make sure that they match.
	*/
	class TestGpuPrimitives : public Test
	{
		private:
			typedef host::types::Map< unsigned int, unsigned int > HostMap;
			typedef gpu::types::Map< unsigned int, unsigned int > DeviceMap;
	
		private:
			void _init( HostMap& );
			bool _compare( const HostMap&, const DeviceMap& );
	
		private:
			bool testSetIntersection();
			bool testSetUnion();
			bool testSetDifference();
			bool testAggregation();
			bool testSelect();
			bool testMap();
			bool doTest();
	
		public:
			unsigned int elements;
			unsigned int keyLimit;
			unsigned int valueLimit;
				
		public:
			TestGpuPrimitives();
	
	};

}

int main( int argc, char** argv );

#endif

