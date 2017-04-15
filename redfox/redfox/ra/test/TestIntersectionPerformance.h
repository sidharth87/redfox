/*!
	\brief TestIntersectionPerformance.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Thursday June 11, 2009
	\brief The header file for the TestIntersectionPerformance class.
*/

#ifndef TEST_INTERSECTION_PERFORMANCE_H_INCLUDED
#define TEST_INTERSECTION_PERFORMANCE_H_INCLUDED

#include <hydrazine/interface/Test.h>
#include <lb/gpu/interface/Map.h>
#include <lb/host/interface/Map.h>

namespace test
{
	
	/*!
		\brief A performance test for the GPU primitive set intersection 
			operator
			
		1) Create two random cpu sets, copy into gpu equivalents, do a set 
			intersection, copy the results back.  Measure the time of
			each operation.
	*/
	class TestIntersectionPerformance : public Test
	{
		private:
			typedef host::types::Map< unsigned int, unsigned int > HostMap;
			typedef gpu::types::Map< unsigned int, unsigned int > DeviceMap;
	
		private:
			void _init( HostMap& );
			bool _compare( const HostMap&, const DeviceMap& );
		
		public:
			unsigned int keyLimit;
			unsigned int valueLimit;
			unsigned int elements;
			double speedup;

		private:
			bool benchmark();
		
			bool doTest();
			
		public:
			TestIntersectionPerformance();
			
	};

}

int main( int argc, char** argv );

#endif

