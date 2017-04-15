################################################################################
# File:   Makefile.am
# Date:   Monday November 1, 2010
# Author: Gregory Diamos <gregory.diamos@gatech.edu>
#
# Brief:  The autoconf source file for the RedFox project.
#
################################################################################

################################################################################
## Global Parameters
################################################################################

################################################################################
## Programs and Libraries
BUILT_SOURCES  = 
bin_PROGRAMS   = RedFox
check_PROGRAMS = 
lib_LIBRARIES  = libredfox.a
################################################################################

################################################################################
## libredfox
libredfox_a_CXXFLAGS = -Wall -ansi -pedantic -Werror -std=c++0x
libredfox_a_SOURCES  = \
	redfox/nvcc/implementation/RelationalAlgebraCompiler.cpp \
	redfox/nvcc/implementation/RelationalAlgebraKernel.cpp \
	redfox/nvcc/implementation/CudaCompilerInterface.cpp \
	redfox/protocol/implementation/HarmonyIR.pb.cc \
	redfox/protocol/implementation/HarmonyIRPrinter.cpp \
	redfox/protocol/implementation/BloxCompiler.pb.cc \
	redfox/protocol/implementation/RelationalAlgebra.pb.cc \
	hydrazine/implementation/ArgumentParser.cpp \
	hydrazine/implementation/Exception.cpp \
	hydrazine/implementation/debug.cpp \
	hydrazine/implementation/LowLevelTimer.cpp \
	hydrazine/implementation/Timer.cpp \
	hydrazine/implementation/string.cpp
################################################################################

################################################################################
## RedFox
RedFox_CXXFLAGS = -Wall -ansi -pedantic -Werror -std=c++0x
RedFox_SOURCES = redfox/driver/implementation/RedFox.cpp 
RedFox_LDADD   = libredfox.a
RedFox_LDFLAGS = 
################################################################################

################################################################################
## Headers
nobase_include_HEADERS = \
	redfox/nvcc/interface/RelationalAlgebraKernel.h \
	redfox/nvcc/interface/CudaCompilerInterface.h \
	redfox/nvcc/interface/RelationalAlgebraCompiler.h \
	redfox/protocol/interface/RelationalAlgebra.pb.h \
	redfox/protocol/interface/BloxCompiler.pb.h \
	redfox/protocol/interface/HarmonyIR.pb.h \
	redfox/driver/interface/RedFox.h \
	redfox/ra/implementation/Map.cu \
	redfox/ra/implementation/DeviceVectorWrapper.cu \
	redfox/ra/implementation/GpuPrimitives.cu \
	redfox/ra/test/TestIntersectionPerformance.cpp \
	redfox/ra/test/TestMap.cpp \
	redfox/ra/test/TestGpuPrimitives.cpp \
	redfox/ra/test/TestGpuPrimitives.h \
	redfox/ra/test/TestMap.h \
	redfox/ra/test/TestIntersectionPerformance.h \
	redfox/ra/interface/SetUnion.h \
	redfox/ra/interface/DataTypes.h \
	redfox/ra/interface/SetCommon.h \
	redfox/ra/interface/MapOperator.h \
	redfox/ra/interface/GpuPrimitives.h \
	redfox/ra/interface/SetIntersection.h \
	redfox/ra/interface/Project.h \
	redfox/ra/interface/DeviceVectorWrapper.h \
	redfox/ra/interface/Map.h \
	redfox/ra/interface/SetIntersectionWithDecompression.h \
	redfox/ra/interface/DetailedImplementations.h \
	redfox/ra/interface/Comparisons.h \
	redfox/ra/interface/Operators.h \
	redfox/ra/interface/SetDifference.h \
	redfox/ra/interface/Aggregate.h
################################################################################


