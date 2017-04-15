################################################################################
# File:   configure.ac
# Date:   Monday November 1, 2010
# Author: Gregory Diamos <gregory.diamos@gatech.edu>
#
# Brief:  The autoconf source file for the RedFox project.
#
################################################################################

################################################################################
## Package specification
# Revision scheme is MajorFeatureSet.MinorFeatureSet.SVNChangelistNumber
AC_PREREQ([2.65])
AC_INIT(redfox, 0.5.41, gregory.diamos@gatech.edu)
AM_INIT_AUTOMAKE([])
################################################################################

################################################################################
## Check For Programs
AC_PROG_CXX([g++ clang++])
AC_PROG_CC([gcc clang])
AC_PROG_RANLIB
################################################################################

################################################################################
## Check for Libraries
AC_CHECK_LIB([rt], [clock_gettime], [],\
	AC_MSG_ERROR("librt required"))
AC_CHECK_LIB([pthread], [pthread_create], [],\
	AC_MSG_ERROR("libpthread required"))
AC_CHECK_LIB([protobuf], [main], [], \
	AC_MSG_ERROR("libprotobuf required"))
################################################################################

################################################################################
## Check for Headers

################################################################################

################################################################################
## Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T
AC_CHECK_TYPES([ptrdiff_t])
################################################################################

################################################################################
## Checks for library functions.
AC_FUNC_ERROR_AT_LINE
AC_CHECK_FUNCS([floor memset pow select sqrt])
################################################################################

################################################################################
## Output Generation
AC_CONFIG_FILES([Makefile])
AC_CONFIG_HEADERS([configure.h])
AC_OUTPUT
################################################################################


