AC_INIT([randdataset],[1.1.0],[Hannes@HannesEder.net])
AC_PREREQ([2.60])
AC_CONFIG_SRCDIR([src/randdataset.c])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_PROG_CC
AC_C_CONST
AC_HEADER_STDC
AC_CHECK_HEADERS([float.h])
AC_CHECK_HEADERS([libgen.h])
AC_CHECK_HEADERS([stdlib.h])
AC_CHECK_HEADERS([string.h])
AC_CHECK_HEADERS([unistd.h])
AC_FUNC_MALLOC
AC_CHECK_FUNCS([basename])
AC_CHECK_FUNCS([getopt])
AC_CHECK_FUNCS([snprintf])
AC_CHECK_FUNCS([sqrt])
AC_CHECK_LIB(m,sqrt)
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile src/Makefile])
AC_OUTPUT
