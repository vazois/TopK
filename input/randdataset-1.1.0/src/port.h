/*-------------------------------------------------------------------------
 *
 * port.h
 *	  Header for src/port/ compatibility functions.
 *
 * Portions Copyright (c) 1996-2007, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 * $PostgreSQL: pgsql/src/include/port.h,v 1.113 2007/09/28 22:25:49 tgl Exp $
 *
 *-------------------------------------------------------------------------
 */
#ifndef PG_PORT_H
#define PG_PORT_H

#include "../config.h"

#ifndef HAVE_GETOPT
extern int	getopt(int nargc, char *const * nargv, const char *ostr);
extern char	   *optarg;
extern int		optind;
#endif

#ifndef HAVE_BASENAME
extern char *basename(const char *path);
#endif

#if !defined(HAVE_SNPRINTF) && _WIN32
#define snprintf _snprintf
#endif

#endif   /* PG_PORT_H */
