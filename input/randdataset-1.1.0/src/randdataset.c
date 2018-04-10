/*
 *	randdataset.c
 *
 *	DESCRIPTION:
 *		Random dataset generator for SKYLINE operator evaluation.
 *
 *	AUTHOR:
 *		based on the work by the authors of [Borzsonyi2001]
 *		modified by Hannes Eder <Hannes@HannesEder.net>
 *		thanks to Donald Kossmann for providing the source code
 *		of the original implementation
 *
 *	REFERENCES:
 *		[Borzsonyi2001] Börzsönyi, S.; Kossmann, D. & Stocker, K.: 
 *		The Skyline Operator, ICDE 2001, 421--432
 */
#include "../config.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#ifdef HAVE_LIBGEN_H
#include <libgen.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "port.h"
#include "randdataset.h"

/*
 * some macros for warnings/errors
 */
const char *progname = NULL;

#define warning(FMT, ...) fprintf(stderr, "%s: warning: " FMT "\n", progname, ##__VA_ARGS__)
#define invalidargs(FMT, ...) do { fprintf(stderr, "%s: error: " FMT "\n", progname, ##__VA_ARGS__); usage(); exit(1); } while (0)
#define fatal(FMT, ...) do { fprintf(stderr, "%s: error: " FMT "\n", progname, ##__VA_ARGS__); exit(1); } while (0)



static int id = 0;
static int opt_id = 0;
static int opt_use_seed = 0;
static int opt_seed = 0;
static int opt_pad = 0;
static int opt_copy = 0;
static int opt_create = 0;

static char    pad_alphabet[26] = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
static size_t	pad_alphabet_len = 0;
static char	   *padding = NULL;


/*
 * sqr
 */
static double
sqr(double a)
{
	return a*a;
}

/*
 * padding_init
 */
static void
padding_init(void)
{
	if (opt_pad)
	{
		size_t	pad_len;
		size_t	i;
		char   *p;

		assert(pad_alphabet);
		pad_alphabet_len = strlen(pad_alphabet);
		assert(pad_alphabet_len > 0);

		pad_len = ((opt_pad+pad_alphabet_len) / pad_alphabet_len + 1) * pad_alphabet_len;

		padding = (char *) malloc(pad_len + 1);
		assert(padding);

		for (p = padding, i = 0; i<pad_len; ++i)
		{
			*p++ = pad_alphabet[i % pad_alphabet_len];
		}
		*p++ = '\0';
	}
}

/*
 * padding_done
 */
static void
padding_done(void)
{
	if (padding)
	{
		free(padding);
		padding = NULL;
	}
}

/*
 * stats_init
 *
 *	inits the statistics, what else ;) 
 */
static void
stats_init(int dim)
{
	int		d;

	/*
	 * We don't explicitly free this buffers, since they are freed when
	 * the program terminates anyway.
	 */
	stats_sum_x = (double *) malloc(dim * sizeof(double));
	stats_sum_x_sqr = (double *) malloc(dim * sizeof(double));
	stats_sum_x_prod = (double *) malloc(dim * dim * sizeof(double));

	stats_vector_count = 0;

	for (d = 0; d < dim; d++)
	{
		int		dd;

		stats_sum_x[d] = 0.0;
		stats_sum_x_sqr[d] = 0.0;
		for (dd = 0; dd < dim; dd++)
			stats_sum_x_prod[d * dim + dd] = 0.0;
	}
}


/*
 * stats_enter
 *
 *	adds the vector x[0..dim-1] into the stats
 */
static void
stats_enter(int dim, double *x)
{
	int		d;
	stats_vector_count++;
	for (d = 0; d < dim; d++)
	{
		int		dd;

		stats_sum_x[d] += x[d];
		stats_sum_x_sqr[d] += x[d] * x[d];
		for (dd = 0; dd < dim; dd++)
			stats_sum_x_prod[d * dim + dd] += x[d] * x[dd];
	}
}

/*
 * stats_output
 *
 *	write the statistics to STDERR
 */
static void
stats_output(int dim)
{
	int		d;

	/*
	 *	mean, var, sd
	 */
	for (d = 0; d < dim; d++)
	{
		double E = stats_sum_x[d] / stats_vector_count;
		double V = stats_sum_x_sqr[d] / stats_vector_count - E * E;
		double s = sqrt(V);
		fprintf(stderr, "-- E[X%d]=%5.2f Var[X%d]=%5.2f s[X%d]=%5.2f\n", d + 1, E, d + 1, V,
			d + 1, s);
	}

	/*
	 * correlation factor matrix
	 */
	fprintf(stderr, "--\n-- correlation factor matrix:\n");
	for (d = 0; d < dim; d++) {
		int		dd;

		fprintf(stderr, "--");
		for (dd = 0; dd < dim; dd++) {
			double cov =
				(stats_sum_x_prod[d * dim + dd] / stats_vector_count) -
				(stats_sum_x[d] / stats_vector_count) * (stats_sum_x[dd] /
				stats_vector_count);
			double cor =
				cov / sqrt(stats_sum_x_sqr[d] / stats_vector_count -
				sqr(stats_sum_x[d] / stats_vector_count)) /
				sqrt(stats_sum_x_sqr[dd] / stats_vector_count -
				sqr(stats_sum_x[dd] / stats_vector_count));
			fprintf(stderr, " %5.2f", cor);
		}
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "--\n-- %d vector(s) generated\n", stats_vector_count);
}

/*
 * random_equal
 *
 *	returns a random value x \in [min,max]	
 */
static double
random_equal(double min, double max)
{
	double x = (double) rand() / RAND_MAX;
	return x * (max - min) + min;
}

/*
 * random_peak
 *
 *	Returns a random value x \in [min,max) as sum of dim equally 
 *	distributed random values.
 */
static double
random_peak(double min, double max, int dim)
{
	int		d;
	double	sum = 0.0;

	for (d = 0; d < dim; d++)
		sum += random_equal(0.1, 0.99);
	sum /= dim;
	return sum * (max - min) + min;
}

/*
 * random_normal
 *
 *	Returns a normally distributed random value x \in (med-var,med+var)
 *	with E[x] = med.
 *
 *	NOTE: This implementation works well if the random values returned by
 *	the underlaying random_equal are sufficiently independent.
 */
static double
random_normal(double med, double var)
{
	return random_peak(med - var, med + var, 12);
}

/*
 * output_vector
 *
 *	Writes the vector x[0..dim-1] to stdout, separated by spaces and with a
 *	trailing linefeed.
 */
static void
output_vector(int dim, double *x)
{
	++id;
	if (opt_id)
	{
		fprintf(stdout, "%d,", id);
	}

	while (dim--)
        //fprintf(stdout, "%.*e%s", DBL_DIG, *(x++), dim ? "," : "");
		fprintf(stdout, "%.16f%s", (*(x++))+0.001, dim ? "," : "");

	if (opt_pad)
	{
		fprintf(stdout, ",'%.*s'", opt_pad, padding+(id-1) % pad_alphabet_len);
	}
	
	fprintf(stdout, "\n");
}

/*
 * is_vector_ok
 *
 *	returns 1 iif all x_i \in [0,1]
 */
static int
is_vector_ok(int dim, double *x)
{
	while (dim--)
	{
		if (*x < 0.0 || *x > 1.0)
			return 0;
		x++;
	}

	return 1;
}

/*
 * generate_indep
 *
 *	Generate count vectors x[0..dim-1] with x_i \in [0,1] independently
 *	and equally distributed and outputs them to STDOUT.
 */
static void
generate_indep(int count, int dim)
{
	double *x = (double *) malloc(sizeof(double) * dim);

	while (count--)
	{
		int		d;

		for (d = 0; d < dim; d++)
			x[d] = random_equal(0, 1);

		output_vector(dim, x);
		stats_enter(dim, x);
	}

	free(x);
}

void generate_indep_inmem(float *data, uint64_t n, uint64_t d, bool transpose){
	if(!transpose){
		for(uint64_t i = 0; i < n;i++){
			for(uint64_t m = 0; m < d;m++){
				data[i*n + m] = random_equal(0, 1);
			}
		}
	}else{
		for(uint64_t i = 0; i < n;i++){
			for(uint64_t m = 0; m < d;m++){
				data[m*n + i] = random_equal(0, 1);
			}
		}
	}
}



/*
 * generate_corr
 *
 *	Generates count vectors x[0..dim-1] with x_i \in [0,1].
 *	The x_i are correlated, i.e. if x is high in one dimension
 *	it is likely that x is high in another.
 */
static void
generate_corr(int count, int dim)
{
	double *x = (double *) malloc(sizeof(double) * dim);

	while (count--)
	{
		do
		{
			int		d;
			double	v = random_peak(0, 1, dim);
			double	l = v <= 0.5 ? v : 1.0 - v;

			for (d = 0; d < dim; d++)
				x[d] = v;

			for (d = 0; d < dim; d++)
			{
				double h = random_normal(0, l);
				x[d] += h;
				x[(d + 1) % dim] -= h;
			}
		} while (!is_vector_ok(dim, x));

		output_vector(dim, x);
		stats_enter(dim, x);
	}

	free(x);
}

void generate_corr_inmem(float *data, uint64_t n, uint64_t d, bool transpose){
	double *x = (double *) malloc(sizeof(double) * d);
	uint64_t count = 0;

	while (count < n)
	{
		do
		{
			double	v = random_peak(0, 1, d);
			double	l = v <= 0.5 ? v : 1.0 - v;

			for (uint64_t m = 0; m < d; m++)
				x[m] = v;

			for (uint64_t m = 0; m < d; m++)
			{
				double h = random_normal(0, l);
				x[m] += h;
				x[(m + 1) % d] -= h;
			}
		} while (!is_vector_ok(d, x));

		if(!transpose){
			for (uint64_t m = 0; m < d; m++){
				data[count*n + m] = x[m];
			}
		}else{
			for (uint64_t m = 0; m < d; m++){
				data[count*m + n] = x[m];
			}
		}
		count++;
	}
	free(x);
}

/*
 * generate_anti
 *
 *	Generates count vectors x[0..dim-1] with x_i in [0,1], such that
 *	if x is high in one dimension it is likely that x is low in another.
 */
static void
generate_anti(int count, int dim)
{
	double *x = (double *) malloc(sizeof(double) * dim);

	while (count--)
	{
		do
		{
			int		d;
			double	v = random_normal(0.5, 0.25);
			double	l = v <= 0.5 ? v : 1.0 - v;
			
			for (d = 0; d < dim; d++)
				x[d] = v;
		
			for (d = 0; d < dim; d++)
			{
				double h = random_equal(-l, l);
				x[d] += h;
				x[(d + 1) % dim] -= h;
			}
		} while (!is_vector_ok(dim, x));

		output_vector(dim, x);
		stats_enter(dim, x);
	}
}

void generate_anti_inmem(float *data,uint64_t n, uint64_t d, bool transpose){
	double *x = (double *) malloc(sizeof(double) * d);
	uint64_t count = 0;

	while (count < n)
	{
		do
		{
			double	v = random_normal(0.5, 0.25);
			double	l = v <= 0.5 ? v : 1.0 - v;

			for (uint64_t m = 0; m < d; m++)
				x[m] = v;

			for (uint64_t m = 0; m < d; m++)
			{
				double h = random_equal(-l, l);
				x[m] += h;
				x[(m + 1) % d] -= h;
			}
		} while (!is_vector_ok(d, x));

		if(!transpose){
			for (uint64_t m = 0; m < d; m++){
				data[count*n + m] = x[m];
			}
		}else{
			for (uint64_t m = 0; m < d; m++){
				data[count*m + n] = x[m];
			}
		}
		count++;
	}
}

/*
 * usage
 */
static void
usage()
{
	fprintf(stderr, 
"\
Test Data Generator for Skyline Operator Evaluation\n\
usage: %s (-i|-c|-a) -d DIM -n COUNT [-s SEED] [-p] [-S] [-h|-?]\n\
\n\
Options:\n\
       -i       independent (dim >= 1)\n\
       -c       correlated (dim >= 2)\n\
       -a       anti-correlated (dim >= 2)\n\
\n\
       -d DIM   dimensions >=1\n\
       -n COUNT number of vectors\n\
       -I       unique id for every vector\n\
       -p PAD   add a padding field, PAD characters long\n\
\n\
       -C       generate SQL COPY statement\n\
       -R       generate SQL CREATE TABLE statement\n\
       -T NAME  use NAME instead of default table name\n\
\n\
       -s SEED  set random generator seed to SEED\n\
\n\
       -S       output stats to stderr\n\
\n\
       -h -?    display this help message and exit\n\
\n\
Examples:\n\
       %s -i -d 3 -n 10 -I -R\n\
       %s -a -d 2 -n 100 -S\n\
\n\
", progname, progname, progname);
}
