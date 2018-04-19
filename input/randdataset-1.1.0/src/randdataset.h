#ifndef RANDDATASET_H
#define RANDDATASET_H

#include <inttypes.h>
#include <iostream>
#include <ctime>


static double sqr(double a);

static void padding_init(void);
static void padding_done(void);

static int stats_vector_count;
static double *stats_sum_x;
static double *stats_sum_x_sqr;
static double *stats_sum_x_prod;

static void stats_init(int dim);
static void stats_enter(int dim, double *x);
static void stats_output(int dim);

static double random_equal(double min, double max);
static double random_peak(double min, double max, int dim);
static double random_normal(double med, double var);

static void output_vector(int dim, double *x);
static int is_vector_ok(int dim, double *x);

static void generate_indep(int count, int dim);
static void generate_corr(int count, int dim);
static void generate_anti(int count, int dim);


void generate_indep_inmem(float *data,uint64_t n, uint64_t d, bool transpose);
void generate_corr_inmem(float *data,uint64_t n, uint64_t d, bool transpose);
void generate_anti_inmem(float *data,uint64_t n, uint64_t d, bool transpose);

//static void generate_corr_inmem(int count, int dim);
//static void generate_anti_inmem(int count, int dim);


static void usage();

#endif
