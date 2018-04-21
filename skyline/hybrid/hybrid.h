/**
 * hybrid.h
 *
 * @date Feb 12, 2014
 * @author Sean Chester (schester)
 */

#ifndef HYBRID_H_
#define HYBRID_H_

#include <cstdio>
#include <inttypes.h>
#include "../common/common.h"
#include "../common/skyline_i.h"

using namespace std;

class Hybrid: public SkylineI {
public:
  Hybrid(uint64_t threads, uint64_t tuples, uint64_t dims,
      const uint64_t accum, const uint64_t q_size );
  virtual ~Hybrid();

  vector<uint64_t> Execute();
  void Init(float** data);

  void printPartitionSizes() {
    printf( "Created %lu non-empty partitions:\n", part_map_.size() );
    for (uint64_t i = 1; i < part_map_.size(); i++) {
      printf( "%" PRIu64 "\n", part_map_.at( i ).second - part_map_.at( i - 1 ).second );
    }
  }

private:
  uint64_t skyline();
  void inline partition();
  void inline compare_to_skyline_points( EPTUPLE &t );
  void inline compare_to_peers( const uint64_t i, const uint64_t start );
  void inline update_partition_map( const uint64_t start, const uint64_t end );

  // Data members:
  const uint64_t num_threads_; /**< Number of threads with which to execute */
  uint64_t n_; /**< Number of input tuples remaining */
  const uint64_t accum_; /**< Size of alpha block of points to concurrently process */
  const uint64_t pq_size_; /**< Number of points to use for each thread in the pre-filter */

  EPTUPLE* data_; /**< Array of input data points */
  vector<uint64_t> skyline_; /**< Vector in which the skyline result will be copied */
  vector<pair<uint64_t, uint64_t> > part_map_; /**< Data structure used in Phase I computation */

};

#endif /* HYBRID_H_ */
