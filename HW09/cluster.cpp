#include "cluster.h"
#include <cmath>

// To avoid false sharing, we use padding so that each thread's data
// occupies its own cache line (typically 64 bytes = 16 floats).
// Each thread writes to dists[tid * PADDING] instead of dists[tid].
// This ensures no two threads share a cache line.

static const size_t CACHE_LINE_FLOATS = 16; // 64 bytes / 4 bytes per float

void cluster(const size_t n, const size_t t, const float *arr,
             const float *centers, float *dists) {
#pragma omp parallel num_threads(t)
  {
    unsigned int tid = omp_get_thread_num();
    float local_dist = 0.0f;
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; i++) {
      local_dist += std::fabs(arr[i] - centers[tid]);
    }
    // Write once to the shared array, padded to avoid false sharing.
    // The caller allocates dists with size t * CACHE_LINE_FLOATS.
    dists[tid * CACHE_LINE_FLOATS] = local_dist;
  }
}
