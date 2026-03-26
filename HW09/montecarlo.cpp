#include "montecarlo.h"
#include <cmath>

int montecarlo(const size_t n, const float *x, const float *y,
               const float radius) {
  int incircle = 0;
  const float r2 = radius * radius;

#ifdef USE_SIMD
#pragma omp parallel for simd reduction(+ : incircle) schedule(static)
#else
#pragma omp parallel for reduction(+ : incircle) schedule(static)
#endif
  for (size_t i = 0; i < n; i++) {
    float dist2 = x[i] * x[i] + y[i] * y[i];
    incircle += (int)(dist2 <= r2);
  }
  return incircle;
}
