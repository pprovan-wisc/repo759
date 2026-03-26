#include "montecarlo.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <random>

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " n t\n";
    return 1;
  }

  const size_t n = static_cast<size_t>(std::atoll(argv[1]));
  const int t    = std::atoi(argv[2]);
  const float radius = 1.0f;

  omp_set_num_threads(t);

  // Create and fill x and y arrays with random floats in [-radius, radius]
  float *x = new float[n];
  float *y = new float[n];

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist_range(-radius, radius);
  for (size_t i = 0; i < n; i++) {
    x[i] = dist_range(rng);
    y[i] = dist_range(rng);
  }

  // Time the montecarlo function
  auto start = std::chrono::high_resolution_clock::now();
  int incircle = montecarlo(n, x, y, radius);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;

  // Estimate pi: pi ~ 4 * incircle / n
  double pi_est = 4.0 * static_cast<double>(incircle) / static_cast<double>(n);

  std::cout << pi_est << "\n";
  std::cout << elapsed.count() << "\n";

  delete[] x;
  delete[] y;
  return 0;
}
