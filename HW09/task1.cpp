#include "cluster.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

// Must match the padding used in cluster.cpp
static const size_t CACHE_LINE_FLOATS = 16;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " n t\n";
    return 1;
  }

  const size_t n = static_cast<size_t>(std::atoll(argv[1]));
  const size_t t = static_cast<size_t>(std::atoll(argv[2]));

  // Create and fill arr with random floats in [0, n]
  float *arr = new float[n];
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist_range(0.0f, static_cast<float>(n));
  for (size_t i = 0; i < n; i++) {
    arr[i] = dist_range(rng);
  }

  // Sort arr
  std::sort(arr, arr + n);

  // Create centers array of length t
  // centers[i] = (2*i + 1) * n / (2*t)  for i = 0, 1, ..., t-1
  // i.e., n/(2t), 3n/(2t), ..., (2t-1)*n/(2t)
  float *centers = new float[t];
  for (size_t i = 0; i < t; i++) {
    centers[i] = static_cast<float>((2 * i + 1) * n) / static_cast<float>(2 * t);
  }

  // Create dists array with padding to avoid false sharing.
  // Padded size: t * CACHE_LINE_FLOATS floats.
  float *dists = new float[t * CACHE_LINE_FLOATS]();

  // Time the cluster function
  auto start = std::chrono::high_resolution_clock::now();
  cluster(n, t, arr, centers, dists);
  auto end = std::chrono::high_resolution_clock::now();

  // Find maximum distance and its thread (partition) ID
  float max_dist = dists[0];
  size_t max_id = 0;
  for (size_t i = 1; i < t; i++) {
    float d = dists[i * CACHE_LINE_FLOATS];
    if (d > max_dist) {
      max_dist = d;
      max_id = i;
    }
  }

  std::chrono::duration<double, std::milli> elapsed = end - start;

  // Output: max distance, partition ID, time in ms
  std::cout << max_dist << "\n";
  std::cout << max_id << "\n";
  std::cout << elapsed.count() << "\n";

  delete[] arr;
  delete[] centers;
  delete[] dists;
  return 0;
}
