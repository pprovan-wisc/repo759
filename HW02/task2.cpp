#include <iostream>
#include <chrono>
#include <random>
#include "convolution.h"

int main(int argc, char* argv[]) {
    if (argc < 3) return 1;

    int n = std::stoi(argv[1]); // [cite: 50]
    int m = std::stoi(argv[2]); // [cite: 52]

    // Allocate memory 
    float* image = new float[n * n];
    float* mask = new float[m * m];
    float* output = new float[n * n];

    // Random number generation 
    std::mt19937 gen(1337); 
    std::uniform_real_distribution<float> dist_img(-10.0, 10.0);
    std::uniform_real_distribution<float> dist_mask(-1.0, 1.0);

    for (int i = 0; i < n * n; i++) image[i] = dist_img(gen);
    for (int i = 0; i < m * m; i++) mask[i] = dist_mask(gen);

    // Timing the convolution [cite: 54]
    auto start = std::chrono::high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;

    // Output results [cite: 54, 55, 56, 62]
    std::cout << duration.count() << std::endl;
    std::cout << output[0] << std::endl;
    std::cout << output[n * n - 1] << std::endl;

    // Cleanup [cite: 57]
    delete[] image;
    delete[] mask;
    delete[] output;

    return 0;
}
