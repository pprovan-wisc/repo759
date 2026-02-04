#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "scan.h"

int main(int argc, char* argv[]) {
    //Check if the user provided n
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <n>" << std::endl;
        return 1;
    }

    //Convert string to integer
    int n = std::atoi(argv[1]);

    //Allocate memory for input and output arrays
    //We use std::vector for automatic deallocation
    std::vector<float> input(n);
    std::vector<float> output(n);
    
    //Random number generator with  distribution between -1.0 and 1.0
    std::mt19937 gen(42); //Seeded with 42 for consistent results
    std::uniform_real_distribution<float> dis(-1.0, 1.0);

    //Fill the input array with random floats
    for (int i = 0; i < n; ++i) {
        input[i] = dis(gen);
    }

    //Record the start time, run scan, then record the end time
auto start = std::chrono::high_resolution_clock::now();
    //Call the function from scan.cpp
    scan(input.data(), output.data(), n);
auto end = std::chrono::high_resolution_clock::now();
    //Calculate the duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;

    //Print the required output values
    //Time taken in milliseconds
    std::cout << duration.count() << std::endl;
    //The first element of the scanned array
    std::cout << output[0] << std::endl;
    //The last element of the scanned array (the total sum)
    std::cout << output[n - 1] << std::endl;

    return 0;
}
