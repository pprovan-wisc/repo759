#include <iostream>
#include <omp.h>

// Helper to compute factorial
long long factorial(int n) {
    long long fact = 1;
    for (int i = 1; i <= n; ++i) {
        fact *= i;
    }
    return fact;
}

int main() {
    // Set OpenMP to use 4 threads
    omp_set_num_threads(4);

    #pragma omp parallel
    {
        // Print total threads only once
        #pragma omp single
        {
            std::cout << "Number of threads: " << omp_get_num_threads() << "\n";
        }

        // Each thread introduces itself
        // Note: Using critical/atomic for cout ensures it doesn't garble output
        #pragma omp critical
        {
            std::cout << "I am thread No. " << omp_get_thread_num() << "\n";
        }

        // Parallelize the factorial loop
        #pragma omp for
        for (int i = 1; i <= 8; ++i) {
            long long result = factorial(i);
            #pragma omp critical
            {
                std::cout << i << "!=" << result << "\n";
            }
        }
    }

    return 0;
}
