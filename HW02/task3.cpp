#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "matmul.h"

using namespace std;
using namespace std::chrono;

int main() {
    const unsigned int n = 1024; // n between 1000 and 2000 [cite: 82]
    
    double* A = new double[n * n];
    double* B = new double[n * n];
    double* C = new double[n * n];
    vector<double> vA(n * n), vB(n * n);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(0.0, 1.0);

    for (unsigned int i = 0; i < n * n; ++i) {
        double valA = dist(gen);
        double valB = dist(gen);
        A[i] = vA[i] = valA;
        B[i] = vB[i] = valB;
    }

    cout << n << endl; // Print number of rows [cite: 85]

    auto run_test = [&](auto func, auto& matA, auto& matB) {
        fill(C, C + (n * n), 0.0);
        auto start = high_resolution_clock::now();
        func(matA, matB, C, n);
        auto end = high_resolution_clock::now();
        duration<double, milli> ms = end - start;
        cout << ms.count() << endl;      // Time in ms [cite: 85]
        cout << C[n * n - 1] << endl;    // Last element [cite: 85]
    };

    run_test(mmul1, A, B);
    run_test(mmul2, A, B);
    run_test(mmul3, A, B);
    run_test(mmul4, vA, vB);

    delete[] A; delete[] B; delete[] C;
    return 0;
}
