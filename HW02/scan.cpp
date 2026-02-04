#include "scan.h"

void scan(const float* input, float* output, int n) {
    if (n <= 0) return; //if array is size zero we have nothing to do so return
    output[0] = input[0]; //The first element of an inclusive scan is always identical to the input
    for (int i = 1; i < n; ++i) {
        //Each new element is the current input plus the sum of everything before
        //We find the sum of everything before it in the previous slot
        output[i] = output[i - 1] + input[i];
    }
}
