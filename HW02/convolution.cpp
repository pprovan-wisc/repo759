#include <iostream>
#include <vector>
#include "convolution.h"

void convolve(const float* f, float* g, int n, const float* w, int m) {
    int radius = (m - 1) / 2;

    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
            float sum = 0.0;

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    // Calculate the coordinates in the original image
                    int img_x = x + i - radius;
                    int img_y = y + j - radius;

                    float val;
                    // Check boundary conditions [cite: 42, 43, 44]
                    bool x_in = (img_x >= 0 && img_x < n);
                    bool y_in = (img_y >= 0 && img_y < n);

                    if (x_in && y_in) {
                        // Inside bounds
                        val = f[img_x * n + img_y];
                    } else if (x_in || y_in) {
                        // One coordinate is out: Edge (pad with 1) [cite: 44]
                        val = 1.0;
                    } else {
                        // Both coordinates are out: Corner (pad with 0) [cite: 44]
                        val = 0.0;
                    }

                    sum += w[i * m + j] * val;
                }
            }
            g[x * n + y] = sum;
        }
    }
}
