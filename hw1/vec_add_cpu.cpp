#include <iostream>
#include <cstdlib>
#include <ctime>

#define N 80

void matrixAdd(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int index = i * n + j;
            c[index] = a[index] + b[index];
        }
    }
}

int main() {
    float *a, *b, *c; // Matrices

    int size = N * N * sizeof(float);

    // Allocate memory for matrices on host
    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);

    // Initialize matrices with random numbers
    srand(time(NULL));
    for (int i = 0; i < N * N; ++i) {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
        b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Start timing
    clock_t start = clock();

    // Perform matrix addition on CPU
    matrixAdd(a, b, c, N);

    // Stop timing
    clock_t end = clock();

    // Calculate elapsed time in milliseconds
    double elapsed_time = double(end - start) / CLOCKS_PER_SEC * 1000.0;

    // Print execution time
    std::cout << "Execution time: " << elapsed_time << " milliseconds" << std::endl;

    // Free host memory
    free(a);
    free(b);
    free(c);

    return 0;
}
