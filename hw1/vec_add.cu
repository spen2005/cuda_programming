#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define N 6400
#define BLOCK_SIZE 4

// CUDA kernel for matrix addition
__global__ void matrixAddition(float *a, float *b, float *result, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        result[index] = a[index] + b[index];
    }
}

void execute(int block_size) {
    // Matrix dimensions
    int rows = N;
    int cols = N;
    int size = rows * cols;

    // Host arrays and initialization with random values
    float *h_a = new float[size];
    float *h_b = new float[size];
    float *h_result = new float[size];

    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device arrays
    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, size * sizeof(float));
    cudaMalloc(&d_b, size * sizeof(float));
    cudaMalloc(&d_result, size * sizeof(float));

    // Copy initialized matrices from host to device
    cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice);

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel for matrix addition and measure time
    dim3 threadsPerBlock(block_size, block_size);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(start);
    matrixAddition<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_result, rows, cols);
    cudaEventRecord(stop);

    // Synchronize events and calculate elapsed time
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Block size: " << block_size << ", Time taken: " << elapsedTime << " ms\n";

    // Copy result back to host
    cudaMemcpy(h_result, d_result, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    // Free host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_result;
}

int main() {
    int block_sizes[] = {4, 8, 10, 16, 20, 32};
    for (int i = 0; i < 6; i++) {
        execute(block_sizes[i]);
    }

    return 0;
}

