#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define N 6400


// CUDA kernel for matrix addition
__global__ void matrixAddition(float **a, float **b, float **result, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        result[row][col] = a[row][col] + b[row][col];
    }
    __syncthreads();
}

void execute(int block_size) {
    // Matrix dimensions
    int rows = N;
    int cols = N;

    // Host arrays and initialization with random values
    float **h_a = new float*[rows];
    float **h_b = new float*[rows];
    float **h_result = new float*[rows];

    for (int i = 0; i < rows; ++i) {
        h_a[i] = new float[cols];
        h_b[i] = new float[cols];
        h_result[i] = new float[cols];
    }

    srand(time(NULL));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            h_a[i][j] = static_cast<float>(rand()) / RAND_MAX;
            h_b[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Device arrays
    float **d_a, **d_b, **d_result;
    cudaMalloc(&d_a, rows * sizeof(float *));
    cudaMalloc(&d_b, rows * sizeof(float *));
    cudaMalloc(&d_result, rows * sizeof(float *));

    float **d_a_data, **d_b_data, **d_result_data;
    cudaMalloc(&d_a_data, rows * cols * sizeof(float));
    cudaMalloc(&d_b_data, rows * cols * sizeof(float));
    cudaMalloc(&d_result_data, rows * cols * sizeof(float));

    cudaMemcpy(d_a, h_a, rows * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, rows * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result, rows * sizeof(float *), cudaMemcpyHostToDevice);

    cudaMemcpy(d_a_data, h_a[0], rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_data, h_b[0], rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Timing
    clock_t start, stop;
    start = clock();

    // Launch kernel for matrix addition and measure time
    dim3 threadsPerBlock(block_size, block_size);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixAddition<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_result, rows, cols);

    cudaDeviceSynchronize(); // Ensure all kernel launches are complete

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds
    std::cout << "Block size: " << block_size << ", Time taken: " << timer_seconds << " ms\n";

    // Copy result back to host
    cudaMemcpy(h_result, d_result, rows * sizeof(float *), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result[0], d_result_data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    cudaFree(d_a_data);
    cudaFree(d_b_data);
    cudaFree(d_result_data);

    // Free host memory
    for (int i = 0; i < rows; ++i) {
        delete[] h_a[i];
        delete[] h_b[i];
        delete[] h_result[i];
    }
    delete[] h_a;
    delete[] h_b;
    delete[] h_result;
}

int main() {
    int a[] = {4, 8, 10, 16, 20, 32};
    for (int i = 0; i < 6; i++) {
        execute(a[i]);
    }

    return 0;
}

