#include <cfloat> // for FLT_MAX
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>


// CUDA kernel to find the minimum absolute value using parallel reduction
__global__ void minAbsKernel(float *array, int n, float *minAbs) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    float localMin = FLT_MAX;
    while (i < n) {
        float val = fabsf(array[i]);
        localMin = fminf(localMin, val);
        i += gridSize;
    }

    sdata[tid] = localMin;
    __syncthreads();

    // Do parallel reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        minAbs[blockIdx.x] = sdata[0];
    }
}

// Host function to generate random array
void RandomInit(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = ((float)rand() / RAND_MAX) * 1000 - 500; // Random float between -500 and 500
    }
}

int main() {
    const int numElements = 81920007;
    const int blockSize = 256;
    int numBlocks;

    // Allocate memory on host
    float *h_data = new float[numElements];

    // Generate random input data
    RandomInit(h_data, numElements);

    // Allocate memory on device
    float *d_data, *d_minAbs;
    cudaMalloc(&d_data, numElements * sizeof(float));
    cudaMalloc(&d_minAbs, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, numElements * sizeof(float), cudaMemcpyHostToDevice);

    // Determine grid size
    numBlocks = (numElements + blockSize - 1) / blockSize;

    // Start time measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    minAbsKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_data, numElements, d_minAbs);

    // Stop time measurement
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    float h_minAbs;
    cudaMemcpy(&h_minAbs, d_minAbs, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_minAbs);

    // Free host memory
    delete[] h_data;

    std::cout << "Minimum absolute value: " << h_minAbs << std::endl;
    std::cout << "Execution time: " << milliseconds << " milliseconds" << std::endl;

    return 0;
}

