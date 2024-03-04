#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>

// CUDA kernel for minimum reduction
__global__ void minAbsReduce(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? fabsf(input[i]) : INFINITY;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// CPU function to find the minimum of an array
float minAbsCPU(float *input, int n) {
    float minVal = INFINITY;
    for (int i = 0; i < n; ++i) {
        minVal = fminf(minVal, fabsf(input[i]));
    }
    return minVal;
}

int main() {
    const int numElements = 81920007;
    const int blockSize = 256;
    const int gridSize = (numElements + blockSize - 1) / blockSize;

    // Allocate memory for the array on host (CPU)
    float *h_input = new float[numElements];

    // Initialize the array with random values
    srand(time(NULL));
    for (int i = 0; i < numElements; ++i) {
        h_input[i] = (rand() % 1000) / 10.0 - 50.0; // Random numbers between -50 and 50
    }

    // Allocate memory for the array on device (GPU)
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, numElements * sizeof(float));
    cudaMalloc((void **)&d_output, gridSize * sizeof(float));

    // Copy the input array from host to device
    cudaMemcpy(d_input, h_input, numElements * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel for parallel reduction
    minAbsReduce<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, numElements);

    // Copy the result back from device to host
    float *h_output = new float[gridSize];
    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Find the minimum of the results from each block on CPU
    float minValGPU = h_output[0];
    for (int i = 1; i < gridSize; ++i) {
        minValGPU = fminf(minValGPU, h_output[i]);
    }

    // Find the minimum of the array on CPU for validation
    float minValCPU = minAbsCPU(h_input, numElements);

    // Output the results
    std::cout << "Minimum absolute value (GPU): " << minValGPU << std::endl;
    std::cout << "Minimum absolute value (CPU): " << minValCPU << std::endl;

    // Free memory
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
