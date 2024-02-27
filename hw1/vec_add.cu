#include <iostream>
#include <cstdlib>
#include <ctime>

#define N 80

__global__ void matrixAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = i * n + j;
    if (i < n && j < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    float *a, *b, *c; // Matrices
    float *d_a, *d_b, *d_c; // Device copies of matrices

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

    // Allocate memory for device copies of matrices
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy matrices from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16); // Starting with a default block size of (16, 16)
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    matrixAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
	
    for(int i=0;i<N;i++){
	printf("%f ",c[i]);
    }
    // Free host memory
    free(a);
    free(b);
    free(c);

    return 0;
}


