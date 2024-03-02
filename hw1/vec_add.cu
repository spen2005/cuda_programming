#include <iostream>
#include <cstdlib>
#include <ctime>
#include <curand_kernel.h>
#include <cuda_runtime.h>
using namespace std ;
#define N 6400
int sz = N*N;


// CUDA kernel for matrix addition
__global__ void matrixAddition(float **a, float **b, float **result, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        result[row][col] = a[row][col] + b[row][col];
    }
    __syncthreads();
}


void execute(int block_size){
    // Matrix dimensions
    int rows = 6400;
    int cols = 6400;


    float **h_a = new float*[rows];
    float **h_b = new float*[rows];
    float **h_result = new float*[rows];

    srand(time(NULL));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            h_a[i][j] = static_cast<float>(rand()) / RAND_MAX;
            h_b[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    float **d_a, **d_b, **d_result;
    cudaMalloc(&d_a, sz   * sizeof(float ));
    cudaMalloc(&d_b, sz   * sizeof(float ));
    cudaMalloc(&d_result, sz   * sizeof(float ));


    cudaMemcpy(d_a, h_a, sz   * sizeof(float ), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sz   * sizeof(float ), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result, sz   * sizeof(float ), cudaMemcpyHostToDevice);


    // CUDA events for timing
    clock_t start, stop;
    start = clock();


    // Launch kernel for matrix addition and measure time
    dim3 threadsPerBlock(block_size,block_size);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixAddition<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_result, rows, cols);
    

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    cout << "took " << timer_seconds << " seconds.\n";
    // Synchronize events and calculate elapsed time

    // Copy result back to host
    cudaMemcpy(h_result, d_result, sz   * sizeof(float ), cudaMemcpyDeviceToHost);


    // FNee device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    // Free host memory
    for (int i = 0; i < rows; ++i) {
        delete[] h_a[i];
        delete[] h_b[i];
        delete[] h_result[i];
    }
    delete[] h_a;
    delete[] h_b;
    delete[] h_result;
    return ;

}

int main(){
	int a[]={4,8,10,16,20,32};
	for(int i=0;i<6;i++){execute(a[i]);cout << "yes" ;}

}
