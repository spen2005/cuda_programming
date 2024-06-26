// Vector Dot
// using multiple GPUs with OpenMP

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>          // header for OpenMP
#include <cuda_runtime.h>

// Variables
float* h_A;   // host vectors
float* h_B;
float* h_C;

// Functions
void RandomInit(float*, int);

// Device code
__global__ void VecDot(const float* A, const float* B, float* C, int N)
{
    extern __shared__ float cache[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float temp = 0.0;  // register for each thread
    while (i < N) {
        temp += A[i]*B[i];
        i += blockDim.x*gridDim.x;  
    }
   
    cache[cacheIndex] = temp;   // set the cache value 

    __syncthreads();

    // perform parallel reduction, threadsPerBlock must be 2^m

    int ib = blockDim.x/2;
    while (ib != 0) {
      if(cacheIndex < ib)
        cache[cacheIndex] += cache[cacheIndex + ib]; 

      __syncthreads();

      ib /=2;
    }
    
    if(cacheIndex == 0)
      C[blockIdx.x] = cache[0];
}

// Host code

int main(void)
{
    printf("\n");
    printf("Vector Dot Product with multiple GPUs \n");
    int N, NGPU, cpu_thread_id=0;
    int *Dev; 
    long mem = 1024*1024*1024;     // 4 Giga for float data type.

    printf("Enter the number of GPUs: ");
    scanf("%d", &NGPU);
    printf("%d\n", NGPU);
    Dev = (int *)malloc(sizeof(int)*NGPU);

    int numDev = 0;
    printf("GPU device number: ");
    for(int i = 0; i < NGPU; i++) {
      scanf("%d", &Dev[i]);
      printf("%d ",Dev[i]);
      numDev++;
      if(getchar() == '\n') break;
    }
    printf("\n");
    if(numDev != NGPU) {
      fprintf(stderr,"Should input %d GPU device numbers\n", NGPU);
      exit(1);
    }

    printf("Enter the size of the vectors: ");
    scanf("%d", &N);        
    printf("%d\n", N);        
    if (3*N > mem) {
        printf("The size of these 3 vectors cannot be fitted into 4 Gbyte\n");
        exit(1);
    }

    // Set the sizes of threads and blocks
    int threadsPerBlock;
    printf("Enter the number of threads per block: ");
    scanf("%d", &threadsPerBlock);
    printf("%d\n", threadsPerBlock);
    if(threadsPerBlock > 1024) {
      printf("The number of threads per block must be less than 1024 ! \n");
      exit(1);
    }
    int blocksPerGrid = (N + threadsPerBlock*NGPU - 1) / (threadsPerBlock*NGPU);
    printf("The number of blocks is %d\n", blocksPerGrid);
    if(blocksPerGrid > 2147483647) {
      printf("The number of blocks must be less than 2147483647 ! \n");
      exit(1);
    }

    long size = N*sizeof(float);
    int sb = blocksPerGrid*NGPU*sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(sb);
    if (! h_A || ! h_B) {
	printf("!!! Not enough memory.\n");
	exit(1);
    }
    
    // Initialize input vectors

    RandomInit(h_A, N);
    RandomInit(h_B, N);

    float Intime, gputime, Outime;

    omp_set_num_threads(NGPU);

    #pragma omp parallel private(cpu_thread_id, Intime, gputime, Outime)
    {
        cudaEvent_t start, stop; // Declare start and stop events
	    float *d_A, *d_B, *d_C;
	    cpu_thread_id = omp_get_thread_num();
	    cudaSetDevice(Dev[cpu_thread_id]);

        // start the timer
        if(cpu_thread_id == 0) {
            cudaEventCreate(&start); // Create start event
            cudaEventCreate(&stop); // Create stop event
            cudaEventRecord(start, 0); // Record start event
        }

	    // Allocate vectors in device memory
	    cudaMalloc((void**)&d_A, size/NGPU);
	    cudaMalloc((void**)&d_B, size/NGPU);
	    cudaMalloc((void**)&d_C, sb/NGPU);

        // Copy vectors from host memory to device memory
	    cudaMemcpy(d_A, h_A+N/NGPU*cpu_thread_id, size/NGPU, cudaMemcpyHostToDevice);
	    cudaMemcpy(d_B, h_B+N/NGPU*cpu_thread_id, size/NGPU, cudaMemcpyHostToDevice);
	    #pragma omp barrier // wait for all threads to complete the data transfer

        // stop the timer
	    if(cpu_thread_id == 0) {
            cudaEventRecord(stop, 0); // Record stop event
            cudaEventSynchronize(stop); // Synchronize stop event
            cudaEventElapsedTime(&Intime, start, stop); // Calculate elapsed time
            printf("Data input time for GPU: %f (ms)\n", Intime);
	    }

        // start the timer
        if(cpu_thread_id == 0) {
            cudaEventRecord(start, 0); // Record start event
        }

        VecDot<<<blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(d_A, d_B, d_C, N/NGPU);
	    cudaDeviceSynchronize();

        // stop the timer
	    if(cpu_thread_id == 0) {
            cudaEventRecord(stop, 0); // Record stop event
            cudaEventSynchronize(stop); // Synchronize stop event
            cudaEventElapsedTime(&gputime, start, stop); // Calculate elapsed time
            printf("Processing time for GPU: %f (ms)\n", gputime);
            printf("GPU Gflops: %f\n", N/(1000000.0*gputime));
	    }

        // Copy result from device memory to host memory
        // h_C contains the result in host memory

        // start the timer
        if(cpu_thread_id == 0) {
            cudaEventRecord(start, 0); // Record start event
        }

        cudaMemcpy(h_C+blocksPerGrid*cpu_thread_id, d_C, sb/NGPU, cudaMemcpyDeviceToHost);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // stop the timer
	    if(cpu_thread_id == 0) {
            cudaEventRecord(stop, 0); // Record stop event
            cudaEventSynchronize(stop); // Synchronize stop event
            cudaEventElapsedTime(&Outime, start, stop); // Calculate elapsed time
            printf("Data output time for GPU: %f (ms)\n", Outime);
	    printf("Total time for GPU: %f (ms) \n",Intime+Outime+gputime); 
	    }
    } 

    double h_G = 0.0;
    for(int i = 0; i < blocksPerGrid*NGPU; i++) 
      h_G += (double) h_C[i];

    

    cudaEvent_t cpu_start, cpu_stop;
    cudaEventCreate(&cpu_start);
    cudaEventCreate(&cpu_stop);
    // start the timer
    cudaEventRecord(cpu_start,0);

    // to compute the reference solution
    
    double h_D=0.0;       
    for(int i = 0; i < N; i++) 
      h_D += (double) h_A[i]*h_B[i];
    

    // stop the timer
    cudaEventRecord(cpu_stop,0);
    cudaEventSynchronize(cpu_stop);
	    float cpu_time;
    cudaEventElapsedTime(&cpu_time,cpu_start,cpu_stop);

    printf("Processing time for CPU: %f (ms) \n",cpu_time);
    printf("CPU Gflops: N/A\n");
    printf("Speed up of GPU = N/A\n");

    
    // destroy the timer
    cudaEventDestroy(cpu_start);
    cudaEventDestroy(cpu_stop);    

    // check result

    printf("Check result:\n");
    double diff = abs( (h_D - h_G)/h_D );
    printf("|(h_G - h_D)/h_D|=%20.15e\n",diff);
    printf("h_G =%20.15e\n",h_G);
    printf("h_D =%20.15e\n",h_D);
    printf("\n");

    free(h_A);
    free(h_B);
    free(h_C);

    for (int i=0; i < NGPU; i++) {
	    cudaSetDevice(i);
	    cudaDeviceReset();
    }

    return 0;
}

// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

