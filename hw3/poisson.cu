// Solve the Poisson equation on a 3D lattice.
//
// compile with the following command:
//
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 -o poisson poisson.cu

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
// field variables
float* h_new;   // host field vectors
float* h_old;   
float* h_C;     // result of diff*diff of each block
float* g_new;   
float* d_new;   // device field vectors
float* d_old;  
float* d_C;

int     MAX=10000000;     // maximum iterations
double  eps=1.0e-10;      // stopping criterion


__global__ void poisson(float* phi_old, float* phi_new, float* C, bool flag)
{
    extern __shared__ float cache[];     
    float  l,r,u,d,f,b    // up, left, right, down ,front , back
    float  diff; 
    int    site, xm1 , ym1 , zm1 , xp1 , yp1 , zp1;

    int Nx = blockDim.x*gridDim.x;
    int Ny = blockDim.y*gridDim.y;
    int Nz = blockDim.z*gridDim.z;
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int z = blockDim.z*blockIdx.z + threadIdx.z;
    int cacheIndex = threadIdx.x + threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;

    site = x + y*Nx + z*Nx*Ny;

    int pos_x,pos_y,pos_z;
    pos_z = site/(L*L);
    pos_y = (site-pos_z*L*L)/L;
    pos_x = site-pos_z*L*L-pos_y*L;

    if(((pos_x == Nx/2 || pos_x == Nx/2-1 ) && (pos_y == Ny/2 || pos_y == Ny/2-1 ) && (pos_z == Nz/2 || pos_z == Nz/2-1 ))|| pos_x==0 || pos_x==Nx-1 || pos_y==0 || pos_y==Ny-1 || pos_z==0 || pos_z==Nz-1) {
        diff = 0.0;
    }
    else {
        xm1 = site - 1;    // x-1
        xp1 = site + 1;    // x+1
        ym1 = site - L;   // y-1
        yp1 = site + L;   // y+1
        zm1 = site - L*L;   // z-1
        zp1 = site + L*L;   // z+1
        if(flag){
          l = phi_old[xm1];
          r = phi_old[xp1];
          u = phi_old[yp1];
          d = phi_old[ym1];
          f = phi_old[zp1];
          b = phi_old[zm1];
          // 1/6(l+r+u+d+f+b)
          phi_new[site] = 0.166*(l+r+u+d+f+b);
        }
        else{
          l = phi_new[xm1];
          r = phi_new[xp1];
          u = phi_new[yp1];
          d = phi_new[ym1];
          f = phi_new[zp1];
          b = phi_new[zm1];
          // 1/6(l+r+u+d+f+b)
          phi_old[site] = 0.166*(l+r+u+d+f+b);
        }
        diff = phi_new[site]-phi_old[site];
    }
    
    cache[cacheIndex]=diff*diff;
    __syncthreads();

    // perform parallel reduction

    int ib = blockDim.x * blockDim.y * blockDim.z/2;
    while (ib != 0) {
        if (cacheIndex < ib) {
            cache[cacheIndex] += cache[cacheIndex + ib];
        }
        __syncthreads();
        ib /= 2;
    }

    int bx = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    if(cacheIndex == 0) {
        C[bx] = cache[0];
    }
}

int main(void)
{

    int gid;              // GPU_ID
    int iter;
    volatile bool flag;   // to toggle between *_new and *_old  
    float cputime;
    float gputime;
    float gputime_tot;
    double flops;
    double error;
    
    printf("Enter the GPU ID (0/1): ");
    scanf("%d",&gid);
    printf("%d\n",gid);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }
    printf("Select GPU with device ID = %d\n", gid);

    cudaSetDevice(gid);

    printf("Solve Poisson equation on a 3D lattice with boundary conditions\n");

    int Nx,Ny,Nz;                // lattice size
    printf("Enter the size (Nx, Ny,Nz) of the 3D lattice: ");
    scanf("%d %d %d",&Nx,&Ny,&Nz);        
    printf("%d %d %d\n",Nx,Ny,Nz);        

    // Set the number of threads (tx,ty) per block
   
    int tx,ty,tz;
    printf("Enter the number of threads (tx,ty) per block: ");
    scanf("%d %d %d",&tx, &ty,&tz);
    printf("%d %d %d\n",tx, ty,tz);
    if( tx*ty*tz > 1024 ) {
      printf("The number of threads per block must be less than 1024 ! \n");
      exit(0);
    }
    dim3 threads(tx,ty,tz);
    
    // The total number of threads in the grid is equal to the total number of lattice sites
    
    int bx = Nx/tx;
    if(bx*tx != Nx) {
      printf("The block size in x is incorrect\n"); 
      exit(0);
    }
    int by = Ny/ty;
    if(by*ty != Ny) {
      printf("The block size in y is incorrect\n"); 
      exit(0);
    }
    int bz = Nz/tz;
    if(bz*tz != Nz) {
      printf("The block size in z is incorrect\n"); 
      exit(0);
    }

    if((bx > 65535) || (by > 65535) || (bz > 65535)) {
      printf("The grid size exceeds the limit ! \n");
      exit(0);
    }
    dim3 blocks(bx,by,bz);
    printf("The dimension of the grid is (%d, %d , %d)\n",bx,by,bz);

    int CPU;    
    printf("To compute the solution vector with CPU/GPU/both (0/1/2) ? ");
    scanf("%d",&CPU);
    printf("%d\n",CPU);
    fflush(stdout);

    // Allocate field vector h_phi in host memory

    int N = Nx*Ny*Nz;
    int size = N*sizeof(float);
    int sb = bx*by*bz*sizeof(float);
    h_old = (float*)malloc(size);
    h_new = (float*)malloc(size);
    g_new = (float*)malloc(size);
    h_C = (float*)malloc(sb);
   
    memset(h_old, 0, size);
    memset(h_new, 0, size);

    // Initialize the field vector with boundary conditions

    /* 
    h(Nx/2,Ny/2,Nz/2) = 1.0
    h(Nx/2,Ny/2,Nz/2-1) = 1.0
    h(Nx/2,Ny/2-1,Nz/2) = 1.0
    h(Nx/2,Ny/2-1,Nz/2-1) = 1.0
    h(Nx/2-1,Ny/2,Nz/2) = 1.0
    h(Nx/2-1,Ny/2,Nz/2-1) = 1.0
    h(Nx/2-1,Ny/2-1,Nz/2) = 1.0
    h(Nx/2-1,Ny/2-1,Nz/2-1) = 1.0
    else h(x,y,z) = 0.0
    */

    int pos_x,pos_y,pos_z;
    for(int i=0; i<N; i++) {
      pos_z = i/(L*L);
      pos_y = (i-pos_z*L*L)/L;
      pos_x = i-pos_z*L*L-pos_y*L;
      if((pos_x == Nx/2 || pos_x == Nx/2-1 ) && (pos_y == Ny/2 || pos_y == Ny/2-1 ) && (pos_z == Nz/2 || pos_z == Nz/2-1 )) {
        h_old[i] = 1.0;
      }
    }

    FILE *out1;                 // save initial configuration in phi_initial.dat
    out1 = fopen("phi_initial.dat","w");

    fprintf(out1, "Inital field configuration:\n");
    for(int k=Nz-1;k>-1;k--) {
      for(int j=Ny-1;j>-1;j--) {
        for(int i=0; i<Nx; i++) {
          fprintf(out1,"%.2e ",h_old[i+j*Nx+k*Nx*Ny]);
        }
        fprintf(out1,"\n");
      }
      fprintf(out1,"\n");
    }
    fclose(out1);

//    printf("\n");
//    printf("Inital field configuration:\n");
//    for(int j=Ny-1;j>-1;j--) {
//      for(int i=0; i<Nx; i++) {
//        printf("%.2e ",h_new[i+j*Nx]);
//      }
//      printf("\n");
//    }

    printf("\n");

    // create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if((CPU==1)||(CPU==2)) {

      // start the timer
      cudaEventRecord(start,0);

      // Allocate vectors in device memory

      cudaMalloc((void**)&d_new, size);
      cudaMalloc((void**)&d_old, size);
      cudaMalloc((void**)&d_C, sb);
  
      // Copy vectors from host memory to device memory

      cudaMemcpy(d_new, h_new, size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_old, h_old, size, cudaMemcpyHostToDevice);
    
      // stop the timer
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);

      float Intime;
      cudaEventElapsedTime( &Intime, start, stop);
      printf("Input time for GPU: %f (ms) \n",Intime);

      // start the timer
      cudaEventRecord(start,0);

      error = 10*eps;  // any value bigger than eps is OK
      iter = 0;        // counter for iterations
      flag = true; 
 
      int sm = tx*ty*tz*sizeof(float);

      while ( (error > eps) && (iter < MAX) ) {

        laplacian<<<blocks,threads,sm>>>(d_old, d_new, d_C, flag);
        cudaMemcpy(h_C, d_C, sb, cudaMemcpyDeviceToHost);
        error = 0.0;
        for(int i=0; i<bx*by*bz; i++) {
          error = error + h_C[i];
        }
        error = sqrt(error);

//        printf("error = %.15e\n",error);
//        printf("iteration = %d\n",iter);

        iter++;
        flag = !flag;

      }

      printf("error (GPU) = %.15e\n",error);
      printf("total iterations (GPU) = %d\n",iter);
    
      // stop the timer
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);

      cudaEventElapsedTime( &gputime, start, stop);
      printf("Processing time for GPU: %f (ms) \n",gputime);
      flops = 7.0*(Nx-2)*(Ny-2)*iter;
      printf("GPU Gflops: %f\n",flops/(1000000.0*gputime));
    
      // Copy result from device memory to host memory
  
      // start the timer
      cudaEventRecord(start,0);

      cudaMemcpy(g_new, d_new, size, cudaMemcpyDeviceToHost);

      cudaFree(d_new);
      cudaFree(d_old);
      cudaFree(d_C);

      // stop the timer
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);

      float Outime;
      cudaEventElapsedTime( &Outime, start, stop);
      printf("Output time for GPU: %f (ms) \n",Outime);

      gputime_tot = Intime + gputime + Outime;
      printf("Total time for GPU: %f (ms) \n",gputime_tot);
      fflush(stdout);

      FILE *outg;                 // save GPU solution in phi_GPU.dat
      outg = fopen("phi_GPU.dat","w");

      fprintf(outg, "GPU field configuration:\n");
      for(int k=Nz-1;k>-1;k--) {
        for(int j=Ny-1;j>-1;j--) {
          for(int i=0; i<Nx; i++) {
            fprintf(outg,"%.2e ",g_new[i+j*Nx+k*Nx*Ny]);
          }
          fprintf(outg,"\n");
        }
        fprintf(outg,"\n");
      }
      fclose(outg);

//      printf("\n");
//      printf("Final field configuration (GPU):\n");
//      for(int j=Ny-1;j>-1;j--) {
//        for(int i=0; i<Nx; i++) {
//          printf("%.2e ",g_new[i+j*Nx]);
//        }
//        printf("\n");
//      }

      printf("\n");

    } 

    if(CPU==1) {      // not to compute the CPU solution 
      free(h_new);
      free(h_old);
      free(g_new);
      free(h_C);
      cudaDeviceReset();
      exit(0);
    }
 
    if((CPU==0)||(CPU==2)) {      // to compute the CPU solution 

      // start the timer
      cudaEventRecord(start,0);

      // to compute the reference solution

      error = 10*eps;      // any value bigger than eps 
      iter = 0;            // counter for iterations
      flag = true;     
      double diff; 

      float  l,r,u,d,f,b    // up, left, right, down ,front , back
      int site, xm1 , ym1 , zm1 , xp1 , yp1 , zp1;

      while ( (error > eps) && (iter < MAX) ) {
        if(flag) {
          error = 0.0;
          for(int x=1;x<Nx-1;x++) {
          for(int y=1;y<Ny-1;y++) {
          for(int z=1;z<Nz-1;z++) {
            if(x == Nx/2 || x == Nx/2-1 ) {
              if(y == Ny/2 || y == Ny/2-1 ) {
                if(z == Nz/2 || z == Nz/2-1 ) {
                  continue;
                }
              }
            }
            site = x + y*Nx + z*Nx*Ny;
            xm1 = site - 1;    // x-1
            xp1 = site + 1;    // x+1
            ym1 = site - Nx;   // y-1
            yp1 = site + Nx;   // y+1
            zm1 = site - Nx*Ny;   // z-1
            zp1 = site + Nx*Ny;   // z+1
            b = h_new[zm1]; 
            l = h_new[xm1]; 
            r = h_new[xp1]; 
            u = h_new[yp1]; 
            f = h_new[zp1]; 
            d = h_new[ym1]; 
            h_new[site] = 0.166*(b+l+r+u+f+d);
            diff = h_new[site]-h_old[site]; 
            error = error + diff*diff;
          }
          }
          }
        }
        else {
          error = 0.0;
          for(int x=1;x<Nx-1;x++) {
          for(int y=1;y<Ny-1;y++) {
          for(int z=1;z<Nz-1;z++) {
            if(x == Nx/2 || x == Nx/2-1 ) {
              if(y == Ny/2 || y == Ny/2-1 ) {
                if(z == Nz/2 || z == Nz/2-1 ) {
                  continue;
                }
              }
            }
            site = x + y*Nx + z*Nx*Ny;
            xm1 = site - 1;    // x-1
            xp1 = site + 1;    // x+1
            ym1 = site - Nx;   // y-1
            yp1 = site + Nx;   // y+1
            zm1 = site - Nx*Ny;   // z-1
            zp1 = site + Nx*Ny;   // z+1
            b = h_old[zm1]; 
            l = h_old[xm1]; 
            r = h_old[xp1]; 
            u = h_old[yp1]; 
            f = h_old[zp1]; 
            d = h_old[ym1]; 
            h_old[site] = 0.166*(b+l+r+u+f+d);
            diff = h_old[site]-h_new[site]; 
            error = error + diff*diff;
          }
          }
          }
        }
        flag = !flag;
        iter++;
        error = sqrt(error);

//        printf("error = %.15e\n",error);
//        printf("iteration = %d\n",iter);

      }                   // exit if error < eps
    
      printf("error (CPU) = %.15e\n",error);
      printf("total iterations (CPU) = %d\n",iter);

      // stop the timer
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);

      cudaEventElapsedTime( &cputime, start, stop);
      printf("Processing time for CPU: %f (ms) \n",cputime);
      flops = 7.0*(Nx-2)*(Ny-2)*(Nz-2)*iter;
      printf("CPU Gflops: %lf\n",flops/(1000000.0*cputime));

      if(CPU==2) {
        printf("Speed up of GPU = %f\n", cputime/(gputime_tot));
        fflush(stdout);
      }

      // destroy the timer
      cudaEventDestroy(start);
      cudaEventDestroy(stop);

      FILE *outc;                 // save CPU solution in phi_CPU.dat
      outc = fopen("phi_CPU.dat","w");

      fprintf(outc, "CPU field configuration:\n");
      for(int k=Nz-1;k>-1;k--) {
        for(int j=Ny-1;j>-1;j--) {
          for(int i=0; i<Nx; i++) {
            fprintf(outc,"%.2e ",h_new[i+j*Nx+k*Nx*Ny]);
          }
          fprintf(outc,"\n");
        }
        fprintf(outc,"\n");
      }
      fclose(outc);

 //     printf("\n");
 //     printf("Final field configuration (CPU):\n");
 //     for(int j=Ny-1;j>-1;j--) {
 //       for(int i=0; i<Nx; i++) {
 //         printf("%.2e ",h_new[i+j*Nx]);
 //       }
 //       printf("\n");
 //     }

      printf("\n");

      free(h_new);
      free(h_old);
      free(g_new);
      free(h_C);

    }

    cudaDeviceReset();
    
}

