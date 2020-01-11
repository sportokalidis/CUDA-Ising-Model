/*
*       Parallel and Distributed Systems
*       Exercise 3
*       V1. GPU with one thread per moment
*       Authors:
*         Portokalidis Stavros, AEM 9334, stavport@ece.auth.gr
*         Christoforidis Savvas, AEM 9147, schristofo@ece.auth.gr
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// block dimensions
#define BX 16
#define BY 16
// window size
#define WS 5

// replace pointer types
#define old(i,j,n) *(old+(i)*n+(j))
#define current(i,j,n) *(current+(i)*n+(j))
#define w(i,j) *(w+(i)*WS+(j))
#define d_w(i,j) *(d_w+(i)*WS+(j))
#define G(i,j,n) *(G+(i)*n+(j))
#define d_current(i,j,n) *(d_current+(i)*n+(j))
#define d_old(i,j,n) *(d_old+(i)*n+(j))

// cuda kernel
__global__ void kernel(int *d_current, int *d_old, double *d_w, int n) {

    // compute column and row global index
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    // check if within bounds.
    if ((c >= n) || (r >= n))
        return;

    double influence = 0; // weighted influence of the neighbors
    for(int ii=0; ii<WS; ii++){
      for(int jj=0; jj<WS; jj++){
        influence +=  d_w(ii,jj) * d_old((r-2+n+ii)%n, (c-2+n+jj)%n, n);
      }
    }

    // magnetic moment gets the value of the SIGN of the weighted influence of its neighbors
    if(fabs(influence) < 10e-7){
      d_current(r,c,n) = d_old(r,c,n); // remains the same in the case that the weighted influence is zero
    }
    else if(influence > 10e-7){
      d_current(r,c,n) = 1;
    }
    else if(influence < 0){
      d_current(r,c,n) = -1;
    }

}

void ising( int *G, double *w, int k, int n){

  dim3 block(BX,BY);  // blockDim
  dim3 grid((n+BX-1)/BX,(n+BY-1)/BY); // gridDim

  int * old = (int*) malloc(n*n*sizeof(int)); // old spin lattice
  int * current = (int*) malloc(n*n*sizeof(int)); // current spin lattice
  if( old==NULL || current==NULL){
    printf("memory allocation failed (CPU)\n");
    exit(1);
  }

  // device variables
  int *d_old, *d_current, *tmp;
  double * d_w;
  if( cudaMalloc(&d_old , n*n*sizeof(int)) != cudaSuccess || cudaMalloc(&d_current , n*n*sizeof(int)) || cudaMalloc(&d_w, WS*WS*sizeof(double))){
    printf("memory allocation failed (GPU)\n");
    exit(1);
  }

  // copy host to device
  cudaMemcpy(d_w, w, WS*WS*sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy(d_old, G, n*n*sizeof(int), cudaMemcpyHostToDevice );

  // run for k steps
  for(int l=0; l<k; l++){

    // kernel execution
    kernel<<<grid,block>>>(d_current, d_old, d_w, n );
    cudaDeviceSynchronize();

    // copy device to host
    cudaMemcpy(old, d_old, n*n*sizeof(int), cudaMemcpyDeviceToHost );
    cudaMemcpy(current, d_current, n*n*sizeof(int), cudaMemcpyDeviceToHost );

    // save result in G
    cudaMemcpy(G , d_current , n*n*sizeof(int), cudaMemcpyDeviceToHost);

    // swap the pointers for the next iteration
    tmp = d_old;
    d_old = d_current;
    d_current = tmp;

     // terminate if no changes are made
    int areEqual = 1;
    for(int i=0; i<n; i++){
      for(int j=0; j<n; j++){
        if(old(i,j,n) != current(i,j,n)){
          areEqual = 0;
          i=n;
          j=n;
        }
      }
    }
    // termination branch
    if(areEqual == 1){
      printf("terminated: spin values stay same (step %d)\n" , l);
      exit(0);
    }

  }

  // free host/device space
  free(old);
  free(current);
  cudaFree(d_old);
  cudaFree(d_current);
  cudaFree(d_w);
}
