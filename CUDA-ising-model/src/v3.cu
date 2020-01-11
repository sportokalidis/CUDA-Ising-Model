/*
*       Parallel and Distributed Systems
*       Exercise 3
*       V3. GPU with multiple thread sharing common input moments
*       Authors:
*         Portokalidis Stavros, AEM 9334, stavport@ece.auth.gr
*         Christoforidis Savvas, AEM 9147, schristofo@ece.auth.gr
*/

#include <stdio.h>
#include <stdlib.h>

// block dimensions
#define BX 16
#define BY 16
// radius of each block
#define RADIUS 2
// window size
#define WS 5

// replace pointer types
#define old(i,j,n) *(old+(i)*n+(j))
#define current(i,j,n) *(current+(i)*n+(j))
#define w(i,j) *(w+(i)*5+(j))
#define d_w(i,j) *(d_w+(i)*5+(j))
#define s_w(i,j) *(s_w+(i)*5+(j))
#define G(i,j,n) *(G+(i)*n+(j))
#define d_current(i,j,n) *(d_current+(i)*n+(j))
#define d_old(i,j,n) *(d_old+(i)*n+(j))
#define s_old(i,j) *(s_old+(i)*(BY+2*RADIUS)+(j))


// cuda kernel
__global__ void kernel(int *d_current, int *d_old, double *d_w, int n){

 // compute column and row global index
 int r = blockIdx.x * blockDim.x + threadIdx.x;
 int c = blockIdx.y * blockDim.y + threadIdx.y;

 // compute column and row local index
 int lxindex = threadIdx.x + RADIUS;
 int lyindex = threadIdx.y + RADIUS;

 double influence;
 // weights and spin values of each block in shared memory
 __shared__ double s_w[WS*WS];
 __shared__ int s_old[(BX + 2*RADIUS)*(BY + 2*RADIUS)];

 // shared weights loaded in parallel
 if(blockDim.x > WS && blockDim.y > WS ){
   if(threadIdx.x < WS && threadIdx.y < WS){
     s_w(threadIdx.x, threadIdx.y) = d_w(threadIdx.x, threadIdx.y);
   }
 }
 // shared threads loaded from thread 0,0
 else{
   if(threadIdx.x == 0 && threadIdx.y == 0){
     for(int i = 0 ; i<WS ; i++){
       for(int j =0; j<WS ; j++){
         s_w(i,j)=d_w(i,j);
       }
     }
   }
 }
 __syncthreads();


 // read input elements into shared memory
 for(int i=r; i<n+RADIUS; i+=blockDim.x*gridDim.x){
   for(int j=c; j<n+RADIUS; j+=blockDim.y*gridDim.y){

     // save old block values in shared block center
     s_old(lxindex,lyindex) = d_old((i+n)%n,(j+n)%n,n);
     __syncthreads();

     // column stencil
     if( threadIdx.y < RADIUS){
       s_old( lxindex , lyindex - RADIUS ) = d_old((i + n)%n , (j-RADIUS+n)%n , n);
       s_old( lxindex  , lyindex + BY ) = d_old( (i + n )%n , (j+BX + n)%n , n);
     }
     // row stencil
     if( threadIdx.x < RADIUS){
       s_old(lxindex  - RADIUS,lyindex ) = d_old( (i-RADIUS+n)%n , (j+n)%n   , n );
       s_old( lxindex + BY,lyindex ) = d_old(( i +BX +n)%n , (j +  n)%n     , n );
     }

     // corner stencil
     if(threadIdx.x < RADIUS && threadIdx.y < RADIUS ){
       // up-left
       s_old(lxindex - RADIUS , lyindex - RADIUS ) = d_old ( ( i - RADIUS + n)%n , (j-RADIUS + n )%n  , n );
       // down-right
       s_old(lxindex +  BX , lyindex + BY  ) = d_old( (i  + BX + n  )%n , (j + BY +n  )%n , n);
       // down-left
       s_old(lxindex - RADIUS, lyindex +  BY  )= d_old ((i - RADIUS + n)%n  , (j  + BX )%n, n);
       // up-right
       s_old(lxindex + BX,lyindex - RADIUS  ) = d_old ((i + BX+n  )%n , (j -RADIUS + n )%n   , n );
     }
     // wait till all threads are finished
     __syncthreads();

     if((i<n)&&(j<n)){
       influence = 0; // weigted influence of the neighbors
       for(int ii=0; ii<WS ; ii++){
         for(int jj=0 ; jj<WS ; jj++){
           if((ii==2) && (jj==2))
           continue;
           // influence calculation
           influence += s_w(ii,jj) * s_old( threadIdx.x + ii,threadIdx.y +  jj );
         }
       }

       // magnetic moment gets the value of the SIGN of the weighted influence of its neighbors
       if(fabs(influence) < 10e-7){
         d_current(i,j,n) =  s_old(lxindex, lyindex); // remains the same in the case that the weighted influence is zero
       }
       else if(influence > 10e-7){
         d_current(i,j,n) = 1;
       }
       else if(influence < 0){
         d_current(i,j,n) = -1;
       }
     }
     __syncthreads();
   }
 }

}

void ising( int *G, double *w, int k, int n){

  dim3 block(BX,BY); // blockDim
  dim3 grid( (n+BX-1)/BX , (n+BY-1)/BY ); // gridDim

  int * old = (int*) malloc(n*n*(size_t)sizeof(int)); // old spin lattice
  int * current = (int*) malloc(n*n*(size_t)sizeof(int)); // current spin lattice
  if(old==NULL || current == NULL){
    printf("memory allocation failed (CPU)\n");
    exit(1);
  }

  // device variables
  int *d_old, *d_current, *tmp;
  double * d_w;
  if( cudaMalloc((void **)&d_old ,n*n*(size_t)sizeof(int)) != cudaSuccess  || cudaMalloc((void **)&d_current,n*n*(size_t)sizeof(int))   != cudaSuccess   || cudaMalloc((void **)&d_w, WS*WS*(size_t)sizeof(double))   != cudaSuccess){
    printf("memory allocation failed (GPU)\n");
    exit(1);
  }

  // copy host to device
  cudaMemcpy(d_w, w, WS*WS*sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy(d_old, G, n*n*sizeof(int), cudaMemcpyHostToDevice );

  // run for k steps
  for(int l=0; l<k; l++){

    // kernel execution
    kernel<<<grid,block>>>(d_current, d_old, d_w, n);
    cudaDeviceSynchronize();

    // copy device to host
    cudaMemcpy(old, d_old, n*n*sizeof(int), cudaMemcpyDeviceToHost );
    cudaMemcpy(current, d_current, n*n*sizeof(int), cudaMemcpyDeviceToHost );

    // save result in G
    memcpy(G , current , n*n*sizeof(int));

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
