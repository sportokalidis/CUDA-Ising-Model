/*
*       Parallel and Distributed Systems
*       Exercise 3
*       V0. Sequential
*       Authors:
*         Portokalidis Stavros, AEM 9334, stavport@ece.auth.gr
*         Christoforidis Savvas, AEM 9147, schristofo@ece.auth.gr
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// window size
#define WS 5

// replace pointer types
#define old(i,j,n) *(old+(i)*n+(j))
#define current(i,j,n) *(current+(i)*n+(j))
#define w(i,j) *(w+(i)*WS+(j))
#define G(i,j,n) *(G+(i)*n+(j))

void ising( int *G, double *w, int k, int n){

  int * old = (int*) malloc(n*n*sizeof(int)); // old spin lattice
  int * current = (int*) malloc(n*n*sizeof(int)); // current spin lattice
  int * tmp;
  if( old==NULL || current==NULL){
    printf("memory allocation failed\n");
    exit(1);
  }
  double influence; // weighted influence of the neighbors


  // initial values saved to old
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
        old(i,j,n) = G(i,j,n);
    }
  }

  // run for k steps
  for(int l=0; l<k; l++){
    // for each current[i][j] point
    for(int i=0; i<n; i++){
      for(int j=0; j<n; j++){

        // calculation of weighted influence
        influence = 0;
        for(int ii=0; ii<WS; ii++){
          for(int jj=0; jj<WS; jj++){
            influence +=  w(ii,jj) * old((i-2+n+ii)%n, (j-2+n+jj)%n, n);
          }
        }

        // magnetic moment gets the value of the SIGN of the weighted influence of its neighbors
        if(fabs(influence) < 10e-7){
          current(i,j,n) = old(i,j,n); // remains the same in the case that the weighted influence is zero
        }
        else if(influence > 10e-7){
          current(i,j,n) = 1;
        }
        else if(influence < 0){
          current(i,j,n) = -1;
        }
        // save result in G
        G(i,j,n) = current(i,j,n);
      }
    }

    // swap the pointers for the next iteration
    tmp = old;
    old = current;
    current = tmp;

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

  free(old);
  free(current);
}


int main(int argc, const char* argv[]){

  int n = 517;
  double weights[] = {0.004, 0.016, 0.026, 0.016, 0.004, 0.016, 0.071, 0.117, 0.071, 0.016, 0.026, 0.117, 0.0, 0.117, 0.026, 0.016, 0.071, 0.117, 0.071, 0.016, 0.004, 0.016, 0.026, 0.016, 0.004};
  int G1[n*n]; // G that changes k times
  int G2[n*n]; // G that changes k times
  int G3[n*n]; // G that changes k times
  int Gk1[n*n];
  int Gk2[n*n];
  int Gk3[n*n];
  FILE *ptr;

  // read initial G
  ptr = fopen("./conf-init.bin","rb");
  size_t arxeio = fread(G1,sizeof(G1),1,ptr);
  fclose(ptr);
  // read initial G
  ptr = fopen("./conf-init.bin","rb");
  arxeio = fread(G2,sizeof(G2),1,ptr);
  fclose(ptr);
  // read initial G
  ptr = fopen("./conf-init.bin","rb");
  arxeio = fread(G3,sizeof(G3),1,ptr);
  fclose(ptr);
  // read k-th Gk
  ptr = fopen("./conf-1.bin","rb"); // allazo onoma arxeiou gia allagi k
  arxeio = fread(Gk1,sizeof(Gk1),1,ptr);
  fclose(ptr);

  ptr = fopen("./conf-4.bin","rb"); // allazo onoma arxeiou gia allagi k
  arxeio = fread(Gk2,sizeof(Gk2),1,ptr);
  fclose(ptr);

  ptr = fopen("./conf-11.bin","rb"); // allazo onoma arxeiou gia allagi k
  arxeio = fread(Gk3,sizeof(Gk3),1,ptr);
  fclose(ptr);

  // execution
  ising(G1, weights, 1, n);

  // check correctness
  int c = 0;
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      if( *(G1+i*n+j) != *(Gk1+i*n+j) ){
        c++;
      }
    }
  }

  if(c!=0){
    printf("k=1 Wrong\n");
  }
  else{
    printf("k=1 Correct\n");
  }

  // execution
  ising(G2, weights, 4, n);

  // check correctness
   c = 0;
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      if( *(G2+i*n+j) != *(Gk2+i*n+j) ){
        c++;
      }
    }
  }
  if(c!=0){
    printf("k=4 Wrong\n");
  }
  else{
    printf("k=4 Correct\n");
  }

  // execution
  ising(G3, weights, 11 , n);

  // check correctness
   c = 0;
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      if( *(G3+i*n+j) != *(Gk3+i*n+j) ){
        c++;
      }
    }
  }
  if(c!=0){
    printf("k=11 Wrong\n");
  }
  else{
    printf("k=11 Correct\n");
  }


  return 0;
}
