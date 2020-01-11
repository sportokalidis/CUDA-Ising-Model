#ifndef ISING
#define ISING

__global__
void kernel2D(int *d_current, int *d_old, double *d_w, int n);
void ising( int *G, double *w, int k, int n);


#endif
