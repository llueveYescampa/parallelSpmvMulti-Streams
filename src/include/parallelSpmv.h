#ifndef PARALLELSPMV_H
#define PARALLELSPMV_H
#include "real.h"

#define USE_TEXTURE

void reader(int *gn, int *gnnz, int *n,  
            int **rPtr,int **cIdx,real **v,
            const char *matrixFile, const int nStreams);

void vectorReader(real *v, const int *n, const char *vectorFile);
int createColIdxMap(int **b,  int *a, const int *n);

__global__ 
#ifdef USE_TEXTURE
void spmv(      real *__restrict__       y, 
          const real *__restrict__ const val,  
          const int  *__restrict__ const row_ptr, 
          const int  *__restrict__ const col_idx, 
          const int nRows,
          const real alpha,
          const real beta
          );
#else
void spmv(      real *__restrict__      y, 
          const real *__restrict__ const x,
          const real *__restrict__ const val,  
          const int  *__restrict__ const row_ptr, 
          const int  *__restrict__ const col_idx, 
          const int nRows,
          const real alpha,
          const real beta
          );
#endif
#endif
