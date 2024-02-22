#ifndef PARALLELSPMV_H
#define PARALLELSPMV_H
#include "floatType.h"


#define USE_TEXTURE

    void reader(unsigned int &gn, 
                unsigned int &gnnz, 
                unsigned int **rPtr,
                unsigned int **cIdx,
                floatType **v,
                const char *const matrixFile);


    void vectorReader(floatType *const v, unsigned int &n, const char *const vectorFile);
    
    
    void getRowsNnzPerStream(      unsigned int *__restrict__ const rowsPerSet, 
                             const unsigned int &global_n, 
                             const unsigned int &global_nnz,  
                             const unsigned int *__restrict__ const rows, 
                             const unsigned int &nRowBlocks);
    
    __global__ 
    #ifdef USE_TEXTURE
        void spmv(      floatType *__restrict__           y, 
                         cudaTextureObject_t              xTex, 
                  const floatType     *__restrict__ const val,  
                  const unsigned int  *__restrict__ const row_ptr, 
                  const unsigned int  *__restrict__ const col_idx, 
                  const unsigned int                      nRows,
                  const floatType                         alpha,
                  const floatType                         beta
                  );
    #else
        void spmv(      floatType     *__restrict__       y, 
                  const floatType     *__restrict__ const x,
                  const floatType     *__restrict__ const val,  
                  const unsigned int  *__restrict__ const row_ptr, 
                  const unsigned int  *__restrict__ const col_idx, 
                  const unsigned int                      nRows,
                  const floatType                         alpha,
                  const floatType                         beta
                  );
                  
    #endif
#endif
