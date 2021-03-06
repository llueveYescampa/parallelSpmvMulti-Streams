#include <stdio.h>
#include <stdlib.h>

#include "real.h"

void getRowsNnzPerStream(int *rowsPS, const int *global_n, const int *global_nnz, const int *row_Ptr, const int nStreams);

void reader( int *n_global, 
             int *nnz_global, 
             int *rowsPS,
             int **rowPtr, int **colIdx, real **val,
             const char *matrixFile,
             const int nStreams
             )
{
    
    
    FILE *filePtr;
    filePtr = fopen(matrixFile, "rb");
    
    // reading global nun rows //
    if ( !fread(n_global, sizeof(int), 1, filePtr) ) exit(0); 

    // reading global nnz //
    if ( !fread(nnz_global, sizeof(int), (size_t) 1, filePtr)) exit(0);

    // reading rowPtr //
    (*rowPtr) = (int *) malloc((*n_global+1)*sizeof(int));    
    //(*rowPtr)[0] = 0;
    // reading rows vector (n+1) values //
    if ( !fread(*rowPtr, sizeof(int), (size_t) (*n_global+1), filePtr)) exit(0);

    getRowsNnzPerStream((rowsPS+1),n_global,nnz_global, *rowPtr, nStreams);

    (*colIdx) = (int *)  malloc( (*nnz_global) * sizeof(int)); 
    // reading colIdx vector (nnz) values //
    if ( !fread(*colIdx, sizeof(int), (size_t) (*nnz_global), filePtr)) exit(0);

    (*val)    = (real *) malloc( (*nnz_global) * sizeof(real)); 
    // reading val vector (nnz) values //
    
    if (sizeof(real) == sizeof(double)) {
        if ( !fread(*val, sizeof(real), (size_t) (*nnz_global), filePtr)) exit(0);
    } else {
        double *temp = (double *) malloc(*nnz_global*sizeof(double)); 
        if ( !fread(temp, sizeof(double), (size_t) (*nnz_global), filePtr)) exit(0);
        for (int i=0; i<*nnz_global; i++) {
            (*val)[i] = (float) temp[i];
        } // end for //    
        free(temp);
    } // end if //
    
    for(int s=0; s<nStreams; ++s) {
        rowsPS[s+1] += rowsPS[s];
    } // end for //
    
    fclose(filePtr);
} // end of reader //
