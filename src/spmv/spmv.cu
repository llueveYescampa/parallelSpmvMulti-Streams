#include <stdio.h>
#include "real.h"
#include "parallelSpmv.h"

#ifdef DOUBLE
    extern texture<int2> xTex;
    //extern texture<int2> valTex;
#else
    extern texture<float> xTex;
    //extern texture<float> valTex;
#endif

#ifdef DOUBLE
    static __inline__ __device__ 
    double fetch_real(texture<int2> t, int i)
    {
	    int2 v = tex1Dfetch(t,i);
	    return __hiloint2double(v.y, v.x);
    } // end of fetch_real() //
#else
    static __inline__ __device__ 
    float fetch_real(texture<float> t, int i)
    {
	    return tex1Dfetch(t,i);
    } // end of fetch_double() //
#endif
__global__ 
void alg3   (real *__restrict__ const y, 
             const real *__restrict__ const val, 
             const int  *__restrict__ const col_idx, 
             const int  *__restrict__ const row_Ptr,
             const int  *__restrict__ const blockRows_d, 
             const int sizeBlockRows,
             const real alpha,
             const real beta
            )
{
    __shared__ real temp_s[SHARED_SIZE];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int starRow=blockRows_d[bid];
    int endRow=blockRows_d[bid+1];
    int firstCol=row_Ptr[starRow];
    int nnz = row_Ptr[endRow] - row_Ptr[starRow];
    
    for (int i=tid; i<nnz; i+=blockDim.x) {
        temp_s[i] = val[firstCol+i] * fetch_real( xTex, col_idx[firstCol+i]);
    } // end for //
    __syncthreads();

    if (tid < (endRow-starRow) ) {
        real sum = 0;
        int row_s = row_Ptr[starRow+tid]   - firstCol;
        int row_e = row_Ptr[starRow+tid+1] - firstCol;
        
        for (int i=row_s; i < row_e; ++i) {
            sum +=temp_s[i];
        } // end for //
        y[starRow+tid] =  beta*y[starRow+tid] +  alpha * sum;
    } // end if //

} // end of alg3() //
