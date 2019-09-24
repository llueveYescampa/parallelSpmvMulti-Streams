//#include <stdio.h>
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
void alg1   (      real *__restrict__ const temp, 
             const real *__restrict__ const val, 
             const int  *__restrict__ const col_idx, 
             const int nnz_global
            )
{
    for (int tid = blockIdx.x*blockDim.x + threadIdx.x; tid<nnz_global; tid+=blockDim.x*gridDim.x) {
        temp[tid] = val[tid] * fetch_real( xTex, col_idx[tid]);
    } // end for //
} // end of alg1() //

__global__ 
void alg2   (real *__restrict__ const y, 
             const real *__restrict__ const temp,
             const int  *__restrict__ const row_Ptr,
             const int nRows,
             const real alpha,
             const real beta
            )
{
    __shared__ int limit;
    __shared__ int row_Ptr_s[MAXTHREADS+1];
    __shared__ real temp_s[SHARED_SIZE];
    
    int tid = threadIdx.x;
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (row < nRows) {
        y[row] = beta*y[row];
        
        row_Ptr_s[tid] = row_Ptr[row];
        if (tid == 0) {
            limit = (nRows-row <  MAXTHREADS) ? nRows-row : MAXTHREADS; 
            row_Ptr_s[limit] = row_Ptr[row+limit];
        } // end if //
        
        __syncthreads();
        
        real sum=0.0;
        int toLoad=row_Ptr_s[limit]-row_Ptr_s[0] ;
        for (int i=row_Ptr_s[0]; i < row_Ptr_s[limit]; i+=SHARED_SIZE) {
        
            int index = tid + i;
            
            __syncthreads();

            for (int j=0;  j <= SHARED_SIZE/limit; ++j) {
                if (tid + j*limit <  (toLoad < SHARED_SIZE ? toLoad :SHARED_SIZE) ) {
                    temp_s[tid + j*limit] = temp[index];
                    index +=limit;
                } // end if //
            } // end for //
            
            __syncthreads();
            
            if (  row_Ptr_s[tid+1] > i  && row_Ptr_s[tid] <= (i+SHARED_SIZE-1) ) {
                int r_s = (row_Ptr_s[tid] - i > 0) ? row_Ptr_s[tid] - i : 0;
                int r_e = (row_Ptr_s[tid+1] - i < SHARED_SIZE) ? row_Ptr_s[tid+1] - i : SHARED_SIZE;
                for (int j=r_s; j < r_e; ++j) {
                    sum += temp_s[j];
                } // end for //
            } // end if //
            
            y[row] = alpha * sum;
            toLoad-=SHARED_SIZE;
        }  // end for //
    } // end if //    
} // end of alg2() //

/*
        if (tid==0 ) {
            printf("tid: %d,blockIdx: %d, limit: %d [%d,%d]\n", tid, blockIdx.x, limit, row_Ptr_s[0], row_Ptr_s[limit]);
        }
*/        


/*
            if (tid==0 && blockIdx.x==0) {
                printf("index and toLoad : (%3d, %3d)---->  ", index, toLoad);
            }
*/


/*            
            if (tid==0 && blockIdx.x==0) {
                for (int j=0;  j < (toLoad < SHARED_SIZE ? toLoad :SHARED_SIZE); ++j) {
                    printf("%f, ", temp_s[j] );
                } // end for //            
                printf("\n");
            }
*/

        //if (blockIdx.x==gridDim.x-1) printf("%d, %d %d \n", tid,  row_Ptr_s[tid], row_Ptr_s[tid+1] );

