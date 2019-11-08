//#include <stdio.h>
#include "parallelSpmv.h"

#ifdef USE_TEXTURE
/*
    #ifdef DOUBLE
        extern texture<int2> xTex;
        //extern texture<int2> valTex;
    #else
        extern texture<float> xTex;
        //extern texture<float> valTex;
    #endif
*/
    #ifdef DOUBLE
        static __inline__ __device__ 
        double fetch_real(cudaTextureObject_t texObject, int i)
        {
            int2 v = tex1Dfetch<int2>(texObject,i);
            return __hiloint2double(v.y, v.x);
        } // end of fetch_real() //
    #else
        static __inline__ __device__ 
        //float fetch_real(texture<float> t, int i)
        float fetch_real(cudaTextureObject_t texObject, int i)
        {
	        return tex1Dfetch<float>(texObject,i);
        } // end of fetch_double() //
    #endif
#endif

__global__ 
#ifdef USE_TEXTURE
void spmv(       real *__restrict__       y,
                 cudaTextureObject_t    xTex, 
           const real *__restrict__ const val, 
           const int  *__restrict__ const row_ptr, 
           const int  *__restrict__ const col_idx, 
           const int nRows,
           const real alpha,
           const real beta
          )
#else
void spmv(       real *__restrict__       y, 
           const real *__restrict__ const x, 
           const real *__restrict__ const val, 
           const int  *__restrict__ const row_ptr, 
           const int  *__restrict__ const col_idx, 
           const int nRows,
           const real alpha,
           const real beta
          )
#endif
{   
    extern __shared__ real shared[]; 


    real *temp = shared; // borrar despues


    int row,col;
    row = blockIdx.x*blockDim.y + threadIdx.y;
    real answer = static_cast<real>(0.0);

    const int warpSize = (blockDim.x < 32) ? blockDim.x: 32;
    int thread_0 = threadIdx.x % warpSize;
    int warpID =  threadIdx.x / warpSize;
    int nWarps = blockDim.x/warpSize;

    if (row < nRows) {
        switch((blockDim.x)) {
            case 1  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    answer += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    answer += (val[col] * x[col_idx[col]]);
#endif
                } // end for //
                y[row] =  beta * y[row] + alpha*answer;
                break; 
            case 2  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    answer += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    answer += (val[col] * x[col_idx[col]]);
#endif
                } // end for //

                answer +=   __shfl_down_sync(0xffffffff, answer, 1);

                if (threadIdx.x == 0) {
                    y[row] =  beta * y[row] + alpha*answer;
                } // end if //   
                break; 
            case 4  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    answer += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    answer += (val[col] * x[col_idx[col]]);
#endif
                } // end for //

                answer +=   __shfl_down_sync(0xffffffff, answer, 2);
                answer +=   __shfl_down_sync(0xffffffff, answer, 1);

                if (threadIdx.x == 0) {
                    y[row] =  beta * y[row] + alpha*answer;
                } // end if //   
                break; 
            case 8  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    answer += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    answer += (val[col] * x[col_idx[col]]);
#endif
                } // end for //
              
                answer +=   __shfl_down_sync(0xffffffff, answer, 4);
                answer +=   __shfl_down_sync(0xffffffff, answer, 2);
                answer +=   __shfl_down_sync(0xffffffff, answer, 1);

                if (threadIdx.x == 0) {
                    y[row] =  beta * y[row] + alpha*answer;
                } // end if //   
                break; 
            case 16  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    answer += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    answer += (val[col] * x[col_idx[col]]);
#endif
                } // end for //
              
                answer +=   __shfl_down_sync(0xffffffff, answer, 8);
                answer +=   __shfl_down_sync(0xffffffff, answer, 4);
                answer +=   __shfl_down_sync(0xffffffff, answer, 2);
                answer +=   __shfl_down_sync(0xffffffff, answer, 1);

                if (threadIdx.x == 0) {
                    y[row] =  beta * y[row] + alpha*answer;
                } // end if //   
                break;
                
            case 32  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    answer += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    answer += (val[col] * x[col_idx[col]]);
#endif
                } // end for //
              
                answer +=   __shfl_down_sync(0xffffffff, answer, 16);
                answer +=   __shfl_down_sync(0xffffffff, answer, 8);
                answer +=   __shfl_down_sync(0xffffffff, answer, 4);
                answer +=   __shfl_down_sync(0xffffffff, answer, 2);
                answer +=   __shfl_down_sync(0xffffffff, answer, 1);

                if (threadIdx.x == 0) {
                    y[row] =  beta * y[row] + alpha*answer;
                } // end if //   
                break; 
            case 64  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    answer += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    answer += (val[col] * x[col_idx[col]]);
#endif
                } // end for //
              
                answer +=   __shfl_down_sync(0xffffffff, answer, 16);
                answer +=   __shfl_down_sync(0xffffffff, answer, 8);
                answer +=   __shfl_down_sync(0xffffffff, answer, 4);
                answer +=   __shfl_down_sync(0xffffffff, answer, 2);
                answer +=   __shfl_down_sync(0xffffffff, answer, 1);

                if (thread_0 == 0) shared[threadIdx.y*nWarps + warpID] = answer;
                __syncthreads();
                answer = (threadIdx.x < nWarps ) ? shared[threadIdx.y*nWarps + thread_0] : 0.0;
                if (warpID == 0) {
                    answer +=   __shfl_down_sync(0xffffffff, answer, 1);
                } // end if //
                
                if (threadIdx.x == 0) {
                    y[row] =  beta * y[row] + alpha*answer;
                } // end if //   
                break; 
            case 128  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    answer += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    answer += (val[col] * x[col_idx[col]]);
#endif
                } // end for //
              
                answer +=   __shfl_down_sync(0xffffffff, answer, 16);
                answer +=   __shfl_down_sync(0xffffffff, answer, 8);
                answer +=   __shfl_down_sync(0xffffffff, answer, 4);
                answer +=   __shfl_down_sync(0xffffffff, answer, 2);
                answer +=   __shfl_down_sync(0xffffffff, answer, 1);

                if (thread_0 == 0) shared[threadIdx.y*nWarps + warpID] = answer;
                __syncthreads();
                answer = (threadIdx.x < nWarps ) ? shared[threadIdx.y*nWarps + thread_0] : 0.0;
                if (warpID == 0) {
                    answer +=   __shfl_down_sync(0xffffffff, answer, 2);
                    answer +=   __shfl_down_sync(0xffffffff, answer, 1);
                } // end if //
                
                if (threadIdx.x == 0) {
                    y[row] =  beta * y[row] + alpha*answer;
                } // end if //   
                break; 
            case 256  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    answer += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    answer += (val[col] * x[col_idx[col]]);
#endif
                } // end for //
              
                answer +=   __shfl_down_sync(0xffffffff, answer, 16);
                answer +=   __shfl_down_sync(0xffffffff, answer, 8);
                answer +=   __shfl_down_sync(0xffffffff, answer, 4);
                answer +=   __shfl_down_sync(0xffffffff, answer, 2);
                answer +=   __shfl_down_sync(0xffffffff, answer, 1);

                if (thread_0 == 0) shared[threadIdx.y*nWarps + warpID] = answer;
                __syncthreads();
                answer = (threadIdx.x < nWarps ) ? shared[threadIdx.y*nWarps + thread_0] : 0.0;
                if (warpID == 0) {
                    answer +=   __shfl_down_sync(0xffffffff, answer, 4);
                    answer +=   __shfl_down_sync(0xffffffff, answer, 2);
                    answer +=   __shfl_down_sync(0xffffffff, answer, 1);
                } // end if //
                
                if (threadIdx.x == 0) {
                    y[row] =  beta * y[row] + alpha*answer;
                } // end if //   
                break; 
            case 512  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    answer += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    answer += (val[col] * x[col_idx[col]]);
#endif
                } // end for //
              
                answer +=   __shfl_down_sync(0xffffffff, answer, 16);
                answer +=   __shfl_down_sync(0xffffffff, answer, 8);
                answer +=   __shfl_down_sync(0xffffffff, answer, 4);
                answer +=   __shfl_down_sync(0xffffffff, answer, 2);
                answer +=   __shfl_down_sync(0xffffffff, answer, 1);

                if (thread_0 == 0) shared[threadIdx.y*nWarps + warpID] = answer;
                __syncthreads();
                answer = (threadIdx.x < nWarps ) ? shared[threadIdx.y*nWarps + thread_0] : 0.0;
                if (warpID == 0) {
                    answer +=   __shfl_down_sync(0xffffffff, answer, 8);
                    answer +=   __shfl_down_sync(0xffffffff, answer, 4);
                    answer +=   __shfl_down_sync(0xffffffff, answer, 2);
                    answer +=   __shfl_down_sync(0xffffffff, answer, 1);
                } // end if //
                
                if (threadIdx.x == 0) {
                    y[row] =  beta * y[row] + alpha*answer;
                } // end if //   
                break; 
        } // end switch //
    } // end if    
} // end of spmv() //
