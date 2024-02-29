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
        double fetch_floatType(cudaTextureObject_t texObject, int i)
        {
            int2 v = tex1Dfetch<int2>(texObject,i);
            return __hiloint2double(v.y, v.x);
        } // end of fetch_floatType() //
    #else
        static __inline__ __device__ 
        //float fetch_floatType(texture<float> t, int i)
        float fetch_floatType(cudaTextureObject_t texObject, int i)
        {
	        return tex1Dfetch<float>(texObject,i);
        } // end of fetch_double() //
    #endif
#endif

__global__ 
#ifdef USE_TEXTURE
void spmv(       floatType *__restrict__           y,
                 cudaTextureObject_t               xTex, 
           const floatType     *__restrict__ const val, 
           const unsigned int  *__restrict__ const row_ptr, 
           const unsigned int  *__restrict__ const col_idx, 
           const unsigned int                      nRows,
           const floatType                         alpha,
           const floatType                         beta
          )
#else
void spmv(       floatType     *__restrict__       y, 
           const floatType     *__restrict__ const x, 
           const floatType     *__restrict__ const val, 
           const unsigned int  *__restrict__ const row_ptr, 
           const unsigned int  *__restrict__ const col_idx, 
           const unsigned int                      nRows,
           const floatType                         alpha,
           const floatType                         beta
          )
#endif
{   
    extern __shared__ floatType temp[];
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const unsigned int sharedMemIndx = blockDim.x*threadIdx.y + threadIdx.x;
    auto sum = static_cast<floatType>(0.0);

    if (row < nRows) {
        for (int col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
          #ifdef USE_TEXTURE
          sum += (val[col] * fetch_floatType( xTex, col_idx[col]));
          //temp[sharedMemIndx] += (val[col] * fetch_floatType( xTex, col_idx[col]));
          #else
          sum += (val[col] * x[col_idx[col]]);
          //temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
          #endif
        } // end for //
        temp[sharedMemIndx] = sum;

        switch(blockDim.x) {
        /*
            case 1024  :
               __syncthreads();
                if (threadIdx.x<512) temp[sharedMemIndx] += temp[sharedMemIndx + 512]; 
         */
            case 512  :
               __syncthreads();
                if (threadIdx.x<256) temp[sharedMemIndx] += temp[sharedMemIndx + 256]; 
                
            case 256  :
               __syncthreads();
                if (threadIdx.x<128) temp[sharedMemIndx] += temp[sharedMemIndx + 128]; 
                
            case 128  :
               __syncthreads();
                if (threadIdx.x < 64) temp[sharedMemIndx] += temp[sharedMemIndx +  64];
                
            case 64  :
               __syncthreads();
                if (threadIdx.x < 32)  temp[sharedMemIndx] += temp[sharedMemIndx + 32];
                
            case 32  :
                if (threadIdx.x < 16)  temp[sharedMemIndx] += temp[sharedMemIndx + 16];
                
            case 16  :
                if (threadIdx.x <  8)  temp[sharedMemIndx] += temp[sharedMemIndx +  8];
        
            case 8  :
                if (threadIdx.x <  4)  temp[sharedMemIndx] += temp[sharedMemIndx +  4];
                
            case 4  :
                if (threadIdx.x <  2)  temp[sharedMemIndx] += temp[sharedMemIndx +  2];

            case 2  :
                if (threadIdx.x <  1)  temp[sharedMemIndx] += temp[sharedMemIndx +  1];
        } // end switch //
        if ( (sharedMemIndx % blockDim.x)  == 0) {
            //y[row] += temp[sharedMemIndx];
            y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
        } // end if //        
    } // end if    
    
} // end of spmv() //
