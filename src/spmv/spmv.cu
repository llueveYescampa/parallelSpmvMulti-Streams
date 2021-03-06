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

template <const unsigned int bs>
__device__ void warpReduce(volatile real * const temp1, const unsigned int sharedMemIndx)
{
    // unrolling warp 
    if (bs >= 64) temp1[sharedMemIndx] += temp1[sharedMemIndx + 32];
    if (bs >= 32) temp1[sharedMemIndx] += temp1[sharedMemIndx + 16];
    if (bs >= 16) temp1[sharedMemIndx] += temp1[sharedMemIndx +  8];
    if (bs >=  8) temp1[sharedMemIndx] += temp1[sharedMemIndx +  4];
    if (bs >=  4) temp1[sharedMemIndx] += temp1[sharedMemIndx +  2];
    if (bs >=  2) temp1[sharedMemIndx] += temp1[sharedMemIndx +  1];
} // end of warpReduce() //

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
    extern __shared__ real temp[];
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const unsigned int sharedMemIndx = blockDim.x*threadIdx.y + threadIdx.x;
    temp[sharedMemIndx] = static_cast<real>(0.0);

    if (row < nRows) {
        switch(blockDim.x) {
            case 1  :
                for (int col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
                    #ifdef USE_TEXTURE
                    temp[sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
                    #else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
                    #endif
                } // end for //

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    //y[row] += temp[sharedMemIndx];
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break;
            case 2  :
                for (int col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
                    #ifdef USE_TEXTURE
                    temp[sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
                    #else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
                    #endif
                } // end for //
                if (threadIdx.x <  1) warpReduce< 2>(temp,sharedMemIndx);

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    //y[row] += temp[sharedMemIndx];
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break; 
            case 4  :
                for (int col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
                    #ifdef USE_TEXTURE
                    temp[sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
                    #else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
                    #endif
                } // end for //
                if (threadIdx.x <  2) warpReduce< 4>(temp,sharedMemIndx);

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    //y[row] += temp[sharedMemIndx];
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break; 
            case 8  :
                for (int col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
                    #ifdef USE_TEXTURE
                    temp[sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
                    #else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
                    #endif
                } // end for //
                if (threadIdx.x <  4) warpReduce< 8>(temp,sharedMemIndx);

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    //y[row] += temp[sharedMemIndx];
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break; 
            case 16  :
                for (int col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
                    #ifdef USE_TEXTURE
                    temp[sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
                    #else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
                    #endif
                } // end for //
                if (threadIdx.x <  8) warpReduce<16>(temp,sharedMemIndx);

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    //y[row] += temp[sharedMemIndx];
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break;
            case 32  :
                for (int col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
                    #ifdef USE_TEXTURE
                    temp[sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
                    #else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
                    #endif
                } // end for //
                if (threadIdx.x < 16) warpReduce<32>(temp,sharedMemIndx);

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    //y[row] += temp[sharedMemIndx];
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break; 
            case 64  :
                for (int col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
                    #ifdef USE_TEXTURE
                    temp[sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
                    #else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
                    #endif
                } // end for //
               __syncthreads();
                if (threadIdx.x < 32) warpReduce<64>(temp,sharedMemIndx);

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    //y[row] += temp[sharedMemIndx];
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break; 
            case 128  :
                for (int col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
                    #ifdef USE_TEXTURE
                    temp[sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
                    #else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
                    #endif
                } // end for //
               __syncthreads();
                if (threadIdx.x < 64) { temp[sharedMemIndx] += temp[sharedMemIndx +  64]; __syncthreads(); }
                if (threadIdx.x < 32) warpReduce<64>(temp,sharedMemIndx);

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    //y[row] += temp[sharedMemIndx];
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break; 
            case 256  :
                for (int col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
                    #ifdef USE_TEXTURE
                    temp[sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
                    #else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
                    #endif
                } // end for //
               __syncthreads();
                if (threadIdx.x<128) { temp[sharedMemIndx] += temp[sharedMemIndx + 128]; __syncthreads(); }
                if (threadIdx.x< 64) { temp[sharedMemIndx] += temp[sharedMemIndx +  64]; __syncthreads(); }
                if (threadIdx.x < 32) warpReduce<64>(temp,sharedMemIndx);

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    //y[row] += temp[sharedMemIndx];
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break; 
            case 512  :
                for (int col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
                    #ifdef USE_TEXTURE
                    temp[sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
                    #else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
                    #endif
                } // end for //
               __syncthreads();
                if (threadIdx.x<256) { temp[sharedMemIndx] += temp[sharedMemIndx + 256]; __syncthreads(); }
                if (threadIdx.x<128) { temp[sharedMemIndx] += temp[sharedMemIndx + 128]; __syncthreads(); }
                if (threadIdx.x< 64) { temp[sharedMemIndx] += temp[sharedMemIndx +  64]; __syncthreads(); }
                if (threadIdx.x < 32) warpReduce<64>(temp,sharedMemIndx);

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    //y[row] += temp[sharedMemIndx];
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break; 
        } // end switch //
    } // end if    
} // end of spmv() //
