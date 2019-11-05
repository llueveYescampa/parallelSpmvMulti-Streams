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
    extern __shared__ real temp[];
    int row,col;
/*
    if (blockDim.y==1) { 
        row = blockIdx.x*blockDim.x + threadIdx.x;
        if (row < nRows)  {
            real dot = (real) 0.0;
            for (col = row_ptr[row]; col < row_ptr[row+1]; ++col ) {
                //dot += (val[col] * x[col_idx[col]]);
                dot += (val[col] * fetch_real( xTex, col_idx[col])); 
            } // end for //
            y[row] = beta * y[row] + alpha*dot;
        } // end if //
        return ;
    } // end if //
*/

    row = blockIdx.x*blockDim.y + threadIdx.y;
    const unsigned int sharedMemIndx = blockDim.x*threadIdx.y + threadIdx.x;
    temp[sharedMemIndx] = (real) 0.0;

    if (row < nRows) {
        volatile real *temp1 = temp;
        
        switch((blockDim.x)) {
            case 1  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    temp[sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
#endif
                } // end for //
                y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                break; 
            case 2  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    temp[ sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
#endif
                } // end for //

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    y[row] =  beta * y[row] + alpha*(temp[sharedMemIndx]+temp[sharedMemIndx+1]) ;
                } // end if //   
                break; 
            case 4  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    temp[ sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
#endif
                } // end for //

                // unrolling warp 
                if (threadIdx.x < 2) {
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 2];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 1];
                } // end if //

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break; 
            case 8  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    temp[ sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
#endif
                } // end for //
              
                // unrolling warp 
                if (threadIdx.x < 4) {
                    temp1[sharedMemIndx] += temp1[sharedMemIndx +  4];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx +  2];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx +  1];
                } // end if //

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break; 
            case 16  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    temp[ sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
#endif
                } // end for //
              
                // unrolling warp 
                if (threadIdx.x < 8) {
                    temp1[sharedMemIndx] += temp1[sharedMemIndx +  8];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx +  4];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx +  2];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx +  1];
                } // end if //

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break;
                
            case 32  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    temp[ sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
#endif
                } // end for //
              
                // unrolling warp 
                if (threadIdx.x < 16) {
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 16];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx +  8];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx +  4];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx +  2];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx +  1];
                } // end if //

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break; 
            case 64  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    temp[ sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
#endif
                } // end for //
               __syncthreads();
               
                // unrolling warp 
                if (threadIdx.x < 32) {
                    temp[sharedMemIndx]  += temp[sharedMemIndx  + 32];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 16];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 8];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 4];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 2];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 1];
                } // end if //

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    //y[row] += temp[sharedMemIndx];
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break; 
            case 128  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    temp[ sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
#endif
                } // end for //
               __syncthreads();
               
                if (threadIdx.x<64) temp[sharedMemIndx] += temp[sharedMemIndx + 64];
                __syncthreads();
                
                // unrolling warp 
                if (threadIdx.x < 32) {
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 32];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 16];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 8];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 4];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 2];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 1];
                } // end if //

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    //y[row] += temp[sharedMemIndx];
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break; 
            case 256  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    temp[ sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
#endif
                } // end for //
               __syncthreads();

                if (threadIdx.x<128) temp[sharedMemIndx] += temp[sharedMemIndx + 128];
                __syncthreads();
               
                if (threadIdx.x<64) temp[sharedMemIndx] += temp[sharedMemIndx + 64];
                __syncthreads();
                
                // unrolling warp 
                if (threadIdx.x < 32) {
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 32];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 16];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 8];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 4];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 2];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 1];
                } // end if //

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    //y[row] += temp[sharedMemIndx];
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break; 
            case 512  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
#ifdef USE_TEXTURE
                    temp[ sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
#else
                    temp[sharedMemIndx] += (val[col] * x[col_idx[col]]);
#endif
                } // end for //
               __syncthreads();

                if (threadIdx.x<256) temp[sharedMemIndx] += temp[sharedMemIndx + 256];
                __syncthreads();

                if (threadIdx.x<128) temp[sharedMemIndx] += temp[sharedMemIndx + 128];
                __syncthreads();
               
                if (threadIdx.x<64) temp[sharedMemIndx] += temp[sharedMemIndx + 64];
                __syncthreads();
                
                // unrolling warp 
                if (threadIdx.x < 32) {
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 32];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 16];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 8];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 4];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 2];
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 1];
                } // end if //

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    //y[row] += temp[sharedMemIndx];
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break; 
        } // end switch //
    } // end if    
} // end of spmv() //
