#include <stdio.h>
#include "real.h"

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
void spmv(real *__restrict__ y, 
           //real *__restrict__ x, 
           real *__restrict__ val, 
           int  *__restrict__ row_ptr, 
           int  *__restrict__ col_idx, 
           const int nRows,
           const real alpha,
           const real beta
          )
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
    volatile real *temp1 = temp;

    if (row < nRows) {
        switch((blockDim.x)) {
            case 1  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
                    //temp[threadIdx.x] += (val[col] * x[col_idx[col]]);
                    temp[ sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
                } // end for //
                y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                break; 
            case 2  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
                    //temp[threadIdx.x] += (val[col] * x[col_idx[col]]);
                    temp[ sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
                } // end for //

                // unrolling warp 
                if (threadIdx.x < 1) {
                    temp1[sharedMemIndx] += temp1[sharedMemIndx + 1];
                } // end if //

                if ((sharedMemIndx % blockDim.x)  == 0) {
                    y[row] =  beta * y[row] + alpha*temp[sharedMemIndx];
                } // end if //   
                break; 
            case 4  :
                for (col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
                    //temp[threadIdx.x] += (val[col] * x[col_idx[col]]);
                    temp[ sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
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
                    //temp[threadIdx.x] += (val[col] * x[col_idx[col]]);
                    temp[ sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
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
                    //temp[threadIdx.x] += (val[col] * x[col_idx[col]]);
                    temp[ sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
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
                    //temp[threadIdx.x] += (val[col] * x[col_idx[col]]);
                    temp[ sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
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
                    //temp[threadIdx.x] += (val[col] * x[col_idx[col]]);
                    temp[ sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
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
                    //temp[threadIdx.x] += (val[col] * x[col_idx[col]]);
                    temp[ sharedMemIndx] += (val[col] * fetch_real( xTex, col_idx[col]));
                } // end for //
               __syncthreads();
               
                if (threadIdx.x<64) temp[sharedMemIndx] += temp[sharedMemIndx + 64];
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
        } // end switch //
    } // end if    
} // end of spmv() //
