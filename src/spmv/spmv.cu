//#include <stdio.h>
#include "real.h"
#include "parallelSpmv.h"

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


__global__ 
void ipcsr  (real *__restrict__ const y, 
             cudaTextureObject_t    xTex, 
             const real *__restrict__ const val, 
             const int  *__restrict__ const col_idx, 
             const int  *__restrict__ const row_Ptr,
             const int  *__restrict__ const blockRows_d, 
             const int  *__restrict__ const wtpb_d, 
             const int sizeBlockRows,
             const real alpha,
             const real beta
            )
{
    __shared__ real temp_s[SHARED_SIZE];
    __shared__ real bVal_s[MAXTHREADS];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int starRow=blockRows_d[bid];
    int endRow=blockRows_d[bid+1];
    int firstCol=row_Ptr[starRow];
    int nnz = row_Ptr[endRow] - row_Ptr[starRow];

    bVal_s[tid] = static_cast<real> (0);
    
    if (nnz<=SHARED_SIZE  && (endRow-starRow) != 1 ) {
        for (int i=tid; i<nnz; i+=blockDim.x) {
            temp_s[i] = val[firstCol+i] * fetch_real( xTex, col_idx[firstCol+i]);
        } // end for //
        __syncthreads();

        int wtpr = wtpb_d[bid]; 
        if (wtpr<2) {
            if (tid < (endRow-starRow) ) {
                real sum = 0;
                int row_s = row_Ptr[starRow+tid]   - firstCol;
                int row_e = row_Ptr[starRow+tid+1] - firstCol;
                
                for (int i=row_s; i < row_e; ++i) {
                    sum +=temp_s[i];
                } // end for //
                y[starRow+tid] =  beta*y[starRow+tid] +  alpha * sum;
            } // end if //
        } else {
            int workingRow=tid/wtpr;
            int line = tid % wtpr;

            if (tid < (endRow-starRow) * wtpr ) {
                real sum = static_cast<real> (0);
                int row_s = row_Ptr[starRow + workingRow ]    - firstCol + line;
                int row_e = row_Ptr[starRow + workingRow + 1] - firstCol;
                
                for (int i=row_s; i < row_e; i+=wtpr) {
                    sum +=temp_s[i];
                } // end for //
                bVal_s[tid] = sum;
                __syncthreads();

                if (line < 256 && wtpr >=512) {
                    bVal_s[tid] += bVal_s[tid+256];
                    __syncthreads();
                } // end if //

                if (line < 128 && wtpr >=256) {
                    bVal_s[tid] += bVal_s[tid+128];
                    __syncthreads();
                } // end if //

                if (line < 64 && wtpr >=128) {
                    bVal_s[tid] += bVal_s[tid+64];
                    __syncthreads();
                } // end if //

                
                if (line < 32 && wtpr >=64) {
                    bVal_s[tid] += bVal_s[tid+32];
                    __syncthreads();
                } // end if //

                volatile real *shem = bVal_s;

                if (line < 16 && wtpr >=32) {
                    shem[tid] += shem[tid+16];
                    //__syncthreads();
                } // end if //
                
            
                if (line < 8 && wtpr >=16) {
                    shem[tid] += shem[tid+8];
                    //__syncthreads();
                } // end if //
                

                if (line < 4 && wtpr >=8) {
                    shem[tid] += shem[tid+4];
                    //__syncthreads();
                } // end if //


                if (line < 2 && wtpr >=4) {
                    shem[tid] += shem[tid+2];
                    //__syncthreads();
                } // end if //
        
                if (line < 1 && wtpr >=2) {
                    y[starRow+workingRow] = beta*y[starRow+workingRow]  + alpha * (bVal_s[tid] + bVal_s[tid+1] );
                } // end if //

            } // end if //
        } // endif    
    } else {
        real sum=static_cast<real>(0);
        for (int i=tid; i<nnz; i+=blockDim.x) {
            sum += val[firstCol+i] * fetch_real( xTex, col_idx[firstCol+i]);
        } // end for //
        bVal_s[tid]=sum;
        
        __syncthreads();

        if (tid < 256 && blockDim.x > 256) {
            bVal_s[tid] += bVal_s[tid+256];
            __syncthreads();
        } // end if //


        if (tid < 128 && blockDim.x > 128) {
            bVal_s[tid] += bVal_s[tid+128];
            __syncthreads();
        } // end if //

        if (tid < 64 && blockDim.x > 64) {
            bVal_s[tid] += bVal_s[tid+64];
            __syncthreads();
        } // end if //
        
        if (tid < 32 && blockDim.x > 32) {
            bVal_s[tid] += bVal_s[tid+32];
            __syncthreads();
        } // end if //
        volatile real *shem = bVal_s;
        if (tid < 16) {
            shem[tid] += shem[tid+16];
            //__syncthreads();
        } // end if //
        
        if (tid < 8) {
            shem[tid] += shem[tid+8];
            //__syncthreads();
        } // end if //
        
        if (tid < 4) {
            shem[tid] += shem[tid+4];
            //__syncthreads();
        } // end if //
        
        if (tid < 2) {
            shem[tid] += shem[tid+2];
            //__syncthreads();
        } // end if //
        
        if (tid < 1) {
            y[starRow] = beta*y[starRow]  + alpha * (bVal_s[tid] + bVal_s[tid+1] );
        } // end if //
    } // end if //
} // end of ipcsr() //
