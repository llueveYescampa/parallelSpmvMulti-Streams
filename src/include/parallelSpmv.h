#define MAXTHREADS 256
#define SHARED_SIZE 768
#ifndef PARALLELSPMV
#define PARALLELSPMV

void reader(int *gn, int *gnnz, int *n,  
            int **rPtr,int **cIdx,real **v,
            const char *matrixFile, const int nStreams);

void vectorReader(real *v, const int *n, const char *vectorFile);
int createColIdxMap(int **b,  int *a, const int *n);


__global__ 
void alg1   (      real *__restrict__ const temp, 
             const real *__restrict__ const val, 
             const int  *__restrict__ const col_idx, 
             const int nnz_global  
            );


__global__ 
void ipcsr  (real *__restrict__ const y, 
             const real *__restrict__ const val, 
             const int  *__restrict__ const col_idx, 
             const int  *__restrict__ const row_Ptr,
             const int  *__restrict__ const blockRows_d, 
             const int  *__restrict__ const wtpb_d, 
             const int sizeBlockRows,
             const real alpha,
             const real beta
            );
#endif
