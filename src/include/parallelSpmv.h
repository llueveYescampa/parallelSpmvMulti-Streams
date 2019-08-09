void reader(int *gn, int *gnnz, int *n,  
            int **rPtr,int **cIdx,real **v,
            const char *matrixFile, const int nStreams);

void vectorReader(real *v, const int *n, const char *vectorFile);
int createColIdxMap(int **b,  int *a, const int *n);

__global__ 
void spmv(real *__restrict__ y, 
          //real *__restrict__ x, 
          real *__restrict__ val,  
          int  *__restrict__ row_ptr, 
          int  *__restrict__ col_idx, 
          const int nRows,
          const real alpha,
          const real beta
          );

