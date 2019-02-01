#include <math.h>

//void getRowsNnzPerProc(int *rowsPerGpu, int *nnzPGPU, const int *global_n, const int *global_nnz,  const int *row_Ptr, const int nStreams)
void getRowsNnzPerProc(int *rowsPerGpu, const int *global_n, const int *global_nnz,  const int *row_Ptr, const int nStreams)
{
    float nnzIncre = (float ) *global_nnz/ (float) nStreams;
    float lookingFor=nnzIncre;
    int startRow=0, endRow;
    int partition=0;

    for (int row=0; row<*global_n; ++row) {    
        if ( (float) row_Ptr[row+1] >=  lookingFor ) { 
            // search for smallest difference
            if (fabs ( lookingFor - row_Ptr[row+1])  <= fabs ( lookingFor - row_Ptr[row])   ) {
                endRow = row;
            } else {
                endRow = row-1;
            } // end if //
            
            rowsPerGpu[partition] = endRow-startRow+1;
            //nnzPGPU[partition]  = row_Ptr[endRow+1] - row_Ptr[startRow];
             
            startRow = endRow+1;
            ++partition;
            if (partition < nStreams-1) {
               lookingFor += nnzIncre;
            } else {
                lookingFor=*global_nnz;
            } // end if //   
        } // end if // 
    } // end for //
} // end of getRowsPerProc //
