#include <math.h>

void getRowsNnzPerStream(int *rowsPerSstream, const int *global_n, const int *global_nnz,  const int *row_Ptr, const int nStreams)
{
    double nnzIncre = (double) *global_nnz/ (double) nStreams;
    double lookingFor=nnzIncre;
    int startRow=0, endRow;
    int partition=0;

    for (int row=0; row<*global_n; ++row) {    
        if ( (double) row_Ptr[row+1] >=  lookingFor ) { 
            // search for smallest difference
            if (fabs ( lookingFor - row_Ptr[row+1])  <= fabs ( lookingFor - row_Ptr[row])   ) {
                endRow = row;
            } else {
                endRow = row-1;
            } // end if //
            
            rowsPerSstream[partition] = endRow-startRow+1;
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
