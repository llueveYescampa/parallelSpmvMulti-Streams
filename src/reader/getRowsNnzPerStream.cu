#define LOW(id,p,n)  ((id)*(n)/(p))
#define HIGH(id,p,n) (LOW((id)+1,p,n)-1)
#define SIZE(id,p,n) (LOW((id)+1,p,n)-LOW(id,p,n)) // eblack

void getRowsNnzPerStream(int *rowsPerSet, const int *global_n, const int *global_nnz,  const int *rows, const int nRowBlocks)
{
    int lowRow=0, upRow;
    int reducedBlockSize= nRowBlocks;
    int reducedNnz=*global_nnz;
    int nnzLimit = rows[lowRow] + SIZE(0,reducedBlockSize, reducedNnz);
    int partition=0;    

    for (int row = 0; row<*global_n; ++row) {
        if ( rows[row+1] >=  nnzLimit ) { 
            if ( ( rows[row+1] - nnzLimit)  <=  nnzLimit - rows[row]   ) {
                upRow = row;
            } else {
                upRow = row-1;
            } // end if //
            rowsPerSet[partition] = upRow-lowRow+1;
            reducedNnz -=  (rows[upRow+1]-rows[lowRow]);
            --reducedBlockSize;
            lowRow=upRow+1;
            if (partition < nRowBlocks-1 ) nnzLimit= rows[lowRow] + SIZE(0,reducedBlockSize, reducedNnz);
            ++partition;
        } // end if // 
        
    } // end for //
} // end of getRowsPerProc //

/*

    double nnzIncre = (double) *global_nnz/ (double) nRowBlocks;
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
            
            rowsPerStream[partition] = endRow-startRow+1;
            //nnzPGPU[partition]  = row_Ptr[endRow+1] - row_Ptr[startRow];
             
            startRow = endRow+1;
            ++partition;
            if (partition < nRowBlocks-1) {
               lookingFor += nnzIncre;
            } else {
                lookingFor=*global_nnz;
            } // end if //   
        } // end if // 
    } // end for //

*/

