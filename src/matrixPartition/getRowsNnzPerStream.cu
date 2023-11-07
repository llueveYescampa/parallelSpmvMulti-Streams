#define LOW(n,p)  ((n)/(p))

void getRowsNnzPerStream(      unsigned int *__restrict__ const rowsPerSet, 
                        const unsigned int &global_n, 
                        const unsigned int &global_nnz,  
                        const unsigned int *__restrict__ const rows, 
                        const unsigned int &nRowBlocks)
{

  unsigned int lowRow=0, upRow;
  unsigned int reducedBlockSize = nRowBlocks;
  unsigned int reducedNnz = global_nnz;
  unsigned int nnzLimit = LOW(global_nnz,nRowBlocks);
  unsigned int partition=1;    

  for (int row = 0; row< global_n; ++row) {
    if ( rows[row+1] >=  nnzLimit ) {
      if ( static_cast<int>( rows[row+1] - nnzLimit)  <=  static_cast<int>(nnzLimit - rows[row])   ) {
        upRow = row;
      } else {
        upRow = row-1;
      } // end if //
      rowsPerSet[partition] = (upRow+1)-lowRow;
      reducedNnz -=  (rows[upRow+1]-rows[lowRow]);
      --reducedBlockSize;            
      lowRow=upRow+1;
      if (partition < nRowBlocks ) nnzLimit = rows[lowRow] + LOW(reducedNnz, reducedBlockSize);
      ++partition;
    } // end if //         
  } // end for //
    
  for(int s=0; s<nRowBlocks; ++s) {
    rowsPerSet[s+1] += rowsPerSet[s];
  } // end for //        
} // end of getRowsPerProc //
