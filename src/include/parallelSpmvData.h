    int n_global,nnz_global;
    int nStreams=1;
    int *starRow = NULL;

    const int basicSize = 32;
    const real parameter2Adjust = 0.5;

    // data for the on_proc solution
    int *row_ptr=NULL;
    int *col_idx=NULL;
    real *val=NULL;
    // end of data for the on_proc solution
    
    cudaStream_t *stream = NULL;
    
    real *meanNnzPerRow=NULL;
    real *sd=NULL;
    
    dim3 *block=NULL;
    dim3 *grid=NULL;

    size_t *sharedMemorySize=NULL;

