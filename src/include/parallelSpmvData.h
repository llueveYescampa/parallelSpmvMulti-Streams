//# define DEFAULT_STREAMS 4
//# define MAX_STREAMS 16

    int n_global,nnz_global;
    int nStreams=1;
    int nRowBlocks=1;
    int *starRowStream = NULL;
    int *starRowBlock = NULL;
    int *blockSize = NULL;

    const int warpSize = 32;
    const real parameter2Adjust = 0.15;

    // data for the on_proc solution
    int *row_ptr=NULL;
    int *col_idx=NULL;
    real *val=NULL;

    int *rows_d, *cols_d;
    real *vals_d;
    real *v_d, *w_d;

    // end of data for the on_proc solution
    
    
    
    cudaStream_t *stream = NULL;
    
    real meanNnzPerRow=0;
    real sd=0;
    
    dim3 *block=NULL;
    dim3 *grid=NULL;

    size_t *sharedMemorySize=NULL;

