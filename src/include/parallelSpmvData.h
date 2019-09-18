# define DEFAULT_STREAMS 4
# define MAX_STREAMS 16

    int n_global,nnz_global;
    int nStreams=DEFAULT_STREAMS;
    int *starRow = NULL;


    // data for the on_proc solution
    int *row_ptr=NULL;
    int *col_idx=NULL;
    real *val=NULL;

    int *rows_d, *cols_d;
    real *vals_d;
    real *temp;
    real *v_d, *w_d;

    // end of data for the on_proc solution
    
    
    
    cudaStream_t *stream = NULL;
    
    real *meanNnzPerRow=NULL;
    real *sd=NULL;
    
    dim3 *block=NULL;
    dim3 *grid=NULL;

    size_t *sharedMemorySize=NULL;

