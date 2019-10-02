
    int n_global,nnz_global;
    int *starRow = NULL;


    // data for the on_proc solution
    int *row_ptr=NULL;
    int *col_idx=NULL;
    real *val=NULL;

    int *rows_d, *cols_d;
    real *vals_d;
    int *blockRows_d;
    real *v_d, *w_d;

    // end of data for the on_proc solution
    
    dim3 block;
    dim3 grid;

