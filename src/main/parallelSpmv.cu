#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "real.h"
#include "parallelSpmv.h"

#define FATAL(msg) \
    do {\
        fprintf(stderr, "[%s:%d] %s\n", __FILE__, __LINE__, msg);\
        exit(-1);\
    } while(0)

#define REP 1000

#ifdef DOUBLE
    texture<int2>  xTex;
    //texture<int2>  valTex;
#else
    texture<float> xTex;
    //texture<float> valTex;
#endif

int main(int argc, char *argv[]) 
{
    if (MAXTHREADS > 512) {
        printf("need to adjust the ipcsr() function to acomodate more than 512 threads per block\nQuitting ....\n");
        exit(-1);
    } // end if //
    #include "parallelSpmvData.h"

    // verifing number of input parameters //
   char exists='t';
   char checkSol='f';
    
    if (argc < 3 ) {
        printf("Use: %s  Matrix_filename InputVector_filename  [SolutionVector_filename  [# of streams] ]  \n", argv[0]);     
        exists='f';
    } // endif //
    
    FILE *fh=NULL;
    // testing if matrix file exists
    if((fh = fopen(argv[1], "rb")  )   == NULL) {
        printf("No matrix file found.\n");
        exists='f';
    } // end if //
    
    // testing if input file exists
    if((fh = fopen(argv[2], "rb")  )   == NULL) {
        printf("No input vector file found.\n");
        exists='f';
    } // end if //

    // testing if output file exists
    if (argc  >3 ) {
        if((fh = fopen(argv[3], "rb")  ) == NULL) {
            printf("No output vector file found.\n");
            exists='f';
        } else {
            checkSol='t';
        } // end if //
    } // end if //
    if (exists == 'f') {
        printf("Quitting.....\n");
        exit(0);
    } // end if //
    fclose(fh);
    
    printf("%s Precision. \n", (sizeof(real) == sizeof(double)) ? "Double": "Single" );
    
    starRow = (int *) malloc(2*sizeof(int) ); 
    starRow[0]=0;
    reader(&n_global,&nnz_global, starRow, 
           &row_ptr,&col_idx,&val,
           argv[1], 1);
    
    // ready to start //    
    cudaError_t cuda_ret;
    
    real *w=NULL;
    real *v=NULL; // <-- input vector to be shared later
    //real *v_off=NULL; // <-- input vector to be shared later
    
    
    v     = (real *) malloc(n_global*sizeof(real));
    w     = (real *) malloc(n_global*sizeof(real)); 

    // reading input vector
    vectorReader(v, &n_global, argv[2]);
//////////////////////////////////////
/* Determining number of rows to be proceced by each block */
    
    int *blockRows=(int *) malloc( n_global*sizeof(int));
    blockRows[0]=0;
    int nRows=0;
    int sizeBlockRows=1;
    int sum=0;
    for (int row=0; row<n_global; ++row) {
        sum += (row_ptr[row+1] - row_ptr[row]);
        ++nRows;
        if ( sum == SHARED_SIZE  ||  (nRows == MAXTHREADS  &&  sum < SHARED_SIZE) ) {
            blockRows[sizeBlockRows] = row+1;
            ++sizeBlockRows;
            nRows=0;
            sum=0;
        } else if (sum > SHARED_SIZE) {
            if (nRows>1) {
                blockRows[sizeBlockRows] = row;
                --row;
            } else {
                blockRows[sizeBlockRows] = row+1;
            } // end if //
            ++sizeBlockRows;
            nRows=0;
            sum=0;
        } // end if //
    } // end for //

    if (blockRows[sizeBlockRows-1] != n_global ) {
        blockRows[sizeBlockRows] = n_global;
    } else {
        --sizeBlockRows;
    } // end if //
    ++sizeBlockRows;

/*
    printf("sizeBlockRows: %d\n", sizeBlockRows); 
    for (int i=0; i<sizeBlockRows; ++i ) {
        printf("%4d", blockRows[i]);
    }
    printf("\n");
    
    
    printf("n_global: %d\n", n_global); 
    printf("\nMAXTHREADS: %d\n", MAXTHREADS); 
    printf("SHARED_SIZE: %d\n", SHARED_SIZE); exit(0);
*/

    int *wtpb     =(int *) malloc( (sizeBlockRows-1)*sizeof(int));
    
    for (int i=0; i<sizeBlockRows-1; ++i) {
        int ratio = MAXTHREADS / (blockRows[i+1] - blockRows[i]);
        int temp = 1;
        do {
            temp = temp<<1;
            //printf("ratio: %d, temp: %d\n",  ratio, temp);
            if (ratio < temp)  {
                wtpb[i] = temp>>1;
                break;
            } // end if //
        } while (  temp <= MAXTHREADS );
    } // end for //
    
    
    cuda_ret = cudaMalloc((void **) &blockRows_d,  (sizeBlockRows)*sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for blockRows_d array");

    cuda_ret = cudaMemcpy(blockRows_d, blockRows, sizeBlockRows*sizeof(int),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix blockRows_d");
    
    cuda_ret = cudaMalloc((void **) &wtpb_d,  (sizeBlockRows-1)*sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for blockRows_d array");

    cuda_ret = cudaMemcpy(wtpb_d, wtpb, (sizeBlockRows-1)*sizeof(int),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix blockRows_d");
    
    
    free(blockRows);
    free(wtpb);
//////////////////////////////////////////

// cuda stuff start here

    // Allocating device memory for input matrices 

    cuda_ret = cudaMalloc((void **) &rows_d,  (n_global+1)*sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for rows_d");
    
    cuda_ret = cudaMalloc((void **) &cols_d,  (nnz_global)*sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for cols_d");
    
    cuda_ret = cudaMalloc((void **) &vals_d,  (nnz_global)*sizeof(real));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for vals_d");

    cuda_ret = cudaMalloc((void **) &v_d,  (n_global)*sizeof(real));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for x_d");

    cuda_ret = cudaMalloc((void **) &w_d,  (n_global)*sizeof(real));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for y_d");

   // Copy the input matrices from the host memory to the device memory

    cuda_ret = cudaMemcpy(rows_d, row_ptr, (n_global+1)*sizeof(int),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix rows_d");

    cuda_ret = cudaMemcpy(cols_d, col_idx, (nnz_global)*sizeof(int),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix cols_d");

    cuda_ret = cudaMemcpy(vals_d, val, (nnz_global)*sizeof(real),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix vals_d");


    cuda_ret = cudaMemcpy(v_d, v, (n_global)*sizeof(real),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix x_d");


    block.x=MAXTHREADS;
    block.y = 1;
    block.z = 1;
    grid.x = sizeBlockRows-1;
    grid.y = 1;
    grid.z = 1;

    block.y=MAXTHREADS/block.x;
    printf("using ipcsr() spmv for on matrix,  blockSize: [%d, %d]\n",block.x,block.y  ) ;

    // Timing should begin here//
    cuda_ret = cudaBindTexture(NULL, xTex, v_d, n_global*sizeof(real));
    //cuda_ret = cudaBindTexture(NULL, valTex, vals_d, nnz_global*sizeof(real));            

    struct timeval tp;                                   // timer
    double elapsed_time;
    gettimeofday(&tp,NULL);  // Unix timer
    elapsed_time = -(tp.tv_sec + tp.tv_usec*1.0e-6);
    for (int t=0; t<REP; ++t) {

        //alg1<<<grid, block >>>(temp,vals_d,cols_d,nnz_global);
        ipcsr<<<grid, block >>>(w_d , vals_d,cols_d, rows_d,blockRows_d,wtpb_d, sizeBlockRows, 1.0, 0.0 );
        cudaStreamSynchronize(NULL);
        
    } // end for //    
    gettimeofday(&tp,NULL);
    elapsed_time += (tp.tv_sec + tp.tv_usec*1.0e-6);
    printf ("Total time was %f seconds, GFLOPS: %f\n", elapsed_time, (2.0*nnz_global+ 3.0*n_global) * 1.0e-9 / (elapsed_time/REP));
    cuda_ret = cudaUnbindTexture(xTex);
    //cuda_ret = cudaUnbindTexture(valTex);

    cuda_ret = cudaMemcpy(w, w_d, (n_global)*sizeof(real),cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix y_d back to host");

// cuda stuff ends here
//////////////////////////////////////
   
    if (checkSol=='t') {
        real *sol=NULL;
        sol     = (real *) malloc((n_global)*sizeof(real)); 
        // reading input vector
        vectorReader(sol, &n_global, argv[3]);
        
        int row=0;
        real tolerance = 1.0e-08;
        if (sizeof(real) != sizeof(double) ) {
            tolerance = 1.0e-02;
        } // end if //

        real error;
        do {
            error =  fabs(sol[row] - w[row]) /fabs(sol[row]);
            if ( error > tolerance ) break;
            ++row;
        } while (row < n_global); // end do-while //
        
        if (row == n_global) {
            printf("Solution match in GPU\n");
        } else {    
            printf("For Matrix %s, solution does not match at element %d in GPU  %20.13e   -->  %20.13e  error -> %20.13e, tolerance: %20.13e \n", 
            argv[1], (row+1), sol[row], w[row], error , tolerance  );
        } // end if //
        free(sol);    
    } // end if //
    free(w);
    free(v);
    
    #include "parallelSpmvCleanData.h" 
    return 0;    
} // end main() //
