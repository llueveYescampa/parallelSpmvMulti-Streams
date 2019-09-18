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

#define REP 1

#ifdef DOUBLE
    texture<int2>  xTex;
    //texture<int2>  valTex;
#else
    texture<float> xTex;
    //texture<float> valTex;
#endif

void meanAndSd(real *mean, real *sd,real *data, int n)
{
    real sum = (real) 0.0; 
    real standardDeviation = (real) 0.0;

    for(int i=0; i<n; ++i) {
        sum += data[i];
    } // end for //

    *mean = sum/n;

    for(int i=0; i<n; ++i) {
        standardDeviation += pow(data[i] - *mean, 2);
    } // end for //
    *sd=sqrt(standardDeviation/n);
} // end of meanAndSd //


int main(int argc, char *argv[]) 
{
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

    nStreams = 1;

    
    
    printf("%s Precision. Solving using %d %s\n", (sizeof(real) == sizeof(double)) ? "Double": "Single", nStreams, (nStreams > 1) ? "streams": "stream"  );

    stream= (cudaStream_t *) malloc(sizeof(cudaStream_t) * nStreams);
    
    starRow = (int *) malloc(sizeof(int) * nStreams+1); 
    starRow[0]=0;
    reader(&n_global,&nnz_global, starRow, 
           &row_ptr,&col_idx,&val,
           argv[1], nStreams);
    
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
// cuda stuff start here

    // Allocating device memory for input matrices 

    cuda_ret = cudaMalloc((void **) &rows_d,  (n_global+1)*sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for rows_d");
    
    cuda_ret = cudaMalloc((void **) &cols_d,  (nnz_global)*sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for cols_d");
    
    cuda_ret = cudaMalloc((void **) &vals_d,  (nnz_global)*sizeof(real));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for vals_d");

    cuda_ret = cudaMalloc((void **) &temp,  (nnz_global)*sizeof(real));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for temp array");

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



    meanNnzPerRow = (real*) malloc(nStreams*sizeof(real));
    sd            = (real*) malloc(nStreams*sizeof(real ));
    block = (dim3 *) malloc(nStreams*sizeof(dim3 )); 
    grid  = (dim3 *) malloc(nStreams*sizeof(dim3 )); 
    sharedMemorySize = (size_t *) calloc(nStreams, sizeof(size_t)); 

    for (int s=0; s<nStreams; ++s) {
        block[s].x = 1;
        block[s].y = 1;
        block[s].z = 1;
        grid[s].x = 1;
        grid[s].y = 1;
        grid[s].z = 1;
    } // end for //

    for (int s=0; s<nStreams; ++s) {
        int nrows = starRow[s+1]-starRow[s];
        /////////////////////////////////////////////////////
        // determining the standard deviation of the nnz per row
        real *temp=(real *) calloc(nrows,sizeof(real));
        
        for (int row=starRow[s], i=0; row<starRow[s]+nrows; ++row, ++i) {
            temp[i] = row_ptr[row+1] - row_ptr[row];
        } // end for //
        meanAndSd(&meanNnzPerRow[s],&sd[s],temp, nrows);
        //printf("file: %s, line: %d, gpu on-prcoc:   %d, mean: %7.3f, sd: %7.3f using: %s\n", __FILE__, __LINE__, s , meanNnzPerRow[s], sd[s], (meanNnzPerRow[s] + 0.5*sd[s] < 32) ? "spmv0": "spmv1" );
        free(temp);
        /////////////////////////////////////////////////////

        //cuda_ret = cudaStreamCreateWithFlags(&stream0[gpu], cudaStreamDefault);
        cuda_ret = cudaStreamCreateWithFlags(&stream[s], cudaStreamNonBlocking ) ;
        if(cuda_ret != cudaSuccess) FATAL("Unable to create stream0 ");
        
        printf("In Stream: %d\n",s);
        

        block[s].x=MAXTHREADS;
        block[s].y=MAXTHREADS/block[s].x;
        grid[s].x = ( (nrows + block[s].x - 1) / block[s].x ) ;
    	sharedMemorySize[s]=block[s].x*block[s].y*sizeof(real);
        printf("using vector spmv for on matrix,  blockSize: [%d, %d] %f, %f\n",block[s].x,block[s].y, meanNnzPerRow[s], sd[s]) ;

    } // end for //

    // Timing should begin here//
    cuda_ret = cudaBindTexture(NULL, xTex, v_d, n_global*sizeof(real));
    //cuda_ret = cudaBindTexture(NULL, valTex, vals_d, nnz_global*sizeof(real));            

    struct timeval tp;                                   // timer
    double elapsed_time;
    gettimeofday(&tp,NULL);  // Unix timer
    elapsed_time = -(tp.tv_sec*1.0e6 + tp.tv_usec);
    for (int t=0; t<REP; ++t) {

        //cuda_ret = cudaMemset(w_d, 0, (size_t) n_global*sizeof(real) );
        //if(cuda_ret != cudaSuccess) FATAL("Unable to set device for matrix w_d");
        
        
        alg1<<<grid[0], block[0] >>>(temp,vals_d,cols_d,nnz_global);
        alg2<<<grid[0], block[0] >>>(w_d , temp,  rows_d, n_global, 1.0, 0.0 );
        cudaStreamSynchronize(NULL);
        
        
    } // end for //    
    gettimeofday(&tp,NULL);
    elapsed_time += (tp.tv_sec*1.0e6 + tp.tv_usec);
    printf ("Total time was %f seconds, GFLOPS: %f\n", elapsed_time*1.0e-6, (2.0*nnz_global+ 3.0*n_global)*REP*1.0e-3/elapsed_time);
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
