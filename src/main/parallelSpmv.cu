#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "parallelSpmv.h"

#define FATAL(msg) \
    do {\
        fprintf(stderr, "[%s:%d] %s\n", __FILE__, __LINE__, msg);\
        exit(-1);\
    } while(0)

#define MAXTHREADS 256
#define REP 1000

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

    if (MAXTHREADS > 512) {
        printf("need to adjust the spmv() function to acomodate more than 512 threads per block\nQuitting ....\n");
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
    
    // reading basic matrix data
    reader(&n_global,&nnz_global, &row_ptr,&col_idx,&val,argv[1]);
    // end of reading basic matrix data

        
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



#ifdef USE_TEXTURE

    cudaTextureDesc td;
    memset(&td, 0, sizeof(td));
    td.normalizedCoords = 0;
    td.addressMode[0] = cudaAddressModeClamp;
    td.readMode = cudaReadModeElementType;


    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = v_d;
    resDesc.res.linear.sizeInBytes = n_global*sizeof(real);
    #ifdef DOUBLE
    resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.linear.desc.y = 32;
    #else
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    #endif
    resDesc.res.linear.desc.x = 32;

    cudaTextureObject_t v_t;
    cuda_ret = cudaCreateTextureObject(&v_t, &resDesc, &td, NULL);
    if(cuda_ret != cudaSuccess) FATAL("Unable to create text memory v_t");

    
/*
    cuda_ret = cudaBindTexture(NULL, v_t, v_d, n_global*sizeof(real));
    //cuda_ret = cudaBindTexture(NULL, valTex, vals_d, nnz_global*sizeof(real));            
*/    
#endif
    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    
    { // determining the number of block rows based on mean and sd of the nnz 
        // opening matrix file to read mean and sd of number of nonzeros per row
        double tmpMean, tmpSD;
        fh = fopen(argv[1], "rb");
        // reading laast two values in file: mean and sd //
        fseek(fh, 0L, SEEK_END);
        long int offset = ftell(fh)-2*sizeof(double);
        fseek(fh, offset, SEEK_SET);
        if ( !fread(&tmpMean, sizeof(double), (size_t) 1, fh)) exit(0);
        if ( !fread(&tmpSD, sizeof(double), (size_t) 1, fh)) exit(0);
        if (fh) fclose(fh);
        
        // determining number of streams based on mean and sd
        real ratio = tmpSD/tmpMean;
        //printf("file: %s, line: %d, tMean nnz: %.2f, SD nnz: %.2f, ratio: %.2f\n", __FILE__, __LINE__ , tmpMean, tmpSD, ratio);
        if        (ratio <= 0.173 ) {
            nRowBlocks = 1;
        } else if (ratio <= 3.5 ) {
            nRowBlocks = round(8.66 * ratio - 0.5);
        } else {
            nRowBlocks = round(58.87 * log(ratio) - 43.92);
        } // end if //
        /*
        if        (ratio <= 0.220 ) {
            nRowBlocks = 1;
        } else if (ratio <= 0.275 ) {
            nRowBlocks = 2;
        } else if (ratio <= 0.420 ) {
            nRowBlocks = 4;
        } else if (ratio <= 0.65 ) {
            nRowBlocks = 8;
        } else if (ratio <= 0.75 ) {
            nRowBlocks = 16;
        } else if (ratio <= 2.20 ) {
            nRowBlocks = 32;
        } else if (ratio <= 8.20 ) {
            nRowBlocks = 64;
        } else if (ratio <= 60.00 ) {
            nRowBlocks = 128;
        } else if (ratio <= 120.00 ) {
            nRowBlocks = 256;
        } else {
            nRowBlocks = 512;
        } // end if //
        */
        printf("nRowBlocks: %d\n", nRowBlocks);
    } // end of determining the number of block rows based on mean and sd of the nnz 
    // the value of  nRowBlocks can be forced by run-time paramenter   
    if (argc  > 4  && atoi(argv[4]) > 0) {
        nRowBlocks = atoi(argv[4]);
    } // end if //
    if (nRowBlocks > n_global) nRowBlocks = n_global;
    
    printf("%s Precision. Solving dividing matrix into %d %s\n", (sizeof(real) == sizeof(double)) ? "Double": "Single", nRowBlocks, (nRowBlocks > 1) ? "blocks": "block"  );
           
    //printf("file: %s, line: %d, n_global: %d, nnz_global: %d, nRowBlocks: %d\n", __FILE__, __LINE__,n_global, nnz_global, nRowBlocks  ); exit(0);



    starRowBlock= (int *) malloc(sizeof(int) * nRowBlocks+1); 
    starRowBlock[0]=0;
           
    getRowsNnzPerStream(starRowBlock,&n_global,&nnz_global, row_ptr, nRowBlocks);

    blockSize= (int *) malloc(sizeof(int) * nRowBlocks); 
    for (int b=0; b<nRowBlocks; ++b) {
        blockSize[b] = 1;
    } // end for //

    for (int b=0; b<nRowBlocks; ++b) {
        int nrows = starRowBlock[b+1]-starRowBlock[b];
        /////////////////////////////////////////////////////
        // determining the standard deviation of the nnz per row
        real *temp=(real *) calloc(nrows,sizeof(real));
        
        for (int row=starRowBlock[b], i=0; row<starRowBlock[b]+nrows; ++row, ++i) {
            temp[i] = row_ptr[row+1] - row_ptr[row];
        } // end for //
        meanAndSd(&meanNnzPerRow,&sd,temp, nrows);
        //printf("file: %s, line: %d, gpu on-prcoc:   %d, mean: %7.3f, sd: %7.3f using: %s\n", __FILE__, __LINE__, s , meanNnzPerRow[s], sd[s], (meanNnzPerRow[s] + 0.5*sd[s] < 32) ? "spmv0": "spmv1" );
        free(temp);
        /////////////////////////////////////////////////////

        // these mean use vector spmv 
        real limit=meanNnzPerRow + parameter2Adjust*sd;
        if ( limit < 4.5  ) {
            blockSize[b]=warpSize/32;
        }  else if (limit < 6.95 ) {
            blockSize[b]=warpSize/16;
        }  else if (limit < 15.5 ) {
            blockSize[b]=warpSize/8;
        }  else if (limit < 74.0 ) {
            blockSize[b]=warpSize/4;
        }  else if (limit < 300.0 ) {
            blockSize[b]=warpSize/2;
        }  else if (limit < 350.0 ) {
            blockSize[b]=warpSize;
        }  else if (limit < 1000.0 ) {
            blockSize[b]=warpSize*2;
        }  else if (limit < 2000.0 ) {
            blockSize[b]=warpSize*4;
        }  else if (limit < 3000.0 ) {
            blockSize[b]=warpSize*8;
        }  else {
            blockSize[b]=warpSize*16;
        } // end if //
        if (blockSize[b] > MAXTHREADS) {
            blockSize[b]=512;
        } // end if //    
        //printf("using vector spmv for on matrix,  blockSize: [%d, %d] %f, %f\n",blockSize[b],blockSize[b], meanNnzPerRow[b], sd[b]) ;
    } // end for //
    
    
    // here comes the consolidation ....
    
    nStreams=1;
    for (int b=1; b<nRowBlocks; ++b) {
        if (blockSize[b] != blockSize[b-1]) {
            ++nStreams;
        } // end if //
    } // end for //
    
    printf("%d Blocks produced %d streams\n", nRowBlocks, nStreams);
    grid  = (dim3 *) malloc(nStreams*sizeof(dim3 )); 
    block = (dim3 *) malloc(nStreams*sizeof(dim3 )); 
    sharedMemorySize = (size_t *) calloc(nStreams, sizeof(size_t)); 
    stream= (cudaStream_t *) malloc(sizeof(cudaStream_t) * nStreams);
    
    starRowStream = (int *) malloc( (nStreams+1) * sizeof(int) ); 
    
    
    for (int s=0; s<nStreams; ++s) {
        block[s].x = 1;
        block[s].y = 1;
        block[s].z = 1;
        grid[s].x = 1;
        grid[s].y = 1;
        grid[s].z = 1;

        //cuda_ret = cudaStreamCreateWithFlags(&stream0[gpu], cudaStreamDefault);
        cuda_ret = cudaStreamCreateWithFlags(&stream[s], cudaStreamNonBlocking ) ;
        if(cuda_ret != cudaSuccess) FATAL("Unable to create stream0 ");
    } // end for //

    block[0].x=blockSize[0];
    starRowStream[0]=starRowBlock[0];
    starRowStream[nStreams]=starRowBlock[nRowBlocks];
    
    if (block[0].x > MAXTHREADS) {
        block[0].x=512;
        block[0].y=1;
    } else {
        block[0].y=MAXTHREADS/block[0].x;
    } // end if //    
    //printf("file: %s, line: %d, using vector spmv for on matrix,  blockSize: [%d, %d]\n", __FILE__, __LINE__, block[0].x,block[0].y) ;
    

    for (int b=1, s=1; b<nRowBlocks; ++b) {
        if (blockSize[b] != blockSize[b-1]) {
            block[s].x=blockSize[b];
            if (block[s].x > MAXTHREADS) {
                block[s].x=512;
                block[s].y=1;
            } else {
                block[s].y=MAXTHREADS/block[s].x;
            } // end if //    
            
            starRowStream[s]=starRowBlock[b];
            //printf("file: %s, line: %d, using vector spmv for on matrix,  blockSize: [%d, %d] \n",  __FILE__, __LINE__,  block[s].x,block[s].y) ;
        	
            ++s;
        } // end if //
    } // end for //

    for (int s=0; s<nStreams; ++s) {
        int nrows = starRowStream[s+1]-starRowStream[s];
        //printf("file: %s, line: %d, using vector spmv for on matrix,  nrows: %d \n",  __FILE__, __LINE__, nrows ) ;
        grid[s].x = ( (nrows + block[s].y - 1) / block[s].y ) ;
        sharedMemorySize[s]=block[s].x*block[s].y*sizeof(real);
    } // end for //

/*
    for (int b=0; b<nRowBlocks; ++b) {
        printf("blockSize: [%d] %d, %d\n",blockSize[b],  starRowBlock[b+1], starRowBlock[b]) ;
    } // end for //
    printf("\n\n");
    //exit(0);    
*/    

    for (int s=0; s<nStreams; ++s) {
        printf("\tblock for stream %3d has size: [%3d, %3d],  %10d rows and %12d non-zeros.\n", 
                               s, 
                               block[s].x, 
                               block[s].y, 
                               starRowStream[s+1]-starRowStream[s],
                               row_ptr[starRowStream[s+1]] - row_ptr[starRowStream[s]] );
    } // end for //
  
    // Timing should begin here//

    struct timeval tp;                                   // timer
    double elapsed_time;
    gettimeofday(&tp,NULL);  // Unix timer
    elapsed_time = -(tp.tv_sec*1.0e6 + tp.tv_usec);
    for (int t=0; t<REP; ++t) {

        //cuda_ret = cudaMemset(w_d, 0, (size_t) n_global*sizeof(real) );
        //if(cuda_ret != cudaSuccess) FATAL("Unable to set device for matrix w_d");
        
        for (int s=0; s<nStreams; ++s) {
            const int sRow = starRowStream[s];
            const int nrows = starRowStream[s+1]-starRowStream[s];
#ifdef USE_TEXTURE
            spmv<<<grid[s], block[s], sharedMemorySize[s], stream[s] >>>((w_d+sRow), v_t, vals_d, (rows_d+sRow), (cols_d), nrows, 1.0,0.0);
#else
            spmv<<<grid[s], block[s], sharedMemorySize[s], stream[s] >>>((w_d+sRow), v_d,  vals_d, (rows_d+sRow), (cols_d), nrows, 1.0,0.0);
#endif
        } // end for //
        
        for (int s=0; s<nStreams; ++s) {
            //cudaStreamSynchronize(NULL);
            cudaStreamSynchronize(stream[s]);
        } // end for //
        
    } // end for //    
    gettimeofday(&tp,NULL);
    elapsed_time += (tp.tv_sec*1.0e6 + tp.tv_usec);
    printf ("Total time was %f seconds, GFLOPS: %f, GBytes/s: %f\n", elapsed_time*1.0e-6, 
                                (2.0*nnz_global+ 3.0*n_global)*REP*1.0e-3/elapsed_time,
                                (nnz_global*(2*sizeof(real) + sizeof(int))+n_global*(sizeof(real)+sizeof(int)))*REP*1.0e-3/elapsed_time );
    

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
#ifdef USE_TEXTURE
    cudaDestroyTextureObject(v_t);
#endif    
    #include "parallelSpmvCleanData.h" 
    return 0;    
} // end main() //
