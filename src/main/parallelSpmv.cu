#include <iostream>
#include <fstream>
#include <chrono>

using std::cout;
using std::ifstream;
using std::ios;

using ns = std::chrono::nanoseconds;

#include "parallelSpmv.h"

#define FATAL(msg) \
    do {\
        fprintf(stderr, "[%s:%d] %s\n", __FILE__, __LINE__, msg);\
        exit(-1);\
    } while(0)

#define MAXTHREADS 256
#define REP 1000

struct str
{
    int value;
    unsigned int index;
};

int ascending(const void *a, const void *b)
{
    struct str *a1 = (struct str *)a;
    struct str *a2 = (struct str *)b;
    
    return ( (*a2).value - (*a1).value );
} // end ascending() //

int descending(const void *a, const void *b)
{
    struct str *a1 = (struct str *)a;
    struct str *a2 = (struct str *)b;
    
    return ( (*a1).value - (*a2).value );
} // end descending() //



  auto meanAndSd = [] (floatType &mean, floatType &sd, const unsigned int *__restrict__ const data,  const unsigned int &n) -> void 
  {
    floatType sum = static_cast<floatType>(0);
    floatType standardDeviation = static_cast<floatType>(0);
    
    for(int row=0; row<n; ++row) {
        sum += (data[row+1] - data[row]);
    } // end for //
    mean = sum/n;
    
    for(int row=0; row<n; ++row) {
        standardDeviation += pow( (data[row+1] - data[row]) - mean, 2);
    } // end for //
    sd = sqrt(standardDeviation/n);
    return;
  }; // end of meanAndSd() lambda function;

int main(int argc, char *argv[]) 
{

    if (MAXTHREADS > 512) {
        printf("need to adjust the spmv() function to acomodate more than 512 threads per block\nQuitting ....\n");
        exit(-1);
    } // end if //
    
    #include "parallelSpmvData.h"

    // verifing number of input parameters //
    auto exists=true;
    auto checkSol=false;

    if (argc < 3 ) {
        cout << "Use: " <<  argv[0] <<   " Matrix_filename InputVector_filename  [SolutionVector_filename  [# of streams ['+'| '-'] ] ]\n";
        exit(-1);
    } // endif //

    ifstream inFile;
    // testing if matrix file exists
    inFile.open(argv[1], ios::in);
    if( !inFile ) {
        cout << "No matrix file found.\n";
        exists=false;
    } // end if //
    inFile.close();
    
    // testing if input file exists
    inFile.open(argv[2], ios::in);
    if( !inFile ) {
        cout << "No input vector file found.\n";
        exists=false;
    } // end if //
    inFile.close();

    // testing if output file exists
    if (argc  >3 ) {
        inFile.open(argv[3], ios::in);
        if( !inFile ) {
            cout << "No output vector file found.\n";
            exists=false;
        } else {
            checkSol=true;
        } // end if //
        inFile.close();        
    } // end if //

    if (!exists) {
        cout << "Quitting.....\n";
        exit(0);
    } // end if //

    if (argc >4 ) { 
      if (strcmp(argv[4],"+") == 0  or strcmp(argv[4],"-") == 0) {
        sort=true;
        if (strcmp(argv[4],"+") == 0) ascen=true;
      } // end if //
    } // end if //
    // reading basic matrix data
    reader(n_global,nnz_global, &row_ptr,&col_idx,&val,argv[1]);
    // end of reading basic matrix data

    // finding the global nnzPerRow and stdDev
    meanAndSd(nnzPerRow, stdDev,row_ptr, n_global);

    v = new floatType[n_global];    
    vectorReader(v, n_global, argv[2]);


////////////////// search for the raw number of block of rows   ////////////////////////////
/////////////////// determining the number of block rows based  ////////////////////////////
/////////////////// on global nnzPerRow and stdDev of the nnz   ////////////////////////////


  floatType temp = round(12.161 * log(stdDev/nnzPerRow) + 14.822);
  if (temp >1) nRowBlocks = temp;

  // the value of  nRowBlocks can be forced by run-time paramenter   
  if (argc > 5  && atoi(argv[5]) > 0) {
      nRowBlocks = atoi(argv[5]);
  } // end if //  
  if (nRowBlocks > n_global) nRowBlocks = n_global;
  
  //cout << "file: " << __FILE__  << " line: " << __LINE__ << "  nRowBlocks:" <<  nRowBlocks << '\n';

/////////////// end of search for the raw number of block of rows   /////////////////////////

  //cout << ( (sizeof(floatType) == sizeof(double)) ? "Double": "Single" )  << " Precision. Solving dividing matrix into " << nRowBlocks << ((nRowBlocks > 1) ? " blocks": " block") << '\n';


////////////////// search for the real number of block of rows   ////////////////////////////
  unsigned int *blockSize = nullptr;
  unsigned int *starRowBlock = nullptr;
     
  blockSize = new unsigned int [nRowBlocks];
  starRowBlock = new unsigned int [nRowBlocks+1];
  
  for (int b=0; b<nRowBlocks; ++b) {
      blockSize[b] = 1;
  } // end for //

  starRowBlock[0]=0;
  getRowsNnzPerStream(starRowBlock,n_global,nnz_global, row_ptr, nRowBlocks);
    
    for (int b=0; b<nRowBlocks; ++b) {
      
      meanAndSd(nnzPerRow, stdDev,&row_ptr[starRowBlock[b]], (starRowBlock[b+1]-starRowBlock[b]));


      // these mean use vector spmv 
      floatType limit=nnzPerRow + parameter2Adjust*stdDev;
      //cout << "b: " << b << ", limit: " << limit << ", nnzPerRow: " << nnzPerRow << ", stdDev: " << stdDev <<  '\n';
      
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
      //printf("using vector spmv for on matrix,  blockSize: [%d, %d] %f, %f\n",blockSize[b],MAXTHREADS/blockSize[b], nnzPerRow, stdDev) ;
    } // end for //
    

    // here comes the consolidation ....
    
    nStreams=1;
    for (int b=1; b<nRowBlocks; ++b) {
        if (blockSize[b] != blockSize[b-1]) {
            ++nStreams;
        } // end if //
    } // end for //
    
   cout << "initial number of row sets: " << nRowBlocks 
        << ", final number of row sets: " << nStreams << '\n';

   
/////////////// end of search for the real number of block of rows //////////////////////////

/////////////// begin  creating stream dependent variables //////////////////////////
    cudaError_t cuda_ret;
    stream = new cudaStream_t[nStreams];

    grid   = new dim3 [nStreams];
    block  = new dim3 [nStreams];
    sharedMemorySize = new size_t [nStreams]();

    starRowStream = new unsigned int [nStreams+1];

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




// sorting before execution by block size or nonzeros

  if (sort) {
    toSortStream = new struct str [nStreams];
    for (int s=0; s<nStreams; ++s) {
        toSortStream[s].index = s;
        
        // sorting by block size
        toSortStream[s].value=block[s].x;
        
        // sorting by non-zeros size
        //toSortStream[s].value = row_ptr[starRowStream[s+1]] - row_ptr[starRowStream[s]];
        
    } // end for //
      if (ascen) {
        qsort(toSortStream, nStreams, sizeof(toSortStream[0]), ascending);
      } else {
        qsort(toSortStream, nStreams, sizeof(toSortStream[0]), descending);
      } // end if //
  } // end if //
    
    
    
// end of sorting before execution by block size or nonzeros


    for (unsigned int s=0; s<nStreams; ++s) {
        unsigned int nrows = starRowStream[s+1]-starRowStream[s];
        //printf("file: %s, line: %d, using vector spmv for on matrix,  nrows: %d \n",  __FILE__, __LINE__, nrows ) ;
        grid[s].x = ( (nrows + block[s].y - 1) / block[s].y ) ;
        sharedMemorySize[s]=block[s].x*block[s].y*sizeof(floatType);
        
        
        /*
        printf("\tblock for stream %3d has size: [%3d, %3d],  %10d rows and %12d non-zeros.\n", 
                               s, 
                               block[s].x, 
                               block[s].y, 
                               starRowStream[s+1]-starRowStream[s],
                               row_ptr[starRowStream[s+1]] - row_ptr[starRowStream[s]] );

*/  
        unsigned int ss=s;
        if (sort) {
          ss = toSortStream[s].index;
        }  // end if //
        cout << "\tblock for stream " << ss
             << "\thas size: [" 
             << block[ss].y
             << ", " 
             << block[ss].x
             << "],\t  and its grid has size: [" 
             << grid[ss].x*block[ss].y  
             << ", "
             << grid[ss].y*block[ss].x
             << "],\t " 
             << starRowStream[ss+1]-starRowStream[ss]
             << " rows and "
             << row_ptr[starRowStream[ss+1]] - row_ptr[starRowStream[ss]]
             << " non-zeros.\n";             

        
    } // end for //

    delete[] starRowBlock;
    delete[] blockSize;

/////////////// end of  creating stream dependent variables //////////////////////////


/////////////// begin  allocating device memory //////////////////////////


    cuda_ret = cudaMalloc((void **) &rows_d,  (n_global+1)*sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for rows_d");
    
    cuda_ret = cudaMalloc((void **) &cols_d,  (nnz_global)*sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for cols_d");
    
    cuda_ret = cudaMalloc((void **) &vals_d,  (nnz_global)*sizeof(floatType));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for vals_d");

    cuda_ret = cudaMalloc((void **) &v_d,  (n_global)*sizeof(floatType));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for x_d");

    cuda_ret = cudaMalloc((void **) &w_d,  (n_global)*sizeof(floatType));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for y_d");        
    
/////////////// end of allocating device memory //////////////////////////


/////////////// begin   copying memory to device  //////////////////////////

    cuda_ret = cudaMemcpy(rows_d, row_ptr, (n_global+1)*sizeof(int),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix rows_d");

    cuda_ret = cudaMemcpy(cols_d, col_idx, (nnz_global)*sizeof(int),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix cols_d");

    cuda_ret = cudaMemcpy(vals_d, val, (nnz_global)*sizeof(floatType),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix vals_d");

    cuda_ret = cudaMemcpy(v_d, v, (n_global)*sizeof(floatType),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix x_d");
    cudaDeviceSynchronize();

/////////////// end of  copying memory to device  //////////////////////////

/////////////// begin   defining texture memory  //////////////////////////
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
    resDesc.res.linear.sizeInBytes = n_global*sizeof(floatType);
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
    

    //cuda_ret = cudaBindTexture(NULL, v_t, v_d, n_global*sizeof(floatType));
    //cuda_ret = cudaBindTexture(NULL, valTex, vals_d, nnz_global*sizeof(floatType));            

    
#endif
    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);   // <---- ???

/////////////// end of  defining texture memory  //////////////////////////


/////////////// begin  testing spmv call //////////////////////////
  // Timing should begin here//

  auto start = std::chrono::steady_clock::now();


    for (int t=0; t<REP; ++t) {

        //cuda_ret = cudaMemset(w_d, 0, (size_t) n_global*sizeof(floatType) );
        //if(cuda_ret != cudaSuccess) FATAL("Unable to set device for matrix w_d");
        
        for (unsigned int ss=0; ss<nStreams; ++ss) {
            unsigned int s=ss;
            if (sort) {
              s = toSortStream[ss].index;
            }  // end if //
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

/////////////// end of testing spmv call //////////////////////////

  auto duration = std::chrono::steady_clock::now() - start;
  auto elapsed_time = std::chrono::duration_cast<ns>(duration).count();
  
  
  cout << "Total time was " << elapsed_time*1.0e-9 
       << " seconds, GFLOPS: " << (2.0*nnz_global+ 3.0*n_global)*REP/elapsed_time
       << ", GBytes/s: " << (nnz_global*(2*sizeof(floatType) + sizeof(int))+n_global*(sizeof(floatType)+sizeof(int)))*REP*1.0/elapsed_time << '\n';

//////////////////////////////////////
// cuda stuff start here
    if (checkSol) {
      w = new floatType[n_global];
      /////////////// begin  copying memory to device  //////////////////////////
      cuda_ret = cudaMemcpy(w, w_d, (n_global)*sizeof(floatType),cudaMemcpyDeviceToHost);
      if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix y_d back to host");
      /////////////// end of copying memory to device  //////////////////////////
    
        floatType *sol=nullptr;
        sol = new floatType[n_global];
        // reading input vector
        vectorReader(sol, n_global, argv[3]);
        
        int row=0;
        floatType tolerance = 1.0e-08;
        if (sizeof(floatType) != sizeof(double) ) {
            tolerance = 1.0e-02;
        } // end if //

        floatType error;
        do {
            error =  fabs(sol[row] - w[row]) /fabs(sol[row]);
            if ( error > tolerance ) break;
            ++row;
        } while (row < n_global); // end do-while //
        
        if (row == n_global) {
          cout << "Solution match in GPU\n";
        } else {
          cout << "For Matrix " << argv[1] << ", solution does not match at element " 
               << (row+1) << " in GPU  " << sol[row] << "   -->  " 
               << w[row] << "  error -> " << error 
               << ", tolerance: " << tolerance << '\n';
        } // end if //
        delete[] sol;
        
    } // end if //

#ifdef USE_TEXTURE
    cudaDestroyTextureObject(v_t);
#endif    

    #include "parallelSpmvCleanData.h" 
    return 0;    
} // end main() //
