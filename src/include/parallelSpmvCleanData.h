    free(starRow);
    free(row_ptr);
    free(col_idx);
    free(val);


    for (int s=0; s<nStreams; ++s) {
        cudaStreamDestroy(stream[s]);
    } // end for /
    
exit(0);
    free(stream);
    
    free(meanNnzPerRow);
    free(sd);
    free(sharedMemorySize);
    free(block);
    free(grid);

