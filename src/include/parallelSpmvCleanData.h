    free(starRow);
    free(row_ptr);
    free(col_idx);
    free(val);

    cudaFree(rows_d);
    cudaFree(cols_d);
    cudaFree(vals_d);
    cudaFree(v_d);
    cudaFree(w_d);

    for (int s=0; s<nStreams; ++s) {
        cudaStreamDestroy(stream[s]);
    } // end for /
    
    free(stream);
    
    free(meanNnzPerRow);
    free(sd);
    free(sharedMemorySize);
    free(block);
    free(grid);

