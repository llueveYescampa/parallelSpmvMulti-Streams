    free(row_ptr_off);
    free(col_idx_off);
    free(val_off);

    for (int s=0; s<nStreams; ++s) {
        cudaStreamDestroy(stream[s]);
    } // end for /
    free(stream);
    
