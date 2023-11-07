    delete[] w;
/////////////// begin  de-allocating device memory //////////////////////////    
    cudaFree(w_d);
    cudaFree(v_d);
    cudaFree(vals_d);
    cudaFree(cols_d);
    cudaFree(rows_d);
/////////////// end  de-allocating device memory //////////////////////////
    delete[] toSortStream;
    delete[] starRowStream;
    delete[] sharedMemorySize;   
    for (int s=0; s<nStreams; ++s) {
        cudaStreamDestroy(stream[s]);
    } // end for /
    delete[] stream;
    delete[] grid;
    delete[] block;    
    delete[] v;    
    delete[] val;
    delete[] col_idx;
    delete[] row_ptr;

