    delete[] w;
/////////////// begin  de-allocating device memory //////////////////////////    
    cudaFree(w_d);
    cudaFree(v_d);
    cudaFree(vals_d);
    cudaFree(cols_d);
    cudaFree(rows_d);
/////////////// end  de-allocating device memory //////////////////////////
    delete[] starRowStream;
    delete[] sharedMemorySize;    
    delete[] stream;
    delete[] grid;
    delete[] block;    
    delete[] v;    
    delete[] val;
    delete[] col_idx;
    delete[] row_ptr;

