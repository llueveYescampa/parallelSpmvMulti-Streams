    free(starRow);
    free(row_ptr);
    free(col_idx);
    free(val);

    cudaFree(rows_d);
    cudaFree(cols_d);
    cudaFree(vals_d);
    cudaFree(blockRows_d);
    cudaFree(wtpb_d);
    cudaFree(v_d);
    cudaFree(w_d);

