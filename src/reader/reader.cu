#include <iostream>
#include <fstream>
#include <cmath>
#include "floatType.h"

using std::ifstream;
using std::ios;

void reader( unsigned int &n_global, 
             unsigned int &nnz_global, 
             unsigned int **rowPtr, unsigned int **colIdx, floatType **val,
             const char *const matrixFile
             )
{   
    ifstream inFile;
    inFile.open(matrixFile, ios::in | ios::binary);

    // reading global nun rows //
    inFile.read((char *) &n_global, sizeof(unsigned int));

    // reading global nnz //
    inFile.read((char *) &nnz_global, sizeof(unsigned int));


    // allocating and reading rowPtr //    
    (*rowPtr) = new unsigned int[n_global+1];    
    inFile.read((char *) (*rowPtr), (n_global+1)*sizeof(unsigned int));

    // allocating and reading colIdx //    
    (*colIdx) = new unsigned int[nnz_global];    
    inFile.read((char *) (*colIdx), (nnz_global)*sizeof(unsigned int));

    // allocating and reading val //    
    (*val) = new floatType[nnz_global];
    if (sizeof(floatType) == sizeof(double)) {
      inFile.read((char *) (*val), (nnz_global)*sizeof(double));
      //inFile.read((char *) &nnzPerRow, sizeof(double));
      //inFile.read((char *) &stdDev, sizeof(double));
    } else {
      double temp;
      for (int i=0; i<nnz_global; ++i) {
        inFile.read((char *) &temp, sizeof(double));
        (*val)[i] = static_cast<floatType>(temp);
      } // end for //
      //inFile.read((char *) &temp, sizeof(double));
      //nnzPerRow = static_cast<floatType>(temp);
      //inFile.read((char *) &temp, sizeof(double));
      //stdDev = static_cast<floatType>(temp);
    } // end if //
/*      
    auto meanAndSd = [] (const unsigned int *__restrict__ const data, const floatType &mean, const int &n) -> floatType 
    {    
      floatType standardDeviation = static_cast<floatType>(0);
      for(int row=0; row<n; ++row) {
          standardDeviation += pow( (data[row+1] - data[row]) - mean, 2);
      } // end for //
      return sqrt(standardDeviation/n);
    }; // end of meanAndSd() lambda function;
    
    nnzPerRow = static_cast<floatType>(nnz_global)/n_global;
    stdDev = meanAndSd((*rowPtr),nnzPerRow, n_global );
*/    
    inFile.close();    
} // end of reader //
