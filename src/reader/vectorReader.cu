#include <iostream>
#include <fstream>
using std::ifstream;
using std::ios;

#include "floatType.h"

void vectorReader(floatType *const v, unsigned int &n, const char *const vectorFile)
{
    ifstream inFile;
    inFile.open(vectorFile, ios::in | ios::binary);

    if (sizeof(floatType) == sizeof(double)) {
      inFile.read((char *) v, (n)*sizeof(double));
    } else {
      double temp;
      for (int i=0; i<n; ++i) {
        inFile.read((char *) &temp, sizeof(double));
        v[i] = static_cast<floatType>(temp);
      } // end for //
    } // end if //
    inFile.close();    
    // end of opening vector file to read values
} // end of vectoReader //
