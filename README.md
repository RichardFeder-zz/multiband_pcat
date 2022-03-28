# pcat-lion
Probabilistic cataloging of astronomical sources from SDSS exposures. Extension of Stephen Portillo's code for simultaneous fit of multiple band photometric data. Herschel-SPIRE data is now supported within pcat_spire.py

Update: A dedicated version of multiband PCAT for handling diffuse emission, PCAT-DE, will be available in a few months in a separate repository with documentation/examples. This will consist of most of the code from pcat_spire.py. 

Usage from Terminal:

python pcat.py [dataname] [verbosity level (0, 1, or 2)], [testpsf (0 or 1)], [datatype (mock, DAOPHOT, etc.)], [multiband (0 or 1)]


To use OpenBLAS:
- Install OpenBLAS through something like ‘git clone https://github.com/xianyi/OpenBLAS’
- Make the library with ‘make’ from within OpenBLAS directory. It should automatically detect the processor for installation
- Install the library with ‘make PREFIX="desired directory" install’
- Compile the library with ‘gcc -shared -o pcat-lion-openblas.so -fPIC pcat-lion-openblas.c -L"desired directory path" -lopenblas’. The -L[path] looks for the installed library in its path, and the -lopenblas searches for anything starting with “lib” that has “openblas” in it. 
