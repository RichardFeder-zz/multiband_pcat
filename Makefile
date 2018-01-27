default:
	icc -mkl -shared -static-intel -liomp5 -fPIC -O2 pcat-lion.c -o pcat-lion.so
