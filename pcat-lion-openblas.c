#include <stdlib.h>
#include <stdbool.h>
//#include <OpenBLAS/cblas.h>
#include "i_malloc.h"
#define max(a,b) \
    ({ typeof (a) _a = (a);    \
	typeof (b) _b = (b);   \
        _a > _b ? _a : _b; })
#define min(a,b) \
    ({ typeof (a) _a = (a);    \
	typeof (b) _b = (b);   \
        _a < _b ? _a : _b; })

// void pcat_imag_acpt(int NX, int NY, float* image, float* image_acpt, int* reg_acpt, int regsize, int margin, int offsetx, int offsety) {
void pcat_imag_acpt(int NX, int NY, float* image, float* image_acpt, int* reg_acpt, int regsize, int margin, int offsetx, int offsety) {
    int NREGX = (NX / regsize) + 1;
    int NREGY = (NY / regsize) + 1;
    int y0, y1, x0, x1, i, j, ii, jj;
    for (j=0 ; j < NREGY ; j++) {
        y0 = max(j*regsize-offsety-margin, 0);
        y1 = min((j+1)*regsize-offsety+margin, NY);
        for (i=0 ; i < NREGX ; i++) {
                x0 = max(i*regsize-offsetx-margin, 0);
                x1 = min((i+1)*regsize-offsetx+margin, NX);
                if (reg_acpt[j*NREGX+i] > 0) {
                    for (jj=y0 ; jj<y1; jj++)
                     for (ii=x0 ; ii<x1; ii++)
                        image_acpt[jj*NX+ii] = image[jj*NX+ii];
                }
        }
    }
}

void pcat_like_eval(int NX, int NY, float* image, float* ref, float* weight, double* diff2, int regsize, int margin, int offsetx, int offsety) {
    int NREGX = (NX / regsize) + 1;
    int NREGY = (NY / regsize) + 1;
    int y0, y1, x0, x1, i, j, ii, jj;
    for (j=0 ; j < NREGY ; j++) {
        y0 = max(j*regsize-offsety-margin, 0);
        y1 = min((j+1)*regsize-offsety+margin, NY);
        for (i=0 ; i < NREGX ; i++) {
                x0 = max(i*regsize-offsetx-margin, 0);
                x1 = min((i+1)*regsize-offsetx+margin, NX);
                diff2[j*NREGX+i] = 0.;
                for (jj=y0 ; jj<y1; jj++)
                 for (ii=x0 ; ii<x1; ii++)
                    diff2[j*NREGX+i] += (image[jj*NX+ii]-ref[jj*NX+ii])*(image[jj*NX+ii]-ref[jj*NX+ii]) * weight[jj*NX+ii];
        }
    }
}

void pcat_model_eval(int NX, int NY, int nstar, int nc, int k, float* A, float* B, float* C, int* x,
	int* y, float* image, float* ref, float* weight, double* diff2, int regsize, int margin,
	int offsetx, int offsety)
{
    int      i,i2,imax,j,j2,jmax,rad,istar,xx,yy;
    float    alpha, beta;

    int n = nc*nc;
    rad = nc/2;

    alpha = 1.0; beta = 0.0;

    // overwrite and shorten A matrix
    // save time if there are many sources per pixel
    int hash[NY*NX];
    for (i=0; i<NY*NX; i++) { hash[i] = -1; }
    int jstar = 0;
    for (istar = 0; istar < nstar; istar++)
    {
        xx = x[istar];
        yy = y[istar];
        int idx = yy*NX+xx;
        if (hash[idx] != -1) {
            for (i=0; i<k; i++) { A[hash[idx]*k+i] += A[istar*k+i]; }
        }
        else {
            hash[idx] = jstar;
            for (i=0; i<k; i++) { A[jstar*k+i] = A[istar*k+i]; }
            x[jstar] = x[istar];
            y[jstar] = y[istar];
            jstar++;
        }
    }
    nstar = jstar;

    //  matrix multiplication
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        nstar, n, k, alpha, A, k, B, n, beta, C, n);

    //  loop over stars, insert psfs into image    
    for (istar = 0 ; istar < nstar ; istar++)
    {
	xx = x[istar];
	yy = y[istar];
	imax = min(xx+rad,NX-1);
	jmax = min(yy+rad,NY-1);
	for (j = max(yy-rad,0), j2 = (istar*nc+j-yy+rad)*nc ; j <= jmax ; j++, j2+=nc)
	    for (i = max(xx-rad,0), i2 = i-xx+rad ; i <= imax ; i++, i2++)
		image[j*NX+i] += C[i2+j2];
    }

    pcat_like_eval(NX, NY, image, ref, weight, diff2, regsize, margin, offsetx, offsety);
}
