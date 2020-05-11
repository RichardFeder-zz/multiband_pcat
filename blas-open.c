#include <stdlib.h>
#include <stdbool.h>
#include "OpenBLAS/cblas.h"
#include "i_malloc.h"
//#include "mkl_cblas.h"
//#include "i_malloc.h"
#define max(a,b) ({ typeof (a) _a = (a); typeof (b) _b = (b); _a > _b ? _a : _b; })
#define min(a,b) ({ typeof (a) _a = (a); typeof (b) _b = (b); _a < _b ? _a : _b; })

void clib_updt_modl(int numbsidexpos, int numbsideypos,
                    float* cntpmodl, float* cntpmodlacpt, int* regiacpt,
                    int sizeregi, int marg, int offsxpos, int offsypos, int booltile){
    
    int NREGY, NREGX;
    if (booltile > 0){
        NREGX = (numbsidexpos / sizeregi) + 1;
        NREGY = (numbsideypos / sizeregi) + 1;
    }
    else {
        NREGX = (numbsidexpos / sizeregi);
        NREGY = (numbsideypos / sizeregi);
    }
    int y0, y1, x0, x1, i, j, ii, jj;
    for (j=0 ; j < NREGY ; j++){
        y0 = max(j*sizeregi-offsypos-marg, 0);
        y1 = min((j+1)*sizeregi-offsypos+marg, numbsideypos);
        for (i=0 ; i < NREGX ; i++){
            x0 = max(i*sizeregi-offsxpos-marg, 0);
            x1 = min((i+1)*sizeregi-offsxpos+marg, numbsidexpos);
            if (regiacpt[j*NREGX+i] > 0){
                for (jj=y0 ; jj<y1; jj++)
                    for (ii=x0 ; ii<x1; ii++)
                        cntpmodlacpt[jj*numbsidexpos+ii] = cntpmodl[jj*numbsidexpos+ii];
            }
        }
    }
}


void clib_eval_llik(int numbsidexpos, int numbsideypos, 
                    float* cntpmodl, float* cntpresi, float* weig, double* chi2,
                    int sizeregi, int marg, int offsxpos, int offsypos, int booltile){
    
    int NREGY, NREGX;
    if (booltile > 0){
        NREGX = (numbsidexpos / sizeregi) + 1;
        NREGY = (numbsideypos / sizeregi) + 1;
    }
    else {
        NREGX = (numbsidexpos / sizeregi);
        NREGY = (numbsideypos / sizeregi);
    }
    //printf("offsxpos, offsypos: %d %d\n", offsxpos, offsypos);
    
    int y0, y1, x0, x1, i, j, ii, jj;
    for (j=0 ; j < NREGY ; j++){
        y0 = max(j*sizeregi-offsypos-marg, 0);
        y1 = min((j+1)*sizeregi-offsypos+marg, numbsideypos);
        for (i=0 ; i < NREGX ; i++){
            x0 = max(i*sizeregi-offsxpos-marg, 0);
            x1 = min((i+1)*sizeregi-offsxpos+marg, numbsidexpos);
            chi2[j*NREGX+i] = 0.;
            for (jj=y0 ; jj<y1; jj++){
                for (ii=x0 ; ii<x1; ii++){
                    //printf("sizeregi offsypos marg numbsideypos numbsidexpos y1 x1: %d %d %d %d %d %d %d \n", sizeregi, offsypos, marg, numbsideypos, numbsidexpos, y1, x1);
                    //printf("j,i,jj,ii: %d %d %d %d\n", j, i, jj, ii);
                    //printf("cntpmodl[jj*numbsidexpos+ii]: %f\n", cntpmodl[jj*numbsidexpos+ii]);
                    //printf("cntpresi[jj*numbsidexpos+ii]: %f\n", cntpresi[jj*numbsidexpos+ii]);
                    //printf("weig[jj*numbsidexpos+ii]: %f\n", weig[jj*numbsidexpos+ii]);
                    //printf("temp: %f\n\n", (cntpmodl[jj*numbsidexpos+ii]-cntpresi[jj*numbsidexpos+ii]) * \
                    //                            (cntpmodl[jj*numbsidexpos+ii]-cntpresi[jj*numbsidexpos+ii]) * weig[jj*numbsidexpos+ii]);
                    chi2[j*NREGX+i] += (cntpmodl[jj*numbsidexpos+ii]-cntpresi[jj*numbsidexpos+ii]) * \
                                       (cntpmodl[jj*numbsidexpos+ii]-cntpresi[jj*numbsidexpos+ii]) * weig[jj*numbsidexpos+ii];
                }
            }
        }
    //printf("chi[0]: %f\n", chi2[0]);
    }
}

void clib_eval_modl(int numbsidexpos, int numbsideypos, int numbphon, int numbpixlpsfnside, int numbparaspix,
                     float* A, float* B, float* C,
                     int* x, int* y, 
                     float* cntpmodl, float* cntpresi, float* weig, double* chi2, 
                     int sizeregi, int marg, int offsxpos, int offsypos, int booltile)
{
    

    int i, m, t, imax, j, r, jmax, rad, p, xposthis, yposthis;
    float alpha, beta;
    int numbpixlpsfn = numbpixlpsfnside * numbpixlpsfnside;
    int numbpixl = numbsidexpos * numbsideypos;
    rad = numbpixlpsfnside / 2;
    alpha = 1.; beta = 0.;

    // save time if there are many phonions per pixel by overwriting and shorting the A matrix
    int hash[numbsideypos*numbsidexpos];
    for (i=0; i<numbsideypos*numbsidexpos; i++)
        hash[i] = -1;

    int numbphonshrt = 0;
    for (p = 0; p < numbphon; p++){
        xposthis = x[p];
        yposthis = y[p];
        int idx = yposthis*numbsidexpos+xposthis;
        if (hash[idx] != -1){
            for (i=0; i<numbparaspix; i++){
                A[hash[idx]*numbparaspix+i] += A[p*numbparaspix+i];
            }
        }
        else{
            hash[idx] = numbphonshrt;
            for (i=0; i<numbparaspix; i++)
                A[numbphonshrt*numbparaspix+i] = A[p*numbparaspix+i];
            x[numbphonshrt] = x[p];
            y[numbphonshrt] = y[p];
            numbphonshrt++;
        }
    }
    numbphon = numbphonshrt;
    
    //  matrix multiplication
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        numbphon, numbpixlpsfn, numbparaspix, alpha, A, numbparaspix, B, numbpixlpsfn, beta, C, numbpixlpsfn);
    
    // int c;
    // double summ;
    // for (p = 0; p < numbphon; p++){
    //     for (i = 0; i < numbpixlpsfn; i++){
    //         summ = 0.;
    //         for (c = 0; c < numbparaspix; c++){
    //             summ = summ + A[p*numbparaspix+c] * B[c*numbpixlpsfn+i];
    //         }
    //         C[p*numbpixlpsfn+i] = summ;
    //     }
    // }
    
    //  loop over phonions, insert psfs into cntpmodl    
    for (p = 0 ; p < numbphon ; p++){
	    xposthis = x[p];
	    yposthis = y[p];
	    imax = min(xposthis+rad, numbsidexpos-1);
	    jmax = min(yposthis+rad, numbsideypos-1);
	    for (j = max(yposthis-rad, 0), r = (p*numbpixlpsfnside+j-yposthis+rad)*numbpixlpsfnside ; j <= jmax ; j++, r+=numbpixlpsfnside){
	        for (i = max(xposthis - rad, 0), m = i-xposthis+rad ; i <= imax ; i++, m++){
                cntpmodl[j*numbsidexpos+i] += C[m+r];
            }
        }
    }
     
    //printf("sizeregi offsypos marg numbsideypos numbsidexpos: %d %d %d %d %d\n", sizeregi, sizeregi, marg, numbsideypos, numbsidexpos);
    clib_eval_llik(numbsidexpos, numbsideypos, cntpmodl, cntpresi, weig, chi2, sizeregi, marg, offsxpos, offsypos, booltile);
}