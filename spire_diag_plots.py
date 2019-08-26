from pcat_spire import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from helpers import *
from image_eval import image_model_eval
import cPickle as pickle
import h5py
import sys
import os

np.seterr(divide='ignore', invalid='ignore')

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def load_param_dict(timestr):
    result_path = '/Users/richardfeder/Documents/multiband_pcat/spire_results/'
    filepath = result_path + timestr
    filen = open(filepath+'/params.txt','r')
    print(filen)
    pdict = pickle.load(filen)

    opt = Struct(**pdict)

    print('param dict load')
    print('width:', opt.width)
    return opt, filepath, result_path



def result_plots(timestr, burn_in_frac=0.5, boolplotsave=1, boolplotshow=0, plttype='pdf'):

    band_dict = dict({0:'250 micron', 1:'350 micron', 2:'500 micron'})
    
    opt, filepath, result_path = load_param_dict(timestr)

    dat = pcat_data(opt)
    dat.load_in_data(opt)

    chain = np.load(filepath+'/chain.npz')

    nsrcs = chain['n']
    xsrcs = chain['x']
    ysrcs = chain['y']
    fsrcs = chain['f']
    chi2 = chain['chi2']
    timestats = chain['times']
    accept_stats = chain['accept']
    diff2s = chain['diff2s']
    burn_in = int(opt.nsamp*burn_in_frac)
    bands = opt.bands

    
    # ------------------- SOURCE NUMBER ---------------------------

    plt.figure()
    plt.title('Posterior Source Number Histogram')
    plt.hist(nsrcs[burn_in:], histtype='step', label='Posterior', color='b', bins=15)
    plt.axvline(np.median(nsrcs[burn_in:]), label='Median=' + str(np.median(nsrcs[burn_in:])), color='b', linestyle='dashed')
    plt.xlabel('nstar')
    plt.legend()
    if boolplotsave:
        plt.savefig(filepath +'/posterior_histogram_nstar.'+plttype, bbox_inches='tight')
    if boolplotshow:
        plt.show()
    plt.close()

    # -------------------- CHI2 ------------------------------------

    sample_number = np.arange(burn_in, opt.nsamp)
    full_sample = xrange(opt.nsamp)
    plt.figure()
    plt.title('Chi-Squared Distribution over Catalog Samples')
    for b in xrange(opt.nbands):
        plt.plot(sample_number, chi2[burn_in:,b], label=band_dict[b])
        plt.axhline(np.min(chi2[burn_in:,b]), linestyle='dashed', alpha=0.5, label=str(np.min(chi2[burn_in:,b]))+' (' + str(band_dict[b]) + ')')
    plt.xlabel('Sample')
    plt.ylabel('Chi2')
    plt.legend()
    if boolplotsave:
        plt.savefig(filepath + '/chi2_sample.'+plttype, bbox_inches='tight')
    if boolplotshow:
        plt.show()
    plt.close()

    # ---------------------------- COMPUTATIONAL RESOURCES --------------------------------

    time_array = np.zeros(3, dtype=np.float32)
    labels = ['Proposal', 'Likelihood', 'Implement']
    print(timestats[0])
    for samp in xrange(opt.nsamp):
        time_array += np.array([timestats[samp][2][0], timestats[samp][3][0], timestats[samp][4][0]])
    plt.figure()
    plt.title('Computational Resources')
    plt.pie(time_array, labels=labels, autopct='%1.1f%%', shadow=True)
    if boolplotsave:
        plt.savefig(filepath+ '/time_resource_statistics.'+plttype, bbox_inches='tight')
    if boolplotshow:
        plt.show()
    plt.close()


    # ------------------------------ ACCEPTANCE FRACTION -----------------------------------------

    proposal_types = ['All', 'Move', 'Birth/Death', 'Merge/Split']
    plt.figure()
    plt.title('Proposal Acceptance Fractions')
    for x in xrange(len(proposal_types)):
        if not np.isnan(accept_stats[0,x]):
            plt.plot(full_sample, accept_stats[:,x], label=proposal_types[x])
    plt.legend()
    plt.xlabel('Sample Number')
    plt.ylabel('Acceptance Fraction')
    if boolplotsave:
        plt.savefig(filepath+'/acceptance_fraction.'+plttype, bbox_inches='tight')
    if boolplotshow:
        plt.show()
    plt.close()



    # -------------------------------- ITERATE OVER BANDS -------------------------------------

    for b in xrange(opt.nbands):

        nbins = 20
        lit_number_counts = np.zeros((opt.nsamp - burn_in, nbins-1)).astype(np.float32)
        raw_number_counts = np.zeros((opt.nsamp - burn_in, nbins-1)).astype(np.float32)

        binz = np.linspace(np.log10(opt.trueminf)+3., 3., nbins)
        print(fsrcs.shape)
        for i, j in enumerate(np.arange(burn_in, opt.nsamp)):
            hist = np.histogram(np.log10(fsrcs[b][j])+3, bins=binz)
            logSv = 0.5*(hist[1][1:]+hist[1][:-1])-3
            binz_Sz = 10**(binz-3)
            dSz = binz_Sz[1:]-binz_Sz[:-1]
            dNdS = hist[0]
            raw_number_counts[i,:] = hist[0]
            n_steradian = 0.11/(180./np.pi)**2 # field covers 0.11 degrees, should change this though for different fields
            n_steradian *= opt.frac # a number of pixels in the image are not actually observing anything
            dNdS_S_twop5 = dNdS*(10**(logSv))**(2.5)
            lit_number_counts[i,:] = dNdS_S_twop5/n_steradian/dSz

        print(np.mean(lit_number_counts, axis=0))
        print(np.std(lit_number_counts, axis=0))

        mean = np.mean(lit_number_counts, axis=0)


        plt.figure()  
        plt.errorbar(logSv+3, np.mean(lit_number_counts, axis=0), yerr=np.array([np.abs(mean - np.percentile(lit_number_counts, 16, axis=0)), np.abs(np.percentile(lit_number_counts, 84, axis=0) - mean)]), marker='.')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('log($S_{\\nu}$) (mJy)')
        plt.ylabel('dN/dS.$S^{2.5}$ ($Jy^{1.5}/sr$)')
        plt.ylim(1e0, 1e5)
        plt.xlim(-1.5, 2.5)
        plt.tight_layout()
        if boolplotsave:
            plt.savefig(filepath+'/posterior_number_counts_histogram_'+str(band_dict[b])+'.'+plttype, bbox_inches='tight')
        if boolplotshow:
            plt.show()
        plt.close()

        mean = np.mean(raw_number_counts, axis=0)

        plt.title('Posterior Flux Distribution - ' + str(band_dict[b]))
        plt.errorbar(logSv+3, np.mean(raw_number_counts, axis=0), yerr=np.array([np.abs(mean-np.percentile(raw_number_counts, 16, axis=0)), np.abs(np.percentile(raw_number_counts, 84, axis=0)-mean)]), fmt='o', label='Posterior')
        plt.legend()
        plt.yscale('log', nonposy='clip')
        plt.xlabel('Magnitude - ' + str(band_dict[b]))
        if boolplotsave:
            plt.savefig(filepath+'/posterior_flux_histogram_'+str(band_dict[b])+'.'+plttype, bbox_inches='tight')
        if boolplotshow:
            plt.show()
        plt.close()



result_plots('20190826-154049', burn_in_frac=0.5, plttype='png')





