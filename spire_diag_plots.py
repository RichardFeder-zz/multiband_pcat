from pcat_spire import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from helpers import *
from image_eval import image_model_eval
import cPickle as pickle
from scipy.ndimage import gaussian_filter
import h5py
import sys
from scipy import stats
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
    opt.auto_resize=False

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
    residuals = chain['residuals']
    burn_in = int(opt.nsamp*burn_in_frac)
    bands = opt.bands


    # ------------------- mean residual ---------------------------


    for b in xrange(opt.nbands):

        residz = residuals[b]
        print(residz.shape)
        median_resid = np.median(residz, axis=0)


        plt.figure(figsize=(10, 5))
        plt.subplot(1,2,1)
        plt.title('Median Residual -- '+band_dict[bands[b]])
        plt.imshow(median_resid, interpolation='none', cmap='Greys', vmin=-0.01, vmax=0.01)
        # plt.imshow(mean_resid, interpolation='none', cmap='Greys', vmin=-0.002, vmax=np.percentile(mean_resid, 90))
        # plt.imshow(mean_resid, interpolation='none', cmap='Greys', vmin=np.percentile(mean_resid, 1), vmax=np.percentile(mean_resid, 99))
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.title('Smoothed Residual')
        smoothed_resid = gaussian_filter(median_resid, sigma=3)
        plt.imshow(smoothed_resid, cmap='Greys', vmin=-0.004, vmax=0.0005)
        # plt.imshow(smoothed_resid, cmap='Greys', vmin=np.percentile(smoothed_resid, 10.), vmax=np.percentile(smoothed_resid, 90))
        plt.colorbar()
        if boolplotsave:
            plt.savefig(filepath +'/median_residual_and_smoothed.'+plttype, bbox_inches='tight')
        if boolplotshow:
            plt.show()
        plt.close()

        median_resid_rav = median_resid[dat.weights[0] != 0.].ravel()


        plt.figure()
        plt.hist(median_resid_rav, bins=np.linspace(-0.02, 0.02, 50))
        if boolplotsave:
            plt.savefig(filepath +'/median_residual_1pt_function.'+plttype, bbox_inches='tight')
        if boolplotshow:
            plt.show()
        plt.close()

        median_resid = median_resid[30:median_resid.shape[1]-30,30:median_resid.shape[0]-30]

        plt.figure()
        plt.imshow(median_resid, vmin=-0.01, vmax=0.01, cmap='Greys', interpolation='none')
        plt.colorbar()
        if boolplotsave:
            plt.savefig(filepath +'/cropped_image.'+plttype, bbox_inches='tight')
        if boolplotshow:
            plt.show()
        plt.close()

        # rbins, radprof, radstd = compute_cl(mean_resid, mean_resid, nbins=18)

        # kmin = 250.
        # plt.figure()
        # plt.plot(rbins*kmin, radprof, marker='.')
        # # plt.plot(rbins*lmin, np.sqrt((rbins*lmin)**2*radprof/(2*np.pi)), marker='.')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('$\\ell$', fontsize=14)
        # plt.ylabel('$C_{\\ell}$', fontsize=14)
        # if boolplotsave:
        #     plt.savefig(filepath +'/residual_power_spectrum.'+plttype, bbox_inches='tight')
        # if boolplotshow:
        #     plt.show()
        # plt.close()

        # plot_radavg_spectrum(rbins, radprofs=[radprof], lmin=500., save=True)

        


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


    nsrc_fov = []

    for b in xrange(opt.nbands):

        nbins = 20
        lit_number_counts = np.zeros((opt.nsamp - burn_in, nbins-1)).astype(np.float32)
        raw_number_counts = np.zeros((opt.nsamp - burn_in, nbins-1)).astype(np.float32)

        binz = np.linspace(np.log10(opt.trueminf)+3., 3., nbins)

        weight = dat.weights[b]

        print(fsrcs.shape)
        print('burnin/nsamp:', burn_in, opt.nsamp)
        print np.arange(burn_in, opt.nsamp)
        for i, j in enumerate(np.arange(burn_in, opt.nsamp)):

            print('nsrcs[j]:', nsrcs[j], j, i)
            print np.array([weight[int(xsrcs[j][k]), int(ysrcs[j][k])] for k in xrange(nsrcs[j])])
            fsrcs_in_fov = np.array([fsrcs[b][j][k] for k in xrange(nsrcs[j]) if weight[int(xsrcs[j][k]),int(ysrcs[j][k])] != 0.])
            nsrc_fov.append(len(fsrcs_in_fov))
            print(len(fsrcs[b][j]), len(fsrcs_in_fov))

            hist = np.histogram(np.log10(fsrcs_in_fov)+3, bins=binz)

            # hist = np.histogram(np.log10(fsrcs[b][j])+3, bins=binz)
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
        plt.xlim(np.log10(opt.trueminf)+3.-0.5, 2.5)
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
        plt.xlabel('log10(Flux) - ' + str(band_dict[b]))
        if boolplotsave:
            plt.savefig(filepath+'/posterior_flux_histogram_'+str(band_dict[b])+'.'+plttype, bbox_inches='tight')
        if boolplotshow:
            plt.show()
        plt.close()

            # ------------------- SOURCE NUMBER ---------------------------

        plt.figure()
        plt.title('Posterior Source Number Histogram')
        plt.hist(nsrc_fov, histtype='step', label='Posterior', color='b', bins=15)
        plt.axvline(np.median(nsrc_fov), label='Median=' + str(np.median(nsrc_fov)), color='b', linestyle='dashed')
        plt.xlabel('nstar')
        plt.legend()
        if boolplotsave:
            plt.savefig(filepath +'/posterior_histogram_nstar.'+plttype, bbox_inches='tight')
        if boolplotshow:
            plt.show()
        plt.close()



result_plots('20190826-191306', burn_in_frac=0.6, plttype='png')





