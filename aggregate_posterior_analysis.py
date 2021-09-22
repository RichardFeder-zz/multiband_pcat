import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from spire_data_utils import *
import pickle
from spire_plotting_fns import grab_extent
import corner
# from pcat_spire import *

import os
import ctypes
import numpy.ctypeslib as npct
from ctypes import c_int, c_double
import numpy.linalg as linalg
from numpy.linalg import inv,det
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter
import scipy
from scipy.optimize import curve_fit


def aggregate_posterior_corner_plot(timestr_list, temp_amplitudes=True, bkgs=True, nsrcs=True, \
                            bkg_bands=[0, 1, 2], temp_bands=[0, 1, 2], n_burn_in=1500, plot_contours=True, tail_name='', pdf_or_png='.pdf', nbands=3, save=True):

    flux_density_conversion_facs = dict({0:86.29e-4, 1:16.65e-3, 2:34.52e-3})
    bkg_labels = dict({0:'$B_{250}$', 1:'$B_{350}$', 2:'$B_{500}$'})
    temp_labels = dict({0:'$A_{250}^{SZ}$', 1:'$A_{350}^{SZ}$', 2:'$A_{500}^{SZ}$'})

    nsrc = []
    bkgs = [[] for x in range(nbands)]
    temp_amps = [[] for x in range(nbands)]
    samples = []

    for i, timestr in enumerate(timestr_list):
        chain = np.load('spire_results/'+timestr+'/chain.npz')

        if nsrcs:
            nsrc.extend(chain['n'][n_burn_in:])
        
        for b in bkg_bands:
            if bkgs:
                bkgs[b].extend(chain['bkg'][n_burn_in:,b].ravel()/flux_density_conversion_facs[b])
        
        for b in temp_bands:
            if temp_amplitudes:
                temp_amps[b].extend(chain['template_amplitudes'][n_burn_in:,b].ravel()/flux_density_conversion_facs[b])


    corner_labels = []

    agg_name = ''

    if bkgs:
        agg_name += 'bkg_'
        for b in bkg_bands:
            samples.append(np.array(bkgs[b]))
            corner_labels.append(bkg_labels[b])
    if temp_amplitudes:
        agg_name += 'temp_amp_'
        for b in temp_bands:
            samples.append(np.array(temp_amps[b]))
            corner_labels.append(temp_labels[b])

    if nsrcs:
        agg_name += 'nsrc_'
        samples.append(np.array(nsrc))
        corner_labels.append("N_{src}")


    samples = np.array(samples).transpose()

    print('samples has shape', samples.shape)


    figure = corner.corner(samples, labels=corner_labels, quantiles=[0.05, 0.16, 0.5, 0.84, 0.95],plot_contours=plot_contours,
                           show_titles=True,title_fmt='.3f', title_kwargs={"fontsize": 14}, label_kwargs={"fontsize":16})



    if save:
        figure.savefig('corner_plot_'+agg_name+tail_name+pdf_or_png, bbox_inches='tight')

    return figure


def compute_gelman_rubin_diagnostic(list_of_chains, i0=0):
    
    list_of_chains = np.array(list_of_chains)
    print('list of chains has shape ', list_of_chains.shape)
    m = len(list_of_chains)
    n = len(list_of_chains[0])-i0
    
    print('n=',n,' m=',m)
    
    B = (n/(m-1))*np.sum((np.mean(list_of_chains[:,i0:], axis=1)-np.mean(list_of_chains[:,i0:]))**2)
    
    W = 0.
    for j in range(m):
        sumsq = np.sum((list_of_chains[j,i0:]-np.mean(list_of_chains[j,i0:]))**2)
                
        W += (1./m)*(1./(n-1.))*sumsq
    
    var_th = ((n-1.)/n)*W + (B/n)
    
    Rhat = np.sqrt(var_th/W)
    
    print("rhat = ", Rhat)
    
    return Rhat, m, n

def compute_hdpi(ts, t_likelihood, frac=0.68):

    ''' computes 1d highest posterior density interval'''
    
    idxs_hdpi = []
    
    idxmax = np.argmax(t_likelihood)
    
    idxs_hdpi.append(idxmax)
    
    psum = t_likelihood[idxmax]
    
    idx0 = np.argmax(t_likelihood)+1
    idx1 = np.argmax(t_likelihood)-1
    while True:
        
        if t_likelihood[idx0] > t_likelihood[idx1]:
            psum += t_likelihood[idx0]
            idxs_hdpi.append(idx0)
            idx0 += 1
        else:
            psum += t_likelihood[idx1]
            idxs_hdpi.append(idx1)
            idx1 -= 1
    
        if psum >= frac:
            print('psum is now ', psum)
            break
        
    ts_credible = np.sort(ts[np.array(idxs_hdpi)])
    
    return ts_credible
    
    
def compute_chain_rhats(all_chains, labels, i0=0, nthin=1):
    
    rhats = []
    for chains in all_chains:
        chains = np.array(chains)
        print(chains.shape)
        rhat, m, n = compute_gelman_rubin_diagnostic(chains[:,::nthin], i0=i0//nthin)
                    
        rhats.append(rhat)
        
    f = plt.figure(figsize=(8,6))
    plt.title('Gelman Rubin statistic $\\hat{R}$ ($N_{c}=$'+str(m)+', $N_s=$'+str(n)+')', fontsize=14)
    barlabel = None
    if nthin > 1:
        barlabel = '$N_{thin}=$'+str(nthin)

    x_pos = [i for i, _ in enumerate(labels)]
    
    plt.bar(x_pos, rhats, width=0.5, alpha=0.4, label=barlabel)
    plt.axhline(1.2, linestyle='dashed', label='$\\hat{R}$=1.2')
    plt.axhline(1.1, linestyle='dashed', label='$\\hat{R}$=1.1')
    plt.xticks(x_pos, labels)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=16)
    plt.ylabel('$\\hat{R}$', fontsize=16)
    plt.tick_params(labelsize=16)
    plt.show()
    
    return f, rhats


def compute_contours_sz(chain_var1, chain_var2, labels, bins=30, level=None, sigma_level=1.0, show=True, save_fpath=None, \
                       xlim=None, ylim=None):

    density, hist2d, extent = compute_posterior_density(chain_var1, chain_var2, bins=bins, norm_mode='max')
    nx, ny = np.meshgrid(hist2d[1], hist2d[2])
    
    if level is None:
        level = 1. - np.exp(-0.5*sigma_level**2)
    print('contour level is ', level)
    
    plt.figure(figsize=(6,6))
    plt.imshow(np.log(density), origin='lower', extent=extent, cmap='Greys', aspect='auto')
    plt.colorbar()
    contours = plt.contour(0.5*(hist2d[1][1:]+hist2d[1][:-1]), 0.5*(hist2d[2][1:]+hist2d[2][:-1]), density, levels=[level])
    segments = contours.collections[0].get_segments()[0]
    plt.scatter(segments[:,0], segments[:,1], color='r')
    plt.legend()
    plt.xlabel(labels[0], fontsize=18)
    plt.ylabel(labels[1], fontsize=18)
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    if show:
        plt.show()
    if save_fpath is not None:
        plt.savefig(save_fpath, bbox_inches='tight')
    
    return contours, segments, density, hist2d

def compute_ellipse_cov(fpath, r2500_conversion_fac=0.163, undo_r2500_conv=True, bins=30, sigma_level=1.0, smaj=None, smin=None, \
                       xlim=None, ylim=None, plot=True, verbose=True, make_corner_plot=True):
    
    '''  
    This function contains the main functionality to 1) load in a collection of posterior samples, 
    2) compute the posterior density over a grid of values, 3) compute the contours corresponding to a desired
    confidence region, 4) compute the best fit ellipse to that contour and rotate the principal components to get
    an estimate of the covariance matrix in the A_SZ basis, assuming a Gaussian probability density. 
        
    '''
    pmw_post, plw_post, nsrc_post = load_post_samples(fpath, r2500_conversion_fac=r2500_conversion_fac, undo_r2500_conv=undo_r2500_conv)

    median_pmw = np.median(pmw_post)
    median_plw = np.median(plw_post)
    if verbose:
        print('Median of PMW is ', median_pmw)
        print('Median of PLW is ', median_plw)

    if make_corner_plot:
        labs = labels=["$A_{350}^{SZ}$", "$A_{500}^{SZ}$", "$N_{src}$", "$B_{250}$", "$B_{350}$", "$B_{500}$"]

        figure = corner.corner(np.array([pmw_post, plw_post, nsrc_post]).transpose(), labels=labs[:3],
                       quantiles=[0.05, 0.16, 0.5, 0.84, 0.95], bins=[20, 20, 15], smooth=True,
                       show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 15}, label_kwargs={"fontsize":18})

    
    # get contours from matplotlib and extract collection of points bounding region
    contours, segments, sz_post_density, hist2d = compute_contours_sz(pmw_post, plw_post,\
                                         ['$A_{SZ}^{350}$ [MJy/sr]', '$A_{SZ}^{500}$ [MJy/sr]'], \
                                        sigma_level=sigma_level, bins=bins, xlim=xlim, ylim=ylim)
    
    # use segment points to fit for an ellipse
    center, phi, axes = find_ellipse(segments[:,0], segments[:,1])

    # rotate the ellipse parameters to obtain a covariance matrix
    if smaj is None:
        smaj = np.argmax(axes)
        smin = np.argmin(axes)

    sz_cov_matrix = reproject_axes_cov_rot(axes, phi, smaj, smin)
    
    ell = Ellipse(xy=center, width=2*axes[0], height=2*axes[1], angle=phi*180/np.pi, alpha=0.5, label='Best fit ellipse')
    if plot:
        plt.figure(figsize=(8,8))
        ax = plt.gca()
        ax.add_patch(ell)
        plt.plot(segments[:,0], segments[:,1])
        plt.scatter(segments[:,0], segments[:,1], color='k')
        plt.scatter([median_pmw], [median_plw], marker='x', label='Posterior median')
        plt.scatter([center[0]], [center[1]], label='Ellipse center')
        plt.legend(fontsize=14, loc=4)
        plt.xlabel('$A_{SZ}^{350}$ [MJy/sr]', fontsize=18)
        plt.ylabel('$A_{SZ}^{500}$ [MJy/sr]', fontsize=18)
        plt.show()
    
    return sz_cov_matrix, sz_post_density, hist2d


def compute_posterior_density(chain_var1, chain_var2, bins=30, norm_mode='max', smooth=False, smooth_sig=5):
    hist2d = np.histogram2d(chain_var1, chain_var2, bins=bins, normed=True)
    density = np.array(hist2d[0]).transpose()
    if smooth:
        density = gaussian_filter(density, sigma=smooth_sig)
    if norm_mode=='max':
        density /= np.max(density)
    elif norm_mode=='sum':
        density /= np.sum(density)
        
    extent = [np.min(hist2d[1]), np.max(hist2d[1]), np.min(hist2d[2]), np.max(hist2d[2])]

    return density, hist2d, extent


def compute_sz_cov_matrix_082021(plw_pmw_cov_input, fixbkgerr=None, gr_stat_fac=None, observed_cov=None, mult_fac=None):
    
    ''' current calculation of the SZ covariance matrix with relevant corrections. '''
    plw_pmw_cov = plw_pmw_cov_input.copy()
    
    if fixbkgerr is not None:
        plw_pmw_cov[0, 0] += fixbkgerr[0]**2
        plw_pmw_cov[1, 1] += fixbkgerr[1]**2
    
    if mult_fac is not None:
        plw_pmw_cov *= mult_fac
        
    if gr_stat_fac is not None and observed_cov is not None:
        print('adding GR factor from observed data')
        
        add_plw = (gr_stat_fac[0]-1.)*observed_cov[0,0]
        add_pmw = (gr_stat_fac[1]-1.)*observed_cov[1,1]
        plw_pmw_cov[0,0] += add_plw
        plw_pmw_cov[1,1] += add_pmw
        
    covmat = np.array([[0., 0., 0.], [0., plw_pmw_cov[0,0], plw_pmw_cov[0, 1]], [0., plw_pmw_cov[1,0], plw_pmw_cov[1, 1]]])

        
    print('new covariance matrix is ')
    print(covmat)
    
    return covmat


''' these functions (fitEllipse(), ellipse_center(), ellipse_angle_of_rotation(), ellipse_axis_length(), find_ellipse()) are taken from stack overflow:
    https://stackoverflow.com/questions/39693869/fitting-an-ellipse-to-a-set-of-data-points-in-python/48002645'''
def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  linalg.eig(np.dot(linalg.inv(S), C))
    n =  np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def ellipse_angle_of_rotation(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))

def ellipse_axis_length(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def find_ellipse(x, y, verbose=True):
    xmean = x.mean()
    ymean = y.mean()
    x = x - xmean
    y = y - ymean
    a = fitEllipse(x,y)
    center = ellipse_center(a)
    center[0] += xmean
    center[1] += ymean
    phi = ellipse_angle_of_rotation(a)
    axes = ellipse_axis_length(a)
    x += xmean
    y += ymean
    
    if verbose:
        print "center = ",  center
        print "angle of rotation (degrees) = ",  phi*180./np.pi
        print "axes = ", axes
    
    return center, phi, axes

def gaussian_chisq(data, model, cov):
    data = np.array(data)
    model = np.array(model)
    cov = np.array(cov)
    sub_dat = data - model
    first = np.dot(sub_dat,inv(cov))
    result = np.dot(first,sub_dat.T)
        
    return result

def GaussSum(x,*p):
    ''' For computing Feldman-Cousins confidence intervals from double Gaussian '''
    n=len(p)//3
    A=p[:n]
    c=p[n:2*n]
    w=p[2*n:3*n]
    y = sum([A[i]*np.exp(-(x-c[i])**2./(2.*(w[i])**2.))/(2*np.pi*w[i]**2)**0.5 for i in range(n)])
    return y


def gather_posteriors_different_amps(timestr_list, label_dict):

    all_temp_posteriors = []
    figs = []


    band_dict = dict({0:'250 micron', 1:'350 micron', 2:'500 micron'})


    temp_mock_amps = [None, 0.3, 0.5] # MJy/sr
    flux_density_conversion_facs = [86.29e-4, 16.65e-3, 34.52e-3]

    ratios = [0.0, 0.5, 1.0, 1.5]
    temp_mins = [-0.003, -0.012]
    temp_maxs = [0.011, 0.027]


    for i in range(2):

        f = plt.figure(figsize=(12, 4))


        for j, timestr in enumerate(timestr_list):
            gdat, filepath, result_path = load_param_dict(timestr, result_path='spire_results/')

            chain = np.load(filepath+'/chain.npz')

            band=band_dict[gdat.bands[i+1]]

            if j==0:
                linelabel = 'Recovered SZ'
                plt.title('Injected and Recovered SZ amplitudes (RX J1347.5-1145, '+band+')', fontsize=18)
            else:
                linelabel = None
            # if j==0:
            #   plt.title(band)
            #   label='Indiv. mock realizations'
            # else:
            #   label = None

            colors = ['C0', 'C1', 'C2', 'C3']

            burn_in = int(gdat.nsamp*gdat.burn_in_frac)

            template_amplitudes = chain['template_amplitudes'][burn_in:, i+1, 0]

            bins = np.linspace(temp_mins[i], temp_maxs[i], 50)

            plt.hist(template_amplitudes, histtype='stepfilled',alpha=0.7,edgecolor='k', bins=bins,label=linelabel,  color=colors[j], linewidth=2)

            plt.axvline(ratios[j]*temp_mock_amps[i+1]*flux_density_conversion_facs[i+1], color=colors[j], label=label_dict[i+1][j], linestyle='dashed', linewidth=3)

            # all_temp_posteriors.extend(template_amplitudes)


        # plt.hist(all_temp_posteriors, label='Aggregate Posterior', histtype='step', bins=30)

        # plt.axvline(0., label='Injected SZ Amplitude', linestyle='dashed', color='g')

        plt.legend()
        plt.xlabel('Template amplitude [mJy/beam]', fontsize=14)
        plt.ylabel('Number of posterior samples', fontsize=14)
        plt.savefig('recover_injected_sz_template_amps_band_'+str(i+1)+'.pdf', bbox_inches='tight')

        # plt.savefig('aggregate_posterior_sz_template_no_injected_signal_band_'+str(i+1)+'.pdf', bbox_inches='tight')
        plt.show()

        figs.append(f)

    return figs



def return_step_func_hist(xvals, hist_bins, hist_vals):
    all_step_vals = np.zeros_like(xvals)
    for i, x in enumerate(xvals):
        for j in range(len(hist_bins)-1):
            if hist_bins[j] <= x < hist_bins[j+1]:
                all_step_vals[i] = hist_vals[j]
                
    return all_step_vals

def handselect_residual(timestr_list_file=None, fmin_subtract=0.01, timestr_list=None, inject_sz_frac=None, tail_name='9_24_20', datatype='real', pdf_or_png ='.png', save=False, dust=False, ref_dust_amp=1.):

    if timestr_list_file is not None:
        timestr_list = np.load(timestr_list_file)['timestr_list']
    elif timestr_list is None:
        print('no list of runs specified, ending now')
        return

    band_dict = dict({0:'250 micron', 1:'350 micron', 2:'500 micron'})

    temp_mock_amps = [0.0111, 0.1249, 0.6912]
    flux_density_conversion_facs = [86.29e-4, 16.65e-3, 34.52e-3]

    temp_mock_amps_dict = dict({'S':0.0111, 'M': 0.1249, 'L': 0.6912})
    figs = []
    median_select_resid_list = []
    for i in np.arange(1):
        all_resids = []

        if inject_sz_frac is not None:
            inject_sz_amp = inject_sz_frac*temp_mock_amps[i]
            ref_vals.append(inject_sz_amp)
            print('inject sz amp is ', inject_sz_amp)

        f = plt.figure(figsize=(10,10))
        plt.title('median residual, $f_{min}$='+str(np.round(fmin_subtract, 4))+', '+band_dict[i], fontsize=18)
        for j, timestr in enumerate(timestr_list):
            if timestr=='':
                continue
            gdat, filepath, result_path = load_param_dict(timestr, result_path='spire_results/')
            
            datapath = gdat.base_path+'/Data/spire/'+gdat.dataname+'/'
            
#            print('filepath:', filepath)
#            print('datapath:', datapath)

            dat = pcat_data(gdat.auto_resize, nregion=gdat.nregion)
            dat.load_in_data(gdat)


            chain = np.load(filepath+'/chain.npz')

            xsrcs = chain['x']
            ysrcs = chain['y']
            fsrcs = chain['f']
            chi2modl = chain['chi2sample']

#            print('Minimum chi2 is ', np.min(chi2modl, axis=0))
#            print(np.array(xsrcs).shape, np.array(fsrcs.shape))

            bkgs = chain['bkg']

            template_amplitudes = chain['template_amplitudes']

            print(np.array(xsrcs).shape, np.array(fsrcs.shape))
            for s in range(100):
            #for s in range(len(xsrcs)):
                
                fmask = (fsrcs[0,:,-s] > fmin_subtract)
                print('fmask has shape', fmask.shape)
                x_fmask = xsrcs[fmask,-s]
                y_fmask = ysrcs[fmask,-s]
                fs_fmask = fsrcs[0,fmask,-s]
                print('x_fmask has shape ', x_fmask.shape)
                print('f_fmask has shape ', fs_fmask.shape)
                print(fs_fmask)
                dtemp = []
                for t, temp in enumerate(dat.template_array[i]):
                    if temp is not None and template_amplitudes[s,t,i]:
                        dtemp.append(template_amplitudes[s,t,i]*temp)

                if len(dtemp) > 0:
                    dtemp = np.sum(np.array(dtemp), axis=0).astype(np.float32)

                libmmult = ctypes.cdll['./blas.so']
                lib = libmmult.clib_eval_modl
                print('imszs:', gdat.imszs[i])
                pixel_per_beam = 2*np.pi*((3.)/2.355)**2
                dmodel, diff2 = image_model_eval(x_fmask, y_fmask, pixel_per_beam*dat.ncs[i]*fs_fmask, bkgs[i,-s], gdat.imszs[i].astype(np.int32), dat.ncs[i],\
                                                 np.array(dat.cfs[i]).astype(np.float32()), weights=dat.weights[i],\
                                                 lib=lib, template=dtemp)
            

                
                r = dat.data_array[i]-dmodel[i]
                print('residual has shape', r.shape)
                print(r)
                all_resids.append(r)

            median_select_resid = np.median(np.array(all_resids), axis=0)
            median_select_resid_list.append(median_select_resid)

            plt.imshow(median_select_resid, cmap='Greys', vmin=np.percentile(median_select_resid, 5), vmax=np.percentile(median_select_resid, 95))
        plt.xlabel('x [pixel]',fontsize=16)
        plt.ylabel('y [pixel]', fontsize=16)
            

        if save:
            plt.savefig('agg_posts/median_residual_fminsub='+str(fmin_subtract)+'_'+tail_name+'.'+pdf_or_png)
        figs.append(f)
        plt.close()

        return median_select_resid_list, figs

#ms, figs = handselect_residual(fmin_subtract=0.01, timestr_list_file='rxj1347_mock_test_9_24_20_10sims.npz')

class pcat_agg():
    
    flux_density_conversion_facs = [86.29e-4, 16.65e-3, 34.52e-3]
    band_dict = dict({0:'250 micron', 1:'350 micron', 2:'500 micron'})

    def __init__(self, base_path='/home/mbzsps/multiband_pcat/', result_path='/home/mbzsps/multiband_pcat/spire_results/'):
        self.base_path = base_path
        self.result_path = result_path
        self.chain = None

    def load_chain(self, timestr, inplace=True):
        chain = np.load(self.result_path+timestr+'/chain.npz')
        if inplace:
            self.chain = chain
        else:
            return chain

    def load_timestr_list(self, timestr_list_file, inplace=False):
        timestr_list = np.load(timestr_list_file)['timestr_list']
        if inplace:
            self.timestr_list = timestr_list
        else:
            return timestr_list

    def grab_acceptance_fracs(self, chain=None):
        if chain is None:
            chain = self.chain
        acceptance_fracs = chain['accept']
        print('acceptance_fracs has shape', acceptance_fracs.shape)
        return acceptance_fracs

    def grab_chi2_stats(self, chain=None):
        if chain is None:
            chain = self.chain
        chi2_stats = chain['chi2']
        print('chi2 stats has shape', chi2_stats.shape)
        return chi2_stats
    
    def grab_bkg_vals(self, chain=None):
        if chain is None:
            chain = self.chain
            
        bkg_vals = chain['bkg']
        #print(bkg_vals)
        print('bkg vals has shape', bkg_vals.shape)
        return bkg_vals

    def compile_stats(self, mode='accept', timestr_list_file=None, inplace=False):
        
        all_stats = []

        if timestr_list_file is not None:
            self.load_timestr_list(timestr_list_file, inplace=True)

        for t, timestr in enumerate(self.timestr_list):
            print(timestr)
            self.load_chain(timestr, inplace=True)
            if mode=='accept':
                stats = self.grab_acceptance_fracs()
            elif mode=='chi2':
                stats = self.grab_chi2_stats()
            elif mode=='bkg':
                stats = self.grab_bkg_vals()
                #if t==0:
            #    stats_shape = stats.shape
            #    all_stats_shape = stats_shape.copy()
            #    all_stats_shape[0] *= len(timestr_list_file)
            #    print('all stats shape is now ', all_stats_shape)
            #    all_stats = np.zeros((all_stats_shape))
            
            all_stats.append(stats)

            #if len(all_stats_shape)==2:
            #    all_stats[i*stats_shape[0]:(i+1)*stats_shape[0],:] = stats
            #elif len(all_stats_shape)==3:
            #    all_stats[i*stats_shape[0]:(i+1)*stats_shape[0],:,:] = stats

        all_stats = np.array(all_stats)

        print('all stats has shape', all_stats.shape)
        if inplace:
            self.all_stats = all_stats
        else:
            return all_stats


# for getting bkg means
get_bkg = False
if get_bkg:
    timestr_sim_file = 'rxj1347_conley_10arcmin_062121_timestrs_fitbkg.npz'
    pcat_agg_obj = pcat_agg()
    burn_in = 1400
    timestr_list = pcat_agg_obj.load_timestr_list(timestr_sim_file, inplace=False)

    all_bkg_vals = pcat_agg_obj.compile_stats(mode='bkg', timestr_list_file=timestr_sim_file)

    print('all_bkg_vals has shape', all_bkg_vals.shape)
    all_bkg_vals = all_bkg_vals[:,burn_in:, :]

    mean_bkg_vals = np.mean(all_bkg_vals, axis=1)

    print('mean bkg vals is ', mean_bkg_vals)
    print(mean_bkg_vals.shape)

    sim_idx_list = []
    mean_list = []

    for timestr in timestr_list:
        gdat, filepath, result_path = load_param_dict(timestr, result_path='spire_results/')
        sim_idx = int(gdat.tail_name[-3:])
        print('sim_idx is ', sim_idx)
    
        sim_idx_list.append(sim_idx)

    print(sim_idx_list)
    print(mean_bkg_vals)

    np.savez('rxj1347_conley_10arcmin_062121_bkg_best_fits.npz', sim_idx_list=sim_idx_list, mean_bkg_vals=mean_bkg_vals)


    exit()



def gather_posteriors(timestr_list=None, timestr_list_file=None, inject_sz_frac=None, tail_name='6_4_20', band_idx0=0, datatype='real', pdf_or_png='.png', save=False, dust=False, ref_dust_amp=1., integrate_sz_prof=False, burn_in_frac=None, spr=False):

    if timestr_list_file is not None:
        timestr_list = np.load(timestr_list_file)['timestr_list']
    elif timestr_list is None:
        print('no list of runs specified, ending now')
        return

#     all_temp_posteriors = []
    figs = []

    band_dict = dict({0:'250 micron', 1:'350 micron', 2:'500 micron'})

    # temp_mock_amps = [None, 0.3, 0.5] # MJy/sr
    #temp_mock_amps = [0.0111, 0.1249, 0.6912]
    temp_mock_amps = [0.03, 0.20, 0.80] # updated values from X-ray measurement of RXJ1347
    flux_density_conversion_facs = [86.29e-4, 16.65e-3, 34.52e-3]

    #temp_mock_amps_dict = dict({'S':0.0111, 'M': 0.1249, 'L': 0.6912})
    temp_mock_amps_dict = dict({'S':0.03, 'M':0.20, 'L':0.80}) # udpated values from X-ray measurement
    medians = []
    pcts_5 = []
    pcts_16, pcts_84, pcts_95 = [[] for x in range(3)]
    ref_vals = []
    dont_include_idxs = []
    #dont_include_idxs = [331, 332, 333, 334]
    #lensed_cat_idxs = np.load('lensed_cat_criteria_45arcsec_20mJy.npz')
    #dont_include_idxs = lensed_cat_idxs['unsatisfied_idxs']
    #print(dont_include_idxs)
    f = plt.figure(figsize=(15, 5), dpi=200)
    plt.suptitle(tail_name, fontsize=20, y=1.04)

    indiv_sigmas_list, indiv_medians_list, indiv_84pcts_list, indiv_16pcts_list = [], [], [], []
    list_of_posts, list_of_chains = [], []

    for i in np.arange(band_idx0, 3):
        mocksim_names, all_temp_posteriors, all_temp_ravel = [], [], []
        indiv_sigmas, indiv_medians, indiv_84pcts, indiv_16pcts = [], [], [], []
        all_temp_chains, all_bkg_chains = [], []
        sim_idx_list = []
        all_nsrc_chains = []
        bkg_means = []
        plt.subplot(1,3, i+1)

        if inject_sz_frac is not None:
            inject_sz_amp = inject_sz_frac*temp_mock_amps[i]
            ref_vals.append(inject_sz_amp)
            print('inject sz amp is ', inject_sz_amp)

        sim_idxs = []
        for j, timestr in enumerate(timestr_list):
  #          print('timestr issssss', timestr)
            gdat, filepath, result_path = load_param_dict(timestr, result_path='spire_results/')
            if j==0:
                print('gdat file name is ', gdat.tail_name, ' and injected sz frac is ', gdat.inject_sz_frac)
            # print('inject sz frac is ', inject_sz_frac, ' while in gdat it is ', gdat.inject_sz_frac)
            #if gdat.tail_name in mocksim_names or '0230' in gdat.tail_name:
            #    print('already have this one, going to next')
            #    continue
            #dont_include_idxs = []
            print(gdat.bands)
            boolvar = 0
            for dont_include_idx in dont_include_idxs:
                if str(dont_include_idx) == gdat.tail_name[-3:]:
                    print('PASS!:', dont_include_idx, gdat.tail_name)
                    boolvar = 1
                    #print(dont_include_idx)
                    #print('pass!')
                    continue

            if gdat.tail_name[-3:] in sim_idx_list and datatype != 'real':
                print('we already have this sim, skipping it', gdat.tail_name[-3:])
                print(sim_idx_list)
                boolvar = 1

            if boolvar==1:
                continue

            print(gdat.tail_name)
            print(gdat.image_extnames)
            
            mocksim_names.append(gdat.tail_name)
            sim_idx_list.append(gdat.tail_name[-3:])    
            chain = np.load(filepath+'/chain.npz')
            #print(chain['template_amplitudes'].shape())
            chi2modl = chain['chi2']
            bkgs = chain['bkg']
            
            bkg_means.append(np.mean(bkgs, axis=0)[i])
            #print('mean_bkg is ', np.mean(bkgs, axis[i])
 #           print('Minimum chi2 is ', np.min(chi2modl, axis=0))
            #print(i)
            #print(i, gdat.bands[i])
            #band=band_dict[gdat.bands[i]]
            band = band_dict[i]
            sim_idxs.append(gdat.tail_name[-3:])
            if j==0:
            #   plt.title(band+', $\\langle \\delta F\\rangle = $'+str(np.median(all_temp)))
                if datatype=='real':
                    label='Indiv. chains'
                else:
                    label='Indiv. mock realizations'

            else:
                label = None

            if burn_in_frac is None:
                burn_in_frac = gdat.burn_in_frac
            burn_in = int(gdat.nsamp*burn_in_frac)
            # burn_in = int(gdat.nsamp*0.75)

            if i==0:
                nsrc_chains = chain['n']
                print(nsrc_chains)
                all_nsrc_chains.append(nsrc_chains)
            
            bkg_chains = chain['bkg']
            
            all_bkg_chains.append(bkg_chains[:,i])

            if dust:
                template_amplitudes = chain['template_amplitudes'][burn_in:, 1, i]
            else:
                template_amplitudes = chain['template_amplitudes'][burn_in:, 0, i]/flux_density_conversion_facs[i]
                full_chain_amps = chain['template_amplitudes'][:,0,i]/flux_density_conversion_facs[i]
                
                if integrate_sz_prof:
                    
                    print('INTEGRATE SZ PROF')
                    t = 0 # use sz template                                                                                                                                                            
                    dat = pcat_data(gdat.auto_resize, nregion=gdat.nregion)
                    dat.load_in_data(gdat)
                    pixel_sizes = dict({'S':6, 'M':8, 'L':12}) # arcseconds                                                                                                                        
                    #print('max of template is ', np.max(dat.template_array[i][t]))
                    npix = dat.template_array[i][t].shape[0]*dat.template_array[i][t].shape[1]
                    geom_fac = (np.pi*pixel_sizes[gdat.band_dict[gdat.bands[i]]]/(180.*3600.))**2
                    #print('geometric factor is ', geom_fac)

                    print('integrating sz profiles..')

                    template_amplitudes = np.array([np.sum(amp*dat.template_array[i][t]) for amp in template_amplitudes])     
                    template_amplitudes *= geom_fac
                    template_amplitudes *= 1e6 # MJy to Jy                                                                                                                                        
                    #print('final template flux densities are ', template_amplitudes)

                    
   #         print('indiv median:', np.median(template_amplitudes))
            indiv_medians.append(np.median(template_amplitudes))
            indiv_sigmas.append(np.std(template_amplitudes))
            indiv_84pcts.append(np.percentile(template_amplitudes, 84))
            indiv_16pcts.append(np.percentile(template_amplitudes, 16))
            all_temp_posteriors.append(template_amplitudes)
            all_temp_ravel.extend(template_amplitudes)
            all_temp_chains.append(full_chain_amps)
        print('length of mocksim_names is ', len(mocksim_names))
        print(mocksim_names)
        print('average background for band ', i, 'is ', np.mean(np.array(bkg_means)), np.std(np.array(bkg_means)))
        print('average sig for band', i, 'is ', np.mean(np.array(indiv_sigmas)))
        all_n, bins, _  = plt.hist(all_temp_ravel, label='Aggregate Posterior', histtype='step', bins=20, color='k')


        for k, t in enumerate(all_temp_posteriors):
            n, _, _ = plt.hist(t, bins=bins, color='black', histtype='stepfilled', linewidth=1.5, alpha=0.15)
            idx = np.argmax(n)
            plt.text(bins[idx], 1.1*n[idx], sim_idxs[k], fontsize=12)
            print('mean/median:', np.mean(t), np.median(t))

        all_temp = np.array(all_temp_ravel)

        print(all_temp.shape, np.median(all_temp), np.std(all_temp), np.percentile(all_temp, 84)-np.median(all_temp), np.median(all_temp)-np.percentile(all_temp, 16))
        
        medians.append(np.median(all_temp_ravel))
        pcts_5.append(np.percentile(all_temp_ravel, 5))
        pcts_16.append(np.percentile(all_temp_ravel, 16))
        pcts_84.append(np.percentile(all_temp_ravel, 84))
        pcts_95.append(np.percentile(all_temp_ravel, 95))

        indiv_medians_list.append(indiv_medians)
        indiv_sigmas_list.append(indiv_sigmas)
        indiv_84pcts_list.append(indiv_84pcts)
        indiv_16pcts_list.append(indiv_16pcts)
        median_str_noinj = str(np.round(np.median(all_temp_ravel), 4))
        str_plus_noinj = str(np.round(np.percentile(all_temp_ravel, 84)-np.median(all_temp_ravel), 4))
        str_minus_noinj = str(np.round(-np.percentile(all_temp_ravel, 16)+np.median(all_temp_ravel), 4))
        if inject_sz_frac is not None:

            
            if dust:
                median_str = str(np.round(np.median(all_temp)-1.,4))

                str_plus = str(np.round(np.percentile(all_temp, 84) - 1., 4))
                str_minus = str(np.round(1. - np.percentile(all_temp, 16), 4))
                unit = ''
            else:
                median_str = str(np.round(np.median(all_temp)-float(inject_sz_amp),4))
                str_plus = str(np.round(np.percentile(all_temp, 84) - np.median(all_temp),4))
                str_minus = str(np.round(np.median(all_temp)-np.percentile(all_temp, 16),4))
                unit = ' MJy/sr'
                
                if integrate_sz_prof:
                    unit = ' Jy'

            if dust:
                plt.title(band+', $\\langle \\delta A_{dust}\\rangle = $'+median_str+'$^{+'+str_plus+'}_{-'+str_minus+'}$'+unit)
            else:
                plt.title(band+', $\\langle \\delta I\\rangle = $'+median_str+'$^{+'+str_plus+'}_{-'+str_minus+'}$'+unit)
        else:
            unit = ' MJy/sr'
            if integrate_sz_prof:
                unit = ' Jy'
                plt.title(band+', $\\langle \\int \\delta I d\\Omega \\rangle = $'+median_str_noinj+'$^{+'+str_plus_noinj+'}_{-'+str_minus_noinj+'}$'+unit)
            else:
                plt.title(band+', $\\langle \\delta I \\rangle = $'+median_str_noinj+'$^{+'+str_plus_noinj+'}_{-'+str_minus_noinj+'}$'+unit)
        #medians.append(np.median(all_temp_ravel))
        #pcts_5.append(np.percentile(all_temp_ravel, 5))
        #pcts_16.append(np.percentile(all_temp_ravel, 16))
        #pcts_84.append(np.percentile(all_temp_ravel, 84))
        #pcts_95.append(np.percentile(all_temp_ravel, 95))
        
        if inject_sz_frac is not None:
            if dust:
                plt.axvline(1., label='Fiducial dust amp.', linestyle='solid', color='k', linewidth=4)
            else:  
                plt.axvline(inject_sz_amp, label='Injected SZ Amplitude', linestyle='solid', color='k', linewidth=4)

        
        plt.axvline(medians[i], label='Median', linestyle='dashed', color='r')

        bin_cents = 0.5*(bins[1:]+bins[:-1])
    
        pm_1sig_fine = np.linspace(pcts_16[i], pcts_84[i], 300)
        all_n_fine_1sig = return_step_func_hist(pm_1sig_fine, bins, all_n)
        pm_2sig_fine = np.linspace(pcts_5[i], pcts_95[i], 300)
        all_n_fine_2sig = return_step_func_hist(pm_2sig_fine, bins, all_n)

        plt.fill_between(pm_1sig_fine, 0, all_n_fine_1sig, interpolate=True, color='royalblue')    
        plt.fill_between(pm_2sig_fine, 0, all_n_fine_2sig, interpolate=True, color='royalblue', alpha=0.4)    

        #plt.xlim(-0.05, 0.2)

        #plt.legend()
        if dust:
            plt.xlabel('Template amplitude')
        else:
            if integrate_sz_prof:
                plt.xlabel('Integrated flux density [Jy]')
            else:
                plt.xlabel('Template amplitude [MJy/sr]')

        plt.ylabel('Number of samples')
        if i==0:
            list_of_chains.append(all_nsrc_chains)
        list_of_posts.append(all_temp_posteriors)
        list_of_chains.append(all_temp_chains)
        list_of_chains.append(all_bkg_chains)
    
    if save:
        if spr:
            plt.savefig('spr_agg_posts/agg_post_'+tail_name+pdf_or_png, bbox_inches='tight')
        elif dust:
            
            plt.savefig('agg_posts/agg_post_dust_'+tail_name+pdf_or_png, bbox_inches='tight')
        else:
            if integrate_sz_prof:
                plt.savefig('agg_posts/agg_post_sz_integrated_'+tail_name+pdf_or_png, bbox_inches='tight')
            else:
                plt.savefig('agg_posts/agg_post_sz_'+tail_name+pdf_or_png, bbox_inches='tight')


    plt.show()


    return f, medians, pcts_16, pcts_84, pcts_5, pcts_95, ref_vals, indiv_medians_list, indiv_sigmas_list, indiv_84pcts_list, indiv_16pcts_list, list_of_posts, list_of_chains



def load_post_samples(fpath, r2500_conversion_fac=0.163, undo_r2500_conv=True):

    ''' Loads SZ/Nsrc posterior samples from file containing compiled samples. '''
    post_samples = np.load(fpath)
    
    pmw_post = post_samples['pmw']
    plw_post = post_samples['plw']
    nsrc_post = post_samples['nsrc']
    
    if undo_r2500_conv:
        pmw_post /= r2500_conversion_fac
        plw_post /= r2500_conversion_fac
        
    return pmw_post, plw_post, nsrc_post

def reproject_axes_cov_rot(axes, phi, smaj, smin, to_rad = False):
    ''' 
    Rotates the eigenvectors of the covariance ellipse by an angle phi to get
    the covariance matrix in the A_SZ basis. 
        
    Inputs
    ------
    phi : 'float' with units of radians
    axes : list of floats, principal axes of the covariance ellipse

    Returns
    -------
    cov_matrix : np.array of size (2, 2)
                Covariance matrix in A_SZ basis
    
    '''
    
    if to_rad:
        phi *= np.pi/180.
    rotation_matrix = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    
    diagonalized_cov_matrix = np.array([[axes[smaj]**2, 0], [0, axes[smin]**2]])
    cov_matrix = np.dot(np.linalg.inv(rotation_matrix), np.dot(diagonalized_cov_matrix, rotation_matrix))
    
    return cov_matrix


def split_timestr_lists_by_simidx(timestr_list):

    list_of_simidxs = []
    list_of_timestr_lists = []
    list_of_simidx_lists = []
    for t, timestr in enumerate(timestr_list):
        gdat, filepath, result_path = load_param_dict(timestr, result_path='spire_results/')
        simidx = gdat.tail_name[-3:]
        if simidx not in list_of_simidxs:
            list_of_simidxs.append(simidx)
            list_of_timestr_lists.append([])
            list_of_simidx_lists.append([])

        timestr_list_idx = list_of_simidxs.index(simidx)
        list_of_timestr_lists[timestr_list_idx].append(timestr)
        list_of_simidx_lists[timestr_list_idx].append(simidx)

    return list_of_simidxs, list_of_timestr_lists, list_of_simidx_lists


def sample_corrected_post(pmw_post_bias, plw_post_bias, pmw_obs, plw_obs, pmw_GR=0.14, plw_GR=0.2, \
                         r2500_conversion_fac=0.163, r2500_central_observed_pmw=0.006, r2500_central_observed_plw=0.009, \
                         pmw_bkg_fix=0.02, plw_bkg_fix=0.008):
    
    ''' This is for propagating samples from the collection of lensed mocks and the observed dI values to a 
        lensing-corrected posterior. This function applies to appropriate penalties for 1) fixing our backgrounds and
        2) imperfect chain convergence, as measured through the Gelman-Rubin statistic
    '''
    
    median_pmw_post_bias = np.median(pmw_post_bias)
    median_plw_post_bias = np.median(plw_post_bias)
    
    pmw_obs_corr_median = pmw_obs + median_pmw_post_bias
    plw_obs_corr_median = plw_obs + median_plw_post_bias
    
    zc_pmw_bias = pmw_post_bias - median_pmw_post_bias
    zc_plw_bias = plw_post_bias - median_plw_post_bias
    
    rescaled_pmw_bias_samps = []
    rescaled_plw_bias_samps = []
    
    for i in range(len(zc_pmw_bias)):
        
        # fixed background additional uncertainty
        zcr_pmw = zc_pmw_bias[i] + np.random.normal(0, pmw_bkg_fix)
        zcr_plw = zc_plw_bias[i] + np.random.normal(0, plw_bkg_fix)
        
        # GR penalty using scatter on real data scaled by R
        zcr_pmw += np.random.normal(0, (pmw_GR)*(r2500_central_observed_pmw/r2500_conversion_fac))
        zcr_plw += np.random.normal(0, (plw_GR)*(r2500_central_observed_plw/r2500_conversion_fac))

        
        rescaled_pmw_bias_samps.append(zcr_pmw)
        rescaled_plw_bias_samps.append(zcr_plw)
        
    return rescaled_pmw_bias_samps+pmw_obs_corr_median, rescaled_plw_bias_samps+plw_obs_corr_median
    

def upsample_log_post(log_post, hist2d=None, upsample_fac=10, order=1):
    ''' 
    Upsample grid of log posterior values
    
    log_post : np.array of floats representing log posterior. Assumes 2D posterior for now
    
    upsample_fac (optional): integer value for upsample factor. Default is 10.
    
    order (optional) : integer representing order of interpolation. Default is 1 (bilinear interpolation)
    
    '''
    log_post_upsampled = scipy.ndimage.zoom(log_post, upsample_fac, order=1)

    
    if hist2d is not None:
        
        bins_var1 = scipy.ndimage.zoom(hist2d[1], upsample_fac, order=1)
        bins_var2 = scipy.ndimage.zoom(hist2d[2], upsample_fac, order=1)
        
        return log_post_upsampled, [bins_var1, bins_var2]
    
    return log_post_upsampled


#timestr_realdat_file = 'rxj1347_realdat_4arcmin_nfcterms=3_timestrs_041621_bkg_vals3.npz'
# timestr_realdat_file = 'rxj1347_realdat_10arcmin_nfcterms=6_timestrs_042021_fitbkg.npz'
#timestr_bkg_file = 'rxj1347_conley_10arcmin_041921_timestrs_fitbkg.npz'

#timestr_bkg_file = 'rxj1347_conley_4arcmin_041921_timestrs_bestfitbkgs.npz'
#timestr_bkg_file = 'rxj1347_conley_4arcmin_unlensed_042021_timestrs_bestfitbkgs.npz'
#timestr_sim_file = 'lensed_1xdust_nfcterms=6_simidx300_30chains_conley_rxj1347_timestrs_3_23_21_narrow_cprior_Fmin=5.npz'
#timestr_sim_file = 'group_timestrs/unlensed_nodust_conley_rxj1347_timestrs_withnoise_3_2_21_narrow_cprior_Fmin=5.npz'
#timestr_sim_file = 'unlensed_1xdust_nfcterms=3_conley_rxj1347_4arcmin_simidx300_withnoise_timestrs_041021.npz'
#timestr_sim_file = 'lensed_1xdust_nfcterms=6_conley_rxj1347_timestrs_withnoise_3_31_21_smallmask_narrow_cprior_Fmin=5.npz'
#truealpha = 2.5
#timestr_dust_file = 'lensed_1xdust_nfcterms=6_truealpha='+str(truealpha)+'_conley_rxj1347_timestrs_3_17_21_narrow_cprior_Fmin=5.npz'
#timestr_dust_file = 'lensed_16xdust_nfcterms='+str(n_fc_terms)+'_conley_rxj1347_timestrs_withnoise_3_22_21_narrow_cprior_Fmin=5.npz'

#fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _, all_temp_posteriors, full_chains = gather_posteriors(timestr_list_file=timestr_sim_file, save=save_figs, tail_name='rxj1347_sims_nfcterms='+str(n_fc_terms)+'_unlensed_4arcmin_041021_smallmask_narrow_cprior_Fmin=5', integrate_sz_prof=False, dust=False, inject_sz_frac=1.0, datatype='real', burn_in_frac=0.5)

#out_tail = '4arcmin_041021_fixbkg_errdivfac=15'
#out_tail = '4arcmin_041021_fixbkg'
#out_tail = '4arcmin_041221_fixbkg_nreg2_longo'

#out_tail = '4arcmin_041921_bestfitbkg_unlensed'
# out_tail = '10arcmin_042021_fitbkg_realdat'
# dirpath = out_tail
# if not os.path.isdir(dirpath):
    # os.makedirs(dirpath)

# pcagg = pcat_agg()
# all_accept_stats = pcagg.compile_stats(mode='accept', timestr_list_file=timestr_realdat_file)
# np.savez(dirpath+'/all_acceptance_stats_'+out_tail+'.npz', all_accept_stats = all_accept_stats, timestr_list_file=timestr_realdat_file)
# all_chi2_stats = pcagg.compile_stats(mode='chi2', timestr_list_file=timestr_realdat_file)
# np.savez(dirpath+'/all_chi2_stats_'+out_tail+'.npz', all_chi2_stats = all_chi2_stats, timestr_list_file=timestr_realdat_file)

# np.savez(dirpath+'/all_stats_'+out_tail+'.npz', all_chi2_stats = all_chi2_stats, all_accept_stats = all_accept_stats, timetsr_list_file=timestr_realdat_file)

# fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _, all_temp_posteriors, full_chains = gather_posteriors(timestr_list_file=timestr_realdat_file, save=save_figs, tail_name=out_tail, integrate_sz_prof=False, dust=False, inject_sz_frac=None, datatype='real', burn_in_frac=0.7)

# ------------ save chains for convergence diagnostic analysis ----------------
#np.savez('list_of_sz_chains_realdat_031621_take1_with_nsrc.npz', all_temp_posterior_sz=all_temp_posteriors, fmin=0.005, n_fc_terms=6, cprior_widths=0.5, nsrc_chains=all_temp_posteriors[-1])
#np.savez('list_of_sz_chains_smallmask_full_realdat_033021_with_nsrc.npz', all_temp_posterior_sz=[full_chains[0], full_chains[1]], sz_band_order=['PMW', 'PLW'], fmin=0.005, n_fc_terms=6, cprior_widths=0.5, nsrc_chains=full_chains[2])
# np.savez(dirpath+'/list_of_chains_full_'+out_tail+'.npz', chains = full_chains, fmin=0.005, n_fc_terms=6, cprior_widths=0.5)
#np.savez('list_of_chains_full_unlensed_smallmask_041021_simidx300.npz', chains=full_chains, fmin=0.005, n_fc_terms=6, cprior_widths=0.5)

#fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _, all_temp_posteriors, full_chains = gather_posteriors(timestr_list_file=timestr_dust_file, save=save_figs, tail_name='lensed_16xdust_nfcterms='+str(n_fc_terms)+'_conley_rxj1347_timestrs_withnoise_3_22_21_narrow_cprior_Fmin=5', integrate_sz_prof=False, dust=False, inject_sz_frac=1.0, datatype='mock')


#fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _, all_temp_posteriors = gather_posteriors(timestr_list_file=timestr_dust_file, save=save_figs, tail_name=timestr_dust_file[:-4], integrate_sz_prof=False, dust=False, inject_sz_frac=1.0, datatype='mock') 

#np.savez('list_of_sz_chains_realdat_nfcterms=6_031621_take2.npz', all_temp_posterior_sz=all_temp_posteriors, fmin=0.005, n_fc_terms=6, cprior_widths=0.5)

#fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _ = gather_posteriors(timestr_list_file="lensed_nodust_conley_rxj1347_timestrs_withnoise_3_2_21_narrow_cprior_Fmin=2p5.npz", save=save_figs, tail_name='lensed_nodust_conley_rxj1347_timestrs_withnoise_3_2_21_narrow_cprior_Fmin=2p5', integrate_sz_prof=False, dust=False, inject_sz_frac=1.0)

#fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _ = gather_posteriors(timestr_list_file="lensed_nodust_conley_rxj1347_timestrs_withnoise_3_2_21_narrow_cprior_Fmin=10.npz", save=save_figs, tail_name='lensed_nodust_conley_rxj1347_timestrs_withnoise_3_2_21_narrow_cprior_Fmin=10', integrate_sz_prof=False, dust=False, inject_sz_frac=1.0)



# figs = aggregate_posterior_corner_plot(timestr_list_threeband_sz2, tail_name='testing', temp_bands=[0, 1, 2], nsrcs=False)

# 
# amplitudes = dict({1:[]})
# labels = dict({1:['No SZ signal', '0.15 MJy/sr', '0.3 MJy/sr', '0.45 MJy/sr'], 2:['No signal', '0.25 MJy/sr', '0.5 MJy/sr', '0.75 MJy/sr']})


# timestrs = ['20200510-230147', '20200512-101717', '20200512-101738', '20200512-103101']

# fs = gather_posteriors_different_amps(timestrs, labels)



# ------------------------- SPORC runs ---------------------------


#timestr_dust_file = 'lensed_1xdust_nfcterms='+str(n_fc_terms)+'_conley_rxj1347_timestrs_withnoise_3_7_21_narrow_cprior_Fmin=5.npz'
#timestr_realdat_file = 'rxj1347_realdat_nfcterms='+str(n_fc_terms)+'_timestrs_3_7_21_narrow_cprior_Fmin=5.npz'
#timestr_realdat_file = 'group_timestrs/rxj1347_realdat_randinit_ASZ_nfcterms=6_timestrs_3_16_21_narrow_cprior_Fmin=5.npz'
#timestr_realdat_file = 'rxj1347_realdat_randinit_ASZ_nfcterms=6_timestrs_3_16_21_take2_narrow_cprior_Fmin=5.npz'
#timestr_realdat_file = 'rxj1347_realdat_smaller_mask_nfcterms=6_timestrs_3_31_21_take3.npz'
#timestr_realdat_file = 'rxj1347_realdat_smaller_mask_nfcterms=6_timestrs_040521_larger_tempfacs.npz'
#timestr_realdat_file = 'rxj1347_realdat_smaller_mask_nfcterms=6_timestrs_040721_fixbkg.npz'
#timestr_realdat_file = 'rxj1347_realdat_6arcmin_mask_nfcterms=6_timestrs_040921_fixbkg.npz'
#timestr_realdat_file = 'rxj1347_realdat_4arcmin_mask_nfcterms=4_timestrs_041021_fixbkg.npz'
#timestr_realdat_file = 'rxj1347_realdat_4arcmin_mask_nfcterms=3_timestrs_041121_fixbkg_errfdivfac=15.npz'
#timestr_realdat_file = 'rxj1347_realdat_4arcmin_mask_nfcterms=3_timestrs_041121_fixbkg_plwstepfac30.npz'
#timestr_realdat_file = 'rxj1347_realdat_4arcmin_mask_nfcterms=3_timestrs_041121_df=5.npz'

#timestr_realdat_file = 'rxj1347_realdat_4arcmin_mask_nfcterms=3_timestrs_041221_nreg=2.npz'
#timestr_realdat_file = 'rxj1347_realdat_4arcmin_mask_nfcterms=3_timestrs_041221_nreg=2_floatbkg.npz'
#timestr_realdat_file = 'rxj1347_realdat_4arcmin_mask_nfcterms=3_timestrs_041221_nreg=2_fixbkg_longo.npz'

#timestr_realdat_file = 'rxj1347_realdat_4arcmin_nfcterms=3_timestrs_041621_bkg_vals3.npz'
#timestr_realdat_file = 'rxj1347_realdat_10arcmin_nfcterms=6_timestrs_042021_fitbkg.npz'
#timestr_realdat_file = 'rxj1347_realdat_4arcmin_nfcterms=3_timestrs_042021_bestfitbkg.npz'

#timestr_bkg_file = 'rxj1347_conley_10arcmin_042121_timestrs_fitbkg.npz'
#timestr_bkg_file = 'rxj1347_conley_4arcmin_042121_timestrs_bestfitbkgs_longo.npz'
#timestr_bkg_file = 'rxj1347_conley_4arcmin_unlensed_042421_timestrs_bestfitbkgs_longo.npz'
#timestr_bkg_file = 'rxj1347_conley_4arcmin_041921_timestrs_bestfitbkgs.npz'
#timestr_bkg_file = 'rxj1347_conley_4arcmin_unlensed_042021_timestrs_bestfitbkgs.npz'
#timestr_sim_file = 'lensed_1xdust_nfcterms=6_simidx300_30chains_conley_rxj1347_timestrs_3_23_21_narrow_cprior_Fmin=5.npz'
#timestr_sim_file = 'group_timestrs/unlensed_nodust_conley_rxj1347_timestrs_withnoise_3_2_21_narrow_cprior_Fmin=5.npz'
#timestr_sim_file = 'unlensed_1xdust_nfcterms=3_conley_rxj1347_4arcmin_simidx300_withnoise_timestrs_041021.npz'
#timestr_sim_file = 'lensed_1xdust_nfcterms=6_conley_rxj1347_timestrs_withnoise_3_31_21_smallmask_narrow_cprior_Fmin=5.npz'

#started_idx_file = 'rxj1347_conley_4arcmin_unlensed_042421_simidxs_bestfitbkgs_longo_spr.npz'
#best_fit_bkg_file = 'rxj1347_conley_10arcmin_062121_bkg_best_fits.npz'
#timestr_list_file = 'rxj1347_conley_4arcmin_unlensed_042421_timestrs_bestfitbkgs_spr.npz'
#sim_idx_list = np.load(best_fit_bkg_file)['sim_idx_list']
#timestr_list = np.load(timestr_list_file)['timestr_list']

#timestr_list_file = 'rxj1347_conley_4arcmin_062121_timestrs_bestfitbkg.npz'
#timestr_list_file = 'rxj1347_conley_10arcmin_062121_timestrs_fitbkg.npz'
#timestr_list_file='rxj1347_realdat_4arcmin_nfcterms=3_timestrs_062221_bestfitbkg.npz'
#timestr_list_file = 'rxj1347_conley_4arcmin_062121_timestrs_bestfitbkg_sig0p25.npz'
# timestr_list_file = 'rxj1347_realdat_4arcmin_nfcterms=3_timestrs_062421_bestfitbkg_fluxalpha=3.npz'
#truealpha = 2.5
#timestr_dust_file = 'lensed_1xdust_nfcterms=6_truealpha='+str(truealpha)+'_conley_rxj1347_timestrs_3_17_21_narrow_cprior_Fmin=5.npz'
#timestr_dust_file = 'lensed_16xdust_nfcterms='+str(n_fc_terms)+'_conley_rxj1347_timestrs_withnoise_3_22_21_narrow_cprior_Fmin=5.npz'

#fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _, all_temp_posteriors, full_chains = gather_posteriors(timestr_list_file=timestr_sim_file, save=save_figs, tail_name='rxj1347_sims_nfcterms='+str(n_fc_terms)+'_unlensed_4arcmin_041021_smallmask_narrow_cprior_Fmin=5', integrate_sz_prof=False, dust=False, inject_sz_frac=1.0, datatype='real', burn_in_frac=0.5)


#losi, lotl, losl = split_timestr_lists_by_simidx(timestr_list)

#print('list of sim idxs is ', losi)
#print('list of timestr lists is ', lotl)
#print('list of simidx lists is ', losl)


#for s, simidx in enumerate(losi):
#    out_tail = '4arcmin_unlensed_042421_simidx='+str(simidx)

#    fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _, all_temp_posteriors, full_chains = gather_posteriors(timestr_list=lotl[s], save=True, tail_name=out_tail, integrate_sz_prof=False, spr=True, dust=False, inject_sz_frac=None, datatype='real', burn_in_frac=0.5)          


#out_tail = '10arcmin_062121_fitbkg_sig0p5'
# out_tail = '4arcmin_062421_fixbkg_sig0p5_conley_fluxalpha=3'
#out_tail = '4arcmin_041021_fixbkg_errdivfac=15'
#out_tail = '4arcmin_unlensed_042421_bestfitbkg_longo'
#out_tail = '4arcmin_041221_fixbkg_nreg2_longo'
#out_tail = '4arcmin_041921_bestfitbkg_unlensed'
#out_tail = '4arcmin_042021_bestfitbkg_realdat'

# dirpath = out_tail
# if not os.path.isdir(dirpath):
#     os.makedirs(dirpath)


#pcagg = pcat_agg()
#all_accept_stats = pcagg.compile_stats(mode='accept', timestr_list_file=timestr_bkg_file)
#np.savez(dirpath+'/all_acceptance_stats_'+out_tail+'.npz', all_accept_stats = all_accept_stats, timestr_list_file=timestr_bkg_file)
#all_chi2_stats = pcagg.compile_stats(mode='chi2', timestr_list_file=timestr_bkg_file)
#np.savez(dirpath+'/all_chi2_stats_'+out_tail+'.npz', all_chi2_stats = all_chi2_stats, timestr_list_file=timestr_bkg_file)

#np.savez(dirpath+'/all_stats_'+out_tail+'.npz', all_chi2_stats = all_chi2_stats, all_accept_stats = all_accept_stats, timetsr_list_file=timestr_bkg_file)

# fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _, all_temp_posteriors, full_chains = gather_posteriors(timestr_list_file=timestr_list_file, save=save_figs, tail_name=out_tail, integrate_sz_prof=False, dust=False, inject_sz_frac=1.0, datatype='real', burn_in_frac=0.5)

# ------------ save chains for convergence diagnostic analysis ----------------
#np.savez('list_of_sz_chains_realdat_031621_take1_with_nsrc.npz', all_temp_posterior_sz=all_temp_posteriors, fmin=0.005, n_fc_terms=6, cprior_widths=0.5, nsrc_chains=all_temp_posteriors[-1])
#np.savez('list_of_sz_chains_smallmask_full_realdat_033021_with_nsrc.npz', all_temp_posterior_sz=[full_chains[0], full_chains[1]], sz_band_order=['PMW', 'PLW'], fmin=0.005, n_fc_terms=6, cprior_widths=0.5, nsrc_chains=full_chains[2])

# np.savez(dirpath+'/list_of_chains_full_'+out_tail+'.npz', chains = full_chains, fmin=0.005, n_fc_terms=n_fc_terms, cprior_widths=0.5)

#np.savez('list_of_chains_full_unlensed_smallmask_041021_simidx300.npz', chains=full_chains, fmin=0.005, n_fc_terms=6, cprior_widths=0.5)

#fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _, all_temp_posteriors, full_chains = gather_posteriors(timestr_list_file=timestr_dust_file, save=save_figs, tail_name='lensed_16xdust_nfcterms='+str(n_fc_terms)+'_conley_rxj1347_timestrs_withnoise_3_22_21_narrow_cprior_Fmin=5', integrate_sz_prof=False, dust=False, inject_sz_frac=1.0, datatype='mock')


#fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _, all_temp_posteriors = gather_posteriors(timestr_list_file=timestr_dust_file, save=save_figs, tail_name=timestr_dust_file[:-4], integrate_sz_prof=False, dust=False, inject_sz_frac=1.0, datatype='mock') 

#np.savez('list_of_sz_chains_realdat_nfcterms=6_031621_take2.npz', all_temp_posterior_sz=all_temp_posteriors, fmin=0.005, n_fc_terms=6, cprior_widths=0.5)

#fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _ = gather_posteriors(timestr_list_file="lensed_nodust_conley_rxj1347_timestrs_withnoise_3_2_21_narrow_cprior_Fmin=2p5.npz", save=save_figs, tail_name='lensed_nodust_conley_rxj1347_timestrs_withnoise_3_2_21_narrow_cprior_Fmin=2p5', integrate_sz_prof=False, dust=False, inject_sz_frac=1.0)

#fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _ = gather_posteriors(timestr_list_file="lensed_nodust_conley_rxj1347_timestrs_withnoise_3_2_21_narrow_cprior_Fmin=10.npz", save=save_figs, tail_name='lensed_nodust_conley_rxj1347_timestrs_withnoise_3_2_21_narrow_cprior_Fmin=10', integrate_sz_prof=False, dust=False, inject_sz_frac=1.0)






