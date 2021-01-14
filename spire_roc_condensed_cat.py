import numpy as np
import matplotlib
matplotlib.use('tKAgg')
import matplotlib.pyplot as plt
import scipy.spatial
from spire_data_utils import *
# from astropy import WCS

from scipy import stats
import astropy.io.fits
import astropy.wcs
import sys
from astropy.io import fits
import time
from astropy import wcs
from scipy import interpolate
import networkx as nx
import os



def mag_from_fluxes(fluxes):
    return -2.5*np.log10(fluxes)

def associate(a, mags_a, b, mags_b, dr, dmag, confs_b = None, sigfs_b = None):
    allmatches = a.query_ball_tree(b, dr)
    goodmatch = np.zeros(mags_a.size, np.bool)
    if confs_b is not None:
        confmatch = np.zeros(mags_a.size)
    if sigfs_b is not None:
        sigfmatch = np.zeros(mags_a.size) + float('inf')

    for i in xrange(len(allmatches)):
        matches = allmatches[i]
        if len(matches):
            mag_a = mags_a[i]
            goodmatch[i] = False
            for j in matches:
                mag_b = mags_b[j]
                if np.abs(mag_a - mag_b) < dmag:
                    goodmatch[i] = True
                    if (confs_b is not None) and (confs_b[j] > confmatch[i]):
                        confmatch[i] = confs_b[j]
                    if (sigfs_b is not None) and (sigfs_b[j] < sigfmatch[i]):
                        sigfmatch[i] = sigfs_b[j]

    if confs_b is not None:
        if sigfs_b is not None:
            return goodmatch, confmatch, sigfmatch
        else:
            return goodmatch, confmatch
    else:
        if sigfs_b is not None:
            return goodmatch, sigfmatch
        else:
            return goodmatch


def clusterize_spire(seed_cat, cat_x, cat_y, cat_n, cat_fs, max_num_sources, nsamp, search_radius):


    print("are there any nans in cat_x? ", np.isnan(cat_x).any())
    print("are there any nans in cat_y? ", np.isnan(cat_y).any())

    nbands = cat_fs.shape[0]
    sorted_posterior_sample = np.zeros((nsamp, max_num_sources, 2+nbands))
    print(cat_x.shape, cat_y.shape)
    print(cat_fs.shape)
    print('nbands=', nbands)
    print 
    for i in xrange(nsamp):

        cat = np.zeros((max_num_sources, 2+nbands))
        cat[:,0] = cat_x[i,:]
        cat[:,1] = cat_y[i,:]
        for b in xrange(nbands):
            cat[:,2+b] = cat_fs[b,i,:]

        cat = np.flipud(cat[cat[:,2].argsort()])
        sorted_posterior_sample[i] = cat

    PCx = sorted_posterior_sample[:,:,0]
    PCy = sorted_posterior_sample[:,:,1]
    PCf = sorted_posterior_sample[:,:,2:]

    stack = np.zeros((np.sum(cat_n), 2))
    j = 0
    for i in xrange(cat_n.size): # don't have to for loop to stack but oh well
        n = cat_n[i]
        stack[j:j+n, 0] = PCx[i, 0:n]
        stack[j:j+n, 1] = PCy[i, 0:n]
        j += n

    #creates tree, where tree is Pcc_stack
    tree = scipy.spatial.KDTree(stack)

    #keeps track of the clusters
    clusters = np.zeros((nsamp, len(seed_cat)*(2+nbands)))

    #numpy mask for sources that have been matched
    mask = np.zeros(len(stack))

    #first, we iterate over all sources in the seed catalog:
    for ct in range(0, len(seed_cat)):

        #query ball point at seed catalog
        matches = tree.query_ball_point(seed_cat[ct], search_radius)

        #in each catalog, find first instance of match w/ desired source (-- first instance okay b/c every catalog is sorted brightest to faintest)
        ##this for loop should naturally populate original seed catalog as well! (b/c all distances 0)
        for i in range(0, nsamp):
            #for specific catalog, find indices of start and end within large tree
            cat_lo_ndx = np.sum(cat_n[:i])
            cat_hi_ndx = np.sum(cat_n[:i+1])
            #want in the form of a numpy array so we can use array slicing/masking  
            matches = np.array(matches)

            #find the locations of matches to ct within specific catalog i
            culled_matches =  matches[np.logical_and(matches >= cat_lo_ndx, matches < cat_hi_ndx)] 

            if culled_matches.size > 0:
                #cut according to mask
                culled_matches = culled_matches[mask[culled_matches] == 0]
                    #if there are matches remaining, we then find the brightest and update
                if culled_matches.size > 0:
                    #find brightest
                    match = np.min(culled_matches)

                    #flag it in the mask
                    mask[match] += 1

                    #find x, y, flux of match
                    x = PCx[i,match-cat_lo_ndx]
                    y = PCy[i,match-cat_lo_ndx]

                    #add information to cluster array
                    clusters[i][ct] = x
                    clusters[i][len(seed_cat)+ct] = y

                    for b in xrange(nbands):
                        clusters[i][(2+b)*len(seed_cat)+ct] = PCf[i,match-cat_lo_ndx, b]

    #we now generate a CLASSICAL CATALOG from clusters
    cat_len = len(seed_cat)

    #arrays to store 'classical' catalog parameters

    mean_x = np.zeros(cat_len)
    mean_y = np.zeros(cat_len)
    mean_fs = np.zeros(shape=(nbands, cat_len))

    err_x = np.zeros(cat_len)
    err_y = np.zeros(cat_len)
    err_fluxes = np.zeros(shape=(nbands, cat_len))

    confidence = np.zeros(cat_len)

    #confidence interval defined for err_(x,y,f)
    hi = 84
    lo = 16

    for i in range(0, len(seed_cat)):
        x = clusters[:,i][np.nonzero(clusters[:,i])]
        y = clusters[:,i+cat_len][np.nonzero(clusters[:,i+cat_len])]
        
        fs=[]
        for b in xrange(nbands):
            fs.append(clusters[:,i+(2+b)*cat_len][np.nonzero(clusters[:,i+(2+b)*cat_len])])
        
        assert x.size == y.size
        assert x.size == fs[0].size
        confidence[i] = x.size/float(nsamp)

        mean_x[i] = np.nanmean(x)
        mean_y[i] = np.nanmean(y)

        for b in xrange(nbands):
            mean_fs[b,i] = np.mean(fs[b])

        if x.size > 1:
            err_x[i] = np.percentile(x, hi) - np.percentile(x, lo)
            err_y[i] = np.percentile(y, hi) - np.percentile(y, lo)

            for b in xrange(nbands):
                err_fluxes[b, i] = np.percentile(fs[b], hi) - np.percentile(fs[b],lo)

    #makes classical catalog

    classical_catalog = np.zeros((cat_len, 5+2*nbands))
    classical_catalog[:,0] = mean_x
    classical_catalog[:,1] = err_x
    classical_catalog[:,2] = mean_y
    classical_catalog[:,3] = err_y
    classical_catalog[:,4] = confidence

    for b in xrange(nbands):
        classical_catalog[:,5+2*b] = mean_fs[b]
        classical_catalog[:,6+2*b] = err_fluxes[b]

    return classical_catalog

        
def get_completeness(test_x, test_y, test_mag, test_n, ref_x, ref_mag, ref_kd, dr=0.5, dmag=0.5):
    complete = np.zeros((test_x.shape[0], ref_x.size))
    for i in xrange(test_x.shape[0]):
        print('B', i)
        n = test_n[i]
        CCc_one = np.zeros((n,2))
        CCc_one[:, 0] = test_x[i,0:n]
        CCc_one[:, 1] = test_y[i,0:n]
        CCmag_one = test_mag[i,0:n]

        complete[i,:] = associate(ref_kd, ref_mag, scipy.spatial.KDTree(CCc_one), CCmag_one, dr, dmag)
    complete = np.sum(complete, axis=0) / float(test_x.shape[0])
    return complete

def plot_fdr(bins, prec_lion=None, prec_condensed=None, show=True, savepath=None, xlabel='$F_{250}$/mJy', labelsize=14):

    f = plt.figure(figsize=(6, 5))

    if prec_lion is not None:
        plt.plot(0.5*(bins[1:]+bins[:-1])*1e3, 1.-prec_lion, label='Catalog ensemble', marker='+', c='b')
    if prec_condensed is not None:
        plt.plot(0.5*(bins[1:]+bins[:-1])*1e3, 1.-prec_condensed, label='Condensed catalog', marker='x', c='r')

    plt.xlabel(xlabel, fontsize=labelsize)
    plt.ylabel('false discovery rate', fontsize=labelsize)
    plt.xscale('log')
    plt.xticks([5, 10, 20, 50, 100], ['5', '10', '20', '50', '100'])

    plt.legend(prop={'size':12}, loc=2, frameon=False)
    plt.tight_layout()
    plt.ylim(-0.05, 1.1)

    if show:
        plt.show()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close()
    return f


def plot_completeness(bins, recl_lion=None, recl_condensed=None, show=True, savepath=None, xlabel='$F_{250}$/mJy', labelsize=14):

    f = plt.figure(figsize=(6, 5))
    if recl_lion is not None:
        plt.plot(0.5*(bins[1:]+bins[:-1])*1e3, recl_lion, c='b', label='Catalog ensemble', marker='+')

    if recl_condensed is not None:
        plt.plot(0.5*(bins[1:]+bins[:-1])*1e3, recl_condensed, c='r', label='Condensed catalog', marker='x')

    plt.xlabel(xlabel, fontsize=labelsize)
    plt.ylabel('completeness', fontsize=labelsize)
    plt.xscale('log')

    plt.ylim(-0.05, 1.1)
    plt.xticks([5, 10, 20, 50, 100], ['5', '10', '20', '50', '100'])
    plt.legend(loc=2, frameon=False, prop={'size':12})
    plt.tight_layout()
    if show:
        plt.show()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')

    plt.close()
    return f


def generate_seed_catalog(kd, cat_all, cat_f0_all, PCi, matching_dist, verbose=False):


    matches = kd.query_ball_tree(kd, matching_dist)

    G = nx.Graph()
    G.add_nodes_from(xrange(0, cat_all.shape[0]))

    for i in xrange(cat_all.shape[0]):
        for j in matches[i]:

            if cat_f0_all[i] != cat_f0_all[j]:

                G.add_edge(i, j)

    current_catalogue = 0

    for i in xrange(cat_all.shape[0]):
        matches[i].sort()
        bincount = np.bincount(PCi[matches[i]]).astype(np.int)
        ending = np.cumsum(bincount).astype(np.int)
        starting = np.zeros(bincount.size).astype(np.int)
        starting[1:bincount.size] = ending[0:bincount.size-1]

        for j in xrange(bincount.size):
            if j == PCi[i]: # do not match to same catalogue
                continue
            if bincount[j] == 0: # no match to catalog j
                continue
            if bincount[j] == 1: # exactly one match to catalog j
                continue
            if bincount[j] > 1:
                dist2 = matching_dist**2
                l = -1
                for k in xrange(starting[j], ending[j]):
                    m = matches[i][k]
                    newdist2 = np.sum((cat_all[i,:] - cat_all[m,:])**2)
                    if newdist2 < dist2:
                        l = m
                        dist2 = newdist2

                if l == -1:
                    print(" didn't find edge even though multiple matches from this catalogue?")
                for k in xrange(starting[j], ending[j]):
                    m = matches[i][k]
                    if m != l:
                        if G.has_edge(i, m):
                            G.remove_edge(i, m)

    seeds = []

    while nx.number_of_nodes(G) > 0:
        deg = nx.degree(G)

        maxmdegr = 0
        i = 0
        for node in G:
            if deg[node] >= maxmdegr:
                i = node
                maxmdegr = deg[node]

        neighbors = nx.all_neighbors(G, i)
        if verbose:
            print('found', i)
        seeds.append([cat_all[i, 0], cat_all[i, 1], deg[i]])
        G.remove_node(i)
        G.remove_nodes_from(neighbors)

    seeds = np.array(seeds)

    return seeds



class cross_match_roc():

    def __init__(self, prev_cut = 0.1, minf=0.005, maxf=0.1, nsamp=100, imdim=100, dr=0.5, dmag=None, nbins=18, \
                filetype='.npz', timestr=None, fbin_mode='logspace', max_noise_level=None, \
                pdf_or_png='png', image=None, error=None, matching_dist=0.75, search_radius=0.75):
        
        for attr, valu in locals().items():
            setattr(self, attr, valu)
            
        self.make_fbins(mode=self.fbin_mode)
            

    def load_in_map(self, path, zero_nans=True):
        
        f = fits.open(path)
        self.image = f[1].data
        self.error = f[2].data

        if zero_nans:
            self.image[np.isnan(self.image)] = 0.0

        self.imdim = self.image.shape[0]
        
        x0 = self.gdat.x0
        y0 = self.gdat.y0

        padded_image = np.zeros(shape=(self.gdat.width, self.gdat.height))
        padded_error = np.zeros(shape=(self.gdat.width, self.gdat.height))

        padded_image[:self.image.shape[0]-x0, : self.image.shape[1]-y0] = self.image[x0:, y0:]
        padded_error[:self.image.shape[0]-x0, : self.image.shape[1]-y0] = self.error[x0:, y0:]
        
        self.image = padded_image
        self.error = padded_error
        
    def make_fbins(self, mode='logspace'):
        if mode=='logspace':
            self.fbins = 10**np.linspace(np.log10(self.minf), np.log10(self.maxf), self.nbins+1)
        else:
            self.fbins = np.linspace(self.minf, self.maxf, self.nbins+1)
                    
         
    def restrict_cat(self, mode='mock_truth'):
        if mode=='mock_truth':
            restrict_mask = (self.mock_cat['x'] < self.gdat.width)&(self.mock_cat['y'] < self.gdat.height)
            print(restrict_mask.shape, np.sum(restrict_mask))
            
            self.mock_cat['x'] = self.mock_cat['x'][restrict_mask]
            self.mock_cat['y'] = self.mock_cat['y'][restrict_mask]
            self.mock_cat['flux'] = self.mock_cat['flux'][restrict_mask]
            print('mock cat has shape', self.mock_cat['x'].shape)
            
                
    def load_cat(self, path=None, mode='mock_truth', restrict_cat=False):

        if self.filetype=='.txt':
            cat = np.loadtxt(path)
            if mode=='mock_truth':
                self.mock_cat = dict({'x':cat[:,0], 'y':cat[:,1], 'f':cat[:,2]})
 
            elif mode=='condensed_cat':
                self.cond_cat = dict({'x':cat[:,0], 'y':cat[:,2], 'f':cat[:, 4]})
        
        elif self.filetype=='.npy':
            print('.npy file')
            cat = np.load(path, allow_pickle=True).item()
            if mode=='mock_truth':
                self.mock_cat = dict({'x':np.array(cat['x']), 'y':np.array(cat['y']), 'f':np.array(cat['f'])})
            
            
    def load_chain(self, path):
        lion = np.load(path, allow_pickle=True)
        self.lion_cat = dict({'n':lion['n'][-self.nsamp:].astype(np.int), 'x':lion['x'][-self.nsamp:,:], 'y':lion['y'][-self.nsamp:,:], \
            'f0':lion['f'][0,-self.nsamp:], 'fs':lion['f'][:,-self.nsamp:]})
      
        self.nbands = lion['f'].shape[0]
        print('self.nbands:', self.nbands)
        

    def compute_kd_tree(self, chain_path=None, mode='lion', mask_hwhm=0., apply_noise_mask=False):
        
        if chain_path is None:
            if mode=='lion':
                cat = self.lion_cat
                print('lion cat has shape', cat['x'].shape)

            elif mode=='mock_truth':
                self.restrict_cat() # for mock
                cat=self.mock_cat

        else:
            self.load_chain(chain_path)
            cat = self.lion_cat

            
        coords = np.zeros((cat['x'].shape[0], 2))
        fs = cat['fs']

        mask = (cat['f0'] > 0)
        mask *= (cat['x'] > mask_hwhm)*(cat['x'] < self.imdim - mask_hwhm)*(cat['y'] > mask_hwhm)*(cat['y'] < self.imdim - mask_hwhm)*(cat['f0'] > self.minf)
        
        if self.image is not None and apply_noise_mask:
            noise_mask = (self.image[cat['y'].astype(np.int), cat['x'].astype(np.int)] != 0.0)
            noise_mask *= (self.error[cat['y'].astype(np.int), cat['x'].astype(np.int)] < self.max_noise_level)
            mask *= noise_mask
            
        coords= np.zeros((np.sum(mask), 2))
    
        coords[:, 0] = cat['x'][mask].flatten()
        coords[:, 1] = cat['y'][mask].flatten()
            

        lion_f0_all = fs[0,mask].flatten()

        kd_tree = scipy.spatial.KDTree(coords)

        PCi,junk = np.mgrid[0:self.nsamp,0:self.gdat.max_nsrc]

        PCi = PCi[mask].flatten()


        if mode == 'lion':

            return kd_tree, coords, fs, PCi, lion_f0_all

        return kd_tree, coords, fs


        lion_kd, lion_r_all, lion_x, lion_y, lion_n, lion_r, lion_all, PCi, lion_f, nb, lion_comments, lion_fs, nmgy = lion_cat_kd(base_path+'/pcat-lion-results/'+run_name+'/chain.npz')


    def save_results(filepath, bins, recl_lion=None, recl_condensed=None, prec_lion=None, prec_condensed=None):

        if recl_lion is not None:
            np.savetxt(filepath+'/completeness_lion_cat_ensemble_dr='+str(self.dr)+'_df='+str(self.dmag)+'.txt', recl_lion)

        if recl_condensed is not None:
            np.savetxt(filepath+'/completeness_lion_condensed_dr='+str(self.dr)+'_df='+str(self.dmag)+'.txt', recl_condensed)

        if prec_lion is not None:
            np.savetxt(filepath+'/fdr_lion_cat_ensemble_dr='+str(self.dr)+'_df='+str(self.dmag)+'.txt', 1.-prec_lion)

        if prec_condensed is not None:
            np.savetxt(filepath+'/fdr_lion_condensed_dr='+str(self.dr)+'_df='+str(self.dmag)+'.txt', 1.-prec_condensed)

    def load_gdat_params(self, timestr):
        gdat, filepath, result_path = load_param_dict(timestr)
        self.gdat=gdat
        self.result_path = result_path
        self.imdim = self.gdat.imsz0[0]
        print('self imdim is now ', self.imdim)
        
    def compute_completeness_fdr(self, ref_path=None, lion_path=None, map_path=None, lion_mode='ensemble', ref_mode='mock_truth',  plot=True, timestr=None):
        
        if timestr is not None:
            self.timestr=timestr
            self.load_gdat_params(self.timestr)
        
        if lion_mode=='ensemble':
            self.load_chain(lion_path)
            lion_kd, lion_coords, lion_fs = self.compute_kd_tree(mode='lion', mask_hwhm=0.)

        elif lion_mode=='condensed_cat':
            self.load_cat(path=lion_path, mode='condensed_cat')
            lion_kd, lion_coords, lion_fs = self.compute_kd_tree(mode='lion')
        print('loading ref cat')
        
        self.load_cat(path=ref_path, mode=ref_mode)
                
        ref_kd, ref_coords, ref_fs = self.compute_kd_tree(mode='mock_truth', mask_hwhm=0.)
        self.reclPC_lion = np.zeros(self.nbins)

        self.prec_lion = np.zeros(self.nbins)
        mags_lion = mag_from_fluxes(lion_fs)
        mags_ref = mag_from_fluxes(ref_fs)
        
        f = plt.figure(figsize=(10,10))
        plt.scatter(ref_coords[:,1], ref_coords[:,0], marker='x', s=3*ref_fs*1e3, label='SIDES Mock Truth')
        plt.scatter(lion_coords[:,1], lion_coords[:,0], marker='+', alpha=0.01, s=3*lion_fs*1e3, color='r', label='PCAT')
        leg = plt.legend(frameon=False, fontsize=16)
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
        plt.show()
        
        goodmatch_lion = associate(lion_kd, mags_lion, ref_kd, mags_ref, self.dr, self.dmag)

        complete_lion = get_completeness(self.lion_cat['x'], self.lion_cat['y'], self.lion_cat['mags'], self.lion_cat['n'], ref_coords[:,0], \
                                         mags_ref, ref_kd, dr=self.dr, dmag=self.dmag)
        

        for i in range(self.nbins):
            rlo = self.fbins[i]
            rhi = self.fbins[i+1]

            inbin = np.logical_and(lion_fs >= rlo, lion_fs < rhi)
            self.prec_lion[i] = np.sum(np.logical_and(inbin, goodmatch_lion)) / float(np.sum(inbin))

        print('prec_lion:', self.prec_lion)

        for i in range(self.nbins):
            rlo = self.fbins[i]
            rhi = self.fbins[i+1]

            inbin = np.logical_and(ref_fs >= rlo, ref_fs < rhi)

            self.reclPC_lion[i] = np.sum(complete_lion[inbin]) / float(np.sum(inbin))
            
            print(rlo, rhi, np.sum(complete_lion[inbin]), float(np.sum(inbin)))

        if lion_mode=='ensemble' and plot:
            
            plot_completeness(self.fbins, recl_lion=self.reclPC_lion)
            plot_fdr(self.fbins, prec_lion=self.prec_lion)
        if plot:
            plot_completeness(mag_from_fluxes(self.fbins), recl_condensed=self.reclPC_lion, xlabel='magnitude')
            plot_fdr(mag_from_fluxes(self.fbins), prec_condensed=self.prec_lion, xlabel='magnitude')
            
        return f   



    def condense_catalogs(self, make_seed_bool=True, seed_fpath=None, nsamp=None, iteration_number=2, matching_dist = None, search_radius = None, prevalence_cut=0.1, \
                            show_figs = True, save_figs = True, save_cats = False, mask_hwhm = 5):
                 
        if matching_dist is not None:
            self.matching_dist = matching_dist

        if search_radius is not None:
            self.search_radius = search_radius


        if nsamp is not None:
            self.nsamp = nsamp
            print('self.nsamp is now ', self.nsamp)

        # get kd tree, coords (x and y), fluxes, PCi, and the raveled first band flux
        lion_kd_tree, lion_coords, lion_fs, PCi, lion_f0_all = self.compute_kd_tree(mode='lion', mask_hwhm = mask_hwhm)

        if not make_seed_bool:
            if seed_fpath is not None:
                dat = np.loadtxt(seed_fpath)
            else:
                print('make_seed_bool is False, but no seed catalog file path is provided. Exiting..')
                return
        else:
            dat = generate_seed_catalog(lion_kd_tree, lion_coords, lion_f0_all, PCi, self.matching_dist)    

        if save_cats and make_seed_bool:
            np.savetxt(self.result_path+'/'+self.timestr+'/raw_seed_catalog_nsamp='+str(self.nsamp)+'_matching_dist='+str(self.matching_dist)+'_maskhwhm='+str(mask_hwhm)+'.txt', dat)

        if show_figs or save_figs:

            chist_plot = plot_confidence_hist(dat, self.nsamp, bins=50, show=show_figs, return_fig=True)

        if save_figs:
            chist_plot.savefig(self.result_path+'/'+self.timestr+'/confidence_histogram_base_cat.'+self.pdf_or_png)

        #performs confidence cut
        x = dat[:,0][dat[:,2] > prevalence_cut*self.nsamp]
        y = dat[:,1][dat[:,2] > prevalence_cut*self.nsamp]
        n = dat[:,2][dat[:,2] > prevalence_cut*self.nsamp]

        assert x.size == y.size
        assert x.size == n.size

        seed_cat = np.zeros((x.size, 2))
        seed_cat[:,0] = x
        seed_cat[:,1] = y
        cat_len = x.size

        condensed_cat = clusterize_spire(seed_cat, self.lion_cat['x'], self.lion_cat['y'], self.lion_cat['n'], lion_fs, self.gdat.max_nsrc, self.nsamp, self.search_radius)

        condensed_x = condensed_cat[:,0]
        condensed_y = condensed_cat[:,1]

        print('min, max of x and y:')
        print(np.amin(condensed_x), np.amax(condensed_x))
        print(np.amin(condensed_y), np.amax(condensed_y))

        if save_cats:
            np.savetxt(self.result_path+'/'+self.timestr+'/condensed_catalog_nsamp='+str(self.nsamp)+'_prevcut='+str(prevalence_cut)+'_searchradius='+str(self.search_radius)+'_maskhwhm='+str(mask_hwhm)+'.txt', condensed_cat)

        return condensed_cat, seed_cat


def plot_confidence_hist(dat, nsamp, bins=50, show=True, return_fig=True):

    fig = plt.gcf()
    fig.set_size_inches(15, 5)
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(1,3,3)
    plt.hist(dat[:,2], bins=bins)
    plt.xlabel("Number of Samples")
    # plt.ylim([0, 1250])
    plt.title("Seed Catalog")
    plt.subplot(1,3,2)
    plt.hist(dat[:,2]/nsamp, bins=bins)
    plt.xlabel("Prevalence")
    # plt.ylim([0, 1250])
    plt.title("Seed Catalog")

    plt.subplot(1,3,1)
    plt.scatter(dat[:,0], dat[:,1], s=50.*(dat[:,2]/nsamp))
    plt.xlabel('x [pix]')
    plt.ylabel('y [pix]')
    plt.title('Seed Catalog')


    if show:
        plt.show()
    if return_fig:
        return fig




