import numpy as np
import matplotlib
matplotlib.use('tKAgg')
import matplotlib.pyplot as plt
import scipy.spatial
from spire_data_utils import *
# from astropy import WCS


def flux_from_mags(mags):
    return 10**(-0.4*mags)

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


class cross_match_roc():

    def __init__(self, prev_cut = 0.1, minf=0.001, maxf=0.1, nsamp=100, imdim=100, dr=0.5, dmag=None, nbins=18, \
                flux_keyword='flux', filetype='.npz', timestr=None, fbin_mode='logspace', max_noise_level=0.003):
        
        for attr, valu in locals().items():
            setattr(self, attr, valu)
            
        self.make_fbins(mode=self.fbin_mode)

        if dmag is None:
            self.dmag = np.abs(2.5*np.log10(self.fbins[0]/self.fbins[1]))
            print('self.dmag:', self.dmag)
            

    def load_in_map(self, path, zero_nans=True):
        
        f = fits.open(path)
        self.image = f[1].data
        self.error = f[2].data

        if zero_nans:
            self.image[np.isnan(self.image)] = 0.0

        self.imdim = self.image.shape[0]
        
        x0 = self.gdat.x0
        y0 = self.gdat.y0
        print(self.gdat.width)
        print(self.gdat.height)
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
        print('here')
        if self.filetype=='.txt':
            cat = np.loadtxt(path)
            if mode=='mock_truth':
                self.mock_cat = dict({'x':cat[:,0], 'y':cat[:,1], 'f':cat[:,2]})
 
            elif mode=='condensed_cat':
                self.cond_cat = dict({'x':cat[:,0], 'y':cat[:,2], 'f':cat[:, 4]})
        
        elif self.filetype=='.npy':
            print('we get here')
            print('.npy file')
            cat = np.load(path, allow_pickle=True).item()
            mags = mag_from_fluxes(cat[self.flux_keyword])
            if mode=='mock_truth':
                self.mock_cat = dict({'x':np.array(cat['x']), 'y':np.array(cat['y']), self.flux_keyword:np.array(cat[self.flux_keyword]), 'mag':mags})
            
            
    def load_chain(self, path):
        lion = np.load(path, allow_pickle=True)
        self.lion_cat = dict({'n':lion['n'][-self.nsamp:].astype(np.int), 'x':lion['x'][-self.nsamp:,:], 'y':lion['y'][-self.nsamp:,:], \
            self.flux_keyword:lion['f'][0,-self.nsamp:], 'mags':mag_from_fluxes(lion['f'][0, -self.nsamp:])})
        self.nbands = lion['f'].shape[0]
        print(type(self.lion_cat[self.flux_keyword]))
        print('self.nbands:', self.nbands)
        


    def compute_kd_tree(self, mode='lion', mask_hwhm=0.):
        if mode=='lion':
            cat = self.lion_cat
            print('lion cat has shape', cat['x'].shape)
            coords = np.zeros((cat['x'].shape[0], 2))

        elif mode=='mock_truth':
            self.restrict_cat() # for mock
            cat=self.mock_cat
            
        coords = np.zeros((cat['x'].shape[0], 2))
        

        fs = cat[self.flux_keyword]
        mask = (cat[self.flux_keyword] > 0)

        mask *= (cat['x'] > mask_hwhm)*(cat['x'] < self.imdim - mask_hwhm)*(cat['y'] > mask_hwhm)*(cat['y'] < self.imdim - mask_hwhm)*(cat['flux'] > self.minf)
        
        if self.image is not None:
            noise_mask = (self.image[cat['y'].astype(np.int), cat['x'].astype(np.int)] != 0.0)
            noise_mask *= (self.error[cat['y'].astype(np.int), cat['x'].astype(np.int)] < self.max_noise_level)
            mask *= noise_mask
            
        coords= np.zeros((np.sum(mask), 2))
    

        coords[:, 0] = cat['x'][mask].flatten()
        coords[:, 1] = cat['y'][mask].flatten()
            
        fs = fs[mask].flatten()


        kd_tree = scipy.spatial.KDTree(coords)

        return kd_tree, coords, fs

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
        
    def compute_completeness_fdr(self, ref_path=None, lion_path=None, map_path=None, lion_mode='ensemble', ref_mode='mock_truth',  plot=True, timestr=None, gdat=None):
        
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
        plt.title('RXJ 1347', fontsize=24)
        plt.scatter(ref_coords[:,1], ref_coords[:,0], marker='x', s=3*ref_fs*1e3, label='SIDES Mock Truth')
        plt.scatter(lion_coords[:,1], lion_coords[:,0], marker='+', alpha=0.01, s=3*lion_fs*1e3, color='r', label='PCAT')
        leg = plt.legend(frameon=False, fontsize=16)
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
#         plt.legend(frameon=False)
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
         






