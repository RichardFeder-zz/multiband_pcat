from astropy import wcs
from astropy.io import fits
import numpy as np


def find_nearest_upper_mod(number, mod_number):
    while number < 10000:
        if np.mod(number, mod_number) == 0:
            return number
        else:
            number += 1
    return False

class wcs_astrometry():
    ''' This class will contain the WCS header and other information necessary to construct arrays for fast 
    astrometric transformations. 
    
    Variables:
        
        all_fast_arrays (list of nd.arrays): contains all astrometry arrays across bands
            e.g. len(all_fast_arrays)= n_bands - 1
            
        dims (list of tuples): contains image dimensions for all observations
        
        wcs_objs (list of astropy.WCS objects): obtained from observation WCS headers
        
        filenames (list): observation names obtained when loading in FITS files
        
        
    Input:
        - 
        -
    
    Functions:
        
        fit_astrom_arrays(): This function computes the mapping for lattices of points from one observation to 
          another, using first differencing to get derivatives for subpixel perturbations.
        
        load_wcs_header(fits): Yes
        
        load_fast_astrom_arrays(fast_arrays): this loads in the arrays to the wcs_astrometry object to be 
          used for 1st order approximation to astrometry
        
        pixel_to_pixel(coords, idx0, idx1): this will take coordinates in one observation (idx0) and convert them to 
          the observation indicated by idx1. Will throw error if arrays to do so do not exist in class object
          
        
    Outputs:
        - 
        - np.array([x', y', dy'/dx, dy'/dy, dx'/dx, dx'/dy])
        - 
    '''
    
    all_fast_arrays = []
    wcs_objs = []
    filenames = []
    dims = []
    verbosity = 0
    
    base_path = '/Users/richardfeder/Documents/multiband_pcat/Data/spire/'

    
    def __init__(self, auto_resize=False, nregion=1):
        self.wcs_objs = []
        self.filenames = []
        self.all_fast_arrays = []
        self.dims = []
        self.auto_resize = auto_resize
        self.nregion = nregion
    
    def change_verbosity(self, verbtype):
        self.verbosity = verbtype

    def change_base_path(self, basepath):
        self.base_path = basepath
        
    def load_wcs_header_and_dim(self, filename=None, head=None, hdu_idx=None):
        if head is None:
            self.filenames.append(filename)
            
            f = fits.open(self.base_path + filename)
            if hdu_idx is None:
                hdu_idx = 0
                
            head = f[hdu_idx].header  

        if self.auto_resize:
            big_dim = np.maximum(head['NAXIS1'], head['NAXIS2'])
            big_pad_dim = find_nearest_upper_mod(big_dim, self.nregion)
            dim = (big_pad_dim, big_pad_dim)
        else:
            dim = (head['NAXIS1'], head['NAXIS2'])
        self.dims.append(dim)
        wcs_obj = wcs.WCS(head)
        self.wcs_objs.append(wcs_obj)
        
    def obs_to_obs(self, idx0, idx1, x, y):
        ra, dec = self.wcs_objs[idx0].all_pix2world(x, y, 0)
        x1, y1 = self.wcs_objs[idx1].all_world2pix(ra, dec, 0)
        return x1, y1
    
    def get_pint_dp(self, p):
        pint = np.floor(p+0.5)
        dp = p - pint
        return pint.astype(int), dp
    
    def get_derivative(self, idx0, idx1, x, y, epsx, epsy):
        
        x1, y1 = self.obs_to_obs(idx0, idx1, x+epsx, y+epsy)
        x0, y0 = self.obs_to_obs(idx0, idx1, x-epsx, y-epsy)
        
        dxp = x1 - x0
        dyp = y1 - y0
        
        if self.verbosity > 0:
            print('dxp:')
            print(dxp)
            print('dyp:')
            print(dyp)
        
        return dxp, dyp
           
    def fit_astrom_arrays(self, idx0, idx1):
        
        x = np.arange(0, self.dims[idx0][0])
        y = np.arange(0, self.dims[idx0][1])
        
        xv, yv = np.meshgrid(x, y)
        
        if self.verbosity > 0:
            print('xv:')
            print(xv)
            print('yv:')
            print(yv)

        dxp_dx, dyp_dx = self.get_derivative(idx0, idx1, xv, yv, 0.5, 0.0)
        dxp_dy, dyp_dy = self.get_derivative(idx0, idx1, xv, yv, 0.0, 0.5)
        xp, yp = self.obs_to_obs(idx0, idx1, xv, yv)
        
        if self.verbosity > 0:
            print('xp:')
            print(xp)
            print('yp:')
            print(yp)
        
        fast_arrays = np.array([xp, yp, dxp_dx, dyp_dx, dxp_dy, dyp_dy])
        self.all_fast_arrays.append(fast_arrays)
        
    def transform_q(self, x, y, idx):
        
        assert len(x)==len(y)
        xtrans, ytrans, dxpdx, dypdx, dxpdy, dypdy = self.all_fast_arrays[idx]
        xints, dxs = self.get_pint_dp(x)
        yints, dys = self.get_pint_dp(y)
        try:
            xnew = xtrans[yints,xints] + dxs*dxpdx[yints,xints] + dys*dxpdy[yints,xints]
            ynew = ytrans[yints,xints] + dxs*dypdx[yints,xints] + dys*dypdy[yints,xints] 
            return np.array(xnew).astype(np.float32), np.array(ynew).astype(np.float32)
        except:
            print(np.max(xints), np.max(yints), xtrans.shape)
            print(xints, dxs, yints, dys)
            raise ValueError('problem accessing elements')






            