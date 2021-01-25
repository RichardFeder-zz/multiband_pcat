from astropy import wcs
from astropy.io import fits
import numpy as np
# from spire_data_utils import *

def find_nearest_mod(number, mod_number, mode='up'):
    '''
    Finds the nearest integer modulo "mod_number". This is used for padding/trimming images appropriate 
    to PCAT's multi-region MCMC approach.

    Parameters
    ----------

    number : 'int'
        Number to be rounded up/down
    mod_number : 'int'
        Base of modulus
    mode : 'string', optional
        Determines whether number is rounded up or down to nearest modulus. 
        Default is 'up'.

    Returns
    -------

    Integer rounded up or down, depending on "mode".

    '''
    if mode=='up':
        return int(mod_number*np.ceil(float(number)/float(mod_number)))
    elif mode=='down':
        return int(mod_number*np.floor(float(number)/float(mod_number)))


class wcs_astrometry():
    ''' 
    This class will contain the WCS header and other information necessary to construct arrays for fast 
    astrometric transformations. 
    
    Parameters
    ----------
        
    all_fast_arrays : list of nd.arrays
        contains all astrometry arrays across bands e.g. len(all_fast_arrays)= n_bands - 1
            
    dims : list of tuples
        contains image dimensions for all observations
        
    wcs_objs : list of astropy.WCS objects
        obtained from observation WCS headers
        
    filenames : list
        observation names obtained when loading in FITS files
        
        
    
    Functions
    ---------
        
    fit_astrom_arrays(): This function computes the mapping for lattices of points from one observation to 
          another, using first differencing to get derivatives for subpixel perturbations.
        
    load_wcs_header(fits): Yes
        
    load_fast_astrom_arrays(fast_arrays): this loads in the arrays to the wcs_astrometry object to be 
          used for 1st order approximation to astrometry
        
    pixel_to_pixel(coords, idx0, idx1): this will take coordinates in one observation (idx0) and convert them to 
          the observation indicated by idx1. Will throw error if arrays to do so do not exist in class object
          
        
    '''
    
    all_fast_arrays = []
    wcs_objs = []
    filenames = []
    dims = []
    verbosity = 0
    
    
    def __init__(self, auto_resize=False, nregion=1, base_path='/Users/richardfeder/Documents/multiband_pcat/Data/spire/'):
        self.wcs_objs = []
        self.filenames = []
        self.all_fast_arrays = []
        self.dims = []
        self.auto_resize = auto_resize
        self.nregion = nregion
        self.base_path = base_path
    
    def change_verbosity(self, verbtype):
        self.verbosity = verbtype

    def change_base_path(self, basepath):
        self.base_path = basepath
        
    def load_wcs_header_and_dim(self, filename=None, head=None, hdu_idx=None, round_up_or_down='up'):
        
        ''' 
        Loads in WCS header information into the wcs_astrometry class

        Parameters
        ----------

        filename : str, optional
            path to file containing desired WCS header
            Default is 'None'.

        head : WCS header, optional
            Can be passed in directly rather than extracted from filename
            Default is 'None'.

        hdu_idx : int, optional
            Integer index specifying the HDU card with the desired WCS header. If not provided, the first two HDU
            cards will be checked.
            Default is 'None'.

        round_up_or_down : str, optional
            regions evaluated are either rounded 'up' or 'down' to be divisible by self.nregions.
            Default is 'up'.

        Returns
        -------

        Nothing! You've been a fooled

        '''

        if head is None:
            if filename is None:
                print('No file or header provided')
                return

            self.filenames.append(filename)
            
            f = fits.open(filename)

            if hdu_idx is None:
                hdu_idx = 0
                
            head = f[hdu_idx].header

        if self.auto_resize:
            try:
                big_dim = np.maximum(head['NAXIS1'], head['NAXIS2'])
            except:
                print('didnt work upping it')
                hdu_idx += 1
                head = f[hdu_idx].header
                big_dim = np.maximum(head['NAXIS1'], head['NAXIS2'])

            big_pad_dim = find_nearest_mod(big_dim, self.nregion, mode=round_up_or_down)

            dim = (big_pad_dim, big_pad_dim)
        else:
            try:
                dim = (head['NAXIS1'], head['NAXIS2'])
            except:
                hdu_idx += 1
                head = f[hdu_idx].header
                dim = (head['NAXIS1'], head['NAXIS2'])
        print('dim:', dim)
        self.dims.append(dim)
        wcs_obj = wcs.WCS(head)
        self.wcs_objs.append(wcs_obj)
        
    def obs_to_obs(self, idx0, idx1, x, y):
        '''
        Transforms set of pixel coordinates from one observation to another

        Parameters
        ----------

        idx0, idx1 : ints
            Indices of initial (idx0) and transformed (idx1) coordinates

        x, y : floats or '~numpy.ndarrays'
            x and y pixel coordinates of initial observation

        Returns
        -------

        x1, y1 : floats or '~numpy.ndarrays'
            x- and y-transformed pixel coordinates 

        '''
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
           
    def fit_astrom_arrays(self, idx0, idx1, bounds0=None, bounds1=None, pos0_pivot=None, pos0=None, x0=None, y0=None):
        '''
        Precomputes set of astrometry arrays used to quickly compute coordinate shifts across bands. 

        Parameters
        ----------

        idx0, idx1 : ints
            Indices for initial (idx0) and transformed (idx1) bands

        bounds0, bounds1 : '~numpy.ndarrays' of shape (2,2), optional
            Can be used to specify image bounds over which to compute fast astrometry arrays. 
            If left unspecified, astrometry arrays computed over full range of initial observation.
            Default is 'None'.

        x0, y0 : ints, optional
            Occasionally we will want to take image cutouts from a large map (e.g. 100x100 pixel cutouts from a larger 5000x5000 pixel map)
            while using the header as copied from the larger map. Because the cutouts will have different zero points,
            the parent WCS header will be offset by some amount (x0, y0). One could maybe modify the WCS header directly, but this 
            is equivalent. 

        Returns
        -------

        self.all_fast_arrays : list of '~numpy.ndarrays' of shape (6, self.dims)
            Contains all sets of astrometry arrays (first order integer approximations + numerical partial derivatives)


        '''

        if bounds0 is not None:
            # if a rectangular mask is provided, then we only need to pre-compute the astrometry arrays over the masked region
            x = np.arange(bounds0[0,0], bounds0[0,1]+self.dims[idx0][0])
            y = np.arange(bounds0[1,0], bounds0[1,1]+self.dims[idx0][1])

        else:

            x = np.arange(0, self.dims[idx0][0])
            y = np.arange(0, self.dims[idx0][1])
        
        xv, yv = np.meshgrid(x, y)

        if pos0_pivot is not None:
            xv += pos0_pivot[0]
            yv += pos0_pivot[1]
        
        if self.verbosity > 0:
            print('xv:')
            print(xv)
            print('yv:')
            print(yv)

        dxp_dx, dyp_dx = self.get_derivative(idx0, idx1, xv, yv, 0.5, 0.0)
        dxp_dy, dyp_dy = self.get_derivative(idx0, idx1, xv, yv, 0.0, 0.5)
        
        xp, yp = self.obs_to_obs(idx0, idx1, xv, yv)

        if bounds1 is not None:
            xp -= bounds1[0,0]
            yp -= bounds1[1,0]

        if pos0 is not None: 
            xp -= pos0[0]
            yp -= pos0[1]
        
        if self.verbosity > 0:
            print('xp:')
            print(xp)
            print('yp:')
            print(yp)
            
        fast_arrays = np.array([xp, yp, dxp_dx, dyp_dx, dxp_dy, dyp_dy])
        self.all_fast_arrays.append(fast_arrays)
        
    def transform_q(self, x, y, idx):
        '''
        Transforms a set of initial coordinates using precomputed fast astrometry arrays.

        Parameters
        ----------

        x, y : floats or '~numpy.ndarrays' of shape (Nsrc,)
            Input pixel coordinates

        idx : int
            Index of transformed observation which has its astrometry precomputed with fit_astrom_arrays

        Returns
        -------

        xnew, ynew : floats or '~numpy.ndarrays' of shape (Nsrc,)
            The transformed set of coordinates.

        '''

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






            