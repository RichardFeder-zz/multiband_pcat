import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double
# in order for visual=True to work, interactive backend should be loaded before importing pyplot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import sys
import os
import warnings
import random
import math
from image_eval import psf_poly_fit, image_model_eval
from helpers import * 
np.seterr(divide='ignore', invalid='ignore')

timestr = time.strftime("%Y%m%d-%H%M%S")
print 'timestr:', timestr
multiple_regions = 0
lin_astrans = 0
fminratio_bool = 0
#generate random seed for initialization
np.random.seed(20170501)


# px and py are for astrometric color correction, don't worry about this

#trueback_dict = dict({"r":179., "i":310., "g":95.}) # run 2583

#swapped riuzg for run 2583
#py = dict({"r":0.02782497, "i":0.10786159, "u":0.01680974, "z":0.0136938, "g":0.00488892})
#px = dict({"r":0.0406713, "i":0.11861364, "u":0.02508619, "z":0.0202184, "g":0.00707397})

#swapped ugriz for run 2583
#py = dict({"r":0.0168097, "i":0.0136938, "g":0.107862})
#px = dict({"r":0.0250862, "i":0.0202184, "g":0.118614})

# swapped riuzg for sparse field, run 8151
px = dict({"r":0.03026955, "i":0.11053126,"u":0.01785904, "z":0.01449279, "g":0.00513858})
py = dict({"r":0.0073814, "i":-0.00581684, "u":0.00432934, "z":0.00352542, "g":0.00125652})

# swapped ugriz for sparse field, run 8151
#px = dict({"g":0.11053126,"r":0.01785904, "i":0.01449279})
#py = dict({"g":-0.00581684, "r":0.00432934, "i":0.00352542})
# backgrounds for sparse field test
trueback_dict = dict({"r":135., "i":215., "g":234., "z":140.}) 



# ------------------------------ these are for input python command, might change this format -------------------------
# script arguments
dataname = str(sys.argv[1])
verbtype = int(sys.argv[2])
# 1 to test, 0 not to test
testpsfn = int(sys.argv[3]) > 0
# 'mock' for simulated, 'mock2' for two source blended mock test
datatype = str(sys.argv[4])
# 1 for multiband, 0 for single band (I think single band is deprecated now but you can do one band fit with multiband=1
multiband = int(sys.argv[5]) > 0

if datatype=='mock2':
    config_type = str(sys.argv[6])
    nrealization = int(sys.argv[7])
    multiple_regions = 0
if datatype=='mock' or datatype=='mock2':
    trueback = []
    lin_astrans = 0

# --------------------------------------------------------------------------------------------------------------------

mock_test_name = 'mock_2star_30'

bands, ncs, nbins, psfs, cfs, pixel_transfer_mats, biases, gains, \
    data_array, data_hdrs, weights, nmgy_per_count, best_fit_astrans, mean_dpos = [[] for x in xrange(14)]


#-------------------------- this portion sets up base directories, probably will need to modify ---------------------

if sys.platform=='darwin':
    base_path = '/Users/richardfeder/Documents/multiband_pcat/pcat-lion-master'
elif sys.platform=='linux2':
    base_path = '/n/fink1/rfeder/mpcat/multiband_pcat'
else:
    base_path = raw_input('Operating system not detected, please enter base_path directory (eg. /Users/.../pcat-lion-master):')
    if not os.path.isdir(base_path):
        raise OSError('Directory chosen does not exist. Please try again.')

if datatype=='mock2':
    result_path = base_path + '/Data'
else:
    if sys.platform=='linux2':
        result_path = '/n/home07/rfederstaehle/pcat-lion-results'
    else:
        result_path = base_path + '/pcat-lion-results'
print 'Results will go in', result_path

#-------------------------- portion below for manually entering in bands used in fit, one by one --------------------

if multiband:
    if datatype=='mock2':
        if config_type=='r' or config_type=='rx3':
            bands = ['r']
        elif config_type=='r+i+g':
            bands = ['r', 'i', 'g']
    else:
        band = raw_input("Enter bands one by one in lowercase ('x' if no more): ")
        while band != 'x':
            bands.append(band)
            band = raw_input("Enter bands one by one in lowercase ('x' if no more): ")
    nbands = len(bands)
    if datatype != 'mock2':
        comments = str(raw_input("Any comments? "))
        with open('comments.txt', 'w') as p:
            p.write(comments)
            p.close()
    print('Loading data for the following bands: ' + str(bands))
else:
    nbands = 1
    bands = ['']


trueback = []       
if datatype != 'mock' and datatype != 'mock2':      
    for band in bands:      
        trueback.append(trueback_dict[band])

print 'datatype:', datatype

start_time = time.clock()


'''now, we want to read in (1) psf template (2) pixel txt file with bias/gain/etc. and (3) the actual image we're cataloging (cts file). Once we do this then we preprocess the data: '''


for b in xrange(nbands):

    if datatype=='mock2':
        base_path = 'Data/'+mock_test_name
        paths = [base_path+'/psfs/'+mock_test_name+'-psf'+str(bands[b])+'.txt', \
                 base_path+'/pixs/'+mock_test_name+'-pix'+str(bands[b])+'.txt']
        paths.append(base_path+'/'+dataname+'/'+dataname+'-nr'+str(nrealization)+'-cts'+ bands[b]+'.txt')
    else:
        #paths = ['Data/'+dataname+'/psfs/'+dataname+'-psf-refit_g.txt', 'Data/'+dataname+'/pixs/'+dataname+'-pix.txt', 'Data/'+dataname+'/cts/'+dataname+'-cts.txt'] # used for refit g band psf for run 2583, not used for sparse field test
        paths = ['Data/'+dataname+'/psfs/'+dataname+'-psf.txt', 'Data/'+dataname+'/pixs/'+dataname+'-pix.txt', 'Data/'+dataname+'/cts/'+dataname+'-cts.txt']
    
        if datatype=='mock':
            paths = ['Data/'+dataname+'/psfs/'+dataname+'-psf.txt', 'Data/'+dataname+'/pixs/'+dataname+'-pix.txt', 'Data/'+dataname+'/cts/'+dataname+'-cts.txt']   
        if multiband:
            for p in xrange(len(paths)):
                if 'refit_g' in paths[p]:
                    print 'Using refit PSF'
                    #paths[p] = paths[p][:-10]+str(bands[b])+paths[p][-10:]
                    paths[p] = paths[p][:-12]+str(bands[b])+paths[p][-12:]
                    print paths[p]
                else:
                    paths[p] = paths[p][:-4]+str(bands[b])+paths[p][-4:]

    psf, nc, cf = get_psf_and_vals(paths[0])
    ncs.append(nc)
    cfs.append(cf)
    psfs.append(psf)

    g = open(paths[1])
    # w and h are the width and height of the image, might need to change a few things in the code that assume the image is a square
    w, h, nb = [np.int32(i) for i in g.readline().split()]
    bias, gain = [np.float32(i) for i in g.readline().split()]
    if multiband:
        a = np.float32(g.readline().split())
        print a, len(a)
        nmgy_per_count.append(a[0])
        if datatype=='mock' or datatype=='mock2':
            #if b>0:
            #    trueback.append(a[1])
            #else:
            #    trueback.append(a[1]+20) # seeing how incorrect background in i band affects joint fit
            trueback.append(a[1])
    biases.append(bias)
    gains.append(gain)
    g.close()


    ''' read in the image, subtract bias, divide by gain to get in units of ADU, and then compute variances
    used when calculate likelihoods with inverse variance weighting. we assume here that the noise is gaussian '''
    data = np.loadtxt(paths[2]).astype(np.float32)
    data -= bias
    data_array.append(data)
    variance = data / gain
    weight = 1. / variance
    weights.append(weight)

    ''' establish image dimensions for first band, then other bands will use these. 
    there might need to be some modifications to the code for non-square images but shouldn't be much of a problem..
    don't worry about the pixel transfer matrices stuff this is just for multiband optical sdss data'''
    if b==0:
        w0=w
        h0=h
        imsz = (w0, h0)
    
    if b > 0:
        # since we're doing color corrections in pcat_multiband_eval, use astrans files with 0p01 default color
        #pathname = 'Data/'+dataname+'/asGrid/asGrid002583-2-0136-0100x0100-'+bands[0]+'-'+bands[b]+'-0310-0630_cterms_0p01_0p01_1203.fits'
        #pathname = 'Data/'+dataname+'/asGrid/asGrid002583-2-0136-0100x0100-'+bands[0]+'-'+bands[b]+'-0310-0630_cterms0_0_nov12.fits'
        pathname = 'Data/'+dataname+'/asGrid/asGrid008151-4-0063-0500x0500-'+bands[0]+'-'+bands[b]+'-0100-0100_cterms_0p01_0p01.fits' #sparse field
        if os.path.isfile(pathname):
            mats = read_astrans_mats(pathname)
            pixel_transfer_mats.append(mats)
            dx, dy = find_mean_offset(pathname, dim=imsz[0])
            dpos = [int(round(dx)), int(round(dy))]
            # data specific adjustment for run 2583
            if dpos[0]==0:
                dpos[0]-= 1
            if dpos[1]==11:
                dpos[1]-= 1.

            #mean_dpos.append([3., 10.])
            mean_dpos.append(dpos)

        else:
            if bands[b]==bands[b-1]:
                mean_dpos.append([0, 0])

            mats = generate_default_astrans([imsz[0], imsz[1]])
            pixel_transfer_mats.append(mats)

        if lin_astrans:
            linex, liney = best_fit_transform(mats)
            best_fit_astrans.append([linex, liney])
        assert w==w0 and h==h0

    if testpsfn:
        testpsf(ncs[b], cf, psfs[b], np.float32(np.random.uniform()*4), np.float32(np.random.uniform()*4), lib=libmmult.pcat_model_eval)



print 'mean_dpos:', mean_dpos
print 'biases:', biases
print 'gains:', gains
print 'nmgy_per_count:', nmgy_per_count
print 'backgrounds:', trueback


''' the multiple regions boolean determines whether subregions are used in each catalog sampling step (i.e. which regions point sources are modified in / where dlogL is calculated at a given step. this subregion approach helps a lot with convergence. the regions are set up in a checkerboard fashion and non overlapping regions are chosen at each step I think '''

if lin_astrans:
    print 'Using linear approximation to asTrans linear interpolation'

''' dont divide image into smaller regions if the image size is small. '''
if imsz[0] < 50:
    multiple_regions = 0

if multiple_regions:
    regsize = imsz[0]/5
    regions_factor = (float(regsize)/float(imsz[0]))**2
else:
    regsize = imsz[0]# single region
    regions_factor = 1
assert imsz[0] % regsize == 0
assert imsz[1] % regsize == 0

''' N_eff is the effective number of pixels in the PSF, I believe this is defined by the PSF FWHM. 
This is then used with the pixel variance (assuming Gaussian noise) to set err_f which is used to set sample step sizes.'''
margin = 10
pixel_variance = trueback[0]/gains[0]
N_eff = 17.5
err_f = np.sqrt(N_eff * pixel_variance)


if datatype=='mock2':
    N_src = 10.
    nsamp = 300
    if config_type=='rx3':
        trueback[0] *= 3
else:
    '''  this is *roughly* the number of sources you think are in the image and is used to set one of the 
    sample step sizes, as long as its within a factor of a few it should be fine''' 
    N_src = 1400.
    #nsamp = 3000
    ''' this sets how many thinned samples are obtained, out of which we usually take the last 300 or something. 
    in total the sampler is taking nsamp*nloop steps but adjacent samples are highly correlated'''
    nsamp = 1500

nloop = 1000 # factor by which sample chain is thinned

''' dont worry about ntemps/temps, was used when we were trying out a simulated annealing approach with galaxies'''
ntemps = 1
temps = np.sqrt(2) ** np.arange(ntemps)


def initialize_c():
    if verbtype > 1:
        print 'initializing c routines and data structs'
    if os.path.getmtime('pcat-lion.c') > os.path.getmtime('pcat-lion.so'):
        warnings.warn('pcat-lion.c modified after compiled pcat-lion.so', Warning)
    array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
    array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    array_2d_double = npct.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
    libmmult.pcat_model_eval.restype = None
    libmmult.pcat_model_eval.argtypes = [c_int, c_int, c_int, c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_1d_int, array_1d_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]
    array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")
    libmmult.pcat_imag_acpt.restype = None
    libmmult.pcat_imag_acpt.argtypes = [c_int, c_int, array_2d_float, array_2d_float, array_2d_int, c_int, c_int, c_int, c_int]
    libmmult.pcat_like_eval.restype = None
    libmmult.pcat_like_eval.argtypes = [c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]

def create_directories(time_string):
    if datatype=='mock2':
        new_dir_name = result_path+'/'+mock_test_name+'/' + str(dataname) + '/results'
    else:
        new_dir_name = result_path +'/'+ str(time_string)
    frame_dir_name = new_dir_name+'/frames'
    if not os.path.isdir(frame_dir_name):
        os.makedirs(frame_dir_name)
    return frame_dir_name, new_dir_name

def flux_proposal(f0, nw, trueminf, b):
    lindf = np.float32(err_f/(regions_factor*np.sqrt(N_src*(2+nbands))))
    logdf = np.float32(0.01/np.sqrt(N_src))
    ff = np.log(logdf*logdf*f0 + logdf*np.sqrt(lindf*lindf + logdf*logdf*f0*f0)) / logdf
    ffmin = np.log(logdf*logdf*trueminf + logdf*np.sqrt(lindf*lindf + logdf*logdf*trueminf*trueminf)) / logdf
    dff = np.random.normal(size=nw).astype(np.float32)
    aboveffmin = ff - ffmin
    oob_flux = (-dff > aboveffmin)
    dff[oob_flux] = -2*aboveffmin[oob_flux] - dff[oob_flux]
    pff = ff + dff
    pf = np.exp(-logdf*pff) * (-lindf*lindf*logdf*logdf+np.exp(2*logdf*pff)) / (2*logdf*logdf)
    return pf


''' the multiband model evaluation looks complicated because we tried out a bunch of things with the astrometry, but could probably rewrite this function. it uses image_model_eval which is written in both python and C (C is faster as you might expect)'''
def pcat_multiband_eval(x, y, f, bkg, imsz, nc, cf, weights, ref, lib, regsize, margin, offsetx, offsety, eps=None, modl_eval_colors=None):
    dmodels = []
    dt_transf = 0
    
    for b in xrange(nbands):
        if b>0:
            t4 = time.clock()

            nmgys = [nmgy_per_count[0], nmgy_per_count[b]]
            colors = adus_to_color(np.abs(f[0]), np.abs(f[b]), nmgys) #need to take absolute value to get colors for negative flux phonions

            if bands[b]==bands[0]:
                xp = x
                yp = y
            elif lin_astrans:
                xp = x + best_fit_astrans[b-1][0](x)
                yp = y + best_fit_astrans[b-1][1](y)
            else:
                if datatype != 'mock' and datatype != 'mock2':
                    x_c = x+colors*px[bands[0]]
                    y_c = y+colors*py[bands[0]]
                    # because we undo the color correction here, some sources may have index out of range
                    x_c[x_c > 499.5] = 499.49 # sparse 500x500 pixel field
                    y_c[y_c > 499.5] = 499.49
                    #x_c[x_c > 99.5] = 99.49 # run 2583
                    #y_c[y_c > 99.5] = 99.49
                    xp, yp = transform_q(x_c, y_c, pixel_transfer_mats[b-1])
                else:
                    xp, yp = transform_q(x, y, pixel_transfer_mats[b-1])
            
            if datatype != 'mock' and datatype !='mock2':             #correcting for different band trimmings, hubble offset
                xp -= mean_dpos[b-1][0]
                yp -= mean_dpos[b-1][1]
            
            if eps is not None: # absolute astrometry proposal perturbs positions
                xp += eps[b-1][0]
                yp += eps[b-1][1]
            
            # make color correction in new band
            if datatype != 'mock' and datatype != 'mock2':
                xp -= colors*px[bands[b]]
                yp -= colors*py[bands[b]]

            if verbtype > 1 and len(xp)>0:
                print b, np.amin(xp), np.amax(xp), np.amin(yp), np.amax(yp)

            dt_transf += time.clock()-t4
            dmodel, diff2 = image_model_eval(xp, yp, f[b], bkg[b], imsz, nc[b], np.array(cf[b]).astype(np.float32()), weights=weights[b], ref=ref[b], lib=libmmult.pcat_model_eval, regsize=regsize, margin=margin, offsetx=offsetx, offsety=offsety)
            diff2s += diff2
        
        else:    
            xp=x
            yp=y
            dmodel, diff2 = image_model_eval(xp, yp, f[b], bkg[b], imsz, nc[b], np.array(cf[b]).astype(np.float32()), weights=weights[b], ref=ref[b], lib=libmmult.pcat_model_eval, regsize=regsize, margin=margin, offsetx=offsetx, offsety=offsety)
            diff2s = diff2
        dmodels.append(dmodel)

    return dmodels, diff2s, dt_transf


# ix, iy = 0. to 3.999
def testpsf(nc, cf, psf, ix, iy, lib=None):
    psf0 = image_model_eval(np.array([12.-ix/5.], dtype=np.float32), np.array([12.-iy/5.], dtype=np.float32), np.array([1.], dtype=np.float32), 0., (25,25), nc, cf, lib=lib)
    plt.subplot(2,2,1)
    plt.imshow(psf0, interpolation='none', origin='lower')
    plt.title('matrix multiply PSF')
    plt.subplot(2,2,2)
    iix = int(np.floor(ix))
    iiy = int(np.floor(iy))
    dix = ix - iix
    diy = iy - iiy
    f00 = psf[iiy:125:5,  iix:125:5]
    f01 = psf[iiy+1:c125:5,iix:125:5]
    f10 = psf[iiy:125:5,  iix+1:125:5]
    f11 = psf[iiy+1:125:5,iix+1:125:5]
    realpsf = f00*(1.-dix)*(1.-diy) + f10*dix*(1.-diy) + f01*(1.-dix)*diy + f11*dix*diy
    plt.imshow(realpsf, interpolation='none', origin='lower')
    plt.title('bilinear interpolate PSF')
    invrealpsf = np.zeros((25,25))
    mask = realpsf > 1e-3
    invrealpsf[mask] = 1./realpsf[mask]
    plt.subplot(2,2,3)
    plt.title('absolute difference')
    plt.imshow(psf0-realpsf, interpolation='none', origin='lower')
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow((psf0-realpsf)*invrealpsf, interpolation='none', origin='lower')
    plt.colorbar()
    plt.title('fractional difference')
    plt.show()

''' neighbours function is used in merge proposal, where you have some source and you want to choose a nearby source with some probability to merge'''
def neighbours(x,y,neigh,i,generate=False):
    neighx = np.abs(x - x[i])
    neighy = np.abs(y - y[i])
    adjacency = np.exp(-(neighx*neighx + neighy*neighy)/(2.*neigh*neigh))
    adjacency[i] = 0.
    neighbours = np.sum(adjacency)
    if generate:
        if neighbours:
            j = np.random.choice(adjacency.size, p=adjacency.flatten()/float(neighbours))
        else:
            j = -1
        return neighbours, j
    else:
        return neighbours

def get_region(x, offsetx, regsize):
    return np.floor(x + offsetx).astype(np.int) / regsize

def idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize):
    match_x = (get_region(x[0:n], offsetx, regsize) % 2) == parity_x
    match_y = (get_region(y[0:n], offsety, regsize) % 2) == parity_y
    return np.flatnonzero(np.logical_and(match_x, match_y))

class Proposal:
    _X = 0
    _Y = 1
    _F = 2

    def __init__(self):
        self.idx_move = None
        self.do_birth = False
        self.idx_kill = None
        self.factor = None
        self.eps_shift = None
        self.goodmove = False
        self.dback = np.zeros(nbands, dtype=np.float32)
        self.xphon = np.array([], dtype=np.float32)
        self.yphon = np.array([], dtype=np.float32)
        self.fphon = []
        self.modl_eval_colors = []
        for x in xrange(nbands):
            self.fphon.append(np.array([], dtype=np.float32))
        self.eps = np.zeros((nbands-1, 2), dtype=np.float32)

    def set_factor(self, factor):
        self.factor = factor

    def in_bounds(self, catalogue):
        return np.logical_and(np.logical_and(catalogue[self._X,:] > 0, catalogue[self._X,:] < (imsz[0] -1)), \
                np.logical_and(catalogue[self._Y,:] > 0, catalogue[self._Y,:] < imsz[1] - 1))

    def assert_types(self):
        assert self.xphon.dtype == np.float32
        assert self.yphon.dtype == np.float32
        assert self.fphon[0].dtype == np.float32

    def __add_phonions_stars(self, stars, remove=False):
        fluxmult = -1 if remove else 1

        self.xphon = np.append(self.xphon, stars[self._X,:])
        self.yphon = np.append(self.yphon, stars[self._Y,:])

        for b in xrange(nbands):
            self.fphon[b] = np.append(self.fphon[b], np.array(fluxmult*stars[self._F+b,:], dtype=np.float32))
        self.assert_types()

    def add_move_stars(self, idx_move, stars0, starsp, modl_eval_colors=[]):
        self.idx_move = idx_move
        self.stars0 = stars0
        self.starsp = starsp
        self.goodmove = True
        inbounds = self.in_bounds(starsp)
        if np.sum(~inbounds)>0:
            starsp[:,~inbounds] = stars0[:,~inbounds]
        self.__add_phonions_stars(stars0, remove=True)
        self.__add_phonions_stars(starsp)
        
    def add_birth_stars(self, starsb):
        self.do_birth = True
        self.starsb = starsb
        self.goodmove = True
        if starsb.ndim == 3:
            starsb = starsb.reshape((starsb.shape[0], starsb.shape[1]*starsb.shape[2]))
        self.__add_phonions_stars(starsb)

    def add_death_stars(self, idx_kill, starsk):
        self.idx_kill = idx_kill
        self.starsk = starsk
        self.goodmove = True
        if starsk.ndim == 3:
            starsk = starsk.reshape((starsk.shape[0], starsk.shape[1]*starsk.shape[2]))
        self.__add_phonions_stars(starsk, remove=True)

    def add_eps_shift(self, stars0): # for absolute astrometry offset, not working currently, don't worry about this
        self.goodmove = True
        self.eps_shift = True
        self.stars0 = stars0
        self.starsp = stars0
        self.__add_phonions_stars(stars0)

    def get_ref_xy(self):
        if self.idx_move is not None:
            return self.stars0[self._X,:], self.stars0[self._Y,:]
        elif self.do_birth:
            bx, by = self.starsb[[self._X,self._Y],:]
            refx = bx if bx.ndim == 1 else bx[:,0]
            refy = by if by.ndim == 1 else by[:,0]
            return refx, refy
        elif self.idx_kill is not None:
            xk, yk = self.starsk[[self._X,self._Y],:]
            refx = xk if xk.ndim == 1 else xk[:,0]
            refy = yk if yk.ndim == 1 else yk[:,0]
            return refx, refy
        elif self.eps_shift is not None:
            return self.stars0[self._X,:], self.stars0[self._Y,:]


class Model:
    nstar = 2000 # this sets the maximum number of sources allowed in thee code, can change depending on the image
    if datatype=='mock2':
        nstar = 20
    trueminf = np.float32(236) # minimum flux for source in ADU, might need to change
    truealpha = np.float32(2) # power law flux slope parameter, might need to change
    alph = 1.0 # used as scalar factor in regularization prior, which determines the penalty in dlogL when adding/subtracting a source
    penalty = 1+0.5*alph*(nbands)
    kickrange = 1. # sets scale for merge proposal i.e. how far you look for neighbors to merge
    eps_prop_sig = 0.001 # don't worry about this
    

    ''' mus and sigs set the gaussian color priors used in optical, won't need these ''' 
    mus = dict({'r-i':0.1, 'r-g':-0.3, 'r-z':0.0, 'r-r':0.0})
    #sigs = dict({'r-i':0.5, 'r-g':5.0, 'r-z':1.0, 'r-r':0.05})
    sigs = dict({'r-i':1.0, 'r-g':1.0, 'r-z':3}) #very broad color prior
    #mus = dict({'r-i':0.0, 'r-g':0.0}) # for same flux different noise tests, r, i, g are all different realizations of r band
    #sigs = dict({'r-i':0.5, 'r-g':0.5})
    color_mus, color_sigs = [], []
    
    for b in xrange(nbands-1):
        col_string = bands[0]+'-'+bands[b+1]
        color_mus.append(mus[col_string])
        color_sigs.append(sigs[col_string])

    print 'Color prior mean/s:', color_mus
    print 'Color prior width/s:', color_sigs

    _X = 0
    _Y = 1
    _F = 2


    split_col_sig = 0.25 # used when splitting sources and determining colors of resulting objects
    k =2.5/np.log(10)

    ''' the init function sets all of the data structures used for the catalog, 
    randomly initializes catalog source values drawing from catalog priors  '''
    def __init__(self):
        self.back = np.zeros(nbands, dtype=np.float32)
        self.regsize = regsize
        self.n = np.random.randint(self.nstar)+1
        self.stars = np.zeros((2+nbands,self.nstar), dtype=np.float32)
        self.stars[:,0:self.n] = np.random.uniform(size=(2+nbands,self.n))  # refactor into some sort of prior function?
        self.stars[self._X,0:self.n] *= imsz[0]-1
        self.stars[self._Y,0:self.n] *= imsz[1]-1
        self.eps = np.zeros((nbands-1, 2), dtype=np.float32)

        for b in xrange(nbands):
            self.back[b] = trueback[b]
            if b==0:
                self.stars[self._F+b,0:self.n] **= -1./(self.truealpha - 1.)
                self.stars[self._F+b,0:self.n] *= self.trueminf
            else:
                new_colors = np.random.normal(loc=self.color_mus[b-1], scale=self.color_sigs[b-1], size=self.n)
                self.stars[self._F+b,0:self.n] = self.stars[self._F,0:self.n]*10**(0.4*new_colors)*nmgy_per_count[0]/nmgy_per_count[b]

    ''' run_sampler() completes nloop samples, so the function is called nsamp times'''
    def run_sampler(self, temperature, nloop=1000, multiband=False):
        t0 = time.clock()
        nmov = np.zeros(nloop)
        movetype = np.zeros(nloop)
        accept = np.zeros(nloop)
        outbounds = np.zeros(nloop)
        dt1 = np.zeros(nloop) # dt1/dt2/dt3/dttq are just for time performance diagnostics
        dt2 = np.zeros(nloop)
        dt3 = np.zeros(nloop)
        dttq = np.zeros(nloop)
        diff2_list = np.zeros(nloop) 

        if multiple_regions:
            self.offsetx = np.random.randint(self.regsize)
            self.offsety = np.random.randint(self.regsize)
        else:
            self.offsetx = 0
            self.offsety = 0
 
        self.nregx = imsz[0] / self.regsize + 1
        self.nregy = imsz[1] / self.regsize + 1

        resids = []
        for b in xrange(nbands):
            resid = data_array[b].copy() # residual for zero image is data
            resids.append(resid)


        evalx = self.stars[self._X,0:self.n]
        evaly = self.stars[self._Y,0:self.n]
        evalf = self.stars[self._F:,0:self.n]
        
        n_phon = evalx.size

        if verbtype > 1:
            print 'beginning of run sampler'
            print 'self.n here'
            print self.n
            print 'n_phon'
            print n_phon

        models, diff2s, dt_transf = pcat_multiband_eval(evalx, evaly, evalf, self.back, imsz, ncs, cfs, weights=weights, ref=resids, lib=libmmult.pcat_model_eval, regsize=self.regsize, margin=margin, offsetx=self.offsetx, offsety=self.offsety, eps = self.eps)
        logL = -0.5*diff2s
        
        if verbtype > 1:
            other_models, diff2s, dt_transf = pcat_multiband_eval(self.stars[self._X,:], self.stars[self._Y,:], self.stars[self._F:,:], self.back, imsz, ncs, cfs, weights=weights, ref=resids, lib=libmmult.pcat_model_eval, regsize=self.regsize, margin=margin, offsetx=self.offsetx, offsety=self.offsety, eps = self.eps)
            
            for b in xrange(nbands):
                df2 = np.sum(weights[b]*(data_array[b]-other_models[b])*(data_array[b]-other_models[b]))
                print 'df2 for band with all of self.stars', b
                print df2

            for b in xrange(nbands):
                df2 = np.sum(weights[b]*(data_array[b]-models[b])*(data_array[b]-models[b]))
                print 'df2 for band', b
                print df2

            print 'beginning diff2s'
            print diff2s
            print 'beginning logL'
            print logL
        

        for b in xrange(nbands):
            resids[b] -= models[b]
        
        '''the proposals here are: move_stars (P) which changes the parameters of existing model sources, 
        birth/death (BD) and merge/split (MS). Don't worry about perturb_astrometry. 
        The moveweights array, once normalized, determines the probability of choosing a given proposal. '''
        moveweights = np.array([80., 40., 40., 0.])
        movetypes = ['P *', 'BD *', 'MS *', 'EPS *']
        movefns = [self.move_stars, self.birth_death_stars, self.merge_split_stars, self.perturb_astrometry]
        moveweights /= np.sum(moveweights)

        if multiple_regions:
            xparities = np.random.randint(2, size=nloop)
            yparities = np.random.randint(2, size=nloop)
        
        rtype_array = np.random.choice(moveweights.size, p=moveweights, size=nloop)
        movetype = rtype_array

        if verbtype > 1:
            for b in xrange(nbands):
                df2 = np.sum(weights[b]*(data_array[b]-models[b])*(data_array[b]-models[b]))
                print 'df2 for band right before nloop loop', b
                print df2

        for i in xrange(nloop):
            t1 = time.clock()
            rtype = rtype_array[i]
            if rtype==3 and verbtype > 1:
                print 'chose rtype, now self.eps is', self.eps
            if verbtype > 1:
                print 'rtype'
                print rtype
            if multiple_regions:
                self.parity_x = xparities[i] # should regions be perturbed randomly or systematically?
                self.parity_y = yparities[i]
            else:
                self.parity_x = 0
                self.parity_y = 0

            #proposal types
            proposal = movefns[rtype]()
            dt1[i] = time.clock() - t1
            
            if proposal.goodmove:
                t2 = time.clock()
                dmodels, diff2s, dt_transf = pcat_multiband_eval(proposal.xphon, proposal.yphon, proposal.fphon, proposal.dback, imsz, ncs, cfs, weights=weights, ref=resids, lib=libmmult.pcat_model_eval, regsize=self.regsize, margin=margin, offsetx=self.offsetx, offsety=self.offsety, eps=proposal.eps)

                dttq[i] = dt_transf
                plogL = -0.5*diff2s                
                plogL[(1-self.parity_y)::2,:] = float('-inf') # don't accept off-parity regions
                plogL[:,(1-self.parity_x)::2] = float('-inf')
                dlogP = plogL - logL
                
                if verbtype > 1:
                    print 'dlogP'
                    print dlogP
                
                assert np.isnan(dlogP).any() == False

                dt2[i] = time.clock() - t2
                t3 = time.clock()
                refx, refy = proposal.get_ref_xy()
                regionx = get_region(refx, self.offsetx, self.regsize)
                regiony = get_region(refy, self.offsety, self.regsize)
                
                if proposal.factor is not None:
                    dlogP[regiony, regionx] += proposal.factor
                else:
                    print 'proposal factor is None'
                acceptreg = (np.log(np.random.uniform(size=(self.nregy, self.nregx))) < dlogP).astype(np.int32)
                acceptprop = acceptreg[regiony, regionx]
                numaccept = np.count_nonzero(acceptprop)
                ''' for each band compute the delta log likelihood between states, theen add these together'''
                for b in xrange(nbands):
                    dmodel_acpt = np.zeros_like(dmodels[b])
                    diff2_acpt = np.zeros_like(diff2s)
                    libmmult.pcat_imag_acpt(imsz[0], imsz[1], dmodels[b], dmodel_acpt, acceptreg, self.regsize, margin, self.offsetx, self.offsety)
                    # using this dmodel containing only accepted moves, update logL
                    libmmult.pcat_like_eval(imsz[0], imsz[1], dmodel_acpt, resids[b], weights[b], diff2_acpt, self.regsize, margin, self.offsetx, self.offsety)   

                    resids[b] -= dmodel_acpt
                    models[b] += dmodel_acpt

                    if b==0:
                        diff2_total1 = diff2_acpt
                    else:
                        diff2_total1 += diff2_acpt

                logL = -0.5*diff2_total1

                if verbtype > 1:
                    print 'diff2_total1'
                    print diff2_total1
                    print 'logL'
                    print logL

                #implement accepted moves
                if proposal.idx_move is not None:
                    if verbtype > 1:
                        print 'acceptprop'
                        print acceptprop

                    starsp = proposal.starsp.compress(acceptprop, axis=1)
                    idx_move_a = proposal.idx_move.compress(acceptprop)
                    if verbtype > 1:
                        print 'idx_move_a'
                        print idx_move_a

                    self.stars[:, idx_move_a] = starsp

                    if verbtype > 1:
                        print 'implementing idx_move proposal'
                        print 'idx_move_a'
                        print idx_move_a
                        print 'size of idx_move_a'
                        print len(idx_move_a), np.count_nonzero(idx_move_a)
                        print 'starsp'
                        print starsp
                
                if proposal.do_birth:
                    if verbtype> 1:
                        print 'implementing birth proposal'
                    starsb = proposal.starsb.compress(acceptprop, axis=1)
                    starsb = starsb.reshape((2+nbands,-1))
                    num_born = starsb.shape[1]
                    self.stars[:, self.n:self.n+num_born] = starsb
                    self.n += num_born
                    if verbtype > 1:
                        print 'num_born'
                        print num_born
                        print 'starsb'
                        print starsb 

                if proposal.idx_kill is not None:
                    idx_kill_a = proposal.idx_kill.compress(acceptprop, axis=0).flatten()
                    num_kill = idx_kill_a.size
                    
                    if verbtype > 1:
                        print 'implementing idx_kill proposal'    
                        print 'num kill'
                        print num_kill
                        print 'stars killed'
                        print self.stars[:,idx_kill_a]

                    # nstar is correct, not n, because x,y,f are full nstar arrays
                    self.stars[:, 0:self.nstar-num_kill] = np.delete(self.stars, idx_kill_a, axis=1)
                    self.stars[:, self.nstar-num_kill:] = 0
                    self.n -= num_kill


                if proposal.eps_shift: # does not work currently
                    if numaccept > 0:
                        print 'implementing it'
                        if verbtype > 1:
                            print 'implementing astrometry offset proposal'
                        self.eps = proposal.eps

                dt3[i] = time.clock() - t3

                if acceptprop.size > 0:
                    accept[i] = np.count_nonzero(acceptprop) / float(acceptprop.size)
                else:
                    accept[i] = 0
            else:
                if verbtype > 1:
                    print 'out of bounds'
                outbounds[i] = 1

            for b in xrange(nbands):
                diff2_list[i] += np.sum(weights[b]*(data_array[b]-models[b])*(data_array[b]-models[b]))
                    
            if verbtype > 1:
                print 'end of Loop', i
                print 'self.n'
                print self.n
                print 'diff2'
                print diff2_list[i]
            
        chi2 = np.zeros(nbands)
        for b in xrange(nbands):
            chi2[b] = np.sum(weights[b]*(data_array[b]-models[b])*(data_array[b]-models[b]))
            
        if verbtype > 1:
            print 'end of sample'
            print 'self.n end'
            print self.n
            
            for b in xrange(nbands):
                df2 = np.sum(weights[b]*(data_array[b]-models[b])*(data_array[b]-models[b]))
                print 'df2 for band at end of run_sampler', b
                print df2

        ''' this section prints out some information at the end of each thinned sample, 
        namely acceptance fractions for the different proposals and some time performance statistics as well. '''
        
        fmtstr = '\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f (Pg) %0.3f (BDg) %0.3f (S-g) %0.3f (gSg) %0.3f (gMS) %0.3f'
        print 'Background', self.back, 'N_star', self.n, 'eps:', self.eps, 'chi^2', list(chi2)
        dt1 *= 1000
        dt2 *= 1000
        dt3 *= 1000
        dttq *= 1000
        accept_fracs = []
        timestat_array = np.zeros((6, 1+len(moveweights)), dtype=np.float32)
        statlabels = ['Acceptance', 'Out of Bounds', 'Proposal (s)', 'Likelihood (s)', 'Implement (s)', 'Coordinates (s)']
        statarrays = [accept, outbounds, dt1, dt2, dt3, dttq]
        for j in xrange(len(statlabels)):
            timestat_array[j][0] = np.sum(statarrays[j])/1000
            if j==0:
                accept_fracs.append(np.sum(statarrays[j])/1000)
            print statlabels[j]+'\t(all) %0.3f' % (np.sum(statarrays[j])/1000),
            for k in xrange(len(movetypes)):
                if j==0:
                    accept_fracs.append(np.mean(statarrays[j][movetype==k]))
                timestat_array[j][1+k] = np.mean(statarrays[j][movetype==k])
                print '('+movetypes[k]+') %0.3f' % (np.mean(statarrays[j][movetype == k])),
            print
            if j == 1:
                print '-'*16
        print '-'*16
        print 'Total (s): %0.3f' % (np.sum(statarrays[2:])/1000)
        print '='*16

        if verbtype > 1:
            for b in xrange(nbands):
                df2 = np.sum(weights[b]*(data_array[b]-models[b])*(data_array[b]-models[b]))
                print 'df2 for band at very very end of run_sampler', b
            print df2

        return self.n, chi2, timestat_array, accept_fracs, diff2_list, rtype_array, accept


    def idx_parity_stars(self):
        return idx_parity(self.stars[self._X,:], self.stars[self._Y,:], self.n, self.offsetx, self.offsety, self.parity_x, self.parity_y, regsize)

    def bounce_off_edges(self, catalogue): # works on both stars and galaxies
        mask = catalogue[self._X,:] < 0
        catalogue[self._X, mask] *= -1
        mask = catalogue[self._X,:] > (imsz[0] - 1)
        catalogue[self._X, mask] *= -1
        catalogue[self._X, mask] += 2*(imsz[0] - 1)
        mask = catalogue[self._Y,:] < 0
        catalogue[self._Y, mask] *= -1
        mask = catalogue[self._Y,:] > (imsz[1] - 1)
        catalogue[self._Y, mask] *= -1
        catalogue[self._Y, mask] += 2*(imsz[1] - 1)
        # these are all inplace operations, so no return value

    def in_bounds(self, catalogue):
        return np.logical_and(np.logical_and(catalogue[self._X,:] > 0, catalogue[self._X,:] < (imsz[0] -1)), \
                np.logical_and(catalogue[self._Y,:] > 0, catalogue[self._Y,:] < imsz[1] - 1))

    def perturb_astrometry(self): # not working currently
        #print '1:', self.eps
        proposal = Proposal()
        old_eps = np.zeros((nbands-1, 2)).astype(np.float32)
        for b in xrange(nbands-1):
            for q in xrange(2):
                old_eps[b,q] = self.eps[b,q]
        band_idx = np.random.randint(nbands-1)
        x_or_y = np.random.randint(2)
        old_eps[band_idx, x_or_y] += np.random.normal(loc=0, scale=self.eps_prop_sig)
        proposal.eps = old_eps
        proposal.add_eps_shift(self.stars)
        #self.regsize = imsz[0] #do full image evaluation for epsilon proposals
        #self.__add_phonion_stars(self.stars)
        proposal.set_factor(0)
        
        return proposal

    def move_stars(self): 
        idx_move = self.idx_parity_stars()
        nw = idx_move.size
        stars0 = self.stars.take(idx_move, axis=1)
        starsp = np.empty_like(stars0)
        
        f0 = stars0[self._F:,:]
        pfs = []
        color_factors = np.zeros((nbands-1, nw)).astype(np.float32)

        for b in xrange(nbands):
            if b==0:
                pf = flux_proposal(f0[b], nw, self.trueminf, b)
            else:
                pf = flux_proposal(f0[b], nw, 1, b) #place a minor minf to avoid negative fluxes in non-pivot bands
            pfs.append(pf)
 
        if (np.array(pfs)<0).any():
            print 'negative flux!'
            print np.array(pfs)[np.array(pfs)<0]
        dlogf = np.log(pfs[0]/f0[0])
        if verbtype > 1:
            print 'average flux difference'
            print np.average(np.abs(f0[0]-pfs[0]))
        factor = -self.truealpha*dlogf

        if np.isnan(factor).any():
            print 'factor nan from flux'
            print 'number of f0 zero elements:', len(f0[0])-np.count_nonzero(np.array(f0[0]))
            if verbtype > 1:
                print 'factor'
                print factor
            #os.exit()
            factor[np.isnan(factor)]=0

        ''' the loop over bands below computes colors and prior factors in color used when sampling the posterior '''
        modl_eval_colors = []

        for b in xrange(nbands-1):
            nmpc = [nmgy_per_count[0], nmgy_per_count[b+1]]
            colors = adus_to_color(pfs[0], pfs[b+1], nmpc)
            orig_colors = adus_to_color(f0[0], f0[b+1], nmpc)
            colors[np.isnan(colors)] = self.color_mus[b] # make nan colors not affect color_factors
            orig_colors[np.isnan(orig_colors)] = self.color_mus[b]
            color_factors[b] -= (colors - self.color_mus[b])**2/(2*self.color_sigs[b]**2)
            color_factors[b] += (orig_colors - self.color_mus[b])**2/(2*self.color_sigs[b]**2)
            modl_eval_colors.append(colors)
    

        assert np.isnan(color_factors).any()==False

        if np.isnan(color_factors).any():
            print 'color factors nan'                

        if verbtype > 1:
            print 'avg abs color_factors:', np.average(np.abs(color_factors))
            print 'avg abs flux factor:', np.average(np.abs(factor))

        factor = np.array(factor) + np.sum(color_factors, axis=0)
        
        if multiple_regions:
            dpos_rms = np.float32(np.sqrt(N_eff/(2*np.pi))*err_f/(np.sqrt(N_src*regions_factor*(2+nbands))))/(np.maximum(f0[0],pfs[0]))
        else:
            dpos_rms = np.float32(np.sqrt(N_eff/(2*np.pi))*err_f/np.sqrt(N_src*(2+nbands)))/(np.maximum(f0[0], pfs[0]))
        
        if verbtype > 1:
            print 'dpos_rms'
            print dpos_rms
        
        dpos_rms[dpos_rms < 1e-3] = 1e-3 #do we need this line? perhaps not
        dx = np.random.normal(size=nw).astype(np.float32)*dpos_rms
        dy = np.random.normal(size=nw).astype(np.float32)*dpos_rms
        starsp[self._X,:] = stars0[self._X,:] + dx
        starsp[self._Y,:] = stars0[self._Y,:] + dy
        

        if verbtype > 1:
            print 'dx'
            print dx
            print 'dy'
            print dy
            print 'mean dx and dy'
            print np.mean(np.abs(dx)), np.mean(np.abs(dy))

        for b in xrange(nbands):
            starsp[self._F+b,:] = pfs[b]
            if (pfs[b]<0).any():
                print 'proposal fluxes less than 0'
                print 'band', b
                print pfs[b]
        self.bounce_off_edges(starsp)

        proposal = Proposal()
        proposal.add_move_stars(idx_move, stars0, starsp, modl_eval_colors)
        
        assert np.isinf(factor).any()==False
        assert np.isnan(factor).any()==False

        proposal.set_factor(factor)
        return proposal

    def birth_death_stars(self):
        lifeordeath = np.random.randint(2)
        nbd = (self.nregx * self.nregy) / 4
        proposal = Proposal()
        # birth
        if lifeordeath and self.n < self.nstar: # need room for at least one source
            nbd = min(nbd, self.nstar-self.n) # add nbd sources, or just as many as will fit
            # mildly violates detailed balance when n close to nstar
            # want number of regions in each direction, divided by two, rounded up
            mregx = ((imsz[0] / regsize + 1) + 1) / 2 # assumes that imsz are multiples of regsize
            mregy = ((imsz[1] / regsize + 1) + 1) / 2
            starsb = np.empty((2+nbands, nbd), dtype=np.float32)
            starsb[self._X,:] = (np.random.randint(mregx, size=nbd)*2 + self.parity_x + np.random.uniform(size=nbd))*regsize - self.offsetx
            starsb[self._Y,:] = (np.random.randint(mregy, size=nbd)*2 + self.parity_y + np.random.uniform(size=nbd))*regsize - self.offsety
            
            for b in xrange(nbands):
                if b==0:
                    starsb[self._F+b,:] = self.trueminf * np.exp(np.random.exponential(scale=1./(self.truealpha-1.),size=nbd))
                else:
                    # draw new source colors from color prior
                    new_colors = np.random.normal(loc=self.color_mus[b-1], scale=self.color_sigs[b-1], size=nbd)
                    if verbtype > 1:
                        print 'new_colors'
                        print new_colors
                    starsb[self._F+b,:] = starsb[self._F,:]*10**(0.4*new_colors)*nmgy_per_count[0]/nmgy_per_count[b]
            
                    if (starsb[self._F+b,:]<0).any():
                        print 'negative birth star fluxes'
                        print 'new_colors'
                        print new_colors
                        print 'starsb fluxes:'
                        print starsb[self._F+b,:]

            # some sources might be generated outside image
            inbounds = self.in_bounds(starsb)
            starsb = starsb.compress(inbounds, axis=1)
            factor = np.full(starsb.shape[1], -self.penalty)
            proposal.add_birth_stars(starsb)
            proposal.set_factor(factor)
            
            assert np.isnan(factor).any()==False
            assert np.isinf(factor).any()==False

        # death
        # does region based death obey detailed balance?
        elif not lifeordeath and self.n > 0: # need something to kill
            idx_reg = self.idx_parity_stars()
            nbd = min(nbd, idx_reg.size) # kill nbd sources, or however many sources remain
            if nbd > 0:
                idx_kill = np.random.choice(idx_reg, size=nbd, replace=False)
                starsk = self.stars.take(idx_kill, axis=1)
                factor = np.full(nbd, self.penalty)
                proposal.add_death_stars(idx_kill, starsk)
                proposal.set_factor(factor)
                assert np.isnan(factor).any()==False
        return proposal


    def merge_split_stars(self):

        splitsville = np.random.randint(2)
        idx_reg = self.idx_parity_stars()
        fracs, sum_fs = [],[]
        idx_bright = idx_reg.take(np.flatnonzero(self.stars[self._F, :].take(idx_reg) > 2*self.trueminf)) # in region!
        bright_n = idx_bright.size
        nms = (self.nregx * self.nregy) / 4
        goodmove = False
        proposal = Proposal()
        # split
        if splitsville and self.n > 0 and self.n < self.nstar and bright_n > 0: # need something to split, but don't exceed nstar
            nms = min(nms, bright_n, self.nstar-self.n) # need bright source AND room for split source
            dx = (np.random.normal(size=nms)*self.kickrange).astype(np.float32)
            dy = (np.random.normal(size=nms)*self.kickrange).astype(np.float32)
            idx_move = np.random.choice(idx_bright, size=nms, replace=False)
            stars0 = self.stars.take(idx_move, axis=1)

            fminratio = stars0[self._F,:] / self.trueminf
 
            if verbtype > 1:
                print 'stars0 at splitsville start'
                print stars0
                print 'fminratio here'
                print fminratio
                print 'dx'
                print dx
                print 'dy'
                print dy
                print 'idx_move'
                print idx_move
                
            fracs.append((1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32))
            
            
            for b in xrange(nbands-1):
                # changed to split similar fluxes
                d_color = np.random.normal(0,self.split_col_sig)
                frac_sim = np.exp(d_color/self.k)*fracs[0]/(1-fracs[0]+np.exp(d_color/self.k)*fracs[0])
                
                fracs.append(frac_sim)
                if verbtype > 1:
                    print 'frac_sim'
                    print frac_sim
                    print 'dcolor for band', b+1, 'is', d_color
                    print 'k:', self.k
                    print 'fracs[0]', fracs[0]
                
            starsp = np.empty_like(stars0)
            starsb = np.empty_like(stars0)

            starsp[self._X,:] = stars0[self._X,:] + ((1-fracs[0])*dx)
            starsp[self._Y,:] = stars0[self._Y,:] + ((1-fracs[0])*dy)
            starsb[self._X,:] = stars0[self._X,:] - fracs[0]*dx
            starsb[self._Y,:] = stars0[self._Y,:] - fracs[0]*dy

            for b in xrange(nbands):
                
                starsp[self._F+b,:] = stars0[self._F+b,:]*fracs[b]
                starsb[self._F+b,:] = stars0[self._F+b,:]*(1-fracs[b])
                if (starsp[self._F+b,:]<0).any():
                    print 'neg starsp in band', b
                    print 'stars0'
                    print stars0
                    print 'fracs[b]'
                    print fracs[b]
                    print 'starsp[self._F+b,:]'
                    print starsp[self._F+b,:]
                if (starsb[self._F+b,:]<0).any():
                    print 'neg starsb in band', b
                    print 'stars0'
                    print stars0
                    print '1-fracs[b]'
                    print (1-fracs[b])
                    print 'starsb[self._F+b,:]'
                    print starsb[self._F+b,:]
            # don't want to think about how to bounce split-merge
            # don't need to check if above fmin, because of how frac is decided
            inbounds = np.logical_and(self.in_bounds(starsp), self.in_bounds(starsb))
            stars0 = stars0.compress(inbounds, axis=1)
            starsp = starsp.compress(inbounds, axis=1)
            starsb = starsb.compress(inbounds, axis=1)
            idx_move = idx_move.compress(inbounds)
            fminratio = fminratio.compress(inbounds)

            for b in xrange(nbands):
                fracs[b] = fracs[b].compress(inbounds)
                sum_fs.append(stars0[self._F+b,:])
            
            nms = idx_move.size
            goodmove = nms > 0
            
            if goodmove:
                proposal.add_move_stars(idx_move, stars0, starsp)
                proposal.add_birth_stars(starsb)
                # can this go nested in if statement? 
            invpairs = np.empty(nms)
            

            if verbtype > 1:
                print 'splitsville happening'
                print 'goodmove:', goodmove
                print 'invpairs'
                print invpairs
                print 'nms:', nms
                print 'sum_fs'
                print sum_fs
                print 'fminratio'
                print fminratio

            for k in xrange(nms):
                xtemp = self.stars[self._X, 0:self.n].copy()
                ytemp = self.stars[self._Y, 0:self.n].copy()
                xtemp[idx_move[k]] = starsp[self._X, k]
                ytemp[idx_move[k]] = starsp[self._Y, k]
                xtemp = np.concatenate([xtemp, starsb[self._X, k:k+1]])
                ytemp = np.concatenate([ytemp, starsb[self._Y, k:k+1]])
                invpairs[k] =  1./neighbours(xtemp, ytemp, self.kickrange, idx_move[k]) #divide by zero
                invpairs[k] += 1./neighbours(xtemp, ytemp, self.kickrange, self.n)
            invpairs *= 0.5
        # merge
        elif not splitsville and idx_reg.size > 1: # need two things to merge!
            nms = min(nms, idx_reg.size/2)
            idx_move = np.empty(nms, dtype=np.int)
            idx_kill = np.empty(nms, dtype=np.int)
            choosable = np.zeros(self.nstar, dtype=np.bool)
            choosable[idx_reg] = True
            nchoosable = float(idx_reg.size)
            invpairs = np.empty(nms)
            
            if verbtype > 1:
                print 'merging two things!'
                print 'nms:', nms
                print 'idx_move', idx_move
                print 'idx_kill', idx_kill
                
            for k in xrange(nms):
                idx_move[k] = np.random.choice(self.nstar, p=choosable/nchoosable)
                invpairs[k], idx_kill[k] = neighbours(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], self.kickrange, idx_move[k], generate=True)
                if invpairs[k] > 0:
                    invpairs[k] = 1./invpairs[k]
                # prevent sources from being involved in multiple proposals
                if not choosable[idx_kill[k]]:
                    idx_kill[k] = -1
                if idx_kill[k] != -1:
                    invpairs[k] += 1./neighbours(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], self.kickrange, idx_kill[k])
                    choosable[idx_move[k]] = False
                    choosable[idx_kill[k]] = False
                    nchoosable -= 2
            invpairs *= 0.5

            inbounds = (idx_kill != -1)
            idx_move = idx_move.compress(inbounds)
            idx_kill = idx_kill.compress(inbounds)
            invpairs = invpairs.compress(inbounds)
            nms = idx_move.size
            goodmove = nms > 0


            stars0 = self.stars.take(idx_move, axis=1)
            starsk = self.stars.take(idx_kill, axis=1)
            f0 = stars0[self._F:,:]
            fk = starsk[self._F:,:]

            for b in xrange(nbands):
                sum_fs.append(f0[b,:] + fk[b,:])
                fracs.append(f0[b,:] / sum_fs[b])
            fminratio = sum_fs[0] / self.trueminf
            
            if verbtype > 1:
                print 'fminratio'
                print fminratio
                print 'nms is now', nms
                print 'sum_fs[0]', sum_fs[0]
                print 'all sum_fs:'
                print sum_fs
                print 'stars0'
                print stars0
                print 'starsk'
                print starsk
                print 'idx_move'
                print idx_move
                print 'idx_kill'
                print idx_kill
                
            starsp = np.empty_like(stars0)
            # place merged source at center of flux of previous two sources
            starsp[self._X,:] = fracs[0]*stars0[self._X,:] + (1-fracs[0])*starsk[self._X,:]
            starsp[self._Y,:] = fracs[0]*stars0[self._Y,:] + (1-fracs[0])*starsk[self._Y,:]
            
            for b in xrange(nbands):
                starsp[self._F+b,:] = f0[b] + fk[b]
            
            if goodmove:
                proposal.add_move_stars(idx_move, stars0, starsp)
                proposal.add_death_stars(idx_kill, starsk)
            
            # turn bright_n into an array
            bright_n = bright_n - (f0[0] > 2*self.trueminf) - (fk[0] > 2*self.trueminf) + (starsp[self._F,:] > 2*self.trueminf)
        
        ''' The lines below are where we compute the prior factors that go into P(Catalog), 
        which we use along with P(Data|Catalog) in order to sample from the posterior. 
        The variable "factor" has the log prior (log(P(Catalog))), and since the prior is a product of 
        individual priors we add log factors to get the log prior.'''
        if goodmove:
            # first three terms are ratio of flux priors, remaining terms come from how we choose sources to merge, and last term is Jacobian for the transdimensional proposal
            factor = np.log(self.truealpha-1) + (self.truealpha-1)*np.log(self.trueminf) - self.truealpha*np.log(fracs[0]*(1-fracs[0])*sum_fs[0]) + np.log(2*np.pi*self.kickrange*self.kickrange) - np.log(imsz[0]*imsz[1]) + np.log(1. - 2./fminratio) + np.log(bright_n) + np.log(invpairs) + np.log(sum_fs[0])

            if np.isnan(factor).any():
                print 'nan before colorfactors'

            if verbtype > 1:
                print 'factor before colors'
                print factor
                print 'sum of factor before colors'
                print np.sum(factor)

            for b in xrange(nbands-1):
                stars0_color = adus_to_color(stars0[self._F,:], stars0[self._F+b+1,:], [nmgy_per_count[0], nmgy_per_count[b]])
                starsp_color = adus_to_color(starsp[self._F,:], starsp[self._F+b+1,:], [nmgy_per_count[0], nmgy_per_count[b]])
                dc = self.k*(np.log(fracs[b+1]/fracs[0]) - np.log((1-fracs[b+1])/(1-fracs[0])))
                # added_fac comes from the transition kernel of splitting colors in the manner that we do
                added_fac = 0.5*np.log(2*np.pi*self.split_col_sig**2)+(dc**2/(2*self.split_col_sig**2))
                if np.isnan(added_fac).any():
                    print 'added fac is nan'
                factor += added_fac
                
                if splitsville:
                    if np.isnan(factor).any():
                        print 'nan right before split'
                    starsb_color = adus_to_color(starsb[self._F,:], starsb[self._F+b+1,:], [nmgy_per_count[0], nmgy_per_count[b]])
                    # colfac is ratio of color prior factors i.e. P(s_0)P(s_1)/P(s_merged), where 0 and 1 are original sources 
                    colfac = (stars0_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (starsp_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (starsb_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2)-0.5*np.log(2*np.pi*self.color_sigs[b]**2)
                        
                    factor += colfac
                    if np.isnan(factor).any():
                        print 'nan on split'
                        print 'starsb'
                        print starsb
                        print 'fracs'
                        print fracs
                        print 'starsb_color'
                        print starsb_color
                        print 'starsp_color'
                        print starsp_color
                        print 'first term'
                        print (stars0_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2)
                        print 'second term'
                        print (starsp_color-self.color_mus[b])**2/(2*self.color_sigs[b]**2)
                        print 'third term'
                        print (starsb_color-self.color_mus[b])**2/(2*self.color_sigs[b]**2)
                        print 'fourth term'
                        print -0.5*np.log(2*np.pi*self.color_sigs[b]**2)
                else:
                    starsk_color = adus_to_color(starsk[self._F,:], starsk[self._F+b+1,:], [nmgy_per_count[0], nmgy_per_count[b]])
                    # same as above but for merging sources
                    colfac = (starsp_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (stars0_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (starsk_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2)-0.5*np.log(2*np.pi*self.color_sigs[b]**2)
                    factor += colfac

                    if np.isnan(factor).any():
                        print 'nan on merge'
            if not splitsville:
                factor *= -1
                factor += self.penalty
            else:
                factor -= self.penalty

            proposal.set_factor(factor)
            
            if np.isnan(factor).any():
                print 'factor'
                print factor
                print 
                
            assert np.isnan(factor).any()==False

            if verbtype > 1:
                print 'kickrange factor', np.log(2*np.pi*self.kickrange*self.kickrange)
                print 'imsz factor', np.log(imsz[0]*imsz[1]) 
                print 'fminratio:', fminratio
                print 'fmin factor', np.log(1. - 2./fminratio)
                print 'kickrange factor', np.log(2*np.pi*self.kickrange*self.kickrange) - np.log(imsz[0]*imsz[1]) + np.log(1. - 2./fminratio)
                print 'factor after colors'
                print factor
        return proposal


''' Here is where we initialize the C libraries and instantiate the arrays that will store our thinned samples and other stats '''
libmmult = npct.load_library('pcat-lion', '.')
initialize_c()
nstar = Model.nstar
cprior_vals = np.array([Model.color_mus, Model.color_sigs])
nsample = np.zeros(nsamp, dtype=np.int32)
xsample = np.zeros((nsamp, nstar), dtype=np.float32)
ysample = np.zeros((nsamp, nstar), dtype=np.float32)
timestats = np.zeros((nsamp, 6, 5), dtype=np.float32)
diff2_all = np.zeros((nsamp, nloop), dtype=np.float32)
accept_all = np.zeros((nsamp, nloop), dtype=np.float32)
rtypes = np.zeros((nsamp, nloop), dtype=np.float32)
accept_stats = np.zeros((nsamp, 5), dtype=np.float32)
tq_times = np.zeros(nsamp, dtype=np.float32)
if multiband:
    fsample = [np.zeros((nsamp, nstar), dtype=np.float32) for x in xrange(nbands)]
    colorsample = [[] for x in xrange(nbands-1)]
else:
    fsample = np.zeros((nsamp, nstar), dtype=np.float32)
chi2sample = np.zeros((nsamp, nbands), dtype=np.int32)
eps_sample = np.zeros((nsamp, nbands-1, 2), dtype=np.float32)
models = [Model() for k in xrange(ntemps)]
#create directory for results

frame_dir, newdir = create_directories(timestr)

# sampling loop
for j in xrange(nsamp):
    chi2_all = np.zeros((ntemps,nbands))
    print 'Sample', j
    #if j < 1350 or j > 1480:
    #    verbtype = 0
    #else:
    #    verbtype = 2
    temptemp = 1.
    for k in xrange(ntemps):
        _, chi2_all[k], statarrays,  accept_fracs, diff2_list, rtype_array, accepts = models[k].run_sampler(temptemp, multiband=multiband)

    nsample[j] = models[0].n
    xsample[j,:] = models[0].stars[Model._X, :]
    ysample[j,:] = models[0].stars[Model._Y, :]
    diff2_all[j,:] = diff2_list
    accept_all[j,:] = accepts
    rtypes[j,:] = rtype_array
    accept_stats[j,:] = accept_fracs
    nmgy_sample = []
    if multiband:
        for b in xrange(nbands):
            fsample[b][j,:] = models[0].stars[Model._F+b,:]
            if b>0:
                nmpc = [nmgy_per_count[0], nmgy_per_count[b]]
                csample = adus_to_color(models[0].stars[Model._F,:], models[0].stars[Model._F+b,:], nmpc)
                csample = np.array([value for value in csample if not math.isnan(value)])
                colorsample[b-1].append(csample)
    else:
        fsample[j,:] = models[0].stars[Model._F, :]
    chi2sample[j] = chi2_all[0]
    timestats[j,:] = statarrays
    eps_sample[j,:,:] = models[0].eps

if not multiband:
    colorsample = []

print 'saving...'

if datatype=='mock2':
    np.savez(result_path + '/'+mock_test_name+'/' + str(dataname) + '/results/'+str(config_type)+'-'+str(nrealization)+'.npz', n=nsample, x=xsample, y=ysample, f=fsample, chi2=np.sum(chi2sample, axis=1), times=timestats, back=trueback, accept=accept_stats, cprior_vals=cprior_vals)
else:
    np.savez(result_path + '/' + str(timestr) + '/chain.npz', n=nsample, x=xsample, y=ysample, f=fsample, colors=colorsample, eps=eps_sample, chi2=chi2sample, times=timestats, accept=accept_stats, nmgy=nmgy_per_count, back=trueback, pixel_transfer_mats=pixel_transfer_mats, diff2s=diff2_all, rtypes=rtypes, accepts=accept_all, cprior_vals=cprior_vals, comments=comments)

dt_total = time.clock()-start_time
print 'Full Run Time (s):', np.round(dt_total,3)
print 'Time String:', str(timestr)

if datatype == 'real':
    datatype == 'DAOPHOT'

result_diag_command = 'python result_diagnostics.py '+dataname+' '+str(timestr)+' '+datatype+' '
for band in bands:
    result_diag_command += band+' '

# UNCOMMENT TO RUN RESULT DIAGNOSTICS which make posterior plots etc. 
#print result_diag_command
#os.system(result_diag_command)
