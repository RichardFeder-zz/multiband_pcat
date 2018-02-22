import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double
import h5py
# in order for visual=True to work, interactive backend should be loaded before importing pyplot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import signal
import time
import astropy.wcs
import astropy.io.fits
import sys
import os
import warnings
from astropy.io import fits
import random
import math
from image_eval import psf_poly_fit, image_model_eval
from result_diagnostics import results, multiband_sample_frame, zoom_in_frame

def generate_default_astrans(imsz):
    astransx, astransy, mat3, mat4, mat5, mat6 = [[] for x in xrange(6)]
    for x in xrange(imsz[0]):
        astransx.append(np.linspace(0, imsz[0]-1, imsz[0]))
        astransy.append(np.full((imsz[1]), x).transpose())
    mat3 = np.full((imsz[0], imsz[1]), 1)
    mat4 = np.zeros((imsz[0], imsz[1]))
    mat5 = np.zeros((imsz[0], imsz[1]))
    mat6 = np.full((imsz[0], imsz[1]), 1)
    pixel_transfer_mats = np.zeros((6, imsz[0],imsz[1]))
    pixel_transfer_mats = np.array([astransx, astransy, mat3, mat4, mat5, mat6])
    return pixel_transfer_mats

# Run, rerun, CamCol, DAOPHOTID, RA, DEC, xu, yu, u (mag), uErr, chi, sharp, flag, xg, yg, g, gErr, chi, sharp, flag, 
# xr, yr, r, rerr, chi, sharp, flag, xi, yi, i, ierr, chi, sharp, flag, xz, yz, z, zerr, chi, sharp, flag

directory_path = "/Users/richardfeder/Documents/multiband_pcat/pcat-lion-results"
timestr = time.strftime("%Y%m%d-%H%M%S")
c = 0
multiple_regions = 1
# actual_background = [179., 239.]
# trueback = [179., 239.]
trueback = [180., 314.]
# trueback = [179., 310.]
# mean_dx = -0.642 #this is for run 2583, camcol 2
# mean_dy = 2.756
mean_dx = 0
mean_dy = 0
np.seterr(divide='ignore', invalid='ignore')


# script arguments
dataname = sys.argv[1]
visual = int(sys.argv[2]) > 0
# 1 to test, 0 not to test
testpsfn = int(sys.argv[3]) > 0
# 'star' for star only, 'stargalx' for star and galaxy
strgmode = sys.argv[4]
# 'mock' for simulated
datatype = sys.argv[5]
# 1 for multiband, 0 for single band
multiband = int(sys.argv[6]) > 0

total_time = time.clock()


bands, ncs, nbins, psfs, cfs, pixel_transfer_mats, biases, gains, data_array, data_hdrs, weights, nmgy_per_count = [[] for x in xrange(12)]

if multiband:
    band = raw_input("Enter bands one by one in lowercase ('x' if no more): ")
    while band != 'x':
        bands.append(band)
        band = raw_input("Enter bands one by one in lowercase ('x' if no more): ")
    nbands = len(bands)
    print('Loading data for the following bands: ' + str(bands))
else:
    nbands = 1
    bands = ['']


print 'Lion mode:', strgmode
print 'datatype:', datatype

#this is sort of dumb
for b in xrange(nbands):
    if multiband:
        f = open('Data/'+dataname+'/'+dataname+'-psf'+str(bands[b])+'.txt')
        psf = np.loadtxt('Data/'+dataname+'/'+dataname+'-psf'+str(bands[b])+'.txt',skiprows=1).astype(np.float32)
        g = open('Data/'+dataname+'/'+dataname +'-pix'+bands[b]+'.txt')
        data = np.loadtxt('Data/'+dataname+'/'+dataname+'-cts'+ bands[b]+'.txt').astype(np.float32)
    else:
        f = open('Data/'+dataname+'/'+dataname+'-psf.txt')
        psf = np.loadtxt('Data/'+dataname+'/'+dataname+'-psf.txt',skiprows=1).astype(np.float32)
        g = open('Data/'+dataname+'/'+dataname+'-pix.txt')
        data = np.loadtxt('Data/'+dataname+'/'+dataname+'-cts.txt').astype(np.float32)

    nc, nbin = [np.int32(i) for i in f.readline().split()]
    f.close()
    psfs.append(psf)
    ncs.append(nc)
    cf = psf_poly_fit(psf, nbin=nbin)
    cfs.append(cf)
    npar = cf.shape[0]
    w, h, nb = [np.int32(i) for i in g.readline().split()]
    bias, gain = [np.float32(i) for i in g.readline().split()]
    if multiband:
        a = np.float32(g.readline().split())
        nmgy_per_count.append(a[0])
    biases.append(bias)
    gains.append(gain)
    g.close()
    data -= bias
    data_array.append(data)
    variance = data / gain
    weight = 1. / variance
    weights.append(weight)

    if b==0:
        w0=w
        h0=h
        imsz = (w0, h0)
    #load asTran files for other bands
    if b > 0:
        pathname = 'Data/'+dataname+'/asGrid002583-2-0136-100x100-'+bands[0]+'-'+bands[b]+'-before_shift.fits'
        if os.path.isfile(pathname):
            transfer_hdu_gr = fits.open('Data/'+dataname+'/asGrid002583-2-0136-100x100-'+bands[0]+'-'+bands[b]+'-before_shift.fits')
            pixel_transfer_mats.append(np.array([transfer_hdu_gr[0].data, transfer_hdu_gr[1].data,transfer_hdu_gr[2].data, transfer_hdu_gr[3].data, transfer_hdu_gr[4].data, transfer_hdu_gr[5].data]))
        else:
            pixel_transfer_mats.append(generate_default_astrans([imsz[0], imsz[1]]))
        #may eventually want transfer_hdu_ba, but as long as we only make proposals from one band and then transform to other bands we should be fine with this
        assert w==w0 and h==h0

    if visual and testpsfn:
        testpsf(ncs[b], cf, psfs[b], np.float32(np.random.uniform()*4), np.float32(np.random.uniform()*4), lib=libmmult.pcat_model_eval)

def initialize_c():
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
    new_dir_name = directory_path + '/' + str(time_string)
    frame_dir_name = new_dir_name + '/frames'
    os.makedirs(frame_dir_name)
    return frame_dir_name

def gaussian(x, mu, sig):
    return -np.power(x - mu, 2.) / (2 * np.power(sig, 2.))

def get_pint_dp(p):
    pint = np.floor(p+0.5)
    dp = p - pint
    return pint.astype(int), dp

def flux_proposal(f0, nw, trueminf):
    pixel_variance = trueback[0]/gains[0]
    N_eff = 17.5
    err_f = np.sqrt(N_eff * pixel_variance)
    N_src = 1000.
    if multiple_regions:
        lindf = np.float32(err_f/(np.sqrt(N_src*0.04*(2+nbands))))
    else:
        lindf = np.float32(5*err_f/np.sqrt(N_src*(2+nbands)))
    if np.random.uniform() < 0.01:
        lindf *= 100
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

def transform_q(x,y, mats):
    if len(x) != len(y):
        print('Unequal number of x and y coordinates')
        return
    xtrans, ytrans, dxpdx, dypdx, dxpdy, dypdy = mats
    xints, dxs = get_pint_dp(x)
    yints, dys = get_pint_dp(y)
    xnew = xtrans[yints,xints] + dxs*dxpdx[yints,xints] + dys*dxpdy[yints,xints]
    ynew = ytrans[yints,xints] + dxs*dypdx[yints,xints] + dys*dypdy[yints,xints] 
    return np.array(xnew).astype(np.float32), np.array(ynew).astype(np.float32)

def pcat_multiband_eval(x, y, f, bkg, imsz, nc, cf, weights, ref, lib, regsize, margin, offsetx, offsety):
    dmodels, diff2s = [[],[]]
    dt_transf = 0
    for b in xrange(nbands):
        if b>0:
            t4 = time.clock()
            xp, yp = transform_q(x, y, pixel_transfer_mats[b-1])
            dt_transf = time.clock()-t4
            dmodel, diff2 = image_model_eval(xp, yp, f[b], bkg[b], imsz, nc[b], np.array(cf[b]).astype(np.float32()), weights=weights[b], ref=ref[b], lib=libmmult.pcat_model_eval, regsize=regsize, margin=margin, offsetx=offsetx-int(mean_dx), offsety=offsety-int(mean_dy))
        else:    
            xp=x
            yp=y
            dmodel, diff2 = image_model_eval(xp, yp, f[b], bkg[b], imsz, nc[b], np.array(cf[b]).astype(np.float32()), weights=weights[b], ref=ref[b], lib=libmmult.pcat_model_eval, regsize=regsize, margin=margin, offsetx=offsetx, offsety=offsety)
        dmodels.append(dmodel)
        diff2s.append(diff2)
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


def neighbours(x,y,neigh,i,generate=False):
    neighx = np.abs(x - x[i])
    neighy = np.abs(y - y[i])
    adjacency = np.exp(-(neighx*neighx + neighy*neighy)/(2.*neigh*neigh))
    oldadj = adjacency.copy()
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

def adus_to_color(flux0, flux1, nm_2_cts):
    colors = adu_to_magnitude(flux0, nm_2_cts[0]) - adu_to_magnitude(flux1, nm_2_cts[1])
    # colors = -2.5*np.log10((np.array(flux0)*nm_2_cts[0])/(np.array(flux1)*nm_2_cts[1]))
    return colors
def adu_to_magnitude(flux, nm_2_cts):
    mags = 22.5-2.5*np.log10((np.array(flux)*nm_2_cts))
    return mags

def get_region(x, offsetx, regsize):
    return np.floor(x + offsetx).astype(np.int) / regsize

def idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize):
    match_x = (get_region(x[0:n], offsetx, regsize) % 2) == parity_x
    match_y = (get_region(y[0:n], offsety, regsize) % 2) == parity_y
    return np.flatnonzero(np.logical_and(match_x, match_y))

def signal_handler(signal, frame):
    # print('Saving plots now..')
    # if multiband:
    #     results(nsample,fsample, truef, colorsample, nsamp, timestats, tq_times, chi2sample, bkgsample, directory_path, timestr, nbands, bands, multiband, nmgy_per_count)
    # else:
    #     results(nsample,fsample, truef, [], nsamp, timestats, tq_times, chi2sample, bkgsample, directory_path, timestr, nbands, bands, multiband, nmgy_per_count)
    os._exit

class Proposal:
    _X = 0
    _Y = 1
    _F = 2

    def __init__(self):
        self.idx_move = None
        self.do_birth = False
        self.idx_kill = None
        self.factor = None
        self.goodmove = False
        self.do_dback = None
        self.dback = np.zeros(nbands, dtype=np.float32)
        self.xphon = np.array([], dtype=np.float32)
        self.yphon = np.array([], dtype=np.float32)
        self.fphon = []
        for x in xrange(nbands):
            self.fphon.append(np.array([], dtype=np.float32))

    def set_factor(self, factor):
        self.factor = factor

    def assert_types(self):
        assert self.xphon.dtype == np.float32
        assert self.yphon.dtype == np.float32
        assert self.fphon[0].dtype == np.float32

    def add_background_shift(self, back, dback, which_band, stars):
        if back[which_band] + dback[which_band] > 0:
            self.do_dback = True
            self.goodmove = True
            self.dback[which_band] = dback[which_band]
            self.stars0 = stars

    def __add_phonions_stars(self, stars, remove=False):
        fluxmult = -1 if remove else 1
        self.xphon = np.append(self.xphon, stars[self._X,:])
        self.yphon = np.append(self.yphon, stars[self._Y,:])
        for b in xrange(nbands):
            self.fphon[b] = np.append(self.fphon[b], np.array(fluxmult*stars[self._F+b,:], dtype=np.float32))
        self.assert_types()

    def add_move_stars(self, idx_move, stars0, starsp):
        self.idx_move = idx_move
        self.stars0 = stars0
        self.starsp = starsp
        self.goodmove = True
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
        #not sure if this is the right thing to do for dback 
        elif self.do_dback is not None:
            return self.stars0[self._X,:], self.stars0[self._Y,:]

class Model:
    # should these be class or instance variables?
    # nstar = int((imsz[0]**2)/5)
    nstar = 2000
    trueminf = np.float32(236) 
    truealpha = np.float32(2)
    penalty = 1+0.5*(nbands) #multiband 
    kickrange = 1.
    r_i_sig = np.sqrt(0.75)
    r_i_mu = 0.75

    _X = 0
    _Y = 1
    _F = 2

    def __init__(self):
        self.back = trueback
        self.n = np.random.randint(self.nstar)+1
        self.stars = np.zeros((2+nbands,self.nstar), dtype=np.float32)
        self.stars[:,0:self.n] = np.random.uniform(size=(2+nbands,self.n))  # refactor into some sort of prior function?
        self.stars[self._X,0:self.n] *= imsz[0]-1
        self.stars[self._Y,0:self.n] *= imsz[1]-1
        for b in xrange(nbands):
            if b==0:
                self.stars[self._F+b,0:self.n] **= -1./(self.truealpha - 1.)
                self.stars[self._F+b,0:self.n] *= self.trueminf
            else:
                new_colors = np.random.normal(loc=self.r_i_mu, scale=self.r_i_sig, size=self.n)
                self.stars[self._F+b,0:self.n] = self.stars[self._F,0:self.n]*10**(0.4*new_colors)*nmgy_per_count[0]/nmgy_per_count[1]
    # should offsetx/y, parity_x/y be instance variables?

    def run_sampler(self, temperature, nloop=1000, visual=False, multiband=False, savefig=False):
        t0 = time.clock()
        nmov = np.zeros(nloop)
        movetype = np.zeros(nloop)
        accept = np.zeros(nloop)
        outbounds = np.zeros(nloop)
        dt1 = np.zeros(nloop)
        dt2 = np.zeros(nloop)
        dt3 = np.zeros(nloop)
        dttq = np.zeros(nloop)
        # single region
        if multiple_regions:
            self.offsetx = np.random.randint(regsize)
            self.offsety = np.random.randint(regsize)
        else:
            self.offsetx = 0
            self.offsety = 0
 
        self.nregx = imsz[0] / regsize + 1
        self.nregy = imsz[1] / regsize + 1

        resids = []
        for b in xrange(nbands):
            resid = data_array[b].copy() # residual for zero image is data
            resids.append(resid)
        if strgmode == 'star':
            evalx = self.stars[self._X,0:self.n]
            evaly = self.stars[self._Y,0:self.n]
            evalf = self.stars[self._F:,0:self.n]
        n_phon = evalx.size
        models, diff2s, dt_transf = pcat_multiband_eval(evalx, evaly, evalf, self.back, imsz, ncs, cfs, weights=weights, ref=resids, lib=libmmult.pcat_model_eval, \
            regsize=regsize, margin=margin, offsetx=self.offsetx, offsety=self.offsety)
 

        if multiband:
            diff2_total = np.sum(np.array(diff2s), axis=0)
            # diff2_total = diff2s[0]
        else:
            diff2_total = diff2s[0]
        logL = -0.5*diff2_total 
        for b in xrange(nbands):
            resids[b] -= models[b]
        # proposal types
        moveweights = np.array([80., 40., 40.,0.]) # 80, 40, 40

        moveweights /= np.sum(moveweights)

        for i in xrange(nloop):
            t1 = time.clock()
            rtype = np.random.choice(moveweights.size, p=moveweights)
            movetype[i] = rtype
            # defaults
            dback = [np.float32(0.), np.float32(0.)]
            pn = self.n
            # should regions be perturbed randomly or systematically?
            if multiple_regions:
                self.parity_x = np.random.randint(2)
                self.parity_y = np.random.randint(2)
            else:
                self.parity_x = 0
                self.parity_y = 0
            movetypes = ['P *', 'BD *', 'MS *', 'BGD *']
            #proposal types
            movefns = [self.move_stars, self.birth_death_stars, self.merge_split_stars, self.background_shift]
            proposal = movefns[rtype]()
            dt1[i] = time.clock() - t1
            if proposal.goodmove:
                t2 = time.clock()
                dmodels, diff2s, dt_transf = pcat_multiband_eval(proposal.xphon, proposal.yphon, proposal.fphon, proposal.dback, imsz, ncs, cfs, weights=weights, ref=resids, lib=libmmult.pcat_model_eval, regsize=regsize, margin=margin, offsetx=self.offsetx, offsety=self.offsety)
                dttq[i] = dt_transf
                # temporary fix
                if multiband:
                    diff2_total = np.sum(np.array(diff2s), axis=0)
                    # diff2_total = diff2s[0]
                else:
                    diff2_total = diff2s[0]
                plogL = -0.5*diff2_total
                plogL[(1-self.parity_y)::2,:] = float('-inf') # don't accept off-parity regions
                plogL[:,(1-self.parity_x)::2] = float('-inf')
                dlogP = (plogL - logL) / temperature
                
                dt2[i] = time.clock() - t2
                t3 = time.clock()
                refx, refy = proposal.get_ref_xy()
                regionx = get_region(refx, self.offsetx, regsize)
                regiony = get_region(refy, self.offsety, regsize)
                
                if proposal.factor is not None:
                    dlogP[regiony, regionx] += proposal.factor
                    acceptreg = (np.log(np.random.uniform(size=(self.nregy, self.nregx))) < dlogP).astype(np.int32)
                    
                    acceptprop = acceptreg[regiony, regionx]
                    numaccept = np.count_nonzero(acceptprop)
                
                #iterate over all bands, is this the right thing to be doing with these? 
                for b in xrange(nbands):
                    dmodel_acpt = np.zeros_like(dmodels[b])
                    diff2s[b].fill(0)
                    offx = self.offsetx
                    offy = self.offsety
                    if b>0:
                        offx -= int(mean_dx)
                        offy -= int(mean_dy)
          
                    libmmult.pcat_imag_acpt(imsz[0], imsz[1], dmodels[b], dmodel_acpt, acceptreg, regsize, margin, offx, offy)
                    # using this dmodel containing only accepted moves, update logL
                    libmmult.pcat_like_eval(imsz[0], imsz[1], dmodel_acpt, resids[b], weights[b], diff2s[b], regsize, margin, offx, offy)   
                    resids[b] -= dmodel_acpt
                    models[b] += dmodel_acpt

                if multiband:
                    diff2_total1 = np.sum(np.array(diff2s), axis=0)
                    # diff2_total1 = np.array(diff2s[0])+np.array(diff2s[1])
                    # diff2_total1 = diff2s[0]
                else:
                    diff2_total1 = np.array(diff2s[0])
                logL = -0.5*diff2_total1
                #implement accepted moves
                if proposal.idx_move is not None:
                    starsp = proposal.starsp.compress(acceptprop, axis=1)
                    idx_move_a = proposal.idx_move.compress(acceptprop)
                    self.stars[:, idx_move_a] = starsp
                if proposal.do_birth:
                    starsb = proposal.starsb.compress(acceptprop, axis=1)
                    starsb = starsb.reshape((2+nbands,-1))
                    num_born = starsb.shape[1]
                    self.stars[:, self.n:self.n+num_born] = starsb
                    self.n += num_born
                if proposal.idx_kill is not None:
                    idx_kill_a = proposal.idx_kill.compress(acceptprop, axis=0).flatten()
                    num_kill = idx_kill_a.size
                    # nstar is correct, not n, because x,y,f are full nstar arrays
                    self.stars[:, 0:self.nstar-num_kill] = np.delete(self.stars, idx_kill_a, axis=1)
                    self.stars[:, self.nstar-num_kill:] = 0
                    self.n -= num_kill
                if proposal.do_dback is not None and acceptprop[0]:
                    self.back += proposal.dback
                dt3[i] = time.clock() - t3

                if acceptprop.size > 0: 
                    accept[i] = np.count_nonzero(acceptprop) / float(acceptprop.size)
                else:
                    accept[i] = 0
            else:
                outbounds[i] = 1
        
        chi2 = []
        for b in xrange(nbands):
            #don't evaluate chi2 on outer periphery
            xmin, xmax = 3, imsz[0]-3
            ymin, ymax = 3, imsz[1]-3
            chi2.append(np.sum(weights[b][xmin:xmax,ymin:ymax]*(data_array[b][xmin:xmax,ymin:ymax]-models[b][xmin:xmax,ymin:ymax])*(data_array[b][xmin:xmax,ymin:ymax]-models[b][xmin:xmax,ymin:ymax])))        
        
        fmtstr = '\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f (Pg) %0.3f (BDg) %0.3f (S-g) %0.3f (gSg) %0.3f (gMS) %0.3f'
        print 'Temperature', temperature, 'background', self.back, 'N_star', self.n, 'N_phon', n_phon, 'chi^2', chi2
        dt1 *= 1000
        dt2 *= 1000
        dt3 *= 1000
        
        othertimes = []
        othertimes.append(np.sum(dttq))
        # transform_q_times = np.sum(dttq)
        timestat_array = np.zeros((5, 1+len(moveweights)), dtype=np.float32)
        statlabels = ['Acceptance', 'Out of Bounds', 'Proposal (s)', 'Likelihood (s)', 'Implement (s)']
        statarrays = [accept, outbounds, dt1, dt2, dt3]
        for j in xrange(len(statlabels)):
            timestat_array[j][0] = np.sum(statarrays[j])/1000
            print statlabels[j]+'\t(all) %0.3f' % (np.sum(statarrays[j])/1000),
            for k in xrange(len(movetypes)):
                timestat_array[j][1+k] = np.mean(statarrays[j][movetype==k])
                print '('+movetypes[k]+') %0.3f' % (np.mean(statarrays[j][movetype == k])),
            print
            if j == 1:
                print '-'*16
        print '-'*16
        print 'Total (s): %0.3f' % (np.sum(statarrays[2:])/1000)
        print '='*16

        tplot = time.clock()
        if visual or savefig:
            multiband_sample_frame(data_array, self.stars[self._X,0:self.n], self.stars[self._Y,0:self.n], self.stars[self._F:,0:self.n], truex, truey, truef, truecolor, resids, weights, \
                bands, nmgy_per_count, nstar, frame_dir, c, pixel_transfer_mats, visual, savefig)
        if savefig:
            bounds = [0, 20, 80, 100]
            # if len(data_array[0])==100:
                # if datatype=='mock':
                #     zoom_in_frame(data_array, self.stars[self._X,0:self.n], self.stars[self._Y,0:self.n], self.stars[self._F:,0:self.n], truex, truey, truef, bounds, frame_dir, c)
                # else:
            zoom_in_frame(data_array, self.stars[self._X,0:self.n], self.stars[self._Y,0:self.n], self.stars[self._F:,0:self.n], hx, hy, hf, bounds, frame_dir, c, nmgy_per_count, pixel_transfer_mats)
        dtplot = time.clock()-tplot
        othertimes.append(dtplot)
        return self.n, chi2, timestat_array, othertimes

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

    def background_shift(self):
        # print('here')
        which_band = np.random.randint(nbands)
        dback = np.zeros(nbands)
        dback[which_band] = np.random.normal(scale=0.01)
        # print(dback)
        factor = gaussian(trueback[which_band]+dback[which_band], trueback[which_band], 2)
        # print(factor)
        proposal = Proposal()
        proposal.add_background_shift(self.back, dback, which_band, self.stars)
        proposal.set_factor(factor)
        return proposal


    def move_stars(self): 
        idx_move = self.idx_parity_stars()
        nw = idx_move.size
        stars0 = self.stars.take(idx_move, axis=1)
        starsp = np.empty_like(stars0)
        f0 = stars0[self._F:,:]
        pfs = []
        color_factors = np.zeros((nbands-1, nw)).astype(np.float32)
        
        pf = flux_proposal(f0[0], nw, self.trueminf)
        pfs.append(pf)

        pixel_variance = trueback[0]/gains[0]
        N_eff = 17.5
        err_f = np.sqrt(N_eff * pixel_variance)
        N_src = 1000.

        for b in xrange(nbands-1):
            colors = adus_to_color(f0[0], f0[b+1], nmgy_per_count)
            # print(colors)
            color_factors[b] -= (gaussian(colors, self.r_i_mu, self.r_i_sig))

            color_step = err_f * 2.5 / (np.log(10) * f0[0]) * np.sqrt(1+0*10**(-0.8*colors)) / np.sqrt((2+nbands)*N_src*0.04)

            colors += np.random.normal(0, color_step, len(colors))
            # print('new colors: ' + str(colors))
            color_factors[b] += (gaussian(colors, self.r_i_mu, self.r_i_sig)) #gaussian prior on r-i color
            pf_1 = pf*10**(0.4*colors)*nmgy_per_count[0]/nmgy_per_count[1]
            # pf_1 = f1_to_f2(pf, nmgy_per_count, colors)
            pfs.append(pf_1)
        dlogf = np.log(pfs[0]/f0[0])
        factor = -self.truealpha*dlogf
        #temporary
        factor = np.array(factor) + color_factors[0]
        if multiple_regions:
            dpos_rms = np.float32(np.sqrt(N_eff/(2*np.pi))*err_f/(np.sqrt(N_src*0.04*(2+nbands))))/(np.maximum(f0[0], pfs[0])) 
        else:
            dpos_rms = np.float32(np.sqrt(N_eff/(2*np.pi))*err_f/np.sqrt(N_src*(2+nbands)))/(np.maximum(f0[0], pfs[0]))
        dx = np.random.normal(size=nw).astype(np.float32)*dpos_rms
        dy = np.random.normal(size=nw).astype(np.float32)*dpos_rms
        starsp[self._X,:] = stars0[self._X,:] + dx
        starsp[self._Y,:] = stars0[self._Y,:] + dy
        for b in xrange(nbands):
            starsp[self._F+b,:] = pfs[b]
        self.bounce_off_edges(starsp)

        proposal = Proposal()
        proposal.add_move_stars(idx_move, stars0, starsp)
        # print "factor", factor
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
                    new_colors = np.random.normal(loc=self.r_i_mu, scale=self.r_i_sig, size=nbd)
                    starsb[self._F+b,:] = starsb[self._F,:]*10**(0.4*new_colors)*nmgy_per_count[0]/nmgy_per_count[1]
            # some sources might be generated outside image
            inbounds = self.in_bounds(starsb)
            starsb = starsb.compress(inbounds, axis=1)
            factor = np.full(starsb.shape[1], -self.penalty)
            proposal.add_birth_stars(starsb)
            proposal.set_factor(factor)
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
        return proposal


    def merge_split_stars(self):
        splitsville = np.random.randint(2)
        idx_reg = self.idx_parity_stars()
        sum_f = 0
        low_n = 0
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
 
            fracs.append((1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32))
            for b in xrange(nbands-1):
                fracs.append(np.random.uniform(size=nms).astype(np.float32))

            starsp = np.empty_like(stars0)
            starsb = np.empty_like(stars0)

            starsp[self._X,:] = stars0[self._X,:] + ((1-fracs[0])*dx)
            starsp[self._Y,:] = stars0[self._Y,:] + ((1-fracs[0])*dy)
            starsb[self._X,:] = stars0[self._X,:] - fracs[0]*dx
            starsb[self._Y,:] = stars0[self._Y,:] - fracs[0]*dy


            for b in xrange(nbands):
                starsp[self._F+b,:] = stars0[self._F+b,:]*fracs[b]
                starsb[self._F+b,:] = stars0[self._F+b,:] * (1-fracs[b])
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

            invpairs = np.empty(nms)
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

            starsp = np.empty_like(stars0)
            starsp[self._X,:] = fracs[0]*stars0[self._X,:] + (1-fracs[0])*starsk[self._X,:]
            starsp[self._Y,:] = fracs[0]*stars0[self._Y,:] + (1-fracs[0])*starsk[self._Y,:]
            for b in xrange(nbands):
                starsp[self._F+b,:] = f0[b] + fk[b]
            if goodmove:
                proposal.add_move_stars(idx_move, stars0, starsp)
                proposal.add_death_stars(idx_kill, starsk)
            # turn bright_n into an array
            bright_n = bright_n - (f0[0] > 2*self.trueminf) - (fk[0] > 2*self.trueminf) + (starsp[self._F,:] > 2*self.trueminf)
        if goodmove:
            factor = np.log(self.truealpha-1) + (self.truealpha-1)*np.log(self.trueminf) - self.truealpha*np.log(fracs[0]*(1-fracs[0])*sum_fs[0]) + \
                np.log(2*np.pi*self.kickrange*self.kickrange) - np.log(imsz[0]*imsz[1]) + np.log(1. - 2./fminratio) + np.log(bright_n) + \
                np.log(invpairs) + np.log(sum_fs[0]) # last term is Jacobian
            if multiband:
                stars0_color = adus_to_color(stars0[self._F,:], stars0[self._F+1,:], nmgy_per_count)
                starsp_color = adus_to_color(starsp[self._F,:], starsp[self._F+1,:], nmgy_per_count)
                factor += np.log(2.5/np.log(10)) - np.log(fracs[1]*(1-fracs[1])) 

                if splitsville:
                    starsb_color = adus_to_color(starsb[self._F,:], starsb[self._F+1,:], nmgy_per_count)
                    factor += (stars0_color - self.r_i_mu)**2/(2*self.r_i_sig**2) - (starsp_color - self.r_i_mu)**2/(2*self.r_i_sig**2) - (starsb_color - self.r_i_mu)**2/(2*self.r_i_sig**2)
                else:
                    starsk_color = adus_to_color(starsk[self._F,:], starsk[self._F+1,:], nmgy_per_count)
                    factor += (starsp_color - self.r_i_mu)**2/(2*self.r_i_sig**2) - (stars0_color - self.r_i_mu)**2/(2*self.r_i_sig**2) - (starsk_color - self.r_i_mu)**2/(2*self.r_i_sig**2)
            if not splitsville:
                factor *= -1
                factor += self.penalty
            else:
                factor -= self.penalty
            proposal.set_factor(factor)
        return proposal

libmmult = npct.load_library('pcat-lion', '.')
initialize_c()

if datatype == 'mock':
    if strgmode == 'star':
        truth = np.loadtxt('Data/'+dataname+'/'+dataname+'-tru.txt')
        truex = truth[:,0]
        truey = truth[:,1]
        truef = truth[:,2:]
        if multiband:
            truecolor = adus_to_color(truef[:,0], truef[:,1], nmgy_per_count)           
    labldata = 'DAOPHOT'
    hx, hy, hf = [], [], []
else:
    labldata = 'HST 606W'
true_h = np.loadtxt('Data/'+dataname+'/HTcat-'+dataname+'.txt')
hx = true_h[:,0]
hy = true_h[:,1]
hf = true_h[:,2:]

# print "n_src: ",len([x for x in truef[:,0] if x>236])
# print "n_src: ",len([x for x in truef[:,1] if x>236])

ntemps = 1
temps = np.sqrt(2) ** np.arange(ntemps)
if multiple_regions:
    regsize = imsz[0]/2
else:
    regsize = imsz[0]# single region
assert imsz[0] % regsize == 0
assert imsz[1] % regsize == 0
margin = 10
nsamp = 100
nloop = 1000
nstar = Model.nstar
nsample = np.zeros(nsamp, dtype=np.int32)
xsample = np.zeros((nsamp, nstar), dtype=np.float32)
ysample = np.zeros((nsamp, nstar), dtype=np.float32)
timestats = np.zeros((nsamp, 5, 5), dtype=np.float32)
tq_times = np.zeros(nsamp, dtype=np.float32)
plt_times = np.zeros(nsamp, dtype=np.float32)
bkgsample = np.zeros((nsamp, nbands), dtype=np.float32)
if multiband:
    fsample = [np.zeros((nsamp, nstar), dtype=np.float32) for x in xrange(nbands)]
    colorsample = []
else:
    fsample = np.zeros((nsamp, nstar), dtype=np.float32)
chi2sample = np.zeros((nsamp, nbands), dtype=np.int32)
models = [Model() for k in xrange(ntemps)]
#create directory for results

frame_dir = create_directories(timestr)
# signal.signal(signal.SIGINT, signal_handler)

plt.ion()
plt.figure(figsize=(15,10))
# sampling loop
for j in xrange(nsamp):
    # signal.signal(signal.SIGINT, signal_handler)
    chi2_all = np.zeros((ntemps,nbands))
    print 'Loop', j
    sf = 0
    if nsamp<15:
        sf = 1
        c+= 1
    elif j%(int(nsamp/15))==0 and nsamp > 14:
        sf = 1
        c+=1
     
    temptemp = 1.#max(50 - 0.1*j, 1)
    for k in xrange(ntemps):
        _, chi2_all[k], statarrays, othertimes = models[k].run_sampler(temptemp, visual=(k==0)*visual, multiband=multiband, savefig=sf)

    tq_times[j] = othertimes[0]
    plt_times[j] = othertimes[1]
    nsample[j] = models[0].n
    xsample[j,:] = models[0].stars[Model._X, :]
    ysample[j,:] = models[0].stars[Model._Y, :]
    bkgsample[j] = models[0].back
    nmgy_sample = []
    if multiband:
        for b in xrange(nbands):
            fsample[b][j,:] = models[0].stars[Model._F+b,:]
        csample = adus_to_color(models[0].stars[Model._F,:], models[0].stars[Model._F+1,:], nmgy_per_count)
        csample = np.array([value for value in csample if not math.isnan(value)])
        colorsample.append(csample)
    else:
        fsample[j,:] = models[0].stars[Model._F, :]
    chi2sample[j] = chi2_all[0]
    timestats[j,:] = statarrays
if not multiband:
    colorsample = []
results(nsample,fsample, truef, colorsample, nsamp, timestats, tq_times, plt_times, chi2sample, bkgsample, directory_path, timestr, nbands, bands, multiband, nmgy_per_count)
dt_total = time.clock()-total_time
print 'Full Run Time (s):', np.round(dt_total,3)
print 'saving...'
np.savez(directory_path + '/' + str(timestr) + '/chain.npz', n=nsample, x=xsample, y=ysample, f=fsample, chi2=np.sum(chi2sample, axis=1), times=timestats)
