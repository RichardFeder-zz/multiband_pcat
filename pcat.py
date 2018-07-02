import numpy as np
import sys
import os
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
import warnings
from astropy.io import fits
import random
import math
from image_eval import psf_poly_fit, image_model_eval
from result_diagnostics import results, multiband_sample_frame
from helpers import *

# Run, rerun, CamCol, DAOPHOTID, RA, DEC, xu, yu, u (mag), uErr, chi, sharp, flag, xg, yg, g, gErr, chi, sharp, flag, 
# xr, yr, r, rerr, chi, sharp, flag, xi, yi, i, ierr, chi, sharp, flag, xz, yz, z, zerr, chi, sharp, flag


timestr = time.strftime("%Y%m%d-%H%M%S")
c = 0
dpi_val = 300
multiple_regions = 1
include_hubble = 0

#generate random seed for initialization
np.random.seed(20170501)

# trueback = [180., 314., 103., 140.] #r, i, g, z
#trueback = [180., 315., 140.]
trueback = [314.]

np.seterr(divide='ignore', invalid='ignore')

# script arguments
dataname = str(sys.argv[1])
visual = int(sys.argv[2]) > 0
# 1 to test, 0 not to test
testpsfn = int(sys.argv[3]) > 0
# 'mock' for simulated, 'mock2' for two source blended mock
datatype = str(sys.argv[4])
# 1 for multiband, 0 for single band
multiband = int(sys.argv[5]) > 0

if datatype=='mock2':
    config_type = str(sys.argv[6])
    nrealization = int(sys.argv[7])
if datatype=='mock' or datatype=='mock2':
    trueback = []

mock_test_name = 'mock_2star_18'

bands, ncs, nbins, psfs, cfs, pixel_transfer_mats, biases, gains, \
    data_array, data_hdrs, weights, nmgy_per_count, mean_dpos = [[] for x in xrange(13)]


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
    result_path = base_path + '/pcat-lion-results'
print 'Results will go in', result_path

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
    print('Loading data for the following bands: ' + str(bands))
else:
    nbands = 1
    bands = ['']

print 'datatype:', datatype

start_time = time.clock()

#could simplify
for b in xrange(nbands):

    if datatype=='mock2':
        base_path = 'Data/'+mock_test_name
        paths = [base_path+'/psfs/'+mock_test_name+'-psf'+str(bands[b])+'.txt', \
                 base_path+'/pixs/'+mock_test_name+'-pix'+str(bands[b])+'.txt']
        paths.append(base_path+'/'+dataname+'/'+dataname+'-nr'+str(nrealization)+'-cts'+ bands[b]+'.txt')
    else:
        paths = ['Data/'+dataname+'/psfs/'+dataname+'-psf.txt', 'Data/'+dataname+'/pixs/'+dataname+'-pix.txt', \
                    'Data/'+dataname+'/cts/'+dataname+'-cts.txt']
        if multiband:
            for p in xrange(len(paths)):
                paths[p] = paths[p][:-4]+str(bands[b])+paths[p][-4:]

    psf, nc, cf = get_psf_and_vals(paths[0])
    ncs.append(nc)
    cfs.append(cf)
    psfs.append(psf)

    g = open(paths[1])
    w, h, nb = [np.int32(i) for i in g.readline().split()]
    imdim = (w,h)
    bias, gain = [np.float32(i) for i in g.readline().split()]
    if multiband:
        a = np.float32(g.readline().split())
        print a, len(a)
        nmgy_per_count.append(a[0])
        if datatype=='mock' or datatype=='mock2':
            trueback.append(a[1])
    biases.append(bias)
    gains.append(gain)
    g.close()

    data = np.loadtxt(paths[2]).astype(np.float32)
    data -= bias
    data_array.append(data)
    variance = data / gain
    weight = 1. / variance
    weights.append(weight)

    # establish image dimensions for first band, then other bands will use these
    if b==0:
        w0=w
        h0=h
        imsz = (w0, h0)
    #load asTran files for other bands
    if b > 0:
        #data-specific
        pathname = 'Data/'+dataname+'/asGrid/asGrid002583-2-0136-100x100-'+bands[0]+'-'+bands[b]+'-0310-0630_cterms0_0.fits'
        if os.path.isfile(pathname):
            pixel_transfer_mats.append(read_astrans_mats(pathname))
            dx, dy = find_mean_offset(pathname, dim=imsz[0])
            dpos = [int(dx), int(dy)]
            mean_dpos.append(dpos)
        else:
            pixel_transfer_mats.append(generate_default_astrans([imsz[0], imsz[1]]))
        assert w==w0 and h==h0

    if visual and testpsfn:
        testpsf(ncs[b], cf, psfs[b], np.float32(np.random.uniform()*4), np.float32(np.random.uniform()*4), lib=libmmult.pcat_model_eval)

if multiple_regions:
    regsize = imsz[0]/2
else:
    regsize = imsz[0]# single region
assert imsz[0] % regsize == 0
assert imsz[1] % regsize == 0
margin = 10


#mock test
if datatype =='mock2':
    nsamp = 100
else:
    nsamp = 500
nloop = 1000

def initialize_c():
    # if os.path.getmtime('pcat-lion-test.c') > os.path.getmtime('pcat-lion-test.so'):
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
    return frame_dir_name

def flux_proposal(f0, nw, trueminf, b):
    pixel_variance = trueback[b]/gains[b]
    N_eff = 17.5
    err_f = np.sqrt(N_eff * pixel_variance)
    #data specific
    N_src = 1400.
    if multiple_regions:
        lindf = np.float32(5*err_f/(np.sqrt(N_src*0.04*(2+nbands))))
        #print 'lindf', np.mean(lindf)
    else:
        lindf = np.float32(err_f/np.sqrt(N_src*(2+nbands)))
    logdf = np.float32(0.01/np.sqrt(N_src))
    ff = np.log(logdf*logdf*f0 + logdf*np.sqrt(lindf*lindf + logdf*logdf*f0*f0)) / logdf
    ffmin = np.log(logdf*logdf*trueminf + logdf*np.sqrt(lindf*lindf + logdf*logdf*trueminf*trueminf)) / logdf
    dff = np.random.normal(size=nw)
    aboveffmin = ff - ffmin
    oob_flux = (-dff > aboveffmin)
    dff[oob_flux] = -2*aboveffmin[oob_flux] - dff[oob_flux]
    pff = ff + dff
    pf = np.exp(-logdf*pff) * (-lindf*lindf*logdf*logdf+np.exp(2*logdf*pff)) / (2*logdf*logdf)
   # print 'std difference is', np.std(f0-pf), np.mean(np.abs(f0-pf))
    return pf

def pcat_multiband_eval(x, y, f, bkg, imsz, nc, cfs, weights, ref, lib, regsize, margin, offsetx, offsety):
    dmodels, diff2s = [[],[]]
    dt_transf = 0
    for b in xrange(nbands):
        if b>0:
            t4 = time.clock()
            xp, yp = transform_q(x, y, pixel_transfer_mats[b-1])
            #correcting for different band trimmings, hubble offset
            xp -= mean_dpos[b-1, 0]
            yp -= mean_dpos[b-1, 1]
            #test
            # xp += 0.5
            # yp += 0.5
            dt_transf += time.clock()-t4
            dmodel, diff2 = image_model_eval(xp, yp, f[b], bkg[b], imsz, nc[b], np.array(cfs[b]).astype(np.float32()), \
             weights=weights[b], ref=ref[b], lib=libmmult.pcat_model_eval, regsize=regsize, margin=margin, offsetx=offsetx, offsety=offsety)
        else:    
            xp=x
            yp=y
            dmodel, diff2 = image_model_eval(xp, yp, f[b], bkg[b], imsz, nc[b], np.array(cfs[b]).astype(np.float32()), \
                weights=weights[b], ref=ref[b], lib=libmmult.pcat_model_eval, regsize=regsize, margin=margin, offsetx=offsetx, offsety=offsety)
            # dmodel, diff2 = image_model_eval(xp, yp, f[b], bkg[b], imsz, nc[b], np.array(cfs[b]).astype(np.float32()), \
            #     weights=weights[b], ref=ref[b], lib=None, regsize=regsize, margin=margin, offsetx=offsetx, offsety=offsety)
        dmodels.append(dmodel)
        diff2s.append(diff2)
    return dmodels, diff2s, dt_transf

# ix, iy = 0. to 3.999
def testpsf(nc, cf, psf, ix, iy, lib=None):
    psf0 = image_model_eval(np.array([12.-ix/5.], dtype=np.float32), np.array([12.-iy/5.], \
        dtype=np.float32), np.array([1.], dtype=np.float32), 0., (25,25), nc, cf, lib=lib)
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

    def in_bounds(self, catalogue):
        return np.logical_and(np.logical_and(catalogue[self._X,:] > 0, catalogue[self._X,:] < (imsz[0] -1)), \
                np.logical_and(catalogue[self._Y,:] > 0, catalogue[self._Y,:] < imsz[1] - 1))

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
        inbounds = self.in_bounds(starsp)
        starsp = starsp.compress(inbounds, axis=1)
        # mask = np.logical_and(np.logical_and(starsp[self._X,:] > 0, starsp[self._X,:]<99.5), np.logical_and(starsp[self._Y,:] > 0, starsp[self._Y,:]<99.5))
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
    #mock test
    nstar = 2000
    if datatype=='mock2':
        nstar = 20
    trueminf = np.float32(236) 
    truealpha = np.float32(2)
    penalty = 1+0.5*(nbands) #multiband 
    kickrange = 1.
    # color_sigs = [np.sqrt(0.75)] # r-i, 
    color_sigs = [0.5, 0.5] # r-i, r-z
    # color_mus = [0.75]
    color_mus = [0.25, 0.1] # r-i, r-z

    _X = 0
    _Y = 1
    _F = 2

    def __init__(self):
        self.back = np.zeros(nbands, dtype=np.float32)
        self.regsize = regsize
        self.n = np.random.randint(self.nstar)+1
        self.stars = np.zeros((2+nbands,self.nstar), dtype=np.float32)
        self.stars[:,0:self.n] = np.random.uniform(size=(2+nbands,self.n))  # refactor into some sort of prior function?
        self.stars[self._X,0:self.n] *= imsz[0]-1
        self.stars[self._Y,0:self.n] *= imsz[1]-1
        for b in xrange(nbands):
            self.back[b] = trueback[b]
            if b==0:
                self.stars[self._F+b,0:self.n] **= -1./(self.truealpha - 1.)
                self.stars[self._F+b,0:self.n] *= self.trueminf
            else:
                new_colors = np.random.normal(loc=self.color_mus[b-1], scale=self.color_sigs[b-1], size=self.n)
                self.stars[self._F+b,0:self.n] = self.stars[self._F,0:self.n]*10**(0.4*new_colors)*nmgy_per_count[0]/nmgy_per_count[b]
    def run_sampler(self, nloop=1000, visual=False, multiband=False, savefig=False):
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
            self.offsetx = np.random.randint(self.regsize)
            self.offsety = np.random.randint(self.regsize)
        else:
            self.offsetx = 0
            self.offsety = 0
 
        self.nregx = imsz[0] / self.regsize + 1
        self.nregy = imsz[1] / self.regsize + 1

        #used when background shift is proposed in multiple regions
        # self.region_params = [0, 0, 0, 0, 0] #offsetx, offsety, regsize, parity

        resids = []
        for b in xrange(nbands):
            resid = data_array[b].copy() # residual for zero image is data
            resids.append(resid)

        evalx = self.stars[self._X,0:self.n]
        evaly = self.stars[self._Y,0:self.n]
        evalf = self.stars[self._F:,0:self.n]
        n_phon = evalx.size
        models, diff2s, dt_transf = pcat_multiband_eval(evalx, evaly, evalf, self.back, imsz, ncs, cfs, weights=weights, ref=resids, lib=libmmult.pcat_model_eval, \
            regsize=self.regsize, margin=margin, offsetx=self.offsetx, offsety=self.offsety)
        # models, diff2s, dt_transf = pcat_multiband_eval(evalx, evaly, evalf, self.back, imsz, ncs, cfs, weights=weights, ref=resids, lib=None, \
        #     regsize=self.regsize, margin=margin, offsetx=self.offsetx, offsety=self.offsety)
 
        if multiband:
            diff2_total = np.sum(np.array(diff2s), axis=0)
        else:
            diff2_total = diff2s[0]
        logL = -0.5*diff2_total 
        for b in xrange(nbands):
            resids[b] -= models[b]
        # proposal types
        moveweights = np.array([70., 40., 40.])
        # moveweights = np.array([80., 40., 40., 0.])
        moveweights /= np.sum(moveweights)

        n_back_prop = 0
        n_back_acpt = 0

        for i in xrange(nloop):
            t1 = time.clock()
            rtype = np.random.choice(moveweights.size, p=moveweights)
            movetype[i] = rtype
            # defaults
            dback = [np.float32(0.) for x in xrange(nbands)]
            pn = self.n
            # should regions be perturbed randomly or systematically?
            if multiple_regions and rtype!=3: #don't do multiple regions if there is a background shift
                self.parity_x = np.random.randint(2)
                self.parity_y = np.random.randint(2)
            else:
                self.parity_x = 0
                self.parity_y = 0

            # #background peturbing, doesn't seem to work right now
            # if rtype==3:
            #     n_back_prop += 1
            #     #save previous parameters and change them for background evaluation
            #     self.region_params = [self.offsetx, self.offsety, self.regsize, self.nregx, self.nregy] #save previous values for next step
            #     # print "self.region_params", self.region_params
            #     self.offsetx = 0
            #     self.offsety = 0
            #     self.regsize = imsz[0]
            #     self.nregx = 1
            #     self.nregy = 1
            #     #recalculate log likelihood over full region
            #     models, diff2s, dt_transf = pcat_multiband_eval(evalx, evaly, evalf, self.back, imsz, ncs, cfs, weights=weights, ref=resids, lib=libmmult.pcat_model_eval, \
            #         regsize=imsz[0], margin=margin, offsetx=0, offsety=0)
            #     # logL = self.calculate_logL(diff2s)
            #     if multiband:
            #         diff2_total = np.sum(np.array(diff2s), axis=0)
            #         # diff2_total = diff2s[2]
            #     else:
            #         diff2_total = diff2s[0]
            #     logL = -0.5*diff2_total 
            #     # print "logL at 507", logL
            movetypes = ['P *', 'BD *', 'MS *']
            # movetypes = ['P *', 'BD *', 'MS *', 'BGD *']
            #proposal types
            movefns = [self.move_stars, self.birth_death_stars, self.merge_split_stars]
            # movefns = [self.move_stars, self.birth_death_stars, self.merge_split_stars, self.background_shift]
            proposal = movefns[rtype]()
            dt1[i] = time.clock() - t1
            if proposal.goodmove:
                t2 = time.clock()
                dmodels, diff2s, dt_transf = pcat_multiband_eval(proposal.xphon, proposal.yphon, proposal.fphon, proposal.dback, \
                                                imsz, ncs, cfs, weights=weights, ref=resids, lib=libmmult.pcat_model_eval, regsize=self.regsize, \
                                                margin=margin, offsetx=self.offsetx, offsety=self.offsety)
                dttq[i] = dt_transf
                # temporary fix
                if multiband:
                    diff2_total = np.sum(np.array(diff2s), axis=0)
                else:
                    diff2_total = diff2s[0]
                plogL = -0.5*diff2_total
                #background
                if rtype != 3:
                    plogL[(1-self.parity_y)::2,:] = float('-inf') # don't accept off-parity regions
                    plogL[:,(1-self.parity_x)::2] = float('-inf')
                dlogP = plogL - logL
                
                dt2[i] = time.clock() - t2
                t3 = time.clock()
                refx, refy = proposal.get_ref_xy()
                regionx = get_region(refx, self.offsetx, self.regsize)
                regiony = get_region(refy, self.offsety, self.regsize)
                
                if proposal.factor is not None:
                    dlogP[regiony, regionx] += proposal.factor
                    acceptreg = (np.log(np.random.uniform(size=(self.nregy, self.nregx))) < dlogP).astype(np.int32)
                    acceptprop = acceptreg[regiony, regionx]
                    numaccept = np.count_nonzero(acceptprop)

                for b in xrange(nbands):
                    dmodel_acpt = np.zeros_like(dmodels[b])
                    diff2s[b].fill(0)
                    offx = self.offsetx
                    offy = self.offsety
          
                    libmmult.pcat_imag_acpt(imsz[0], imsz[1], dmodels[b], dmodel_acpt, acceptreg, self.regsize, margin, offx, offy)
                    # using this dmodel containing only accepted moves, update logL
                    libmmult.pcat_like_eval(imsz[0], imsz[1], dmodel_acpt, resids[b], weights[b], diff2s[b], self.regsize, margin, offx, offy)   
                    resids[b] -= dmodel_acpt
                    models[b] += dmodel_acpt

                if multiband:
                    diff2_total1 = np.sum(np.array(diff2s), axis=0)
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
                    for idx in idx_kill_a:
                        self.stars[:,idx] = self.stars[:,self.n-1]
                        self.stars[:,self.n-1] = 0
                        self.n -= 1

                # if proposal.do_dback is not None:
                #     n_back_acpt += 1
                #     self.back += proposal.dback
                #     #reset parameters and calculate 
                #     self.offsetx = self.region_params[0]
                #     self.offsety = self.region_params[1]
                #     self.regsize = self.region_params[2]
                #     self.nregx = self.region_params[3]
                #     self.nregy = self.region_params[4]
                #     # models, diff2s, dt_transf = pcat_multiband_eval(evalx, evaly, evalf, self.back, imsz, ncs, cfs, weights=weights, ref=resids, lib=libmmult.pcat_model_eval, \
                #     #     regsize=self.regsize, margin=margin, offsetx=self.offsetx, offsety=self.offsety)
                #     models, diff2s, dt_transf = pcat_multiband_eval(evalx, evaly, evalf, self.back, imsz, ncs, cfs, weights=weights, ref=resids, lib=None, \
                #         regsize=self.regsize, margin=margin, offsetx=self.offsetx, offsety=self.offsety)
                #     if multiband:
                #         diff2_total = np.sum(np.array(diff2s), axis=0)
                #     else:
                #         diff2_total = diff2s[0]
                #     logL = -0.5*diff2_total

                dt3[i] = time.clock() - t3

                if acceptprop.size > 0: 
                    #background
                    if rtype != 3:
                        accept[i] = np.count_nonzero(acceptprop) / float(acceptprop.size)
                    else:
                        accept[i] = float(n_back_acpt)/float(n_back_prop)
                else:
                    accept[i] = 0
            else:
                outbounds[i] = 1
        
        chi2 = []
        for b in xrange(nbands):
            #don't evaluate chi2 on outer periphery, though this is mainly important for edge effects in multiband case
            # could be 'if multiband:'
            if datatype != 'mock':
                xmin, xmax = 2, imsz[0]-2
                ymin, ymax = 2, imsz[1]-2
            else:
                xmin, xmax = 0, imsz[0]
                ymin, ymax = 0, imsz[1]
            chi2.append(np.sum(weights[b][xmin:xmax,ymin:ymax]*(data_array[b][xmin:xmax,ymin:ymax]-models[b][xmin:xmax,ymin:ymax])*(data_array[b][xmin:xmax,ymin:ymax]-models[b][xmin:xmax,ymin:ymax])))        
        
        
        # ---------------- Printing diagnostics here ---------------------

        fmtstr = '\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f (Pg) %0.3f (BDg) %0.3f (S-g) %0.3f (gSg) %0.3f (gMS) %0.3f'
        print 'Background', self.back, 'N_star', self.n, 'N_phon', n_phon, 'chi^2', chi2
        dt1 *= 1000
        dt2 *= 1000
        dt3 *= 1000
        accept_fracs = []
        othertimes = []
        othertimes.append(np.sum(dttq))
        timestat_array = np.zeros((5, 1+len(moveweights)), dtype=np.float32)
        statlabels = ['Acceptance', 'Out of Bounds', 'Proposal (s)', 'Likelihood (s)', 'Implement (s)']

        #timestat_arrays = [dt1, dt2, dt3, dt_transf]

        # ### CLEAN THIS UP

        statarrays = [accept, outbounds, dt1, dt2, dt3]
        #print 'dt2:', np.sum(dt2)
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


        # ------------------------------------------------------------------

        tplot = time.clock()
        if visual or savefig:
            if include_hubble:
                hfs = hf[:,0]
            else:
                hfs = []

            multiband_sample_frame(data_array, self.stars[self._X,0:self.n], self.stars[self._Y,0:self.n], self.stars[self._F:,0:self.n], truex, truey, truef, truecolors, hubble_coords, hfs, resids, weights, \
                bands, nmgy_per_count, nstar, frame_dir, c, pixel_transfer_mats, mean_dpos, visual, savefig, include_hubble, datatype)

        dtplot = time.clock()-tplot
        othertimes.append(dtplot)
        return self.n, chi2, timestat_array, othertimes, accept_fracs

    # def calculate_logL(diff2s, len_diff20):
    #     if multiband:
    #         diff2_recon = []
    #         a = 0
    #         b = len_diff20-1
    #         print diff2s[b]
    #         while diff2s[b] is not None:
    #             print diff2s[b]
    #             diff2_recon.append(diff2s[a:b])
    #             a+= len_diff20+1
    #             b+= len_diff20+1
    #         print diff2_recon
    #         diff2_total = np.sum(np.array(diff2_recon), axis=0)
    #     else:
    #         diff2_total = diff2s[0]
    #     logL = -0.5*diff2_total 
    #     return logL

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

    # def background_shift(self):
    #     which_band = 0 #for finding g band stuff
    #     dback = np.zeros(nbands)
    #     # dback[which_band] = np.random.normal(scale=0.01)
    #     pixel_variance = trueback[which_band]/gains[which_band]
    #     bkg_step_size = np.sqrt(pixel_variance)/(imsz[0]*imsz[1])
    #     # bkg_step_size = 0.5
    #     dback[which_band] = np.random.normal(scale=bkg_step_size)
    #     # factor = gaussian(trueback[which_band]+dback[which_band], trueback[which_band], 1)*imsz[0]
    #     factor = 0
    #     # print(factor)
    #     proposal = Proposal()
    #     proposal.add_background_shift(self.back, dback, which_band, self.stars)
    #     proposal.set_factor(factor)
    #     return proposal


    def move_stars(self): 
        idx_move = self.idx_parity_stars()
        nw = idx_move.size
        stars0 = self.stars.take(idx_move, axis=1)
        starsp = np.empty_like(stars0)
        f0 = stars0[self._F:,:]
        pfs = []
        color_factors = np.zeros((nbands-1, nw)).astype(np.float32)

        for b in xrange(nbands):
            pf = flux_proposal(f0[b], nw, 0, b)
            pfs.append(pf)

        pixel_variance = trueback[0]/gains[0]
        N_eff = 17.5
        err_f = np.sqrt(N_eff * pixel_variance)
        # data specific
        N_src = 1400.
 
        dlogf = np.log(pfs[0]/f0[0])
        factor = -self.truealpha*dlogf

        factor = np.array(factor) + np.sum(color_factors, axis=0)
        if multiple_regions:
            dpos_rms = np.float32(np.sqrt(N_eff/(2*np.pi))*err_f/(np.sqrt(N_src*0.04*(2+nbands))))/(np.maximum(f0[0], pfs[0])) 
        else:
            dpos_rms = np.float32(np.sqrt(N_eff/(2*np.pi))*err_f/np.sqrt(N_src*(2+nbands)))/(np.maximum(f0[0], pfs[0]))
        dx = np.random.normal(size=nw)*dpos_rms
        dy = np.random.normal(size=nw)*dpos_rms
       # print 'dx average: ', np.mean(np.abs(dx))
        starsp[self._X,:] = stars0[self._X,:] + dx
        starsp[self._Y,:] = stars0[self._Y,:] + dy
        for b in xrange(nbands):
            starsp[self._F+b,:] = pfs[b]
        self.bounce_off_edges(starsp)

        proposal = Proposal()
        proposal.add_move_stars(idx_move, stars0, starsp)
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
                    new_colors = np.random.normal(loc=self.color_mus[b-1], scale=self.color_sigs[b-1], size=nbd)
                    starsb[self._F+b,:] = starsb[self._F,:]*10**(0.4*new_colors)*nmgy_per_count[0]/nmgy_per_count[b]
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
            dx = (np.random.normal(size=nms)*self.kickrange)
            dy = (np.random.normal(size=nms)*self.kickrange)
            idx_move = np.random.choice(idx_bright, size=nms, replace=False)
            stars0 = self.stars.take(idx_move, axis=1)
            fminratio = stars0[self._F,:] / self.trueminf
 
            fracs.append((1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32))
            for b in xrange(nbands-1):
                fracs.append(np.random.uniform(size=nms))

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
                for b in xrange(nbands-1):
                    stars0_color = adus_to_color(stars0[self._F,:], stars0[self._F+b+1,:], [nmgy_per_count[0], nmgy_per_count[b]])
                    starsp_color = adus_to_color(starsp[self._F,:], starsp[self._F+b+1,:], [nmgy_per_count[0], nmgy_per_count[b]])
                    factor += np.log(2.5/np.log(10)) - np.log(fracs[b+1]*(1-fracs[b+1])) 

                    if splitsville:
                        starsb_color = adus_to_color(starsb[self._F,:], starsb[self._F+b+1,:], [nmgy_per_count[0], nmgy_per_count[b]])
                        factor += (stars0_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (starsp_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (starsb_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2)
                    else:
                        starsk_color = adus_to_color(starsk[self._F,:], starsk[self._F+b+1,:], [nmgy_per_count[0], nmgy_per_count[b]])
                        factor += (starsp_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (stars0_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (starsk_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2)
            if not splitsville:
                factor *= -1
                factor += self.penalty
            else:
                factor -= self.penalty
            proposal.set_factor(factor)
        return proposal

# --------------------------------- start executing the program now ---------------------------------------------

# libmmult = npct.load_library('pcat-lion-test', '.')
libmmult = npct.load_library('pcat-lion', '.')
initialize_c()

if datatype=='mock2':
    truth = np.loadtxt('Data/'+mock_test_name+'/'+dataname+'/'+dataname+'-tru.txt')
else:
    truth = np.loadtxt('Data/'+dataname+'/truth/'+dataname+'-tru.txt')
truex = truth[:,0]
truey = truth[:,1]
truef = truth[:,2:]
truecolors = []
if multiband:
    for b in xrange(nbands-1):
        nmpc = [nmgy_per_count[0], nmgy_per_count[b+1]]
        truecolor = adus_to_color(truef[:,0], truef[:,b+1], nmpc) 
        truecolors.append(truecolor)

if include_hubble:
    true_h = fits.open('Data/'+dataname+'/hubble_pixel_coords-2583-2-0136.fits')

    xoff = 310
    yoff = 630

    hxr = true_h[0].data-xoff
    hyr = true_h[1].data-yoff
    hxi = true_h[2].data-xoff
    hyi = true_h[3].data-yoff
    hxg = true_h[4].data-xoff
    hyg = true_h[5].data-yoff

    hubble_coords = [hxr, hyr, hxi, hyi, hxg, hyg]

    true_h = np.loadtxt('Data/'+dataname+'/HTcat-'+dataname+'.txt')
    hx = true_h[:,0]
    hy = true_h[:,1]
    hf = true_h[:,2:]
else:
    hx, hy, hf = [], [], []
    hubble_coords = [hx, hy, hf]
    mean_dpos = np.zeros((nbands, 2), dtype=np.float32)


nstar = Model.nstar
nsample = np.zeros(nsamp, dtype=np.int32)
xsample = np.zeros((nsamp, nstar), dtype=np.float32)
ysample = np.zeros((nsamp, nstar), dtype=np.float32)
dt2s = np.zeros(nsamp, dtype=np.float32)
timestats = np.zeros((nsamp, 5, 4), dtype=np.float32)
accept_stats = np.zeros((nsamp, 4), dtype=np.float32)
tq_times = np.zeros(nsamp, dtype=np.float32)
plt_times = np.zeros(nsamp, dtype=np.float32)
bkgsample = np.zeros((nsamp, nbands), dtype=np.float32)
if multiband:
    fsample = [np.zeros((nsamp, nstar), dtype=np.float32) for x in xrange(nbands)]
    colorsample = [[] for x in xrange(nbands-1)]
else:
    fsample = np.zeros((nsamp, nstar), dtype=np.float32)
chi2sample = np.zeros((nsamp, nbands), dtype=np.int32)
model = Model()

#create directory for results
frame_dir = create_directories(timestr)

if visual:
    plt.ion()
    plt.figure(1, figsize=(15, min(2, nbands)*5))
# sampling loop
for j in xrange(nsamp):
    chi2_all = np.zeros(nbands)
    print 'Loop', j
    sf = 0
    # if nsamp<10:
    #     sf = 1
    #     c+= 1
    # else:
    #     if j%10==0:
    #         sf = 1
    #         c+=1
     
    _, chi2_all, statarrays, othertimes, accept_fracs = model.run_sampler(visual=visual, multiband=multiband, savefig=sf)

    tq_times[j] = othertimes[0]
    plt_times[j] = othertimes[1]
    nsample[j] = model.n
    xsample[j,:] = model.stars[Model._X, :]
    ysample[j,:] = model.stars[Model._Y, :]
    bkgsample[j] = model.back

    accept_stats[j,:] = accept_fracs
    nmgy_sample = []
    if multiband:
        for b in xrange(nbands):
            fsample[b][j,:] = model.stars[Model._F+b,:]
            if b>0:
                nmpc = [nmgy_per_count[0], nmgy_per_count[b]]
                csample = adus_to_color(model.stars[Model._F,:], model.stars[Model._F+b,:], nmpc)
                csample = np.array([value for value in csample if not math.isnan(value)])
                colorsample[b-1].append(csample)

    else:
        fsample[j,:] = model.stars[Model._F, :]

    chi2sample[j] = chi2_all
    timestats[j,:] = statarrays
   # dt2s[j] = dt2_val
if not multiband:
    colorsample = []
print 'saving...'

#print np.mean(dt2s), np.std(dt2s)

#mock test
if datatype=='mock2':
    np.savez(result_path + '/'+mock_test_name+'/' + str(dataname) + '/results/'+str(config_type)+'-'+str(nrealization)+'.npz', n=nsample, x=xsample, y=ysample, f=fsample, chi2=np.sum(chi2sample, axis=1), times=timestats, accept=accept_stats)
    result_dir = result_path + '/'+mock_test_name+'/' + str(dataname) + '/results'
else:
    np.savez(result_path + '/' + str(timestr) + '/chain.npz', n=nsample, x=xsample, y=ysample, f=fsample, chi2=np.sum(chi2sample, axis=1), times=timestats, accept=accept_stats)
    result_dir = result_path + '/' + timestr


# results(nsample,fsample, truef, colorsample, nsamp, timestats, tq_times, plt_times, chi2sample, bkgsample, np.array(accept_stats), result_dir, nbands, bands, multiband, nmgy_per_count, datatype)

dt_total = time.clock()-start_time
print 'Full Run Time (s):', np.round(dt_total,3)
