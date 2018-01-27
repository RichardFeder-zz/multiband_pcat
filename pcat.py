import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double
import h5py
# in order for visual=True to work, interactive backend should be loaded before importing pyplot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
from galaxy import to_moments, from_moments, retr_sers, retr_tranphon

# Run, rerun, CamCol, DAOPHOTID, RA, DEC, xu, yu, u (mag), uErr, chi, sharp, flag, xg, yg, g, gErr, chi, sharp, flag, 
# xr, yr, r, rerr, chi, sharp, flag, xi, yi, i, ierr, chi, sharp, flag, xz, yz, z, zerr, chi, sharp, flag

directory_path = "/Users/richardfeder/Documents/multiband_pcat/pcat-lion-results"
timestr = time.strftime("%Y%m%d-%H%M%S")
c = 0
trueback = 179
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

bands, ncs, nbins, psfs, cfs, pixel_transfer_mats, biases, gains, data_array, data_hdrs, weights = [[] for x in xrange(11)]

if multiband:
    band = raw_input("Enter bands one by one in lowercase ('done' if no more): ")
    while band != 'done':
        bands.append(band)
        band = raw_input("Enter bands one by one in lowercase ('done' if no more): ")
    nbands = len(bands)
    print('Loading data for the following bands: ' + str(bands))
else:
    nbands = 1
    bands = ['']

def magnitudes_to_counts(frame_file, mags):
    fits_frame = fits.open(frame_file)
    frame_header = fits_frame[0].header
    nanomaggy_per_count = frame_header['NMGY']
    nmgy = 10**((22.5-np.array(mags))/2.5)
    source_counts = [ x/nanomaggy_per_count for x in nmgy ]
    return source_counts

def extract_sdss_catalog(catalog_file, frame_file_r, frame_file_i, outfile, bounds):
    with open(catalog_file, 'r') as p:
        lines = p.read().splitlines()
    sources = []
    for line in lines:
        sources.append(line.split())
    subregion_sources = [x for x in sources if float(x[20])>bounds[0] and float(x[20])<bounds[1] and float(x[21])>bounds[2] and float(x[21])<bounds[3]]
    catalog_sources = [[float(x[20]), float(x[21])] for x in subregion_sources if float(x[22]) != 99.999 and float(x[29]) != 99.999] 
    r_mags = [float(x[22]) for x in subregion_sources]
    i_mags = [float(x[29]) for x in subregion_sources]
    
    r_counts = magnitudes_to_counts(frame_file_r, r_mags)
    i_counts = magnitudes_to_counts(frame_file_i, i_mags)
    
    for s in xrange(len(catalog_sources)):
        catalog_sources[s].append(r_counts[s])
        catalog_sources[s].append(i_counts[s])
    
    print(str(len(subregion_sources) - len(catalog_sources)) + ' sources in subregion were undetected in either r or i band.')
    print(str(len(catalog_sources)) + ' sources included in catalog from subregion.')
    with open(outfile, 'w') as file:
        file.writelines(' '.join(str(j) for j in source) + '\n' for source in catalog_sources)

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
    if visual:
        frame_dir_name = new_dir_name + '/frames'
    os.makedirs(frame_dir_name)
    return frame_dir_name

#frame_dir = create_directories(timestr)

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def get_pint_dp(p):
    pint = np.floor(p+0.5)
    dp = p - pint
    return pint.astype(int), dp

def flux_proposal(f0, nw, trueminf):
    lindf = np.float32(60.*134/np.sqrt(25.))
    logdf = np.float32(0.01/np.sqrt(25.))
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
    return xnew, ynew

def pcat_multiband_eval(x, y, f, f1, bkg, imsz, nc, cf, weights, ref, lib, regsize, margin, offsetx, offsety):
    dmodels, diff2s = [[],[]]
    for b in xrange(2):
        if b>0:
            xp, yp = transform_q(x, y, pixel_transfer_mats[b-1])
            dmodel, diff2 = image_model_eval(xp, yp, f1, bkg, imsz, nc, cf.astype(np.float32()), weights=weights[b], ref=ref[b], lib=libmmult.pcat_model_eval, regsize=regsize, margin=margin, offsetx=offsetx, offsety=offsety)
        else:    
            xp=x
            yp=y
            dmodel, diff2 = image_model_eval(xp, yp, f, bkg, imsz, nc, cf.astype(np.float32()), weights=weights[b], ref=ref[b], lib=libmmult.pcat_model_eval, regsize=regsize, margin=margin, offsetx=offsetx, offsety=offsety)

        # dmodel, diff2 = image_model_eval(xp, yp, f[b], 0, imsz, nc, cf.astype(np.float32()), weights=weights[b], ref=ref[b], lib=libmmult.pcat_model_eval, regsize=regsize, margin=margin, offsetx=offsetx, offsety=offsety)
        dmodels.append(dmodel)
        diff2s.append(diff2)
    return dmodels, diff2s

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
    f01 = psf[iiy+1:125:5,iix:125:5]
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

def results(nchain, fchain, truef, color, nsamp):
    print('nsamp: ' + str(nsamp))
    plt.figure()
    plt.title('Posterior Source Number Histogram')
    plt.hist(nchain, histtype='step', label='Posterior', color='b')
    plt.axvline(np.median(nchain), label='Median', color='b', linestyle='dashed')
    plt.xlabel('nstar')
    plt.axvline(len(truef), label='DAOPHOT', linestyle='dashed', color='g')
    plt.legend()
    plt.savefig(directory_path + '/' + timestr + '/posterior_histogram_nstar.png')
    plt.show()

    if multiband:
        plt.figure()
        plt.title('Posterior Color Distribution (Normalized to 1)')
        plt.hist(np.log10(np.array(truef[:,0])/np.array(truef[:,1])), histtype='step', label='DAOPHOT', color='g', normed=1)
        plt.hist(np.concatenate(np.array(color)), histtype='step', label='Posterior', color='b', normed=1)
        plt.legend()
        plt.xlabel(str(bands[0]) + '-' + str(bands[1]))
        plt.savefig(directory_path + '/' + timestr + '/posterior_histogram_r_i_color.png')
        plt.show()


    
    # if multiband:
    #     for b in xrange(nbands):
    #         plt.figure()
    #         plt.title('Posterior Flux Distribution - ' + str(bands[b]))
    #         (n,bins,patches) = plt.hist(truef[:,b], histtype='step', label='SDSS', color='g')
    #         plt.hist(np.array(fmedian_samples[:,:,b])/len(fmedian_samples), bins=bins, histtype='step', label='Posterior', color='navy')
    #         plt.legend()
    #         plt.savefig(directory_path + '/' + timestr + '/posterior_flux_histogram_' + str(bands[b]) + '.png')
    # else:
    #     plt.figure()
    #     plt.title('Posterior Flux Distribution')
    #     (n, bins, patches) = plt.hist(truef, histtype='step', label='SDSS', color='g')
    #     plt.hist(np.array(fmedian_samples[:,:])/len(fmedian_samples), bins=bins, histtype='step', label='Posterior', color='navy')
    #     plt.legend()
    #     plt.savefig(directory_path + '/' + timestr + '/posterior_flux_histogram.png')



class Proposal:
    gridphon, amplphon = retr_sers(sersindx=2.)
    _X = 0
    _Y = 1
    _F = 2

    def __init__(self):
        self.idx_move = None
        self.do_birth = False
        self.idx_kill = None
        self.factor = None
        self.goodmove = False

        self.xphon = np.array([], dtype=np.float32)
        self.yphon = np.array([], dtype=np.float32)
        #this is a temporary fix, should figure out how to do this properly to generalize
        self.fphon = np.array([], dtype=np.float32)
        if multiband:
            self.fphon1 = np.array([], dtype=np.float32)

    def set_factor(self, factor):
        self.factor = factor

    def assert_types(self):
        assert self.xphon.dtype == np.float32
        assert self.yphon.dtype == np.float32
        assert self.fphon.dtype == np.float32
        if multiband:
            assert self.fphon1.dtype == np.float32
        #for b in xrange(nbands):
        #    assert self.fphon[b].dtype == np.float32

    def __add_phonions_stars(self, stars, remove=False):
        fluxmult = -1 if remove else 1
        self.xphon = np.append(self.xphon, stars[self._X,:])
        self.yphon = np.append(self.yphon, stars[self._Y,:])
        for b in xrange(nbands):
            if b==0:
                self.fphon = np.append(self.fphon, np.array(fluxmult*stars[self._F+b,:], dtype=np.float32))
            else:
                self.fphon1 = np.append(self.fphon1, np.array(fluxmult*stars[self._F+b,:], dtype=np.float32))
            #print('flux mult array ' + str(b) + ' is = ' + str(np.array(fluxmult*stars[self._F+b,:], dtype=np.float32())))
            #self.fphon[b] = np.array(self.fphon[b], dtype=np.float32())
            #self.fphon[b] = np.append(self.fphon[b], np.array(fluxmult*stars[self._F+b,:], dtype=np.float32))
        #print(self.fphon)
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

class Model:
    # should these be class or instance variables?
    nstar = 1000
    trueminf = np.float32(236)
    truealpha = np.float32(2.00)

    gridphon, amplphon = retr_sers(sersindx=2.)

    penalty = 1.5
    kickrange = 1.

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
            self.stars[self._F+b,0:self.n] **= -1./(self.truealpha - 1.)
            self.stars[self._F+b,0:self.n] *= self.trueminf

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

        self.offsetx = np.random.randint(regsize)
        self.offsety = np.random.randint(regsize)
        self.nregx = imsz[0] / regsize + 1
        self.nregy = imsz[1] / regsize + 1

        resids = []
        for b in xrange(nbands):
            resid = data_array[b].copy() # residual for zero image is data
            resids.append(resid)
        #resid = data.copy() # residual for zero image is data
        if strgmode == 'star':
            evalx = self.stars[self._X,0:self.n]
            evaly = self.stars[self._Y,0:self.n]
            evalf = self.stars[self._F:,0:self.n]
        else:
            xposphon, yposphon, specphon = retr_tranphon(self.gridphon, self.amplphon, self.galaxies[:,0:self.ng])
            evalx = np.concatenate([self.stars[self._X,0:self.n], xposphon]).astype(np.float32)
            evaly = np.concatenate([self.stars[self._Y,0:self.n], yposphon]).astype(np.float32)
            evalf = np.concatenate([self.stars[self._F,0:self.n], specphon]).astype(np.float32)
        n_phon = evalx.size
        #temporary solution 2 bands
        models, diff2s = pcat_multiband_eval(evalx, evaly, evalf[0], evalf[1], self.back, imsz, nc, cf, weights=weights, ref=resids, lib=libmmult.pcat_model_eval, \
            regsize=regsize, margin=margin, offsetx=self.offsetx, offsety=self.offsety)

        logLs = []
        for b in xrange(nbands):
            logLs.append(-0.5*diff2s[b])
            resids[b] -= models[b]
        logL = np.sum(logLs)

        moveweights = np.array([80., 40., 40.])
        moveweights /= np.sum(moveweights)

        for i in xrange(nloop):
            t1 = time.clock()
            rtype = np.random.choice(moveweights.size, p=moveweights)
            movetype[i] = rtype
            # defaults
            dback = np.float32(0.)
            pn = self.n

            # should regions be perturbed randomly or systematically?
            self.parity_x = np.random.randint(2)
            self.parity_y = np.random.randint(2)

            movetypes = ['P *', 'BD *', 'MS *']

            #does it calculate each one of these and then choose one? 
            movefns = [self.move_stars, self.birth_death_stars, self.merge_split_stars]
            proposal = movefns[rtype]()

            dt1[i] = time.clock() - t1

            if proposal.goodmove:
                t2 = time.clock()
                dmodels, diff2s = pcat_multiband_eval(proposal.xphon, proposal.yphon, proposal.fphon, proposal.fphon1, dback, imsz, nc, cf, weights=weights, ref=resids, lib=libmmult.pcat_model_eval, regsize=regsize, margin=margin, offsetx=self.offsetx, offsety=self.offsety)
                #temporary fix
                if multiband:
                    diff2_total = np.array(diff2s[0]) + np.array(diff2s[1])
  
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
                    libmmult.pcat_imag_acpt(imsz[0], imsz[1], dmodels[b], dmodel_acpt, acceptreg, regsize, margin, self.offsetx, self.offsety)
                    # using this dmodel containing only accepted moves, update logL
                    diff2s[b].fill(0)
                    libmmult.pcat_like_eval(imsz[0], imsz[1], dmodel_acpt, resid, weight, diff2s[b], regsize, margin, self.offsetx, self.offsety)
                    logLs[b] = -0.5*diff2s[b]
                    resids[b] -= dmodel_acpt
                    models[b] += dmodel_acpt
                logL = np.sum(logLs)

                # implement accepted moves. I think here is where the problem comes into play since i'm just using one acceptprop
                if proposal.idx_move is not None:
                    # print('acceptprop: ' + str(acceptprop))
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
                dt3[i] = time.clock() - t3

                # hmm...
                #back += dback
                if acceptprop.size > 0: 
                    accept[i] = np.count_nonzero(acceptprop) / float(acceptprop.size)
                else:
                    accept[i] = 0
            else:
                outbounds[i] = 1
        
        #does it make sense to add both of these? I think so but honestly doesn't matter too much I guess in PCAT context.
        chi2 = 0
        for b in xrange(nbands):
            chi2 += np.sum(weights[b]*(data_array[b]-models[b])*(data_array[b]-models[b]))        
        
        fmtstr = '\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f (Pg) %0.3f (BDg) %0.3f (S-g) %0.3f (gSg) %0.3f (gMS) %0.3f'
        print 'Temperature', temperature, 'background', self.back, 'N_star', self.n, 'N_phon', n_phon, 'chi^2', chi2
        dt1 *= 1000
        dt2 *= 1000
        dt3 *= 1000
        timestat_array = np.zeros((5, 4), dtype=np.float32)
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
        if visual:
            if multiband:
                sizefac = 10.*136
                
                plt.gcf().clear()
                plt.figure(1)

                plt.subplot(1,5,1)
                plt.imshow(data, origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data), vmax=np.percentile(data, 95))
                sizefac = 10.*136
                plt.scatter(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], marker='x', s=self.stars[self._F, 0:self.n]/(2*sizefac), color='r')
                if datatype == 'mock':
                    if strgmode == 'star' or strgmode == 'galx':
                        mask = truef[:,0] > 25 # will have to change this for other data sets
                        plt.scatter(truex[mask], truey[mask], marker='+', s=2*truef[mask] / sizefac, color='lime')
                        mask = np.logical_not(mask)
                        plt.scatter(truex[mask], truey[mask], marker='+', s=2*truef[mask] / sizefac, color='g')
                plt.xlim(-0.5, imsz[0]-0.5)
                plt.ylim(-0.5, imsz[1]-0.5)
            
                plt.subplot(1, 5, 2)
                plt.title('Residual in ' + str(bands[0]) + ' band')
                plt.imshow(resids[0]*np.sqrt(weights[0]), origin='lower', interpolation='none', cmap='bwr', vmax=100, vmin=-100)
                
                plt.subplot(1,5, 3)
                plt.title('Posterior Flux Histogram (' + str(bands[0]) + ')')
                (n, bins, patches) = plt.hist(np.log10(truef[:,0]), log=True, alpha=0.5, label=labldata,  color='g', histtype='step')
                plt.hist(np.log10(self.stars[self._F, 0:self.n]), log=True, alpha=0.5, color='r', bins=bins, label='Chain - ' + bands[0], histtype='step')
                plt.legend()
                plt.xlabel('log10 flux')
                plt.ylim((0.5, nstar))
          
                plt.subplot(1, 5, 4)
                plt.title('Posterior Flux Histogram (' + str(bands[1]) + ')')

                if datatype == 'mock':
                    (n, bins, patches) = plt.hist(np.log10(truef[:,1]), log=True, alpha=0.5, color='g', label=labldata, histtype='step')
                    plt.hist(np.log10(self.stars[self._F+1, 0:self.n]), color='gold', bins=bins, log=True, alpha=0.5, label='Chain - ' + bands[1], histtype='step')
                else:
                    plt.hist(np.log10(f[0:n]), range=(np.log10(trueminf), np.ceil(np.log10(np.max(f[0:n])))), log=True, alpha=0.5, label='Chain', histtype='step')
                plt.legend()
                plt.xlabel('log10 flux')
                plt.ylim((0.5, nstar))

                #Color histogram
                color_bins = np.linspace(-2, 2, 10)
                plt.subplot(1, 5, 5)
                plt.title('Posterior Color Histogram')
                colors = np.array(self.stars[self._F+1, 0:self.n])/np.array(self.stars[self._F, 0:self.n])
                pos_color = [color for color in colors if color > 0]
                #print(str(len(pos_color)) + ' of ' + str(len(colors)) + ' samples with r - i > 0')
                plt.hist(np.log10(pos_color), label='Chain', log=True, alpha=0.5, bins = color_bins, histtype='step', color='r')
                plt.hist(np.log10(np.array(truef[:,0])/np.array(truef[:,1])), label='True', bins=color_bins, color='g', histtype='step')
                plt.legend()
                plt.xlabel(bands[0] + ' - ' + bands[1])
                if visual:
                    plt.draw()
                if savefig:
                    plt.savefig(frame_dir + '/frame_' + str(c) + '.png')
                plt.pause(1e-5)

            else:
                plt.gcf().clear()
                plt.figure(1)
                plt.clf()
                plt.subplot(1,3,1)
                plt.imshow(data, origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data), vmax=np.percentile(data, 95))
                sizefac = 10.*136
                if datatype == 'mock':
                    if strgmode == 'star' or strgmode == 'galx':
                        mask = truef > 250 # will have to change this for other data sets
                        plt.scatter(truex[mask], truey[mask], marker='+', s=truef[mask] / sizefac, color='lime')
                        mask = np.logical_not(mask)
                        plt.scatter(truex[mask], truey[mask], marker='+', s=truef[mask] / sizefac, color='g')
                plt.scatter(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], marker='x', s=self.stars[self._F, 0:self.n]/sizefac, color='r')
                plt.xlim(-0.5, imsz[0]-0.5)
                plt.ylim(-0.5, imsz[1]-0.5)
                plt.subplot(1,3,2)
                plt.imshow(resid*np.sqrt(weight), origin='lower', interpolation='none', cmap='bwr', vmin=-100, vmax=100)
                plt.colorbar()
                if j == 0:
                    plt.tight_layout()
                plt.subplot(1,3,3)
                if datatype == 'mock':
                    plt.hist(np.log10(truef), range=(np.log10(self.trueminf), np.log10(np.max(truef))), log=True, alpha=0.5, label=labldata, histtype='step')
                    plt.hist(np.log10(self.stars[self._F, 0:self.n]), range=(np.log10(self.trueminf), np.log10(np.max(truef))), log=True, alpha=0.5, label='Chain', histtype='step')
                else:
                    plt.hist(np.log10(self.stars[self._F, 0:self.n]), range=(np.log10(self.trueminf), np.ceil(np.log10(np.max(self.f[0:self.n])))), log=True, alpha=0.5, label='Chain', histtype='step')
                plt.legend()
                plt.xlabel('log10 flux')
                plt.ylim((0.5, self.nstar))
                plt.draw()
                if savefig:
                    plt.savefig(frame_dir + '/frame_' + str(c) + '.png')
                plt.pause(1e-5)


        return self.n, chi2, timestat_array

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

    def move_stars(self): 
        idx_move = self.idx_parity_stars()
        nw = idx_move.size
        stars0 = self.stars.take(idx_move, axis=1)
        starsp = np.empty_like(stars0)
        f0 = stars0[self._F:,:]


        pfs = []
        for b in xrange(nbands):
            pf = flux_proposal(f0[b], nw, self.trueminf)
            pfs.append(pf)
        # calculate flux distribution prior factor, but only for first flux bin. the rest are parameterized by color priors
        # what to do in case of nbands > 2? I guess multiple color priors
        # FINE TUNE NEEDED for color prior center
        dlogf = np.log(pfs[0]/f0[0])
        #print('dlogf: ' + str(dlogf))
        factor = -self.truealpha*dlogf
        #print('factor 629: ' + str(factor))
        if multiband and nbands>1:
            colors = np.log(np.array(self.stars[self._F, 0:self.n])/np.array(self.stars[self._F+1, 0:self.n]))
            color_factor = np.sum(((colors-0.5)**2/(2*2))) #gaussian prior
            factor = np.array(factor)+color_factor

        dpos_rms = np.float32(60.*134/np.sqrt(25.))/(np.maximum(f0[0], pfs[0]))
        dx = np.random.normal(size=nw).astype(np.float32)*dpos_rms
        dy = np.random.normal(size=nw).astype(np.float32)*dpos_rms
        starsp[self._X,:] = stars0[self._X,:] + dx
        starsp[self._Y,:] = stars0[self._Y,:] + dy
        for b in xrange(nbands):
            starsp[self._F+b,:] = pfs[b]
        #starsp[self._F,:] = pf
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
                starsb[self._F+b,:] = self.trueminf * np.exp(np.random.exponential(scale=1./(self.truealpha-1.),size=nbd))
            # starsb[self._F,:] = self.trueminf * np.exp(np.random.exponential(scale=1./(self.truealpha-1.),size=nbd))

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
            #how should I unpack this? should I have it as just one array and then index appropriately for position and then iteratively for fluxes?
            if multiband:
                x0, y0, f0, f1 = stars0
            else:
                x0, y0, f0 = stars0
            # what should I do with fminratio for multiband? same for frac
            fminratio = f0 / self.trueminf
            frac = (1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32)
            
            starsp = np.empty_like(stars0)
            starsb = np.empty_like(stars0)

            starsp[self._X,:] = x0 + ((1-frac)*dx)
            starsp[self._Y,:] = y0 + ((1-frac)*dy)
            starsb[self._X,:] = x0 - frac*dx
            starsb[self._Y,:] = y0 - frac*dy
            for b in xrange(nbands):
                if b==0:
                    starsp[self._F,:] = f0 * frac
                    starsb[self._F,:] = f0 * (1-frac)
                else:
                    starsp[self._F+b,:] = f1 * frac
                    starsb[self._F+b,:] = f1 * (1-frac)

            # don't want to think about how to bounce split-merge
            # don't need to check if above fmin, because of how frac is decided
            inbounds = np.logical_and(self.in_bounds(starsp), self.in_bounds(starsb))
            stars0 = stars0.compress(inbounds, axis=1)
            starsp = starsp.compress(inbounds, axis=1)
            starsb = starsb.compress(inbounds, axis=1)
            idx_move = idx_move.compress(inbounds)
            fminratio = fminratio.compress(inbounds)
            frac = frac.compress(inbounds)
            nms = idx_move.size
            goodmove = nms > 0
            if goodmove:
                proposal.add_move_stars(idx_move, stars0, starsp)
                proposal.add_birth_stars(starsb)

            # need to calculate factor
            sum_f = stars0[self._F,:]
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
            # this is different type of f0 than usual
            f0 = stars0[self._F:,:]
            fk = starsk[self._F:,:]
            sum_f = f0[0] + fk[0]
            fminratio = sum_f / self.trueminf
            frac = f0[0] / sum_f
            starsp = np.empty_like(stars0)
            starsp[self._X,:] = frac*stars0[self._X,:] + (1-frac)*starsk[self._X,:]
            starsp[self._Y,:] = frac*stars0[self._Y,:] + (1-frac)*starsk[self._Y,:]
            for b in xrange(nbands):
                starsp[self._F+b,:] = f0[b] + fk[b]
            if goodmove:
                proposal.add_move_stars(idx_move, stars0, starsp)
                proposal.add_death_stars(idx_kill, starsk)
            # turn bright_n into an array
            bright_n = bright_n - (f0[0] > 2*self.trueminf) - (fk[0] > 2*self.trueminf) + (starsp[self._F,:] > 2*self.trueminf)
        if goodmove:
            factor = np.log(self.truealpha-1) + (self.truealpha-1)*np.log(self.trueminf) - self.truealpha*np.log(frac*(1-frac)*sum_f) + np.log(2*np.pi*self.kickrange*self.kickrange) - np.log(imsz[0]*imsz[1]) + np.log(1. - 2./fminratio) + np.log(bright_n) + np.log(invpairs) + np.log(sum_f) # last term is Jacobian
            if not splitsville:
                factor *= -1
                factor += self.penalty
            else:
                factor -= self.penalty
            proposal.set_factor(factor)
        return proposal

libmmult = npct.load_library('pcat-lion', '.')
initialize_c()

print 'Lion mode:', strgmode
print 'datatype:', datatype

for b in xrange(nbands):
    print('Retrieving data from Data/'+dataname+'/'+dataname+'-psf'+str(bands[b])+'.txt')
    f = open('Data/'+dataname+'/'+dataname+'-psf'+str(bands[b])+'.txt')
    #f = open('Data/'+dataname+'_psf.txt')
    nc, nbin = [np.int32(i) for i in f.readline().split()]
    f.close()
    #psf = np.loadtxt('Data/'+dataname+'_psf.txt', skiprows=1).astype(np.float32)
    psf = np.loadtxt('Data/'+dataname+'/'+dataname+'-psf'+str(bands[b])+'.txt',skiprows=1).astype(np.float32)
    psfs.append(psf)
    ncs.append(nc)
    cf = psf_poly_fit(psf, nbin=nbin)
    cfs.append(cfs)
    npar = cf.shape[0]

    g = open('Data/'+dataname+'/'+dataname+'-pix'+bands[b]+'.txt')
    w, h, nb = [np.int32(i) for i in g.readline().split()]
    bias, gain = [np.float32(i) for i in g.readline().split()]
    biases.append(bias)
    gains.append(gain)
    g.close()

    data = np.loadtxt('Data/'+dataname+'/'+dataname+'_subregion_cts'+ bands[b]+'.txt').astype(np.float32)
    data -= bias
    print('orig data has shape ' + str(data.shape))
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
        transfer_hdu_gr = fits.open('Data/'+dataname+'/asGrid002583-2-0136-100x100-'+bands[0]+'-'+bands[b]+'.fits')
        #may eventually want transfer_hdu_ba, but as long as we only make proposals from one band and then transform to other bands we should be fine with this
        pixel_transfer_mats.append(np.array([transfer_hdu_gr[0].data, transfer_hdu_gr[1].data,transfer_hdu_gr[2].data, transfer_hdu_gr[3].data, transfer_hdu_gr[4].data, transfer_hdu_gr[5].data]))
        #check image dimensions are consistent across bands
        assert w==w0 and h==h0

    if visual and testpsfn:
        testpsf(ncs[b], cf, psfs[b], np.float32(np.random.uniform()*4), np.float32(np.random.uniform()*4), lib=libmmult.pcat_model_eval)

if datatype == 'mock':
    if strgmode == 'star':
        truth = np.loadtxt('Data/'+dataname+'/'+dataname+'_tru.txt')
        truex = truth[:,0]
        truey = truth[:,1]
        truef = truth[:,2:]
    labldata = 'True'
else:
    labldata = 'HST 606W'

ntemps = 1
temps = np.sqrt(2) ** np.arange(ntemps)

regsize = 50
assert imsz[0] % regsize == 0
assert imsz[1] % regsize == 0
margin = 10

nsamp = 30
nloop = 1000
nstar = Model.nstar
nsample = np.zeros(nsamp, dtype=np.int32)
xsample = np.zeros((nsamp, nstar), dtype=np.float32)
ysample = np.zeros((nsamp, nstar), dtype=np.float32)
timestats = np.zeros((nsamp, 5, 4), dtype=np.float32)
if multiband:
    #fsample = np.zeros((nsamp, nbands, nstar), dtype=np.float32)
    fsample = []
    colorsample = []
else:
    fsample = np.zeros((nsamp, nstar), dtype=np.float32)
chi2sample = np.zeros(nsamp, dtype=np.int32)

models = [Model() for k in xrange(ntemps)]

#create directory for results
frame_dir = create_directories(timestr)

if visual:
    plt.ion()
    plt.figure(figsize=(15,5))
# sampling loop
for j in xrange(nsamp):
    chi2_all = np.zeros(ntemps)
    print 'Loop', j
    sf = 0
    if j%(nsamp/2)==0:
        sf = 1
        c+=1
    temptemp = max(50 - 0.1*j, 1)
    for k in xrange(ntemps):
        _, chi2_all[k], statarrays = models[k].run_sampler(temptemp, visual=(k==0)*visual, multiband=multiband, savefig=sf)

    for k in xrange(ntemps-1, 0, -1):
        logfac = (chi2_all[k-1] - chi2_all[k]) * (1./temps[k-1] - 1./temps[k]) / 2.
        if np.log(np.random.uniform()) < logfac:
            print 'swapped', k-1, k
            models[k-1], models[k] = models[k], models[k-1]

    nsample[j] = models[0].n
    xsample[j,:] = models[0].stars[Model._X, :]
    ysample[j,:] = models[0].stars[Model._Y, :]
    
    if multiband:
        fsample.append(models[0].stars[Model._F:,:])

        csample = np.log10(models[0].stars[Model._F,:]/models[0].stars[Model._F+1,:])
        csample = np.array([value for value in csample if not math.isnan(value)])
        colorsample.append(csample)
    else:
        fsample[j,:] = models[0].stars[Model._F, :]

    chi2sample[j] = chi2_all[0]
    timestats[j,:] = statarrays

results(nsample, fsample, truef, colorsample, nsamp)

print 'saving...'
np.savez(directory_path + '/' + str(timestr) + '/chain.npz', n=nsample, x=xsample, y=ysample, f=fsample, chi2=chi2sample, times=timestats)
