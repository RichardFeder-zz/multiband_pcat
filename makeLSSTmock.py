import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int
import os
from image_eval import image_model_eval, psf_poly_fit
from galaxy import retr_sers, retr_tranphon, to_moments

pixel_scale = 0.2 # arcsec
sigma_psf = 0.297 # arcsec, fwhm=0.7/2.355
sigma_in_pix = sigma_psf / pixel_scale
nc = 25 # PSF postage stamp size
nbin = 5 # upsampling

x = np.arange(-(nc/2), (nc/2)+1, 1/float(nbin))
r2 = x[:,None]*x[:,None] + x[None,:]*x[None,:]
psf = np.exp(-r2/(2.*sigma_in_pix*sigma_in_pix)) / (2*np.pi*sigma_in_pix*sigma_in_pix)
cf = psf_poly_fit(psf, nbin=nbin)
npar = cf.shape[0]

types = np.loadtxt('Data/phoSim_example.txt', skiprows=9, usecols=12, dtype='string')
stars = (types == 'point')
galxs = (types == 'sersic2d')
mags = np.loadtxt('Data/phoSim_example.txt', skiprows=9, usecols=4, dtype=np.float32)
star_mags = mags[stars]
galx_mags = mags[galxs]
rads = np.loadtxt('Data/phoSim_example.txt', skiprows=9, usecols=13, dtype='string')
galx_rads = rads[galxs].astype(np.float32)

#np.random.seed(20171005) # set seed to always get same catalogue
imsz = (100, 100) # image size width, height
nstar = int(np.sum(stars) / float(4000*4000) * imsz[0]*imsz[1])
truex = (np.random.uniform(size=nstar)*(imsz[0]-1)).astype(np.float32)
truey = (np.random.uniform(size=nstar)*(imsz[1]-1)).astype(np.float32)
cmags = np.random.choice(star_mags, size=nstar)
print np.min(cmags), np.max(cmags), 'stars', nstar
truef = (10**(0.4*(25-cmags)) * 336*250).astype(np.float32) # notebook II 23

trueback = np.float32(445*250) # notebook II 23
gain = np.float32(1.) # deal in photoelectrons

ngalx = int(np.sum(galxs) / float(4000*4000) * imsz[0]*imsz[1])
truexg = (np.random.uniform(size=ngalx)*(imsz[0]-1)).astype(np.float32)
trueyg = (np.random.uniform(size=ngalx)*(imsz[1]-1)).astype(np.float32)
rg = (np.random.choice(galx_rads, size=ngalx) / pixel_scale).astype(np.float32)
print 'rad (pix)', np.min(rg), np.max(rg)
ug = np.random.uniform(low=3e-4, high=1., size=ngalx).astype(np.float32) #3e-4 for numerics
thetag = np.arccos(ug).astype(np.float32)
phig = (np.random.uniform(size=ngalx)*np.pi - np.pi/2.).astype(np.float32)
truexxg, truexyg, trueyyg = to_moments(rg, thetag, phig)
assert (truexxg*trueyyg > truexyg*truexyg).all()
cmags = np.random.choice(galx_mags, size=ngalx)
print np.min(cmags), np.max(cmags), 'galxs', ngalx
truefg = (10**(0.4*(25-cmags)) * 336*250).astype(np.float32) # notebook II 23
gridphon, amplphon = retr_sers(sersindx=2.)
xphon, yphon, fphon = retr_tranphon(gridphon, amplphon, truexg, trueyg, truefg, truexxg, truexyg, trueyyg)

if os.path.getmtime('pcat-lion.c') > os.path.getmtime('pcat-lion.so'):
    warnings.warn('pcat-lion.c modified after compiled pcat-lion.so', Warning)

array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
array_2d_double = npct.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
libmmult = npct.load_library('pcat-lion', '.')
libmmult.pcat_model_eval.restype = None
libmmult.pcat_model_eval.argtypes = [c_int, c_int, c_int, c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_1d_int, array_1d_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]

noise = np.random.normal(size=(imsz[1],imsz[0])).astype(np.float32)
mock = image_model_eval(np.concatenate([truex,xphon]), np.concatenate([truey,yphon]), np.concatenate([truef,fphon]), trueback, imsz, nc, cf, lib=libmmult.pcat_model_eval)
mock[mock < 1] = 1. # maybe some negative pixels
variance = mock / gain
oldmock = mock.copy() 
mock += (np.sqrt(variance) * np.random.normal(size=(imsz[1],imsz[0]))).astype(np.float32)

diff = mock - oldmock
print np.sum(diff*diff/variance)

f = open('Data/mockL_pix.txt', 'w')
f.write('%1d\t%1d\t1\n0.\t%0.3f' % (imsz[0], imsz[1], gain))
f.close()

np.savetxt('Data/mockL_cts.txt', mock)

np.savetxt('Data/mockL_psf.txt', psf, header='%1d\t%1d' % (nc, nbin), comments='')

truth = np.array([truex, truey, truef]).T
np.savetxt('Data/mockL_str.txt', truth)
truth = np.array([truexg, trueyg, truefg, truexxg, truexyg, trueyyg]).T
np.savetxt('Data/mockL_gal.txt', truth)
