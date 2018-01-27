import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int
import os
from image_eval import image_model_eval, psf_poly_fit
from galaxy import retr_sers, retr_tranphon, to_moments

f = open('Data/sdss.0921_psf.txt')
nc, nbin = [np.int32(i) for i in f.readline().split()]
f.close()
psf = np.loadtxt('Data/sdss.0921_psf.txt', skiprows=1).astype(np.float32)
cf = psf_poly_fit(psf, nbin=nbin)
npar = cf.shape[0]

np.random.seed(20170501) # set seed to always get same catalogue
imsz = (100, 100) # image size width, height
nstar = 60
truex = (np.random.uniform(size=nstar)*(imsz[0]-1)).astype(np.float32)
truey = (np.random.uniform(size=nstar)*(imsz[1]-1)).astype(np.float32)
truealpha = np.float32(2.0)
trueminf = np.float32(250.)
truelogf = np.random.exponential(scale=1./(truealpha-1.), size=nstar).astype(np.float32)
truef = trueminf * np.exp(truelogf)
trueback = np.float32(179.)
gain = np.float32(4.62)
ngalx = 60
truexg = (np.random.uniform(size=ngalx)*(imsz[0]-1)).astype(np.float32)
trueyg = (np.random.uniform(size=ngalx)*(imsz[1]-1)).astype(np.float32)
truelogf = np.random.exponential(scale=1./(truealpha-1.), size=ngalx).astype(np.float32)
truermin_g = np.float32(1.00)
truealpha_g = np.float32(4.00)
rg = truermin_g*np.exp(np.random.exponential(scale=1./(truealpha_g-1.),size=ngalx)).astype(np.float32)
#mask = rg > 100 # clip large galaxies
#rg[mask] /= 100.
#mask = rg > 10
#rg[mask] /= 10.
ug = np.random.uniform(low=3e-4, high=1., size=ngalx).astype(np.float32) #3e-4 for numerics
thetag = np.arccos(ug).astype(np.float32)
phig = (np.random.uniform(size=ngalx)*np.pi - np.pi/2.).astype(np.float32)
truexxg, truexyg, trueyyg = to_moments(rg, thetag, phig)
assert (truexxg*trueyyg > truexyg*truexyg).all()
truefg = trueminf * np.exp(truelogf)
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

f = open('Data/mockg4_pix.txt', 'w')
f.write('%1d\t%1d\t1\n0.\t%0.3f' % (imsz[0], imsz[1], gain))
f.close()

np.savetxt('Data/mockg4_cts.txt', mock)

np.savetxt('Data/mockg4_psf.txt', psf, header='%1d\t%1d' % (nc, nbin), comments='')

truth = np.array([truex, truey, truef]).T
np.savetxt('Data/mockg4_str.txt', truth)
truth = np.array([truexg, trueyg, truefg, truexxg, truexyg, trueyyg]).T
np.savetxt('Data/mockg4_gal.txt', truth)
