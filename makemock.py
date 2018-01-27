import numpy as np
from image_eval import image_model_eval, psf_poly_fit

f = open('Data/sdss.0921_psf.txt')
nc, nbin = [np.int32(i) for i in f.readline().split()]
f.close()
psf = np.loadtxt('Data/sdss.0921_psf.txt', skiprows=1).astype(np.float32)
cf = psf_poly_fit(psf, nbin=nbin)
npar = cf.shape[0]

np.random.seed(20170501) # set seed to always get same catalogue
imsz = (300, 300) # image size width, height
nstar = 9000
truex = (np.random.uniform(size=nstar)*(imsz[0]-1)).astype(np.float32)
truey = (np.random.uniform(size=nstar)*(imsz[1]-1)).astype(np.float32)
truealpha = np.float32(2.0)
trueminf = np.float32(250.)
truelogf = np.random.exponential(scale=1./(truealpha-1.), size=nstar).astype(np.float32)
truef = trueminf * np.exp(truelogf)
trueback = np.float32(179.)
gain = np.float32(4.62)

noise = np.random.normal(size=(imsz[1],imsz[0])).astype(np.float32)
mock = image_model_eval(truex, truey, truef, trueback, imsz, nc, cf)
mock[mock < 1] = 1. # maybe some negative pixels
variance = mock / gain
mock += (np.sqrt(variance) * np.random.normal(size=(imsz[1],imsz[0]))).astype(np.float32)

f = open('Data/mock300_pix.txt', 'w')
f.write('%1d\t%1d\t1\n0.\t%0.3f' % (imsz[0], imsz[1], gain))
f.close()

np.savetxt('Data/mock300_cts.txt', mock)

np.savetxt('Data/mock300_psf.txt', psf, header='%1d\t%1d' % (nc, nbin), comments='')

truth = np.array([truex, truey, truef]).T
np.savetxt('Data/mock300_tru.txt', truth)
