import numpy as np
from image_eval import image_model_eval, psf_poly_fit
import os

f = open('Data/sdss.0921/sdss.0921_psf.txt')
nc, nbin = [np.int32(i) for i in f.readline().split()]
f.close()
psf = np.loadtxt('Data/sdss.0921/sdss.0921_psf.txt', skiprows=1).astype(np.float32)
cf = psf_poly_fit(psf, nbin=nbin)
npar = cf.shape[0]

np.random.seed(20170501) # set seed to always get same catalogue
imsz = (20, 20) # image size width, height
nstar = imsz[0]**2 / 4
truex = (np.random.uniform(size=nstar)*(imsz[0]-1)).astype(np.float32)
truey = (np.random.uniform(size=nstar)*(imsz[1]-1)).astype(np.float32)
truealpha = np.float32(2.0)
trueminf = np.float32(50.)
truelogf = np.random.exponential(scale=1./(truealpha-1.), size=nstar).astype(np.float32)
truef = trueminf * np.exp(truelogf)

#for i band flux calculation
color_mus = [1, -0.5]
color_sigs = [0.3, 0.2]

r_i = np.random.normal(loc=color_mus[0], scale=color_sigs[0], size=nstar)
r_g = np.random.normal(loc=color_mus[1], scale=color_sigs[1], size=nstar)
truefi = truef*(10**(0.4*r_i))
truefg = truef*(10**(0.4*r_g))

truebackr = np.float32(179.)
truebacki = np.float32(310.)
truebackg = np.float32(115.)

nmgy_to_cts = 0.00546689
gain = np.float32(4.62)
noise = np.random.normal(size=(imsz[1],imsz[0])).astype(np.float32)

dir_name = 'mock' + str(imsz[0])

if not os.path.isdir('Data/' + dir_name):
	os.makedirs('Data/' + dir_name)


mockr = image_model_eval(truex, truey, truef, truebackr, imsz, nc, cf)
mockr[mockr < 1] = 1. # maybe some negative pixels
variance = mockr / gain
mockr += (np.sqrt(variance) * np.random.normal(size=(imsz[1],imsz[0]))).astype(np.float32)

mocki = image_model_eval(truex, truey, truefi, truebacki, imsz, nc, cf)
mocki[mocki < 1] = 1. # maybe some negative pixels
variance = mocki / gain
mocki += (np.sqrt(variance) * np.random.normal(size=(imsz[1],imsz[0]))).astype(np.float32)

mockg = image_model_eval(truex, truey, truefg, truebackg, imsz, nc, cf)
mockg[mockg < 1] = 1. # maybe some negative pixels
variance = mocki / gain
mockg += (np.sqrt(variance) * np.random.normal(size=(imsz[1],imsz[0]))).astype(np.float32)

f = open('Data/' + dir_name + '/' + dir_name+'-pixr.txt', 'w')
f.write('%1d\t%1d\t1\n0.\t%0.3f\n%0.8f\t%1d' % (imsz[0], imsz[1], gain, nmgy_to_cts, truebackr))
f.close()

f = open('Data/' + dir_name + '/' + dir_name+'-pixi.txt', 'w')
f.write('%1d\t%1d\t1\n0.\t%0.3f\n%0.8f\t%1d' % (imsz[0], imsz[1], gain, nmgy_to_cts, truebacki))
f.close()

f = open('Data/' + dir_name + '/' + dir_name+'-pixg.txt', 'w')
f.write('%1d\t%1d\t1\n0.\t%0.3f\n%0.8f\t%1d' % (imsz[0], imsz[1], gain, nmgy_to_cts, truebackg))
f.close()

np.savetxt('Data/' + dir_name + '/' + dir_name+'-ctsr.txt', mockr)
np.savetxt('Data/' + dir_name + '/' + dir_name+'-ctsi.txt', mocki)
np.savetxt('Data/' + dir_name + '/' + dir_name+'-ctsg.txt', mockg)

np.savetxt('Data/' + dir_name + '/' + dir_name+'-psfr.txt', psf, header='%1d\t%1d' % (nc, nbin), comments='')
np.savetxt('Data/' + dir_name + '/' + dir_name+'-psfi.txt', psf, header='%1d\t%1d' % (nc, nbin), comments='')
np.savetxt('Data/' + dir_name + '/' + dir_name+'-psfg.txt', psf, header='%1d\t%1d' % (nc, nbin), comments='')


truth = np.array([truex, truey, truef, truefi, truefg]).T
np.savetxt('Data/' + dir_name + '/' + dir_name+'-tru.txt', truth)
