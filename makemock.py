import numpy as np
from image_eval import image_model_eval, psf_poly_fit
import os

n_realizations = 3
two_star_blend = 0
two_star_mode = 'rx3' # r, r_i_g, rx3 are the three modes
dim = 100
bands = ['r', 'i', 'g']

f = open('Data/sdss.0921/sdss.0921_psf.txt')
nc, nbin = [np.int32(i) for i in f.readline().split()]
f.close()
psf = np.loadtxt('Data/sdss.0921/sdss.0921_psf.txt', skiprows=1).astype(np.float32)
cf = psf_poly_fit(psf, nbin=nbin)
npar = cf.shape[0]

np.random.seed(20170501) # set seed to always get same catalogue
imsz = (dim, dim) # image size width, height


truebacks = [np.float32(179.) for x in xrange(len(bands))] #assume same background
nmgy_to_cts = 0.00546689
gain = np.float32(4.62)

# for testing two source separations
if two_star_blend:
	dir_name = 'mock_2star_' + str(imsz[0])
	nstar = 2
	offsets = np.array([0.5, 0.75, 1.0, 1.5], dtype=np.float32)
	flux_ratios = np.array([1.0, 2.0, 5.0], dtype=np.float32)
	r_i_colors = [0.3, 0.0]
	r_g_colors = [-1.0, 0.0]
	r_fluxes = np.array([250., 500., 1000.], dtype=np.float32)

else:
	dir_name = 'mock' + str(imsz[0])

if not os.path.isdir('Data/' + dir_name):
	os.makedirs('Data/' + dir_name)


# generate specified number of noise realizations
noise_realizations = []
for n in xrange(n_realizations):
	noise_realizations.append(np.random.normal(size=(imsz[1],imsz[0])).astype(np.float32))


back_pix = truebacks
if two_star_mode=='rx3':
		back_pix[0] *= 3



for b in xrange(len(bands)):
	# pix
	f = open('Data/'+dir_name+'/'+dir_name+'-pix'+bands[b]+'.txt', 'w')
	f.write('%1d\t%1d\t1\n0.\t%0.3f\n%0.8f\t%1d' % (imsz[0], imsz[1], gain, nmgy_to_cts, back_pix[b]))
	f.close()
	# psf
	np.savetxt('Data/'+dir_name+'/'+dir_name+'-psf'+bands[b]+'.txt', psf, header='%1d\t%1d' % (nc, nbin), comments='')


if two_star_blend:
	for offset in offsets:
		truex = np.zeros(nstar, dtype=np.float32)
		truey = np.zeros(nstar, dtype=np.float32)
		truer = np.zeros(nstar, dtype=np.float32)
		truei = np.zeros(nstar, dtype=np.float32)
		trueg = np.zeros(nstar, dtype=np.float32)

		truex = [imsz[0]/2, imsz[0]/2 + offset]
		truey = [imsz[1]/2, imsz[1]/2]

		# ----------------- ITERATE THROUGH FLUXES ---------------------
		
		true_fluxes = [[],[],[]]

		for flux in r_fluxes:
			truer[0] = flux
			if two_star_mode=='rx3':
				truer[0] *= (1+10**(0.4*r_i_colors[0])+10**(-0.4*r_g_colors[0]))
			else:
				truei[0] = flux*10**(0.4*r_i_colors[0])
				trueg[0] = flux*10**(-0.4*r_g_colors[0])

			# ------------------- ITERATE THROUGH FLUX RATIOS -----------------------

			for flux_ratio in flux_ratios:
				# if two_star_mode=='rx3':
				# 	subdir = 'Data/' + dir_name + '/' + dir_name + '-' + str(offset)+'-'+str(flux*3)+'-'+str(flux_ratio)
				# else:

				truer[1] = flux*flux_ratio*(1+10**(0.4*r_i_colors[1])+10**(-0.4*r_g_colors[1]))
				truei[1] = flux*flux_ratio*10**(0.4*r_i_colors[1])
				trueg[1] = flux*flux_ratio*10**(-0.4*r_g_colors[1])

				true_fluxes = [truer, truei, trueg]

				mock_counts = []

				f = flux
				if two_star_mode=='rx3':
					f *=3
				subdir = 'Data/' + dir_name + '/' + dir_name + '-' + str(offset)+'-'+str(f)+'-'+str(flux_ratio)
				if not os.path.isdir(subdir):
					os.makedirs(subdir)

				for b in xrange(len(bands)):
					mock = image_model_eval(np.array(truex, dtype=np.float32), np.array(truey, dtype=np.float32), np.array(true_fluxes[b],dtype=np.float32), truebacks[b], imsz, nc, cf)
					mock[mock < 1] = 1. # maybe some negative pixels
					variance = mock / gain

					for n in xrange(n_realizations):
						mock += (np.sqrt(variance)*noise_realizations[n])

						np.savetxt(subdir+'/'+dir_name+'-'+str(offset)+'-'+str(f)+'-'+str(flux_ratio)+'-nr'+str(n+1)+'-cts'+bands[b]+'.txt', mock)
				
				truth = np.array([np.array(truex), np.array(truey), np.array(truer), np.array(truei), np.array(trueg)], dtype=np.float32).T
				np.savetxt(subdir + '/'+dir_name+'-'+str(offset)+'-'+str(f)+'-'+str(flux_ratio)+'-tru.txt', truth)

# for standard nstar mock data
else:
	n_second_pop = 0
	nstar = dim*dim/4
	color_mus = [1, -0.5]
	color_sigs = [0.3, 0.2]
	truealpha = np.float32(2.0)
	trueminf = np.float32(250.)
	truelogf = np.random.exponential(scale=1./(truealpha-1.), size=nstar).astype(np.float32)

	truebackr = np.float32(179.)
	truebacki = np.float32(310.)
	truebackg = np.float32(115.)
	truebacks = [truebackr, truebacki, truebackg]

	true_params = np.zeros(shape=(nstar, 2+len(bands)),dtype=np.float32)
	true_params[:,0] = np.random.uniform(size=nstar)*(imsz[0]-1) # x coordinate
	true_params[:,1] = np.random.uniform(size=nstar)*(imsz[1]-1) # y coordinate
	true_params[:,2] = trueminf * np.exp(truelogf) # r band flux

	# add other fluxes as needed
	for b in xrange(len(bands)-1):
		colors = np.random.normal(loc=color_mus[b], scale=color_sigs[b], size=nstar-n_second_pop)
		colors = np.append(colors, np.random.normal(loc=-0.5, scale=0.1, size=n_second_pop))
		true_params[:,3+b] = true_params[:,2]*(10**(0.4*colors))


	dir_name = 'mock' + str(imsz[0])
	if not os.path.isdir('Data/' + dir_name):
		os.makedirs('Data/' + dir_name)

	for b in xrange(len(bands)):
		mock = image_model_eval(true_params[:,0], true_params[:,1], true_params[:,2+b], truebacks[b], imsz, nc, cf)
		mock[mock < 1] = 1.
		variance = mock / gain
		mock += (np.sqrt(variance) * np.random.normal(size=(imsz[1],imsz[0]))).astype(np.float32)
		np.savetxt('Data/'+dir_name+'/'+dir_name+'-cts'+str(bands[b])+'.txt', mock)

		np.savetxt('Data/' + dir_name + '/' + dir_name+'-psf'+str(bands[b])+'.txt', psf, header='%1d\t%1d' % (nc, nbin), comments='')

		f = open('Data/' + dir_name + '/' + dir_name+'-pix'+str(bands[b])+'.txt', 'w')
		f.write('%1d\t%1d\t1\n0.\t%0.3f\n%0.8f\t%1d' % (imsz[0], imsz[1], gain, nmgy_to_cts, truebacks[b]))
		f.close()

	np.savetxt('Data/'+dir_name+'/'+dir_name+'-tru.txt', true_params)

	# truth = np.array([truex, truey, truef, truefi, truefg]).T
	# np.savetxt('Data/' + dir_name + '/' + dir_name+'-tru.txt', truth)
