from astropy.io import fits
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import astropy
import math
import astropy.io.fits
import astropy.wcs
from matplotlib.colors import LogNorm 
from scipy.optimize import curve_fit
import pandas as pd
from copy import deepcopy
# from __future__ import division
import scipy.signal
import pyfits as pyf
from matplotlib import rc


offsets = np.array([0.5,0.75, 1.0, 1.5], dtype=np.float32)
flux_ratios = np.array([1.0, 2.0, 5.0], dtype=np.float32)
r_fluxes = np.array([250.0, 500.0, 1000.0], dtype=np.float32)
imsz = 16
nsamp = 100
pdf_or_png = 'png'
ncases = 3
cases = ['r+i+g', 'r', 'rx3']


mock_test_name = 'mock_2star_16'
directory_path = "/Users/richardfeder/Documents/multiband_pcat/pcat-lion-master"

def find_offsets_errs(f, rat, plist, offsets, num, case):
	sublist = [p for p in plist if p[1]==f and p[2]==rat and p[3]==case]
	offs, errs = [], []
	for offset in offsets:
		entry = [p for p in sublist if p[0]==offset]
		if len(entry)>0:
			offs.append(offset)
			errs.append(entry[0][num])
		else:
			offs.append(np.nan)
			errs.append(np.nan)
	return offs, errs


def find_min_ds(xs, ys, fs, source, case, flux_ratios, r_fluxes, offsets, a, b, c):
	abs_nonz_dx = np.abs(xs-source[0])
	abs_nonz_dy = np.abs(ys-source[1])
	ds = np.square(abs_nonz_dx)+np.square(abs_nonz_dy)
	min_ds_arg = np.argmin(ds)
	minds = np.sqrt(ds[min_ds_arg]) # if ds within 2 offset separations of source, associate it, otherwise don't
	fminds = fs[min_ds_arg]
	if minds < 2*offsets[a]:
		if case==2:
			frac_f = np.abs(fminds-3*r_fluxes[b]*flux_ratios[c])/(3*r_fluxes[b]*flux_ratios[c])         	# need to calculate this more carefully
		else:
			frac_f = np.abs(fminds-r_fluxes[b]*flux_ratios[c])/(r_fluxes[b]*flux_ratios[c])
		return minds, frac_f, min_ds_arg
	else:
		return [0, 0, 0]

def load_arrays(a, b, c):
	dataname = mock_test_name+'-' + str(offsets[a])+'-'+str(r_fluxes[b])+'-'+str(flux_ratios[c])

	chain_types = ['chain3', 'chain1', 'chain1x3']

	all_x, all_y, all_f, ns = [[] for x in xrange(4)]

	for chain in chain_types:
		if chain == 'chain1x3':
			dataname = mock_test_name+'-' + str(offsets[a])+'-'+str(3.*r_fluxes[b])+'-'+str(flux_ratios[c])

		p = np.load(directory_path+'/Data/'+mock_test_name+'/'+dataname+'/results/'+chain+'.npz')
		all_x.append(p['x'][-nsamp:])
		all_y.append(p['y'][-nsamp:])
		all_f.append(p['f'][0,-nsamp:])
		ns.append(p['n'][-nsamp:])

	return all_x, all_y, all_f, ns



def flux_position_errors(imsz, nsamp):
	pos_error_list = [] #offset, r_flux, flux_ratio, mean_flux_err, mean_flux_err1, mean_flux_err3
	for a in xrange(len(offsets)):
		for b in xrange(len(r_fluxes)):
			for c in xrange(len(flux_ratios)):

				all_x, all_y, all_f, ns = load_arrays(a,b,c)
				
				src1 = [int(imsz/2), int(imsz/2)] # true positions of source 1
				src2 = [int(imsz/2)+offsets[a], int(imsz/2)] # true positions of source 2
				
				pos_errs1, pos_errs2, ferrs1, ferrs2 = [[] for x in xrange(4)]
				
				for case in xrange(len(cases)):
					
					dss1, dss2, dfs1, dfs2 = [[] for x in xrange(4)] # for each case (r, r+i+g, rx3), go through each sample and do source association
					
					for s in xrange(len(all_x[0])):

						# in both cases, we first associate to the brighter of the two sources (this will be source 2 by design), 
						# given a specific criterion, then we associate to source 1
						
						if ns[case][s]>1: # two sources or more in given sample

							nonz_x = all_x[case][s][np.nonzero(all_x[case][s])]
							nonz_y = all_y[case][s][np.nonzero(all_x[case][s])]
							nonz_f = all_f[case][s][np.nonzero(all_x[case][s])]
							minds2, frac_f2, min_ds_arg2 = find_min_ds(nonz_x, nonz_y, nonz_f, src2, case, flux_ratios, r_fluxes, offsets, a, b, c)
							
							# if a source is associated, then log its position/flux errors and remove it from sample
							if minds2 > 0:
								dss2.append(minds2) # dss2[case].append(minds2)
								dfs2.append(frac_f2)
								nonz_x = np.delete(nonz_x, min_ds_arg2) #remove first sample and repeat
								nonz_y = np.delete(nonz_y, min_ds_arg2)
								nonz_f = np.delete(nonz_f, min_ds_arg2)

							# if frac_f2 > 0:
							# 	dfs2.append(frac_f2)
							# nonz_x = np.delete(nonz_x, min_ds_arg2) #remove first sample and repeat
							# nonz_y = np.delete(nonz_y, min_ds_arg2)
							# nonz_f = np.delete(nonz_f, min_ds_arg2)
							minds1, frac_f1, min_ds_arg1 = find_min_ds(nonz_x, nonz_y, nonz_f, src1, case, flux_ratios, r_fluxes, offsets, a, b, c)
							if minds1 > 0:
								dss1.append(minds1)
							if frac_f1 > 0:
								dfs1.append(frac_f1)

						elif ns[case][s]==1: #if only one source in the sample
							nonz_x = all_x[case][s][0]
							nonz_y = all_y[case][s][0]
							nonz_f = all_f[case][s][0]
							ds2 = np.sqrt(np.square(nonz_x-src2[0])+np.square(nonz_y-src2[1]))
							if case==2:
								frac_f = np.abs(nonz_f-3*r_fluxes[b]*flux_ratios[c])/(3*r_fluxes[b]*flux_ratios[c])
							else:
								frac_f = np.abs(nonz_f-r_fluxes[b]*flux_ratios[c])/(r_fluxes[b]*flux_ratios[c])
							
							if flux_ratios[c] > 1: #give it directly to brighter source
								dss2.append(ds2)
								dfs2.append(frac_f)
							else:
								ds1 = np.sqrt(np.square(nonz_x-src1[0])+np.square(nonz_y-src1[1]))
								weights = [1/ds1**2, 1/ds2**2]
								weights /= np.sum(weights)
								choose_source = np.random.choice(weights.size, p=weights)
								if choose_source==0:
									dss1.append(ds1)
									dfs1.append(frac_f)
								else:
									dss2.append(ds2)
									dfs2.append(frac_f)
					pos_error_list.append([offsets[a], r_fluxes[b], flux_ratios[c], case, np.mean(dss1), np.mean(dss2), np.mean(dfs1), np.mean(dfs2)])
	return pos_error_list    




plist = flux_position_errors(imsz, nsamp)

for source in xrange(2): # calculate errors for each source
	# POSITION DEVIATION 
	c=1
	plt.figure(figsize=(10,10), dpi=200)  
	for flux in r_fluxes:
		for ratio in flux_ratios:
			plt.subplot(3,3,c)
			plt.title('$f_1$ = '+str(flux)+', $f_2/f_1$ = '+str(ratio))
			for case in xrange(ncases):
				off, err = find_offsets_errs(flux, ratio, plist, offsets, 4+source, case)
				plt.plot(off, err)
				plt.scatter(off, err, label=cases[case])
			if c==3:
				plt.legend(loc=1)
			if c%3==1:
				plt.ylabel('Position Error (pixels)')
			if c > 6:
				plt.xlabel('Source Separation (pixels)')
			plt.ylim(0, 0.8)
			c+=1
	plt.tight_layout()
	plt.savefig(directory_path+'/Data/'+mock_test_name+'/position_error_src'+str(source+1)+'_py.'+pdf_or_png, bbox_inches='tight')
	# FLUX DEVIATION
	c=1
	plt.figure(figsize=(10,10), dpi=200)  
	for flux in r_fluxes:
		for ratio in flux_ratios:
			plt.subplot(3,3,c)
			plt.title('$f_1$ = '+str(flux)+', $f_2/f_1$ = '+str(ratio))
			for case in xrange(ncases):
				off, err = find_offsets_errs(flux, ratio, plist, offsets, 6+source, case)
				plt.plot(off, err)
				plt.scatter(off, err, label=cases[case])
			if c==3:
				plt.legend(loc=1)
			if c%3==1:
				plt.ylabel('Fractional Flux Error')
			if c > 6:
				plt.xlabel('Source Separation (pixels)')
			plt.ylim(0, 1.2)
			c+=1
	plt.tight_layout()
	plt.savefig(directory_path+'/Data/'+mock_test_name+'/flux_error_src'+str(source+1)+'_py.'+pdf_or_png, bbox_inches='tight')


# # PREVALENCE PLOTS 

nstar_vals = []
tol = 3
for a in xrange(len(offsets)):
	xmin = (imsz/2)-tol
	ymin = (imsz/2)-tol
	xmax = (imsz/2)+offsets[a]+tol
	ymax = (imsz/2)+tol
	for b in xrange(len(r_fluxes)):
		for c in xrange(len(flux_ratios)):
			all_x, all_y, all_f, ns = load_arrays(a,b,c)
			for case in xrange(ncases):
				nstar = []
				for n in xrange(len(ns[case])):
					if np.any(all_x[case][n]>xmin) and np.any(all_x[case][n])<xmax and np.any(all_y[case][n]>ymin) and np.any(all_y[case][n]<ymax):
						nstar.append(ns[case][n])

				onestar_prevalence = float(len([x for x in nstar if x==1]))/float(len(nstar))
				twostar_prevalence = float(len([x for x in nstar if x==2]))/float(len(nstar))
				morestar_prevalence = float(len([x for x in nstar if x>2]))/float(len(nstar))
				nstar_vals.append([offsets[a], r_fluxes[b], flux_ratios[c], case, twostar_prevalence, onestar_prevalence, morestar_prevalence])


for flux in r_fluxes:
	c = 1
	plt.figure(figsize=(10,10), dpi=200) 
	for ratio in flux_ratios:
		for numcase in xrange(3):
			plt.subplot(3,3,c)
			plt.title('($f_1$ = '+str(flux)+', $f_2/f_1$ = ' +str(ratio)+')')
			plt.ylim(-0.1,1.1)
			for case in xrange(ncases):
				off, prev = find_offsets_errs(flux, ratio, nstar_vals, offsets, 4+numcase, case) 
				plt.plot(off, prev)
				plt.scatter(off, prev, label=cases[case])
			if c==3:
				plt.legend(loc=1)
			if c%3 ==1:
				plt.ylabel('2-Source Prevalence')
			elif c%3 ==2:
				plt.ylabel('1-Source Prevalence')
				plt.tick_params(labelleft=False)
			elif c%3 == 0:
				plt.ylabel('> 2-Source Prevalence')
				plt.tick_params(labelleft=False)
			if c > 6:
				plt.xlabel('Source Separation (pixels)')
			else:
				plt.tick_params(labelbottom=False)
			c += 1
	plt.tight_layout()
	plt.savefig(directory_path+'/Data/'+mock_test_name+'/prevalence_panels_'+str(flux)+'_py.'+pdf_or_png, bbox_inches='tight')
