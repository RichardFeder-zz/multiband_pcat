import numpy as np
import numpy.ctypeslib as npct
import ctypes
from ctypes import c_int, c_double
# in order for visual=True to work, interactive backend should be loaded before importing pyplot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import sys
import os
import argparse
import warnings
import scipy.stats as stats
import random
from astropy.convolution import Gaussian2DKernel
import scipy.signal
from scipy.ndimage import gaussian_filter
from image_eval import psf_poly_fit, image_model_eval
from helpers import * 
from fast_astrom import *
# import cPickle as pickle
import pickle
np.seterr(divide='ignore', invalid='ignore')


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

#generate random seed for initialization
np.random.seed(20170502)

class gdatstrt(object):

	def __init__(self):
		pass
	
	def __setattr__(self, attr, valu):
		super(gdatstrt, self).__setattr__(attr, valu)


def save_params(dir, gdat):
	# save parameters as dictionary, then pickle them to txt file
	param_dict = vars(gdat)
	print('param_dict:')
	print(param_dict)
	
	with open(dir+'/params.txt', 'w') as file:
		# file.write(pickle.dumps(param_dict))
		file.write(str(param_dict))
	file.close()
	
	with open(dir+'/params_read.txt', 'w') as file2:
		for key in param_dict:
			file2.write(key+': '+str(param_dict[key])+'\n')
	file2.close()

def fluxes_to_color(flux1, flux2):
	return 2.5*np.log10(flux1/flux2)


def get_gaussian_psf_template(pixel_fwhm=3., nbin=5):
	nc = 25
	psfnew = Gaussian2DKernel((pixel_fwhm/2.355)*nbin, x_size=125, y_size=125).array.astype(np.float32)
	psfnew *= nc
	cf = psf_poly_fit(psfnew, nbin=nbin)
	return psfnew, cf, nc, nbin


def load_in_map(gdat, band=0, astrom=None):
	# file_path = gdat.base_path+'/Data/spire/'+gdat.dataname+'_sim200P'+band_dict[band]+'W_noise+sz.fits'
	# file_path = gdat.base_path+'/Data/spire/'+gdat.dataname+'_sim200P'+band_dict[band]+'W_noise.fits'
	# file_path = gdat.base_path+'/Data/spire/'+gdat.dataname+'_P'+band_dict[band]+'W_sim0200.fits'
	file_path = gdat.base_path+'/Data/spire/'+gdat.dataname+'_P'+gdat.band_dict[band]+'W_nr_1.fits'
	# file_path = gdat.base_path+'/Data/spire/'+gdat.dataname+'_P'+gdat.band_dict[band]+'W.fits'

	print('file_path:', file_path)

	if astrom is not None:
		print('loading from ', gdat.band_dict[band])
		astrom.load_wcs_header_and_dim(gdat.dataname+'_P'+gdat.band_dict[band]+'W_nr_1.fits', hdu_idx=3)
		# astrom.load_wcs_header_and_dim(gdat.dataname+'_P'+gdat.band_dict[band]+'W.fits', hdu_idx=3)

	spire_dat = fits.open(file_path)
	image = np.nan_to_num(spire_dat[1].data)
	error = np.nan_to_num(spire_dat[2].data)
	exposure = spire_dat[3].data
	mask = spire_dat[4].data

	return image, error, exposure, mask


def initialize_c(gdat, libmmult, cblas=False):

	if gdat.verbtype > 1:
		print('initializing c routines and data structs')

	if cblas:
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

	else:
		if os.path.getmtime('blas.c') > os.path.getmtime('blas.so'):
			warnings.warn('blas.c modified after compiled blas.so', Warning)		
		
		array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
		array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
		array_2d_double = npct.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
		libmmult.clib_eval_modl.restype = None
		libmmult.clib_eval_modl.argtypes = [c_int, c_int, c_int, c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_1d_int, array_1d_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]
		array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")
		libmmult.clib_updt_modl.restype = None
		libmmult.clib_updt_modl.argtypes = [c_int, c_int, array_2d_float, array_2d_float, array_2d_int, c_int, c_int, c_int, c_int]
		libmmult.clib_eval_llik.restype = None
		libmmult.clib_eval_llik.argtypes = [c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]


def create_directories(gdat):
	new_dir_name = gdat.result_path+'/'+gdat.timestr
	frame_dir_name = new_dir_name+'/frames'
	if not os.path.isdir(frame_dir_name):
		os.makedirs(frame_dir_name)
	return frame_dir_name, new_dir_name

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


def load_param_dict(timestr):
	
	result_path = '/Users/richardfeder/Documents/multiband_pcat/spire_results/'
	filepath = result_path + timestr
	filen = open(filepath+'/params.txt','r')
	print(filen)
	pdict = pickle.load(filen)
	print(pdict)
	opt = objectview(pdict)

	print('param dict load')
	return opt, filepath, result_path


def result_plots(timestr=None, burn_in_frac=0.5, boolplotsave=True, boolplotshow=False, plttype='pdf', gdat=None):

	
	band_dict = dict({0:'250 micron', 1:'350 micron', 2:'500 micron'})
	lam_dict = dict({0:250, 1:350, 2:500})

	
	if gdat is None:
		gdat, filepath, result_path = load_param_dict(timestr)
		gdat.burn_in_frac = burn_in_frac
		gdat.boolplotshow = boolplotshow
		gdat.boolplotsave = boolplotsave
		gdat.filepath = filepath
		gdat.result_path = result_path
		gdat.timestr = timestr
	else:
		gdat.result_path = '/Users/richardfeder/Documents/multiband_pcat/spire_results/'
		gdat.filepath = gdat.result_path + gdat.timestr
	# gdat.auto_resize=False


	dat = pcat_data()
	dat.load_in_data(gdat)

	chain = np.load(gdat.filepath+'/chain.npz')

	nsrcs = chain['n']
	xsrcs = chain['x']
	ysrcs = chain['y']
	fsrcs = chain['f']
	chi2 = chain['chi2']
	timestats = chain['times']
	accept_stats = chain['accept']
	diff2s = chain['diff2s']
	# residuals = chain['residuals0']
	burn_in = int(gdat.nsamp*burn_in_frac)
	bands = gdat.bands


	# ------------------- mean residual ---------------------------

	print(gdat.nbands)
	for b in range(gdat.nbands):

		residz = chain['residuals'+str(b)]

		median_resid = np.median(residz, axis=0)
		smoothed_resid = gaussian_filter(median_resid, sigma=3)

		minpct = np.percentile(median_resid[dat.weights[b] != 0.], 5.)
		maxpct = np.percentile(median_resid[dat.weights[b] != 0.], 95.)

		minpct_smooth = np.percentile(smoothed_resid[dat.weights[b] != 0.], 5.)
		maxpct_smooth = np.percentile(smoothed_resid[dat.weights[b] != 0.], 95.)


		plt.figure(figsize=(10, 5))
		plt.subplot(1,2,1)
		plt.title('Median Residual -- '+band_dict[bands[b]])
		# plt.imshow(median_resid, interpolation='none', cmap='Greys', vmin=-0.005, vmax=0.005)
		plt.imshow(median_resid, interpolation='none', cmap='Greys', vmin=minpct, vmax=maxpct)
		# plt.imshow(mean_resid, interpolation='none', cmap='Greys', vmin=np.percentile(mean_resid, 1), vmax=np.percentile(mean_resid, 99))
		plt.colorbar()
		# plt.scatter(xsrcs[-1], ysrcs[-1], s=1e2*fsrcs[-1], color='r', marker='x', label='Last Sample')
		plt.legend()
		plt.subplot(1,2,2)
		plt.title('Smoothed Residual')
		# plt.imshow(smoothed_resid, cmap='Greys', vmin=-0.004, vmax=0.0005)
		plt.imshow(smoothed_resid, cmap='Greys', vmin=minpct_smooth, vmax=maxpct_smooth)
		plt.colorbar()
		# plt.scatter(xsrcs[-1], ysrcs[-1], s=1e3*fsrcs[-1], color='r', marker='x', label='Last Sample', alpha=0.5)
		if boolplotsave:
			plt.savefig(gdat.filepath +'/median_residual_and_smoothed_band'+str(b)+'.'+plttype, bbox_inches='tight')
		if boolplotshow:
			plt.show()
		plt.close()

		median_resid_rav = median_resid[dat.weights[b] != 0.].ravel()


		plt.figure()
		plt.title('Median residual 1pt function -- '+band_dict[bands[b]])
		plt.hist(median_resid_rav, bins=np.linspace(-0.02, 0.02, 50))
		plt.xlabel('data - model (Jy)')
		if boolplotsave:
			plt.savefig(gdat.filepath +'/median_residual_1pt_function_band'+str(b)+'.'+plttype, bbox_inches='tight')
		if boolplotshow:
			plt.show()
		plt.close()

		median_resid = median_resid[30:median_resid.shape[1]-30,30:median_resid.shape[0]-30]

		plt.figure()
		plt.imshow(median_resid, vmin=-0.01, vmax=0.01, cmap='Greys', interpolation='none')
		plt.colorbar()
		if boolplotsave:
			plt.savefig(gdat.filepath +'/cropped_image.'+plttype, bbox_inches='tight')
		if boolplotshow:
			plt.show()
		plt.close()

	

	# -------------------- CHI2 ------------------------------------

	sample_number = np.arange(burn_in, gdat.nsamp)
	full_sample = range(gdat.nsamp)
	plt.figure()
	plt.title('Chi-Squared Distribution over Catalog Samples')
	for b in range(gdat.nbands):
		plt.plot(sample_number, chi2[burn_in:,b], label=band_dict[bands[b]])
		plt.axhline(np.min(chi2[burn_in:,b]), linestyle='dashed', alpha=0.5, label=str(np.min(chi2[burn_in:,b]))+' (' + str(band_dict[bands[b]]) + ')')
	plt.xlabel('Sample')
	plt.ylabel('Chi2')
	plt.legend()
	if boolplotsave:
		plt.savefig(gdat.filepath + '/chi2_sample.'+plttype, bbox_inches='tight')
	if boolplotshow:
		plt.show()
	plt.close()

	# ---------------------------- COMPUTATIONAL RESOURCES --------------------------------

	time_array = np.zeros(3, dtype=np.float32)
	labels = ['Proposal', 'Likelihood', 'Implement']
	print(timestats[0])
	for samp in range(gdat.nsamp):
		time_array += np.array([timestats[samp][2][0], timestats[samp][3][0], timestats[samp][4][0]])
	plt.figure()
	plt.title('Computational Resources')
	plt.pie(time_array, labels=labels, autopct='%1.1f%%', shadow=True)
	if boolplotsave:
		plt.savefig(gdat.filepath+ '/time_resource_statistics.'+plttype, bbox_inches='tight')
	if boolplotshow:
		plt.show()
	plt.close()


	# ------------------------------ ACCEPTANCE FRACTION -----------------------------------------

	proposal_types = ['All', 'Move', 'Birth/Death', 'Merge/Split']
	plt.figure()
	plt.title('Proposal Acceptance Fractions')
	for x in range(len(proposal_types)):
		if not np.isnan(accept_stats[0,x]):
			plt.plot(full_sample, accept_stats[:,x], label=proposal_types[x])
	plt.legend()
	plt.xlabel('Sample Number')
	plt.ylabel('Acceptance Fraction')
	if boolplotsave:
		plt.savefig(gdat.filepath+'/acceptance_fraction.'+plttype, bbox_inches='tight')
	if boolplotshow:
		plt.show()
	plt.close()



	# -------------------------------- ITERATE OVER BANDS -------------------------------------


	nsrc_fov = []
	color_post = []
	# color_post_bins = 10**np.linspace(-1, 2, 20)
	color_post_bins = np.linspace(-1, 1, 30)
	color_lin_post_bins = np.linspace(0.0, 5.0, 30)

	for b in range(gdat.nbands):

		color_post = []
		color_lin_post = []


		nbins = 20
		lit_number_counts = np.zeros((gdat.nsamp - burn_in, nbins-1)).astype(np.float32)
		raw_number_counts = np.zeros((gdat.nsamp - burn_in, nbins-1)).astype(np.float32)

		binz = np.linspace(np.log10(gdat.trueminf)+3.-1., 3., nbins)

		weight = dat.weights[b]
		
		for i, j in enumerate(np.arange(burn_in, gdat.nsamp)):

			if b > 0:
				color_post.append(np.histogram(np.log10(fsrcs[0][j]/fsrcs[0][j]), bins=color_post_bins)[0]+0.01)
				color_lin_post.append(np.histogram(fsrcs[b-1][j]/fsrcs[b][j], bins=color_lin_post_bins)[0]+0.01)

				
			fsrcs_in_fov = np.array([fsrcs[b][j][k] for k in range(nsrcs[j]) if dat.weights[0][int(xsrcs[j][k]),int(ysrcs[j][k])] != 0.])
			nsrc_fov.append(len(fsrcs_in_fov))

			hist = np.histogram(np.log10(fsrcs_in_fov)+3, bins=binz)

			# hist = np.histogram(np.log10(fsrcs[b][j])+3, bins=binz)
			logSv = 0.5*(hist[1][1:]+hist[1][:-1])-3
			binz_Sz = 10**(binz-3)
			dSz = binz_Sz[1:]-binz_Sz[:-1]
			dNdS = hist[0]
			raw_number_counts[i,:] = hist[0]
			n_steradian = 0.11/(180./np.pi)**2 # field covers 0.11 degrees, should change this though for different fields
			n_steradian *= gdat.frac # a number of pixels in the image are not actually observing anything
			dNdS_S_twop5 = dNdS*(10**(logSv))**(2.5)
			lit_number_counts[i,:] = dNdS_S_twop5/n_steradian/dSz

		mean = np.mean(lit_number_counts, axis=0)


		plt.figure()  
		plt.errorbar(logSv+3, np.mean(lit_number_counts, axis=0), yerr=np.array([np.abs(mean - np.percentile(lit_number_counts, 16, axis=0)), np.abs(np.percentile(lit_number_counts, 84, axis=0) - mean)]), marker='.')
		plt.yscale('log')
		plt.legend()
		plt.xlabel('log($S_{\\nu}$) (mJy)')
		plt.ylabel('dN/dS.$S^{2.5}$ ($Jy^{1.5}/sr$)')
		plt.ylim(1e0, 1e5)
		plt.xlim(np.log10(gdat.trueminf)+3.-0.5-1.0, 2.5)
		plt.tight_layout()
		if boolplotsave:
			plt.savefig(gdat.filepath+'/posterior_number_counts_histogram_'+str(band_dict[bands[b]])+'.'+plttype, bbox_inches='tight')
		if boolplotshow:
			plt.show()
		plt.close()

		mean = np.mean(raw_number_counts, axis=0)

		plt.title('Posterior Flux Distribution - ' + str(band_dict[bands[b]]))
		plt.errorbar(logSv+3, np.mean(raw_number_counts, axis=0), yerr=np.array([np.abs(mean-np.percentile(raw_number_counts, 16, axis=0)), np.abs(np.percentile(raw_number_counts, 84, axis=0)-mean)]), fmt='o', label='Posterior')
		plt.legend()
		plt.yscale('log', nonposy='clip')
		plt.xlabel('log10(Flux) - ' + str(band_dict[bands[b]]))
		if boolplotsave:
			plt.savefig(gdat.filepath+'/posterior_flux_histogram_'+str(band_dict[bands[b]])+'.'+plttype, bbox_inches='tight')
		if boolplotshow:
			plt.show()
		plt.close()


		if b > 0:
			medians = np.median(np.array(color_post), axis=0)
			medians /= (np.sum(medians)*(color_post_bins[1]-color_post_bins[0]))  
			lin_medians = np.median(np.array(color_lin_post), axis=0)
			lin_medians /= (np.sum(lin_medians)*(color_lin_post_bins[1]-color_lin_post_bins[0]))  
			err = np.std(np.array(color_post), axis=0)
			bincentres = [(color_post_bins[i]+color_post_bins[i+1])/2. for i in range(len(color_post_bins)-1)]
			lin_bincentres = [(color_lin_post_bins[i]+color_lin_post_bins[i+1])/2. for i in range(len(color_lin_post_bins)-1)]

			plt.title('Posterior Color Distribution', fontsize=14)
			# plt.step(bincentres, medians, where='mid', color='b', label='Posterior', alpha=0.5)
			plt.step(lin_bincentres, lin_medians, where='mid', color='b', label='Posterior', alpha=0.5)

			# plt.errorbar(bincentres, medians/np.sum(medians), yerr=err/np.sum(medians), label='Posterior')
			# plt.xscale('log')
			# finevals = np.linspace(-1, 1, 1000)
			# plt.plot(bincentres, stats.norm.pdf(bincentres, -0.8/2.5, 0.5/2.5), label='Prior')
			# plt.xlabel('$\\log_{10}(f_{'+str(lam_dict[bands[b]])+'}/f_{'+str(lam_dict[bands[0]])+'})$', fontsize=14)
			plt.xlabel('$S_{'+str(lam_dict[bands[b-1]])+'}/S_{'+str(lam_dict[bands[b]])+'}$', fontsize=14)

			plt.ylabel('Normalized PDF')
			plt.legend()
			if boolplotsave:
				plt.savefig(gdat.filepath +'/posterior_color_dist_'+str(lam_dict[bands[b-1]])+'_'+str(lam_dict[bands[b]])+'.'+plttype, bbox_inches='tight')
			if boolplotshow:
				plt.show()
			plt.close()



			# ------------------- SOURCE NUMBER ---------------------------

		plt.figure()
		plt.title('Posterior Source Number Histogram')
		plt.hist(nsrc_fov, histtype='step', label='Posterior', color='b', bins=15)
		plt.axvline(np.median(nsrc_fov), label='Median=' + str(np.median(nsrc_fov)), color='b', linestyle='dashed')
		plt.xlabel('nstar')
		plt.legend()
		if boolplotsave:
			plt.savefig(gdat.filepath +'/posterior_histogram_nstar.'+plttype, bbox_inches='tight')
		if boolplotshow:
			plt.show()
		plt.close()


   
class Proposal:
	_X = 0
	_Y = 1
	_F = 2

	def __init__(self, gdat):
		self.idx_move = None
		self.do_birth = False
		self.idx_kill = None
		self.factor = None
		self.goodmove = False
		self.dback = np.zeros(gdat.nbands, dtype=np.float32)
		self.xphon = np.array([], dtype=np.float32)
		self.yphon = np.array([], dtype=np.float32)
		self.fphon = []
		self.modl_eval_colors = []
		for x in range(gdat.nbands):
			self.fphon.append(np.array([], dtype=np.float32))
		self.gdat = gdat
	def set_factor(self, factor):
		self.factor = factor

	def in_bounds(self, catalogue):
		return np.logical_and(np.logical_and(catalogue[self._X,:] > 0, catalogue[self._X,:] < (self.gdat.imsz0[0] -1)), \
				np.logical_and(catalogue[self._Y,:] > 0, catalogue[self._Y,:] < self.gdat.imsz0[1] - 1))

	def assert_types(self):
		assert self.xphon.dtype == np.float32
		assert self.yphon.dtype == np.float32
		assert self.fphon[0].dtype == np.float32

	def __add_phonions_stars(self, stars, remove=False):
		fluxmult = -1 if remove else 1

		self.xphon = np.append(self.xphon, stars[self._X,:])
		self.yphon = np.append(self.yphon, stars[self._Y,:])

		for b in range(self.gdat.nbands):
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
	# trueminf = np.float32(236) # minimum flux for source in ADU, might need to change
	#truealpha = np.float32(2) # power law flux slope parameter, might need to change
	
	_X = 0
	_Y = 1
	_F = 2

	k =2.5/np.log(10)

	mus = dict({'S-M':-0.8, 'M-L':-0.8, 'L-S':1.9, 'M-S':0.8, 'S-L':1.9, 'L-M':0.8})
	#sigs = dict({'r-i':0.5, 'r-g':5.0, 'r-z':1.0, 'r-r':0.05})
	sigs = dict({'S-M':2.5, 'M-L':0.5, 'L-S':0.5, 'M-S':0.5, 'S-L':0.5, 'L-M':0.5}) #very broad color prior
	#mus = dict({'r-i':0.0, 'r-g':0.0}) # for same flux different noise tests, r, i, g are all different realizations of r band
	#sigs = dict({'r-i':0.5, 'r-g':0.5})
	color_mus, color_sigs = [], []
	
	''' the init function sets all of the data structures used for the catalog, 
	randomly initializes catalog source values drawing from catalog priors  '''
	def __init__(self, gdat, dat, libmmult=None):
		self.back = np.zeros(gdat.nbands, dtype=np.float32)
		self.err_f = gdat.err_f
		self.imsz0 = gdat.imsz0 # this is just for first band, where proposals are first made
		self.imszs = gdat.imszs # this is list of image sizes for all bands, not just first one
		self.kickrange = gdat.kickrange
		self.margins = np.array(gdat.margins).astype(np.int)
		self.max_nsrc = gdat.max_nsrc
		self.moveweights = np.array([80., 40., 40.])
		self.movetypes = ['P *', 'BD *', 'MS *']
		self.n = np.random.randint(gdat.max_nsrc)+1
		self.nbands = gdat.nbands
		self.nloop = gdat.nloop
		self.nregion = gdat.nregion
		self.penalty = 1+0.5*gdat.alph*gdat.nbands
		self.regsizes = np.array(gdat.regsizes).astype(np.int)
		self.stars = np.zeros((2+gdat.nbands,gdat.max_nsrc), dtype=np.float32)
		self.stars[:,0:self.n] = np.random.uniform(size=(2+gdat.nbands,self.n))
		self.stars[self._X,0:self.n] *= gdat.imsz0[0]-1
		self.stars[self._Y,0:self.n] *= gdat.imsz0[1]-1
		self.verbtype = gdat.verbtype
		self.nominal_nsrc = gdat.nominal_nsrc
		self.regions_factor = gdat.regions_factor
		self.truealpha = gdat.truealpha
		self.trueminf = gdat.trueminf
		self.bkg = np.array([gdat.bias for b in range(gdat.nbands)])
		self.gdat = gdat
		self.dat = dat
		self.libmmult = libmmult
		self.offsetxs = np.zeros(self.nbands).astype(np.int)
		self.offsetys = np.zeros(self.nbands).astype(np.int)
		self.margins = np.zeros(self.nbands).astype(np.int)

		for b in range(self.nbands-1):

			col_string = self.gdat.band_dict[self.gdat.bands[0]]+'-'+self.gdat.band_dict[self.gdat.bands[b+1]]
			print('col string is ', col_string)
			self.color_mus.append(self.mus[col_string])
			self.color_sigs.append(self.sigs[col_string])

		if gdat.load_state_timestr is None:
			for b in range(gdat.nbands):
				if b==0:
					self.stars[self._F+b,0:self.n] **= -1./(self.truealpha - 1.)
					self.stars[self._F+b,0:self.n] *= self.trueminf
				else:
					new_colors = np.random.normal(loc=self.color_mus[b-1], scale=self.color_sigs[b-1], size=self.n)
					self.stars[self._F+b,0:self.n] = self.stars[self._F,0:self.n]*10**(0.4*new_colors)
		else:
			print('Loading in catalog from run with timestr='+gdat.load_state_timestr+'...')
			catpath = gdat.result_path+'/'+gdat.load_state_timestr+'/final_state.npz'
			self.stars = np.load(catpath)['cat']
			self.n = np.count_nonzero(self.stars[self._F,:])

	''' this function prints out some information at the end of each thinned sample, 
	namely acceptance fractions for the different proposals and some time performance statistics as well. '''
   
	def print_sample_status(self, dts, accept, outbounds, chi2, movetype):    
		fmtstr = '\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f'
		print('Background', self.bkg, 'N_star', self.n, 'chi^2', list(chi2))
		dts *= 1000

		accept_fracs = []
		timestat_array = np.zeros((6, 1+len(self.moveweights)), dtype=np.float32)
		statlabels = ['Acceptance', 'Out of Bounds', 'Proposal (s)', 'Likelihood (s)', 'Implement (s)', 'Coordinates (s)']
		statarrays = [accept, outbounds, dts[0,:], dts[1,:], dts[2,:], dts[3,:]]
		for j in range(len(statlabels)):
			timestat_array[j][0] = np.sum(statarrays[j])/1000
			if j==0:
				accept_fracs.append(np.sum(statarrays[j])/1000)
			#print statlabels[j]+'\t(all) %0.3f' % (np.sum(statarrays[j])/1000),
			for k in range(len(self.movetypes)):
				if j==0:
					accept_fracs.append(np.mean(statarrays[j][movetype==k]))
				timestat_array[j][1+k] = np.mean(statarrays[j][movetype==k])
				#print '('+self.movetypes[k]+') %0.3f' % (np.mean(statarrays[j][movetype == k])),
			print
			if j == 1:
				print('-'*16)
		print('-'*16)
		print('Total (s): %0.3f' % (np.sum(statarrays[2:])/1000))
		print('='*16)

		return timestat_array, accept_fracs


		''' the multiband model evaluation looks complicated because we tried out a bunch of things with the astrometry, but could probably rewrite this function. it uses image_model_eval which is written in both python and C (C is faster as you might expect)'''
	def pcat_multiband_eval(self, x, y, f, nc, cf, weights, ref, lib):
		dmodels = []
		dt_transf = 0

		for b in range(self.nbands):
			if b>0:
				t4 = time.clock()
				if self.gdat.bands[b] != self.gdat.bands[0]:
					xp, yp = self.dat.fast_astrom.transform_q(x, y, b-1)
				else:
					xp = x
					yp = y
				dt_transf += time.clock()-t4
				
				dmodel, diff2 = image_model_eval(xp, yp, 25*f[b], self.bkg[b], self.imszs[b], \
												nc[b], np.array(cf[b]).astype(np.float32()), weights=self.dat.weights[b], \
												ref=ref[b], lib=lib, regsize=self.regsizes[b], \
												margin=self.margins[b], offsetx=self.offsetxs[b], offsety=self.offsetys[b])
				diff2s += diff2
			else:    
				xp=x
				yp=y
				dmodel, diff2 = image_model_eval(xp, yp, 25*f[b], self.bkg[b], self.imszs[b], \
												nc[b], np.array(cf[b]).astype(np.float32()), weights=self.dat.weights[b], \
												ref=ref[b], lib=lib, regsize=self.regsizes[b], \
												margin=self.margins[b], offsetx=self.offsetxs[b], offsety=self.offsetys[b])
			
				diff2s = diff2
			# dmodels.append(dmodel.transpose())
			# dmodel[weights[0]==0.] = 0.
			dmodels.append(dmodel)
		return dmodels, diff2s, dt_transf



	''' run_sampler() completes nloop samples, so the function is called nsamp times'''
	def run_sampler(self):
		
		t0 = time.clock()
		nmov = np.zeros(self.nloop)
		movetype = np.zeros(self.nloop)
		accept = np.zeros(self.nloop)
		outbounds = np.zeros(self.nloop)
		dts = np.zeros((4, self.nloop)) # array to store time spent on different proposals
		diff2_list = np.zeros(self.nloop) 

		''' I'm a bit concerned about setting the offsets for multiple observations with different sizes. 
		For now what I'll do is choose an offset for the pivot band and then compute scaled offsets for the other bands
		based on the relative sub region size, this will be off by at most 0.5 pixel, which hopefully shouldn't affect 
		things too negatively. There might be some edge effects though. '''
		
		if self.nregion > 1:
			self.offsetxs[0] = np.random.randint(self.regsizes[0])
			self.offsetys[0] = np.random.randint(self.regsizes[0])
			self.margins[0] = self.gdat.margin
			
			for b in range(self.gdat.nbands - 1):
				reg_ratio = float(self.imszs[b+1][0])/float(self.imszs[0][0])
				self.offsetxs[b+1] = int(self.offsetxs[0]*reg_ratio)
				self.offsetys[b+1] = int(self.offsetys[0]*reg_ratio)
				self.margins[b+1] = int(self.margins[0]*reg_ratio)
				if self.gdat.verbtype > 1:
					print(self.offsetxs[b+1], self.offsetys[b+1], self.margins[b+1])
		else:

			self.offsetxs = np.array([0 for b in range(self.gdat.nbands)])
			self.offsetys = np.array([0 for b in range(self.gdat.nbands)])


		self.nregx = int(self.imsz0[0] / self.regsizes[0] + 1)
		self.nregy = int(self.imsz0[1] / self.regsizes[0] + 1)

		resids = []
		for b in range(self.nbands):
			resid = self.dat.data_array[b].copy() # residual for zero image is data
			if self.gdat.verbtype > 1:
				print('resid has shape:', resid.shape)
			resids.append(resid)


		evalx = self.stars[self._X,0:self.n]
		evaly = self.stars[self._Y,0:self.n]
		evalf = self.stars[self._F:,0:self.n]
		
		n_phon = evalx.size

		if self.gdat.verbtype > 1:
			print('beginning of run sampler')
			print('self.n here')
			print(self.n)
			print('n_phon')
			print(n_phon)


		if self.gdat.cblas:
			lib = self.libmmult.pcat_model_eval
		else:
			lib = self.libmmult.clib_eval_modl

		models, diff2s, dt_transf = self.pcat_multiband_eval(evalx, evaly, evalf, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, ref=resids, lib=lib)
		model = models[0]
		logL = -0.5*diff2s
	   
		for b in range(self.nbands):
			resids[b] -= models[b]
		
		'''the proposals here are: move_stars (P) which changes the parameters of existing model sources, 
		birth/death (BD) and merge/split (MS). Don't worry about perturb_astrometry. 
		The moveweights array, once normalized, determines the probability of choosing a given proposal. '''

		movefns = [self.move_stars, self.birth_death_stars, self.merge_split_stars]
		self.moveweights /= np.sum(self.moveweights)
		if self.gdat.nregion > 1:
			xparities = np.random.randint(2, size=self.nloop)
			yparities = np.random.randint(2, size=self.nloop)
		
		rtype_array = np.random.choice(self.moveweights.size, p=self.moveweights, size=self.nloop)
		movetype = rtype_array

		for i in range(self.nloop):
			t1 = time.clock()
			rtype = rtype_array[i]
			
			if self.verbtype > 1:
				print('rtype: ', rtype)
			if self.nregion > 1:
				self.parity_x = xparities[i] # should regions be perturbed randomly or systematically?
				self.parity_y = yparities[i]
			else:
				self.parity_x = 0
				self.parity_y = 0

			#proposal types
			proposal = movefns[rtype]()
			dts[0,i] = time.clock() - t1
			
			if proposal.goodmove:
				t2 = time.clock()

				if self.gdat.cblas:
					lib = self.libmmult.pcat_model_eval
				else:
					lib = self.libmmult.clib_eval_modl

				dmodels, diff2s, dt_transf = self.pcat_multiband_eval(proposal.xphon, proposal.yphon, proposal.fphon, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, ref=resids, lib=lib)

				plogL = -0.5*diff2s                
				plogL[(1-self.parity_y)::2,:] = float('-inf') # don't accept off-parity regions
				plogL[:,(1-self.parity_x)::2] = float('-inf')
				dlogP = plogL - logL
				if self.verbtype > 1:
					print('dlogP')
					print(dlogP)
				
				assert np.isnan(dlogP).any() == False
				
				dts[1,i] = time.clock() - t2
				t3 = time.clock()
				refx, refy = proposal.get_ref_xy()

				regionx = get_region(refx, self.offsetxs[0], self.regsizes[0])
				regiony = get_region(refy, self.offsetys[0], self.regsizes[0])

				
				if proposal.factor is not None:
					dlogP[regiony, regionx] += proposal.factor
				else:
					print('proposal factor is None')
				acceptreg = (np.log(np.random.uniform(size=(self.nregy, self.nregx))) < dlogP).astype(np.int32)
				acceptprop = acceptreg[regiony, regionx]
				numaccept = np.count_nonzero(acceptprop)
				''' for each band compute the delta log likelihood between states, theen add these together'''
				for b in range(self.nbands):
					dmodel_acpt = np.zeros_like(dmodels[b])
					diff2_acpt = np.zeros_like(diff2s)

					if self.gdat.cblas:

						self.libmmult.pcat_imag_acpt(self.imszs[b][0], self.imszs[b][1], dmodels[b], dmodel_acpt, acceptreg, self.regsizes[b], self.margins[b], self.offsetxs[b], self.offsetys[b])
						# using this dmodel containing only accepted moves, update logL
						self.libmmult.pcat_like_eval(self.imszs[b][0], self.imszs[b][1], dmodel_acpt, resids[b], self.dat.weights[b], diff2_acpt, self.regsizes[b], self.margins[b], self.offsetxs[b], self.offsetys[b])   
					else:

						self.libmmult.clib_updt_modl(self.imszs[b][0], self.imszs[b][1], dmodels[b], dmodel_acpt, acceptreg, self.regsizes[b], self.margins[b], self.offsetxs[b], self.offsetys[b])
						# using this dmodel containing only accepted moves, update logL
						self.libmmult.clib_eval_llik(self.imszs[b][0], self.imszs[b][1], dmodel_acpt, resids[b], self.dat.weights[b], diff2_acpt, self.regsizes[b], self.margins[b], self.offsetxs[b], self.offsetys[b])   

					resids[b] -= dmodel_acpt
					models[b] += dmodel_acpt

					if b==0:
						diff2_total1 = diff2_acpt
					else:
						diff2_total1 += diff2_acpt

				logL = -0.5*diff2_total1


				#implement accepted moves
				if proposal.idx_move is not None:
					starsp = proposal.starsp.compress(acceptprop, axis=1)
					idx_move_a = proposal.idx_move.compress(acceptprop)

					self.stars[:, idx_move_a] = starsp

				
				if proposal.do_birth:
					starsb = proposal.starsb.compress(acceptprop, axis=1)
					starsb = starsb.reshape((2+self.nbands,-1))
					num_born = starsb.shape[1]
					self.stars[:, self.n:self.n+num_born] = starsb
					self.n += num_born

				if proposal.idx_kill is not None:
					idx_kill_a = proposal.idx_kill.compress(acceptprop, axis=0).flatten()
					num_kill = idx_kill_a.size
				   
					# nstar is correct, not n, because x,y,f are full nstar arrays
					self.stars[:, 0:self.max_nsrc-num_kill] = np.delete(self.stars, idx_kill_a, axis=1)
					self.stars[:, self.max_nsrc-num_kill:] = 0
					self.n -= num_kill


				dts[2,i] = time.clock() - t3

				if acceptprop.size > 0:
					accept[i] = np.count_nonzero(acceptprop) / float(acceptprop.size)
				else:
					accept[i] = 0
			else:
				if self.verbtype > 1:
					print('out of bounds')
				outbounds[i] = 1

			for b in range(self.nbands):
				diff2_list[i] += np.sum(self.dat.weights[b]*(self.dat.data_array[b]-models[b])*(self.dat.data_array[b]-models[b]))
					
			if self.verbtype > 1:
				print('end of Loop', i)
				print('self.n')
				print(self.n)
				print('diff2')
				print(diff2_list[i])
			
		chi2 = np.zeros(self.nbands)
		for b in range(self.nbands):
			chi2[b] = np.sum(self.dat.weights[b]*(self.dat.data_array[b]-models[b])*(self.dat.data_array[b]-models[b]))
			
		if self.verbtype > 1:
			print('end of sample')
			print('self.n end')
			print(self.n)


		timestat_array, accept_fracs = self.print_sample_status(dts, accept, outbounds, chi2, movetype)

		if self.gdat.visual:

			if self.gdat.nbands == 1:
				self.plot_single_band_frame(resids, models)
			else:
				self.plot_multiband_frame(resids, models)

		return self.n, chi2, timestat_array, accept_fracs, diff2_list, rtype_array, accept, resids


	def plot_single_band_frame(self, resids, models):
		plt.gcf().clear()
		plt.figure(1, figsize=(9, 4))
		plt.clf()
		plt.subplot(2,3,1)
		plt.title('Data')
		plt.imshow(self.dat.data_array[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(self.dat.data_array[0]), vmax=np.percentile(self.dat.data_array[0], 99.9))
		plt.colorbar()
		sizefac = 10.*136
		plt.scatter(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], marker='x', s=self.stars[self._F, 0:self.n]*100, color='r')
		plt.xlim(-0.5, self.imsz0[0]-0.5)
		plt.ylim(-0.5, self.imsz0[1]-0.5)
		plt.subplot(2,3,2)
		plt.title('Model')
		plt.imshow(models[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(self.dat.data_array[0]), vmax=np.percentile(self.dat.data_array[0], 99.9))
		# plt.imshow(models[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(models[0]), vmax=np.percentile(models[0], 99.9))
		plt.colorbar()
		plt.subplot(2,3,3)
		plt.title('Residual')
		# plt.imshow(models[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data_array[0]), vmax=np.percentile(data_array[0], 99.9))
		# plt.imshow(models[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(models[0]), vmax=np.percentile(models[0], 99.9))
		# plt.imshow(resids[0], origin='lower', interpolation='none', cmap='Greys',  vmin=np.min(data_array[0]), vmax=np.percentile(data_array[0], 99.9))
		# plt.imshow(resids[0], origin='lower', interpolation='none', cmap='Greys',  vmin = np.percentile(resids[0][weights[0] != 0.], 5), vmax=np.percentile(resids[0][weights[0] != 0.], 95))
		if self.gdat.weighted_residual:
			plt.imshow(resids[0]*np.sqrt(self.dat.weights[0]), origin='lower', interpolation='none', cmap='Greys', vmin=-5, vmax=5)
		else:
			plt.imshow(resids[0], origin='lower', interpolation='none', cmap='Greys', vmin = np.percentile(resids[0][self.dat.weights[0] != 0.], 5), vmax=np.percentile(resids[0][self.dat.weights[0] != 0.], 95))

		plt.colorbar()
		plt.scatter(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], marker='x', s=self.stars[self._F, 0:self.n]*100, color='r')

		plt.subplot(2,3,4)
		plt.title('Data (zoomed in)')
		plt.imshow(self.dat.data_array[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(self.dat.data_array[0]), vmax=np.percentile(self.dat.data_array[0], 99.9))
		plt.colorbar()
		plt.scatter(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], marker='x', s=self.stars[self._F, 0:self.n]*100, color='r')
		plt.ylim(90, 140)
		plt.xlim(70, 120)
		plt.subplot(2,3,5)
		plt.title('Residual (zoomed in)')

		if self.gdat.weighted_residual:
			plt.imshow(resids[0]*np.sqrt(self.dat.weights[0]), origin='lower', interpolation='none', cmap='Greys', vmin=-5, vmax=5)
		else:
			plt.imshow(resids[0], origin='lower', interpolation='none', cmap='Greys', vmin = np.percentile(resids[0][self.dat.weights[0] != 0.], 5), vmax=np.percentile(resids[0][self.dat.weights[0] != 0.], 95))
		plt.colorbar()
		plt.ylim(90, 140)
		plt.xlim(70, 120)
		plt.subplot(2,3,6)

		binz = np.linspace(np.log10(self.trueminf)+3., np.ceil(np.log10(np.max(self.stars[self._F, 0:self.n]))+3.), 20)
		hist = np.histogram(np.log10(self.stars[self._F, 0:self.n])+3, bins=binz)
		logSv = 0.5*(hist[1][1:]+hist[1][:-1])-3
		binz_Sz = 10**(binz-3)
		dSz = binz_Sz[1:]-binz_Sz[:-1]
		dNdS = hist[0]

		if self.gdat.raw_counts:
	
			plt.plot(logSv+3, dNdS, marker='.')
			plt.ylabel('dN/dS')
			plt.ylim(5e-1, 3e3)

		else:
			n_steradian = 0.11/(180./np.pi)**2 # field covers 0.11 degrees, should change this though for different fields
			n_steradian *= self.gdat.frac # a number of pixels in the image are not actually observing anything
			dNdS_S_twop5 = dNdS*(10**(logSv))**(2.5)
			plt.plot(logSv+3, dNdS_S_twop5/n_steradian/dSz, marker='.')
			plt.ylabel('dN/dS.$S^{2.5}$ ($Jy^{1.5}/sr$)')
			plt.ylim(1e0, 1e5)


		plt.yscale('log')
		plt.legend()
		plt.xlabel('log($S_{\\nu}$) (mJy)')
		plt.xlim(np.log10(self.trueminf)+3.-0.5, 2.5)
		plt.tight_layout()
		plt.draw()
		# if savefig:
			# plt.savefig(frame_dir + '/frame_' + str(c) + '.png')
		plt.pause(1e-5)

	def plot_multiband_frame(self, resids, models):
		plt.gcf().clear()
		plt.figure(1, figsize=(9, 4))
		plt.clf()
		plt.subplot(2,3,1)
		plt.title('Data')
		plt.imshow(self.dat.data_array[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(self.dat.data_array[0]), vmax=np.percentile(self.dat.data_array[0], 99.9))
		plt.colorbar()
		plt.scatter(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], marker='x', s=self.stars[self._F, 0:self.n]*100, color='r')
		plt.xlim(-0.5, self.imsz0[0]-0.5)
		plt.ylim(-0.5, self.imsz0[1]-0.5)
		plt.subplot(2,3,2)
		plt.title('Data (second band)')
		xp, yp = self.dat.fast_astrom.transform_q(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], 0)
		# xd, yd = self.dat.fast_astrom.transform_q(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], 1)

		plt.imshow(self.dat.data_array[1], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(self.dat.data_array[1]), vmax=np.percentile(self.dat.data_array[1], 99.9))
		plt.colorbar()
		plt.scatter(xp, yp, marker='x', s=self.stars[self._F+1, 0:self.n]*100, color='r')
		plt.xlim(-0.5, self.imszs[1][0]-0.5)
		plt.ylim(-0.5, self.imszs[1][1]-0.5)

		plt.subplot(2,3,3)
		plt.title('Residual')
		if self.gdat.weighted_residual:
			plt.imshow(resids[0]*np.sqrt(self.dat.weights[0]), origin='lower', interpolation='none', cmap='Greys', vmin=-5, vmax=5)
		else:
			plt.imshow(resids[1], origin='lower', interpolation='none', cmap='Greys', vmin = np.percentile(resids[1][self.dat.weights[1] != 0.], 5), vmax=np.percentile(resids[1][self.dat.weights[1] != 0.], 95))
		# plt.imshow(self.dat.data_array[2], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(self.dat.data_array[2]), vmax=np.percentile(self.dat.data_array[2], 99.9))

		plt.colorbar()
		# plt.scatter(xd, yd, marker='x', s=self.stars[self._F+1, 0:self.n]*100, color='r')
		# plt.scatter(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], marker='x', s=self.stars[self._F, 0:self.n]*100, color='r')

		plt.subplot(2,3,4)
		plt.title('Data (zoomed in)')
		plt.imshow(self.dat.data_array[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(self.dat.data_array[0]), vmax=np.percentile(self.dat.data_array[0], 99.9))
		plt.colorbar()
		plt.scatter(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], marker='x', s=self.stars[self._F, 0:self.n]*100, color='r')
		plt.ylim(90, 140)
		plt.xlim(70, 120)
		# plt.title('Residual')
		# if self.gdat.weighted_residual:
		# 	plt.imshow(resids[0]*np.sqrt(self.dat.weights[0]), origin='lower', interpolation='none', cmap='Greys', vmin=-5, vmax=5)
		# else:
		# 	plt.imshow(resids[2], origin='lower', interpolation='none', cmap='Greys', vmin = np.percentile(resids[2][self.dat.weights[2] != 0.], 5), vmax=np.percentile(resids[2][self.dat.weights[2] != 0.], 95))

		# plt.colorbar()
		# plt.scatter(xd, yd, marker='x', s=self.stars[self._F+2, 0:self.n]*100, color='r')
		plt.subplot(2,3,5)
		plt.title('Residual (zoomed in)')

		if self.gdat.weighted_residual:
			plt.imshow(resids[0]*np.sqrt(self.dat.weights[0]), origin='lower', interpolation='none', cmap='Greys', vmin=-5, vmax=5)
		else:
			plt.imshow(resids[0], origin='lower', interpolation='none', cmap='Greys', vmin = np.percentile(resids[0][self.dat.weights[0] != 0.], 5), vmax=np.percentile(resids[0][self.dat.weights[0] != 0.], 95))
		plt.colorbar()
		plt.ylim(90, 140)
		plt.xlim(70, 120)
		plt.subplot(2,3,6)

		binz = np.linspace(np.log10(self.trueminf)+3., np.ceil(np.log10(np.max(self.stars[self._F, 0:self.n]))+3.), 20)
		hist = np.histogram(np.log10(self.stars[self._F, 0:self.n])+3, bins=binz)
		logSv = 0.5*(hist[1][1:]+hist[1][:-1])-3
		binz_Sz = 10**(binz-3)
		dSz = binz_Sz[1:]-binz_Sz[:-1]
		dNdS = hist[0]

		if self.gdat.raw_counts:
	
			plt.plot(logSv+3, dNdS, marker='.')
			plt.ylabel('dN/dS')
			plt.ylim(5e-1, 3e3)

		else:
			n_steradian = 0.11/(180./np.pi)**2 # field covers 0.11 degrees, should change this though for different fields
			n_steradian *= self.gdat.frac # a number of pixels in the image are not actually observing anything
			dNdS_S_twop5 = dNdS*(10**(logSv))**(2.5)
			plt.plot(logSv+3, dNdS_S_twop5/n_steradian/dSz, marker='.')
			plt.ylabel('dN/dS.$S^{2.5}$ ($Jy^{1.5}/sr$)')
			plt.ylim(1e0, 1e5)


		plt.yscale('log')
		plt.legend()
		plt.xlabel('log($S_{\\nu}$) (mJy)')
		plt.xlim(np.log10(self.trueminf)+3.-0.5, 2.5)
		plt.tight_layout()
		plt.draw()
		# if savefig:
			# plt.savefig(frame_dir + '/frame_' + str(c) + '.png')
		plt.pause(1e-5)



	def idx_parity_stars(self):
		return idx_parity(self.stars[self._X,:], self.stars[self._Y,:], self.n, self.offsetxs[0], self.offsetys[0], self.parity_x, self.parity_y, self.regsizes[0])
		# return idx_parity(self.stars[self._X,:], self.stars[self._Y,:], self.n, self.offsetx, self.offsety, self.parity_x, self.parity_y, self.regsize)

	def bounce_off_edges(self, catalogue): # works on both stars and galaxies
		mask = catalogue[self._X,:] < 0
		catalogue[self._X, mask] *= -1
		mask = catalogue[self._X,:] > (self.imsz0[0] - 1)
		catalogue[self._X, mask] *= -1
		catalogue[self._X, mask] += 2*(self.imsz0[0] - 1)
		mask = catalogue[self._Y,:] < 0
		catalogue[self._Y, mask] *= -1
		mask = catalogue[self._Y,:] > (self.imsz0[1] - 1)
		catalogue[self._Y, mask] *= -1
		catalogue[self._Y, mask] += 2*(self.imsz0[1] - 1)
		# these are all inplace operations, so no return value

	def in_bounds(self, catalogue):
		return np.logical_and(np.logical_and(catalogue[self._X,:] > 0, catalogue[self._X,:] < (self.imsz0[0] -1)), \
				np.logical_and(catalogue[self._Y,:] > 0, catalogue[self._Y,:] < self.imsz0[1] - 1))


	def flux_proposal(self, f0, nw, trueminf=None):
		if trueminf is None:
			trueminf = self.trueminf
		lindf = np.float32(self.err_f/(self.regions_factor*np.sqrt(self.gdat.nominal_nsrc*(2+self.nbands))))
		logdf = np.float32(0.01/np.sqrt(self.gdat.nominal_nsrc))
		ff = np.log(logdf*logdf*f0 + logdf*np.sqrt(lindf*lindf + logdf*logdf*f0*f0)) / logdf
		ffmin = np.log(logdf*logdf*trueminf + logdf*np.sqrt(lindf*lindf + logdf*logdf*trueminf*trueminf)) / logdf
		dff = np.random.normal(size=nw).astype(np.float32)
		aboveffmin = ff - ffmin
		oob_flux = (-dff > aboveffmin)
		dff[oob_flux] = -2*aboveffmin[oob_flux] - dff[oob_flux]
		pff = ff + dff
		pf = np.exp(-logdf*pff) * (-lindf*lindf*logdf*logdf+np.exp(2*logdf*pff)) / (2*logdf*logdf)
		return pf


	def move_stars(self): 
		idx_move = self.idx_parity_stars()
		nw = idx_move.size
		stars0 = self.stars.take(idx_move, axis=1)
		starsp = np.empty_like(stars0)
		
		f0 = stars0[self._F:,:]
		pfs = []
		color_factors = np.zeros((self.nbands-1, nw)).astype(np.float32)

		for b in range(self.nbands):
			if b==0:
				pf = self.flux_proposal(f0[b], nw)
			else:
				pf = self.flux_proposal(f0[b], nw, trueminf=0.0001) #place a minor minf to avoid negative fluxes in non-pivot bands
			pfs.append(pf)
 
		if (np.array(pfs)<0).any():
			print('negative flux!')
			print(np.array(pfs)[np.array(pfs)<0])

		dlogf = np.log(pfs[0]/f0[0])

		if self.verbtype > 1:
			print('average flux difference')
			print(np.average(np.abs(f0[0]-pfs[0])))

		factor = -self.truealpha*dlogf

		if np.isnan(factor).any():
			print('factor nan from flux')
			print('number of f0 zero elements:', len(f0[0])-np.count_nonzero(np.array(f0[0])))
			if self.verbtype > 1:
				print('factor')
				print(factor)
			factor[np.isnan(factor)]=0

		''' the loop over bands below computes colors and prior factors in color used when sampling the posterior
		come back to this later  '''
		modl_eval_colors = []
		for b in range(self.nbands-1):
			colors = fluxes_to_color(pfs[0], pfs[b+1])
			orig_colors = fluxes_to_color(f0[0], f0[b+1])
			colors[np.isnan(colors)] = self.color_mus[b] # make nan colors not affect color_factors
			orig_colors[np.isnan(orig_colors)] = self.color_mus[b]
			color_factors[b] -= (colors - self.color_mus[b])**2/(2*self.color_sigs[b]**2)
			color_factors[b] += (orig_colors - self.color_mus[b])**2/(2*self.color_sigs[b]**2)
			modl_eval_colors.append(colors)
	
		assert np.isnan(color_factors).any()==False       

		if self.verbtype > 1:
			print('avg abs color_factors:', np.average(np.abs(color_factors)))
			print('avg abs flux factor:', np.average(np.abs(factor)))

		factor = np.array(factor) + np.sum(color_factors, axis=0)
		
		dpos_rms = np.float32(np.sqrt(self.gdat.N_eff/(2*np.pi))*self.err_f/(np.sqrt(self.nominal_nsrc*self.regions_factor*(2+self.nbands))))/(np.maximum(f0[0],pfs[0]))

		if self.verbtype > 1:
			print('dpos_rms')
			print(dpos_rms)
		
		dpos_rms[dpos_rms < 1e-3] = 1e-3 #do we need this line? perhaps not
		dx = np.random.normal(size=nw).astype(np.float32)*dpos_rms
		dy = np.random.normal(size=nw).astype(np.float32)*dpos_rms
		starsp[self._X,:] = stars0[self._X,:] + dx
		starsp[self._Y,:] = stars0[self._Y,:] + dy
		
		if self.verbtype > 1:
			print('dx')
			print(dx)
			print('dy')
			print(dy)
			print('mean dx and mean dy')
			print(np.mean(np.abs(dx)), np.mean(np.abs(dy)))

		for b in range(self.nbands):
			starsp[self._F+b,:] = pfs[b]
			if (pfs[b]<0).any():
				print('proposal fluxes less than 0')
				print('band', b)
				print(pfs[b])
		self.bounce_off_edges(starsp)

		proposal = Proposal(self.gdat)
		proposal.add_move_stars(idx_move, stars0, starsp, modl_eval_colors)
		
		assert np.isinf(factor).any()==False
		assert np.isnan(factor).any()==False

		proposal.set_factor(factor)
		return proposal



	def birth_death_stars(self):
		lifeordeath = np.random.randint(2)
		nbd = (self.nregx * self.nregy) / 4
		proposal = Proposal(self.gdat)
		# birth
		if lifeordeath and self.n < self.max_nsrc: # need room for at least one source
			nbd = int(min(nbd, self.max_nsrc-self.n)) # add nbd sources, or just as many as will fit
			# mildly violates detailed balance when n close to nstar
			# want number of regions in each direction, divided by two, rounded up

			
			mregx = int(((self.imsz0[0] / self.regsizes[0] + 1) + 1) / 2) # assumes that imsz are multiples of regsize
			mregy = int(((self.imsz0[1] / self.regsizes[0] + 1) + 1) / 2)

			starsb = np.empty((2+self.nbands, nbd), dtype=np.float32)
			starsb[self._X,:] = (np.random.randint(mregx, size=nbd)*2 + self.parity_x + np.random.uniform(size=nbd))*self.regsizes[0] - self.offsetxs[0]
			starsb[self._Y,:] = (np.random.randint(mregy, size=nbd)*2 + self.parity_y + np.random.uniform(size=nbd))*self.regsizes[0] - self.offsetys[0]
			
			for b in range(self.nbands):
				if b==0:
					starsb[self._F+b,:] = self.trueminf * np.exp(np.random.exponential(scale=1./(self.truealpha-1.),size=nbd))
				else:
					# draw new source colors from color prior
					new_colors = np.random.normal(loc=self.color_mus[b-1], scale=self.color_sigs[b-1], size=nbd)
					starsb[self._F+b,:] = starsb[self._F,:]*10**(0.4*new_colors)
			
					if (starsb[self._F+b,:]<0).any():
						print('negative birth star fluxes')
						print('new_colors')
						print(new_colors)
						print('starsb fluxes:')
						print(starsb[self._F+b,:])

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
			nbd = int(min(nbd, idx_reg.size)) # kill nbd sources, or however many sources remain
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
		nms = int((self.nregx * self.nregy) / 4)
		goodmove = False
		proposal = Proposal(self.gdat)
		# split
		if splitsville and self.n > 0 and self.n < self.max_nsrc and bright_n > 0: # need something to split, but don't exceed nstar
			nms = min(nms, bright_n, self.max_nsrc-self.n) # need bright source AND room for split source
			dx = (np.random.normal(size=nms)*self.kickrange).astype(np.float32)
			dy = (np.random.normal(size=nms)*self.kickrange).astype(np.float32)
			idx_move = np.random.choice(idx_bright, size=nms, replace=False)
			stars0 = self.stars.take(idx_move, axis=1)

			fminratio = stars0[self._F,:] / self.trueminf
 
			if self.verbtype > 1:
				print('stars0 at splitsville start')
				print(stars0)
				print('fminratio here')
				print(fminratio)
				print('dx')
				print(dx)
				print('dy')
				print(dy)
				print('idx_move')
				print(idx_move)
				
			fracs.append((1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32))
			
			# color stuff, look at later
			for b in range(self.nbands-1):
				# changed to split similar fluxes
				d_color = np.random.normal(0,self.gdat.split_col_sig)
				frac_sim = np.exp(d_color/self.k)*fracs[0]/(1-fracs[0]+np.exp(d_color/self.k)*fracs[0])
				fracs.append(frac_sim)

				
			starsp = np.empty_like(stars0)
			starsb = np.empty_like(stars0)

			starsp[self._X,:] = stars0[self._X,:] + ((1-fracs[0])*dx)
			starsp[self._Y,:] = stars0[self._Y,:] + ((1-fracs[0])*dy)
			starsb[self._X,:] = stars0[self._X,:] - fracs[0]*dx
			starsb[self._Y,:] = stars0[self._Y,:] - fracs[0]*dy

			for b in range(self.nbands):
				
				starsp[self._F+b,:] = stars0[self._F+b,:]*fracs[b]
				starsb[self._F+b,:] = stars0[self._F+b,:]*(1-fracs[b])
				if (starsp[self._F+b,:]<0).any():
					print('neg starsp in band', b)
					print('stars0')
					print(stars0)
					print('fracs[b]')
					print(fracs[b])
					print('starsp[self._F+b,:]')
					print(starsp[self._F+b,:])
				if (starsb[self._F+b,:]<0).any():
					print('neg starsb in band', b)
					print('stars0')
					print(stars0)
					print('1-fracs[b]')
					print(1-fracs[b])
					print('starsb[self._F+b,:]')
					print(starsb[self._F+b,:])
			# don't want to think about how to bounce split-merge
			# don't need to check if above fmin, because of how frac is decided
			inbounds = np.logical_and(self.in_bounds(starsp), self.in_bounds(starsb))
			stars0 = stars0.compress(inbounds, axis=1)
			starsp = starsp.compress(inbounds, axis=1)
			starsb = starsb.compress(inbounds, axis=1)
			idx_move = idx_move.compress(inbounds)
			fminratio = fminratio.compress(inbounds)

			for b in range(self.nbands):
				fracs[b] = fracs[b].compress(inbounds)
				sum_fs.append(stars0[self._F+b,:])
			
			nms = idx_move.size
			goodmove = nms > 0
			
			if goodmove:
				proposal.add_move_stars(idx_move, stars0, starsp)
				proposal.add_birth_stars(starsb)
				# can this go nested in if statement? 
			invpairs = np.empty(nms)
			

			if self.verbtype > 1:
				print('splitsville happening')
				print('goodmove:', goodmove)
				print('invpairs')
				print(invpairs)
				print('nms:', nms)
				print('sum_fs')
				print(sum_fs)
				print('fminratio')
				print(fminratio)

			for k in range(nms):
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
			nms = int(min(nms, idx_reg.size/2))
			idx_move = np.empty(nms, dtype=np.int)
			idx_kill = np.empty(nms, dtype=np.int)
			choosable = np.zeros(self.max_nsrc, dtype=np.bool)
			choosable[idx_reg] = True
			nchoosable = float(idx_reg.size)
			invpairs = np.empty(nms)
			
			if self.verbtype > 1:
				print('merging two things!')
				print('nms:', nms)
				print('idx_move', idx_move)
				print('idx_kill', idx_kill)
				
			for k in range(nms):
				idx_move[k] = np.random.choice(self.max_nsrc, p=choosable/nchoosable)
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

			for b in range(self.nbands):
				sum_fs.append(f0[b,:] + fk[b,:])
				fracs.append(f0[b,:] / sum_fs[b])
			fminratio = sum_fs[0] / self.trueminf
			
			if self.verbtype > 1:
				print('fminratio')
				print(fminratio)
				print('nms is now', nms)
				print('sum_fs[0]', sum_fs[0])
				print('all sum_fs:')
				print(sum_fs)
				print('stars0')
				print(stars0)
				print('starsk')
				print(starsk)
				print('idx_move')
				print(idx_move)
				print('idx_kill')
				print(idx_kill)
				
			starsp = np.empty_like(stars0)
			# place merged source at center of flux of previous two sources
			starsp[self._X,:] = fracs[0]*stars0[self._X,:] + (1-fracs[0])*starsk[self._X,:]
			starsp[self._Y,:] = fracs[0]*stars0[self._Y,:] + (1-fracs[0])*starsk[self._Y,:]
			
			for b in range(self.nbands):
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
			factor = np.log(self.truealpha-1) + (self.truealpha-1)*np.log(self.trueminf) - self.truealpha*np.log(fracs[0]*(1-fracs[0])*sum_fs[0]) + np.log(2*np.pi*self.kickrange*self.kickrange) - np.log(self.imsz0[0]*self.imsz0[1]) + np.log(1. - 2./fminratio) + np.log(bright_n) + np.log(invpairs) + np.log(sum_fs[0])

			for b in range(self.nbands-1):
				stars0_color = fluxes_to_color(stars0[self._F,:], stars0[self._F+b+1,:])
				starsp_color = fluxes_to_color(starsp[self._F,:], starsp[self._F+b+1,:])
				dc = self.k*(np.log(fracs[b+1]/fracs[0]) - np.log((1-fracs[b+1])/(1-fracs[0])))
				# added_fac comes from the transition kernel of splitting colors in the manner that we do
				added_fac = 0.5*np.log(2*np.pi*self.gdat.split_col_sig**2)+(dc**2/(2*self.gdat.split_col_sig**2))
				factor += added_fac
				
				if splitsville:
					starsb_color = fluxes_to_color(starsb[self._F,:], starsb[self._F+b+1,:])
					# colfac is ratio of color prior factors i.e. P(s_0)P(s_1)/P(s_merged), where 0 and 1 are original sources 
					colfac = (stars0_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (starsp_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (starsb_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2)-0.5*np.log(2*np.pi*self.color_sigs[b]**2)
						
					factor += colfac
			 
				else:
					starsk_color = fluxes_to_color(starsk[self._F,:], starsk[self._F+b+1,:])
					# same as above but for merging sources
					colfac = (starsp_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (stars0_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (starsk_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2)-0.5*np.log(2*np.pi*self.color_sigs[b]**2)
					factor += colfac

			if not splitsville:
				factor *= -1
				factor += self.penalty
			else:
				factor -= self.penalty

			proposal.set_factor(factor)
							
			assert np.isnan(factor).any()==False

			if self.verbtype > 1:
				print('kickrange factor', np.log(2*np.pi*self.kickrange*self.kickrange))
				print('imsz factor', np.log(self.imsz0[0]*self.imsz0[1]))
				print('fminratio:', fminratio)
				print('fmin factor', np.log(1. - 2./fminratio))
				print('kickrange factor', np.log(2*np.pi*self.kickrange*self.kickrange) - np.log(self.imsz0[0]*self.imsz0[1]) + np.log(1. - 2./fminratio))
				print('factor after colors')
				print(factor)
		return proposal



class Samples():

	def __init__(self, gdat):
		self.nsample = np.zeros(gdat.nsamp, dtype=np.int32)
		self.xsample = np.zeros((gdat.nsamp, gdat.max_nsrc), dtype=np.float32)
		self.ysample = np.zeros((gdat.nsamp, gdat.max_nsrc), dtype=np.float32)
		self.timestats = np.zeros((gdat.nsamp, 6, 4), dtype=np.float32)
		self.diff2_all = np.zeros((gdat.nsamp, gdat.nloop), dtype=np.float32)
		self.accept_all = np.zeros((gdat.nsamp, gdat.nloop), dtype=np.float32)
		self.rtypes = np.zeros((gdat.nsamp, gdat.nloop), dtype=np.float32)
		self.accept_stats = np.zeros((gdat.nsamp, 4), dtype=np.float32)
		self.tq_times = np.zeros(gdat.nsamp, dtype=np.float32)
		self.fsample = [np.zeros((gdat.nsamp, gdat.max_nsrc), dtype=np.float32) for x in range(gdat.nbands)]
		self.colorsample = [[] for x in range(gdat.nbands-1)]
		# self.residuals = np.zeros((gdat.nbands, gdat.residual_samples, gdat.width, gdat.height))
		self.residuals = [np.zeros((gdat.residual_samples, gdat.imszs[i][0], gdat.imszs[i][1])) for i in range(gdat.nbands)]
		self.chi2sample = np.zeros((gdat.nsamp, gdat.nbands), dtype=np.int32)
		self.nbands = gdat.nbands
		self.gdat = gdat

	def add_sample(self, j, model, diff2_list, accepts, rtype_array, accept_fracs, chi2_all, statarrays, resids):
		
		self.nsample[j] = model.n
		self.xsample[j,:] = model.stars[Model._X, :]
		self.ysample[j,:] = model.stars[Model._Y, :]
		self.diff2_all[j,:] = diff2_list
		self.accept_all[j,:] = accepts
		self.rtypes[j,:] = rtype_array
		self.accept_stats[j,:] = accept_fracs
		self.chi2sample[j] = chi2_all
		self.timestats[j,:] = statarrays

		for b in range(self.nbands):
			self.fsample[b][j,:] = model.stars[Model._F+b,:]
			if self.gdat.nsamp - j < self.gdat.residual_samples+1:
				self.residuals[b][-(self.gdat.nsamp-j),:,:] = resids[b] 

	def save_samples(self, result_path, timestr):

		if self.gdat.nbands > 1:
			if self.gdat.nbands > 2:
				np.savez(result_path + '/' + str(timestr) + '/chain.npz', n=self.nsample, x=self.xsample, y=self.ysample, f=self.fsample, \
				chi2=self.chi2sample, times=self.timestats, accept=self.accept_stats, diff2s=self.diff2_all, rtypes=self.rtypes, \
				accepts=self.accept_all, residuals0=self.residuals[0], residuals1=self.residuals[1], residuals2=self.residuals[2])
			else:
				print('self.residuals[0] has shape', self.residuals[0].shape, self.residuals[1].shape)
				np.savez(result_path + '/' + str(timestr) + '/chain.npz', n=self.nsample, x=self.xsample, y=self.ysample, f=self.fsample, \
				chi2=self.chi2sample, times=self.timestats, accept=self.accept_stats, diff2s=self.diff2_all, rtypes=self.rtypes, \
				accepts=self.accept_all, residuals0=self.residuals[0], residuals1=self.residuals[1])


		else:
			np.savez(result_path + '/' + str(timestr) + '/chain.npz', n=self.nsample, x=self.xsample, y=self.ysample, f=self.fsample, \
				chi2=self.chi2sample, times=self.timestats, accept=self.accept_stats, diff2s=self.diff2_all, rtypes=self.rtypes, \
				accepts=self.accept_all, residuals0=self.residuals[0])

''' This class sets up the data structures for data/data-related information. 
load_in_data() loads in data, generates the PSF template and computes weights from the noise model
'''
class pcat_data():

	def __init__(self, auto_resize=False, nregion=1):
		self.ncs = []
		self.nbins = []
		self.psfs = []
		self.cfs = []
		self.biases = []
		self.data_array = []
		self.weights = []
		self.masks = []
		self.exposures = []
		self.errors = []
		self.fast_astrom = wcs_astrometry(auto_resize, nregion=nregion)
		self.widths = []
		self.heights = []


	def find_lowest_mod(self, number, mod_number):
		while number > 0:
			if np.mod(number, mod_number) == 0:
				return number
			else:
				number -= 1
		return False

	def find_nearest_upper_mod(self, number, mod_number):
		while number < 10000:
			if np.mod(number, mod_number) == 0:
				return number
			else:
				number += 1
		return False

	def load_in_data(self, gdat, map_object=None):

		gdat.imszs = []
		gdat.regsizes = []
		gdat.margins = []

		for i, band in enumerate(gdat.bands):
			print('band:', band)

			if map_object is not None:

				obj = map_object[band]
				image = np.nan_to_num(obj['signal'])
				error = np.nan_to_num(obj['error'])

				exposure = obj['exp'].data
				mask = obj['mask']
				print('pixsize:', obj['pixsize'])
				gdat.psf_pixel_fwhm = obj['widtha']/obj['pixsize']# gives it in arcseconds and neet to convert to pixels
				self.fast_astrom.load_wcs_header_and_dim(head=obj['shead'])
				gdat.dataname = obj['name']
				print('gdat.dataname:', gdat.dataname)
				if i > 0:
					self.fast_astrom.fit_astrom_arrays(0, i)


			elif gdat.mock_name is None:
				image, error, exposure, mask = load_in_map(gdat, band, astrom=self.fast_astrom)
				
				if i > 0:
					print('we have more than one band:', gdat.bands[0], band)
					# self.fast_astrom.fit_astrom_arrays(gdat.bands[0], i)
					self.fast_astrom.fit_astrom_arrays(0, i)


			else:
				image, error, exposure, mask = load_in_mock_map(gdat.mock_name, band)
			
			if gdat.auto_resize:
				smaller_dim = np.min([image.shape[0]-gdat.x0, image.shape[1]-gdat.y0]) # option to include lower left corner
				larger_dim = np.max([image.shape[0]-gdat.x0, image.shape[1]-gdat.y0])

				print('smaller dim is', smaller_dim)
				print('larger dim is ', larger_dim)

				gdat.width = self.find_nearest_upper_mod(larger_dim, gdat.nregion)
				# gdat.width = self.find_lowest_mod(smaller_dim, gdat.nregion)
				gdat.height = gdat.width
				image_size = (gdat.width, gdat.height)


				padded_image = np.zeros(shape=(gdat.width, gdat.height))
				padded_error = np.zeros(shape=(gdat.width, gdat.height))
				padded_exposure = np.zeros(shape=(gdat.width, gdat.height))
				padded_mask = np.zeros(shape=(gdat.width, gdat.height))

				padded_image[:image.shape[0]-gdat.x0, : image.shape[1]-gdat.y0] = image[gdat.x0:, gdat.y0:]
				padded_error[:image.shape[0]-gdat.x0, : image.shape[1]-gdat.y0] = error[gdat.x0:, gdat.y0:]
				padded_exposure[:image.shape[0]-gdat.x0, : image.shape[1]-gdat.y0] = exposure[gdat.x0:, gdat.y0:]
				padded_mask[:image.shape[0]-gdat.x0, : image.shape[1]-gdat.y0] = mask[gdat.x0:, gdat.y0:]

				variance = padded_error**2

				variance[variance==0.]=np.inf
				weight = 1. / variance

				self.weights.append(weight.astype(np.float32))
				self.errors.append(padded_error.astype(np.float32))
				self.data_array.append(padded_image.astype(np.float32)+gdat.mean_offset)
				self.exposures.append(padded_exposure.astype(np.float32))



			elif gdat.width > 0:

				image = image[gdat.x0:gdat.x0+gdat.width,gdat.y0:gdat.y0+gdat.height]
				error = error[gdat.x0:gdat.x0+gdat.width,gdat.y0:gdat.y0+gdat.height]
				exposure = exposure[gdat.x0:gdat.x0+gdat.width,gdat.y0:gdat.y0+gdat.height]
				mask = mask[gdat.x0:gdat.x0+gdat.width,gdat.y0:gdat.y0+gdat.height]
				image_size = (gdat.width, gdat.height)
				variance = error**2
				variance[variance==0.]=np.inf
				weight = 1. / variance
				self.weights.append(weight.astype(np.float32))
				self.errors.append(error.astype(np.float32))
				self.data_array.append(image.astype(np.float32)+gdat.mean_offset) # constant offset, may need to change
				self.exposures.append(exposure.astype(np.float32))

			else:
				image_size = (image.shape[0], image.shape[1])
				variance = error**2
				variance[variance==0.]=np.inf
				weight = 1. / variance

				self.weights.append(weight.astype(np.float32))
				self.errors.append(error.astype(np.float32))
				self.data_array.append(image.astype(np.float32)+gdat.mean_offset) # constant offset, may need to change
				self.exposures.append(exposure.astype(np.float32))

			if i==0:
				gdat.imsz0 = image_size
			gdat.imszs.append(image_size)
			gdat.regsizes.append(image_size[0]/gdat.nregion)


			gdat.frac = np.count_nonzero(weight)/float(gdat.width*gdat.height)
			
			psf, cf, nc, nbin = get_gaussian_psf_template(pixel_fwhm=gdat.psf_pixel_fwhm)

			print('sum of PSF is ', np.sum(psf))
			self.psfs.append(psf)
			self.cfs.append(cf)
			self.ncs.append(nc)
			self.nbins.append(nbin)
			self.biases.append(gdat.bias)


		# gdat.regsize = gdat.imsz0[0]/gdat.nregion
		gdat.regions_factor = 1./float(gdat.nregion**2)
		print(gdat.imsz0[0], gdat.regsizes[0], gdat.regions_factor)
		assert gdat.imsz0[0] % gdat.regsizes[0] == 0 
		assert gdat.imsz0[1] % gdat.regsizes[0] == 0 

		pixel_variance = np.median(self.errors[0]**2)
		print('pixel_variance:', pixel_variance)
		gdat.N_eff = 4*np.pi*(gdat.psf_pixel_fwhm/2.355)**2
		gdat.err_f = np.sqrt(gdat.N_eff * pixel_variance)/5


		# return gdat

# -------------------- actually execute the thing ----------------

class lion():

	gdat = gdatstrt()


	def __init__(self, 
			# resizes images to largest square dimension modulo nregion
			auto_resize = True, \

			#specify these if you want to fix the dimension of incoming image
			width = 0, \
			height = 0, \
			
			# these set x/y coordinate of lower left corner if cropping image
			x0 = 0, \
			y0 = 0, \

			#indices of bands used in fit, where 0->250um, 1->350um and 2->500um.
			band0 = 0, \
			band1 = None, \
			band2 = None, \
			
			# absolute level subtracted from SPIRE model image, bias and mean_offset are 
			# redundant at the moment but don't worry for now
			bias = 0.0, \
			mean_offset = 0.0035, \

			psf_pixel_fwhm = 3.0, \

			# use if loading data from object and not from saved fits files in directories
			map_object = None, \

			# Configure these for individual directory structure
			base_path = '/Users/richardfeder/Documents/multiband_pcat/', \
			result_path = '/Users/richardfeder/Documents/multiband_pcat/spire_results',\
			
			# name of cluster being analyzed
			dataname = 'a0370', \

			# mock dataset name
			mock_name = None, \
			
			# filepath for previous catalog if using as an initial state. loads in .npy files
			load_state_timestr = None,\
			
			# set flag to True if you want posterior plots/catalog samples/etc from run saved
			save = True, \

			# number of thinned samples
			nsamp = 500, \

			# factor by which the chain is thinned
			nloop = 1000, \

			# scalar factor in regularization prior, scales dlogL penalty when adding/subtracting a source
			alph = 1.0, \

			# scale for merge proposal i.e. how far you look for neighbors to merge
			kickrange = 1.0, \

			# used in subregion model evaluation
			margin = 10, \

			# maximum number of sources allowed in the code, might change depending on the image
			max_nsrc = 2000, \

			# nominal number of sources expected in a given image, helps set sample step sizes during MCMC
			nominal_nsrc = 1000, \

			# splits up image into subregions to do proposals within
			nregion = 5, \

			# used when splitting sources and determining colors of resulting objects
			split_col_sig = 0.25, \

			# number counts power law slope for sources
			truealpha = 3.0, \

			# minimum flux allowed in fit for SPIRE sources (Jy)
			trueminf = 0.001, \

			# interactive backend should be loaded before importing pyplot
			visual = False, \

			# used for visual mode
			weighted_residual = False, \

			# to show raw number counts set to True
			raw_counts = False, \

			# verbosity during program execution
			verbtype = 0, \

			# number of residual samples to average for final product
			residual_samples = 100, \

			# set to True to automatically make posterior/diagnostic plots after the run 
			make_post_plots = False, \

			# used for computing posteriors
			burn_in_frac = 0.6, 

			# save posterior plots
			bool_plot_save = True, \

			# return median model image from last 'residual_samples' samples 
			return_median_model = False, \

			# set to True if using CBLAS library
			cblas=False):


		for attr, valu in locals().items():
		# for attr, valu in locals().iteritems():
			if '__' not in attr and attr != 'gdat' and attr != 'map_object':
				setattr(self.gdat, attr, valu)

		self.gdat.band_dict = dict({0:'S',1:'M',2:'L'}) # for accessing different wavelength filenames
		self.gdat.timestr = time.strftime("%Y%m%d-%H%M%S")
		self.gdat.bands = []
		self.gdat.bands.append(self.gdat.band0)
		if self.gdat.band1 is not None:
			self.gdat.bands.append(self.gdat.band1)
			if self.gdat.band2 is not None:
				self.gdat.bands.append(self.gdat.band2)
		self.gdat.nbands = len(self.gdat.bands)

		self.data = pcat_data(self.gdat.auto_resize, self.gdat.nregion)



		self.data.load_in_data(self.gdat, map_object=map_object)


		if self.gdat.save:
			#create directory for results, save config file from run
			frame_dir, newdir = create_directories(self.gdat)
			self.gdat.frame_dir = frame_dir
			self.gdat.newdir = newdir
			save_params(newdir, self.gdat)


	def main(self):

		''' Here is where we initialize the C libraries and instantiate the arrays that will store our thinned samples and other stats '''
		if self.gdat.cblas:
			libmmult = npct.load_library('pcat-lion', '.')
		else:
			libmmult = ctypes.cdll['blas.so'] # not sure how stable this is, trying to find a good Python 3 fix to deal with path configuration
			# libmmult = npct.load_library('blas', '.')
			# libmmult = npct.load_library('blas', self.gdat.base_path+'pcat-lion-master/')

		initialize_c(self.gdat, libmmult, cblas=self.gdat.cblas)

		start_time = time.clock()

		samps = Samples(self.gdat)
		model = Model(self.gdat, self.data, libmmult)


		# run sampler for gdat.nsamp thinned states

		for j in range(self.gdat.nsamp):
			print('Sample', j)

			_, chi2_all, statarrays,  accept_fracs, diff2_list, rtype_array, accepts, resids = model.run_sampler()
			samps.add_sample(j, model, diff2_list, accepts, rtype_array, accept_fracs, chi2_all, statarrays, resids)


		if self.gdat.save:
			print('saving...')

			# save catalog ensemble and other diagnostics
			samps.save_samples(self.gdat.result_path, self.gdat.timestr)

			# save final catalog state
			np.savez(self.gdat.result_path + '/'+str(self.gdat.timestr)+'/final_state.npz', cat=model.stars)

		if self.gdat.make_post_plots:
			result_plots(gdat = self.gdat)


		dt_total = time.clock() - start_time
		print('Full Run Time (s):', np.round(dt_total,3))
		print('Time String:', str(self.gdat.timestr))

		if self.gdat.return_median_model:
			models = []
			for b in range(self.gdat.nbands):
				model_samples = np.array([self.data.data_array[b]-samps.residuals[b][i] for i in range(self.gdat.residual_samples)])
				median_model = np.median(model_samples, axis=0)
				models.append(median_model)

			return models


'''These lines below here are to instantiate the script without having to load an updated 
version of the lion module every time I make a change, but when Lion is wrapped within another pipeline
these should be moved out of the script and into the pipeline'''

# ob = lion(raw_counts=True, auto_resize=True, visual=True)
# ob = lion(band0=0, band1=2, cblas=True, auto_resize=True, make_post_plots=True, nsamp=1000, residual_samples=100)
# ob = lion(band0=0, cblas=True, auto_resize=True, make_post_plots=True, nsamp=1000, residual_samples=100)
# ob.main()

# result_plots(timestr='20190919-113227', burn_in_frac=0.5, boolplotsave=True, boolplotshow=False, plttype='png', gdat=None)




