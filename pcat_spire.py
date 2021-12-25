from __future__ import print_function
import numpy as np
import numpy.ctypeslib as npct
import ctypes
from ctypes import c_int, c_double
# in order for visual=True to work, interactive backend should be loaded before importing pyplot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import os
import os.path
from os import path
import sys
import warnings
import scipy.stats as stats
from scipy.ndimage import gaussian_filter
from image_eval import psf_poly_fit, image_model_eval
from fast_astrom import *
import pickle
from spire_data_utils import *
from spire_roc_condensed_cat import *
from spire_plotting_fns import *
from fourier_bkg_modl import *


np.seterr(divide='ignore', invalid='ignore')

class objectview(object):
	def __init__(self, d):
		self.__dict__ = d

class gdatstrt(object):
	''' Initializes global data object used throughout PCAT. '''
	def __init__(self):
		pass
	
	def __setattr__(self, attr, valu):
		super(gdatstrt, self).__setattr__(attr, valu)

def add_directory(dirpath):
	if not os.path.isdir(dirpath):
		os.makedirs(dirpath)
	return dirpath

def create_directories(gdat):
	''' 
	Makes initial directory structure for new PCAT run.

	Parameters
	----------

	gdat : global data object

	Returns
	-------

	frame_dir_name : `str'
		New PCAT frame directory name

	new_dir_name : `str'
		New PCAT parent directory name

	timestr : `str'
		Time string associated with new PCAT run

	'''
	new_dir_name = gdat.result_path+'/'+gdat.timestr
	timestr = gdat.timestr
	if os.path.isdir(gdat.result_path+'/'+gdat.timestr):
		i = 0
		time.sleep(np.random.uniform(0, 5))
		while os.path.isdir(gdat.result_path+'/'+gdat.timestr+'_'+str(i)):
			time.sleep(np.random.uniform(0, 2))
			i += 1
		
		timestr = gdat.timestr+'_'+str(i)
		new_dir_name = gdat.result_path+'/'+timestr
		
	frame_dir_name = new_dir_name+'/frames'
	
	if not os.path.isdir(frame_dir_name):
		os.makedirs(frame_dir_name)
	

	print('timestr:', timestr)
	return frame_dir_name, new_dir_name, timestr



def verbprint(verbose, text, file=None, verbthresh=0):
	''' 
	This function is a wrapped print function that accommodates various levels of verbosity. 
	This is an in place operation.

	Parameters
	----------

	verbose : 'int'. Level of verbosity. If verbthresh is None, verbose=1 will result in a statement being printed, otherwise verbose needs to be greater than verbthresh.
	text : 'str'. Text to print. 
	file (optional) : 'str'. User can specifiy file to write logs to. (I'm not sure if this fully works).
			Default is 'None'.
	verbthresh (optional) : Verbosity threshold. Default is 'None' (meaning the verbosity threshold is >0).

	'''
	if verbthresh is not None:
		if verbose > verbthresh:
			print(text, file=file)
	else:
		if verbose:
			print(text, file=file)


def compute_Fstat_alph(imszs, nbands, nominal_nsrc):
	''' 
	Computes expected improvement in log-likelihood per degree of freedom (DOF) in the finite DOF limit through the F-statistic (https://en.wikipedia.org/wiki/F-test).

	Parameters
	----------

	imszs : 'np.array' of 'floats'.
	nbands : 'int'. Number of bands in fit.
	nominal_nsrc : 'int'. Number of expected sources in the fit. This could be adapted at a later point in the chain, but is currently fixed ab initio.

	Returns
	-------

	alph : 'float'. Expected improvement in log-likelihood per degree of freedom.

	'''

	npix = np.sum(np.array([imszs[b][0]*imszs[b][1] for b in range(nbands)]))
	alph = 0.5*(2.+nbands)*npix/(npix - (2.+nbands)*nominal_nsrc)
	alph /= 0.5*(2.+nbands) # regularization prior is normalized relative to limit with infinite data, per degree of freedom

	return alph

def fluxes_to_color(flux1, flux2):

	return 2.5*np.log10(flux1/flux2)

def get_band_weights(band_idxs):
	weights = []
	for idx in band_idxs:
		if idx is None:
			weights.append(0.)
		else:
			weights.append(1.)
	weights /= np.sum(weights)

	return weights 

def initialize_c(gdat, libmmult, cblas=False):

	''' 

	This function initializes the C library needed for the core numerical routines in PCAT. 
	This is an in place operation.
	
	Parameters
	----------
	
	gdat : global data object usued by PCAT

	libmmult : Matrix multiplication library

	cblas (optional) : If True, use CBLAS matrix multiplication routines in model evaluation

	'''

	verbprint(gdat.verbtype, 'initializing c routines and data structs', file=gdat.flog, verbthresh=1)
	# if gdat.verbtype > 1:
	# 	print('initializing c routines and data structs', file=gdat.flog)

	array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
	array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
	array_2d_double = npct.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
	array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")

	if cblas:
		if os.path.getmtime('pcat-lion.c') > os.path.getmtime('pcat-lion.so'):
			warnings.warn('pcat-lion.c modified after compiled pcat-lion.so', Warning)		
				
		libmmult.pcat_model_eval.restype = None
		libmmult.pcat_model_eval.argtypes = [c_int, c_int, c_int, c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_1d_int, array_1d_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]
		libmmult.pcat_imag_acpt.restype = None
		libmmult.pcat_imag_acpt.argtypes = [c_int, c_int, array_2d_float, array_2d_float, array_2d_int, c_int, c_int, c_int, c_int]
		libmmult.pcat_like_eval.restype = None
		libmmult.pcat_like_eval.argtypes = [c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]

	else:
		if os.path.getmtime('blas.c') > os.path.getmtime('blas.so'):
			warnings.warn('blas.c modified after compiled blas.so', Warning)		
		
		libmmult.clib_eval_modl.restype = None
		libmmult.clib_eval_modl.argtypes = [c_int, c_int, c_int, c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_1d_int, array_1d_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]
		libmmult.clib_updt_modl.restype = None
		libmmult.clib_updt_modl.argtypes = [c_int, c_int, array_2d_float, array_2d_float, array_2d_int, c_int, c_int, c_int, c_int]
		libmmult.clib_eval_llik.restype = None
		libmmult.clib_eval_llik.argtypes = [c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]


def icdf_dpow(unit, minm, maxm, brek, sloplowr, slopuppr):
    
    ''' Inverse CDF for double power law, taken from https://github.com/tdaylan/pcat/blob/master/pcat/main.py

	Parameters
	----------
	
	unit : 'np.array' of type 'float'. Uniform draws from CDF

	minm/maxm : 'floats'. Minimum/maximum bounds on flux density distribution. These parameters are used to normalize the distribution so that prior samples can be drawn.

	brek : 'float'. Pivot value for the flux distribution

	sloplowr/slopuppr : 'floats'. Power law parameters for lower/upper end of the CDF

	Returns
	-------

	para : 'np.array' of type 'float'. Sample of flux densities corresponding to CDF draws (unit).

    '''
    
    if np.isscalar(unit):
        unit = np.array([unit])
    
    faca = 1. / (brek**(sloplowr - slopuppr) * (brek**(1. - sloplowr) - minm**(1. - sloplowr)) \
                                / (1. - sloplowr) + (maxm**(1. - slopuppr) - brek**(1. - slopuppr)) / (1. - slopuppr))
    facb = faca * brek**(sloplowr - slopuppr) / (1. - sloplowr)

    para = np.empty_like(unit)
    cdfnbrek = facb * (brek**(1. - sloplowr) - minm**(1. - sloplowr))
    indxlowr = np.where(unit <= cdfnbrek)[0]
    indxuppr = np.where(unit > cdfnbrek)[0]
    if indxlowr.size > 0:
        para[indxlowr] = (unit[indxlowr] / facb + minm**(1. - sloplowr))**(1. / (1. - sloplowr))
    if indxuppr.size > 0:
        para[indxuppr] = ((1. - slopuppr) * (unit[indxuppr] - cdfnbrek) / faca + brek**(1. - slopuppr))**(1. / (1. - slopuppr))
    
    return para

def pdfn_dpow(xdat, minm, maxm, brek, sloplowr, slopuppr):
    
    ''' PDF for double power law, also taken from https://github.com/tdaylan/pcat/blob/master/pcat/main.py'''

    if np.isscalar(xdat):
        xdat = np.array([xdat])
    
    faca = 1. / (brek**(sloplowr - slopuppr) * (brek**(1. - sloplowr) - minm**(1. - sloplowr)) / \
                                            (1. - sloplowr) + (maxm**(1. - slopuppr) - brek**(1. - slopuppr)) / (1. - slopuppr))
    facb = faca * brek**(sloplowr - slopuppr) / (1. - sloplowr)
    
    pdfn = np.empty_like(xdat)
    indxlowr = np.where(xdat <= brek)[0]
    indxuppr = np.where(xdat > brek)[0]
    if indxlowr.size > 0:
        pdfn[indxlowr] = faca * brek**(sloplowr - slopuppr) * xdat[indxlowr]**(-sloplowr)
    if indxuppr.size > 0:
        pdfn[indxuppr] = faca * xdat[indxuppr]**(-slopuppr)
    
    return pdfn

def save_params(directory, gdat):
	''' 
	Save parameters as dictionary, then pickle them to .txt file. This also produces a more human-readable file, params_read.txt. 
	This is an inplace operation. 

	Parameters
	----------

	directory : 'str'. MCMC run result directory to store parameter configuration file.
	gdat : global data object used by PCAT.
	'''
	param_dict = vars(gdat).copy()
	param_dict['fc_templates'] = None # these take up too much space and not necessary
	param_dict['truth_catalog'] = None 
	
	with open(directory+'/params.txt', 'wb') as file:
		file.write(pickle.dumps(param_dict))

	file.close()

	with open(directory+'/params_read.txt', 'w') as file2:
		for key in param_dict:
			file2.write(key+': '+str(param_dict[key])+'\n')
	file2.close()


def neighbours(x,y,neigh,i,generate=False):
	''' 
	Neighbours function is used in merge proposal, where you have some source and you want to choose a nearby
	source with some probability to merge. 
	'''

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
	return (np.floor(x + offsetx).astype(np.int) / regsize).astype(np.int)

def idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize):
	match_x = (get_region(x[0:n], offsetx, regsize) % 2) == parity_x
	match_y = (get_region(y[0:n], offsety, regsize) % 2) == parity_y
	return np.flatnonzero(np.logical_and(match_x, match_y))

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
		self.change_bkg_bool = False
		self.change_template_amp_bool = False
		self.change_fourier_comp_bool = False
		self.perturb_band_idx = None
		
		self.dback = np.zeros(gdat.nbands, dtype=np.float32)
		self.dtemplate = None

		if gdat.float_fourier_comps:
			self.dfc = np.zeros((gdat.n_fourier_terms, gdat.n_fourier_terms, 4))
			self.dfc_rel_amps = np.zeros(gdat.nbands, dtype=np.float32)
			self.fc_rel_amp_bool = False

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

	def add_move_stars(self, idx_move, stars0, starsp):
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

	def change_bkg(self, perturb_band_idx=None):
		self.goodmove = True
		self.change_bkg_bool = True
		if perturb_band_idx is not None:
			self.perturb_band_idx = perturb_band_idx

	def change_template_amplitude(self, perturb_band_idx=None):
		self.goodmove = True
		self.change_template_amp_bool = True
		if perturb_band_idx is not None:
			self.perturb_band_idx = perturb_band_idx

	def change_fourier_comp(self):
		self.goodmove = True
		self.change_fourier_comp_bool = True

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
		elif self.change_bkg_bool:
			return self.stars0[self._X,:], self.stars0[self._Y,:]


class Model:

	_X = 0
	_Y = 1
	_F = 2

	k = 2.5/np.log(10)

	color_mus, color_sigs = [], []
	
	''' the init function sets all of the data structures used for the catalog, 
	randomly initializes catalog source values drawing from catalog priors  '''
	def __init__(self, gdat, dat, libmmult=None, newsrc_minmax_range=500):

		self.dat = dat
		self.gdat = gdat
		self.libmmult = libmmult
		self.newsrc_minmax_range = newsrc_minmax_range
		self.err_f = gdat.err_f
		self.pixel_per_beam = [2*np.pi*(psf_pixel_fwhm/2.355)**2 for psf_pixel_fwhm in self.gdat.psf_fwhms] # variable pixel fwhm
		self.linear_flux = gdat.linear_flux
		self.imsz0 = gdat.imsz0 # this is just for first band, where proposals are first made
		self.imszs = gdat.imszs # this is list of image sizes for all bands, not just first one
		self.kickrange = gdat.kickrange
		self.margins = np.zeros(gdat.nbands).astype(np.int)
		self.max_nsrc = gdat.max_nsrc
		self.bkg = np.array(gdat.bias)

		print('at initialization of model, self.bkg is ', self.bkg)

		# the last weight, used for background amplitude sampling, is initialized to zero and set to be non-zero by lion after some preset number of samples, 
		# so don't change its value up here. There is a bkg_sample_weight parameter in the lion() class
		
		self.moveweights = np.array([0., 0., 0., 0., 0., 0.]) # fourier comp, movestar. weights are specified in lion __init__()
		self.movetypes = ['P *', 'BD *', 'MS *', 'BKG', 'TEMPLATE', 'FC']

		self.n_templates = gdat.n_templates

		self.temp_amplitude_sigs = dict({'sze':0.001, 'dust':0.1, 'planck':0.05, 'fc':0.004})
		if self.gdat.sz_amp_sig is not None:
			self.temp_amplitude_sigs['sze'] = self.gdat.sz_amp_sig
		if self.gdat.fc_amp_sig is not None:
			self.temp_amplitude_sigs['fc'] = self.gdat.fc_amp_sig
		
		# this is for perturbing the relative amplitudes of a fixed fourier comp model across bands
		self.fourier_amp_sig = gdat.fourier_amp_sig
		self.template_amplitudes = np.zeros((self.n_templates, gdat.nbands))
		self.init_template_amplitude_dicts = self.gdat.init_template_amplitude_dicts
		self.dtemplate = np.zeros_like(self.template_amplitudes)

		for i, key in enumerate(self.gdat.template_order):
			for b, band in enumerate(gdat.bands):
				self.template_amplitudes[i][b] = self.init_template_amplitude_dicts[key][gdat.band_dict[band]]
		
		if self.gdat.float_fourier_comps:
			if self.gdat.init_fourier_coeffs is not None:
				self.fourier_coeffs = self.gdat.init_fourier_coeffs.copy()
			
			self.fourier_templates = self.gdat.fc_templates
			
			if self.gdat.bkg_moore_penrose_inv:

				print('Moore Penrose inverse is happening!!!!!')
				self.dat.data_array[0] -= np.nanmean(self.dat.data_array[0]) # this is done to isolate the fluctuation component
				self.bkg[0] = 0.

				_, _, _, bt_siginv_b_inv, mp_coeffs = compute_marginalized_templates(self.gdat.MP_order, self.dat.errors[0],\
														fourier_templates=self.gdat.fc_templates[0][:self.gdat.MP_order,:self.gdat.MP_order,:], \
														data = self.dat.data_array[0], mean_sig=self.gdat.mean_sig, ridge_fac=self.gdat.ridge_fac)

				self.fourier_coeffs[:self.gdat.MP_order, :self.gdat.MP_order, :] = mp_coeffs.copy()

				# this is for quick marginalization of the Fourier components with the data.
				# _, _, _, bt_siginv_b_inv, A_hat = compute_Ahat_templates(self.gdat.MP_order, self.dat.errors[0],\
				# 														fourier_templates=self.gdat.fc_templates[0][:self.gdat.MP_order,:self.gdat.MP_order,:], \
				# 														data = self.dat.data_array[0], mean_sig=self.gdat.mean_sig, ridge_fac=self.gdat.ridge_fac)

				# init_coeffs = np.empty((self.gdat.MP_order,self.gdat.MP_order,4))
				# count = 0
				# for i in range(self.gdat.MP_order):
				# 	for j in range(self.gdat.MP_order):
				# 		for k in range(4):
				# 			init_coeffs[i,j,k] = A_hat[count]
				# 			count += 1

				# self.fourier_coeffs[:self.gdat.MP_order, :self.gdat.MP_order, :] = A_hat.copy()

			self.n_fourier_terms = self.gdat.n_fourier_terms
			self.dfc = np.zeros((self.n_fourier_terms, self.n_fourier_terms, 4))
			self.dfc_rel_amps = np.zeros((gdat.nbands))
			self.fc_rel_amps = self.gdat.fc_rel_amps
		else:
			self.fc_rel_amps = None
			self.fourier_coeffs = None
		
		if self.gdat.nsrc_init is not None:
			self.n = self.gdat.nsrc_init
		else:
			self.n = np.random.randint(gdat.max_nsrc)+1

		self.nbands = gdat.nbands
		self.nloop = gdat.nloop
		self.nominal_nsrc = gdat.nominal_nsrc
		self.nregion = gdat.nregion
		self.offsetxs = np.zeros(self.nbands).astype(np.int)
		self.offsetys = np.zeros(self.nbands).astype(np.int)
		
		self.penalty = (2.+gdat.nbands)*0.5*gdat.alph
		print('PENALTY is ', self.penalty)

		self.regions_factor = gdat.regions_factor
		self.regsizes = np.array(gdat.regsizes).astype(np.int)
		
		self.stars = np.zeros((2+gdat.nbands,gdat.max_nsrc), dtype=np.float32)
		self.stars[:,0:self.n] = np.random.uniform(size=(2+gdat.nbands,self.n))
		self.stars[self._X,0:self.n] *= gdat.imsz0[0]-1
		self.stars[self._Y,0:self.n] *= gdat.imsz0[1]-1

		self.truealpha = gdat.truealpha

		# additional parameters for double power law
		self.alpha_1 = gdat.alpha_1
		self.alpha_2 = gdat.alpha_2
		self.pivot_dpl = gdat.pivot_dpl
		self.trueminf = gdat.trueminf
		self.verbtype = gdat.verbtype

		self.bkg_prop_sigs = np.array([self.gdat.bkg_sig_fac[b]*np.nanmedian(self.dat.errors[b][self.dat.errors[b]>0])/np.sqrt(self.dat.fracs[b]*self.imszs[b][0]*self.imszs[b][1]) for b in range(gdat.nbands)])
		print('BKG PROP SIGS is ', self.bkg_prop_sigs)
		if gdat.bkg_prior_mus is not None:
			self.bkg_prior_mus = gdat.bkg_prior_mus
		else:
			self.bkg_prior_mus = self.bkg.copy()
		self.bkg_prior_sig = gdat.bkg_prior_sig

		self.dback = np.zeros_like(self.bkg)
		
		if self.gdat.color_mus is not None:
			self.mus = self.gdat.color_mus
		else:			
			self.mus = dict({'S-M':0.0, 'M-L':0.5, 'L-S':0.5, 'M-S':0.0, 'S-L':-0.5, 'L-M':-0.5})
		if self.gdat.color_sigs is not None:
			self.sigs = self.gdat.color_sigs
		else:
			self.sigs = dict({'S-M':1.5, 'M-L':1.5, 'L-S':1.5, 'M-S':1.5, 'S-L':1.5, 'L-M':1.5}) #very broad color prior

		for b in range(self.nbands-1):

			if self.linear_flux:
				col_string = self.gdat.band_dict[self.gdat.bands[0]]+'/'+self.gdat.band_dict[self.gdat.bands[b+1]]
				self.color_mus.append(self.linear_mus[col_string])
				self.color_sigs.append(self.linear_sigs[col_string])
			else:
				col_string = self.gdat.band_dict[self.gdat.bands[0]]+'-'+self.gdat.band_dict[self.gdat.bands[b+1]]
				self.color_mus.append(self.mus[col_string])
				self.color_sigs.append(self.sigs[col_string])
			

		print('self.color_mus : ', self.color_mus)
		print('self.color_sigs : ', self.color_sigs)

		# unless previous model state provided to PCAT (for example, from a previous run), draw fluxes from specified flux prior.
		if gdat.load_state_timestr is None:
			for b in range(gdat.nbands):
				if b==0:
					if self.gdat.flux_prior_type=='double_power_law':

						self.stars[self._F+b,0:self.n] = icdf_dpow(self.stars[self._F+b,0:self.n],\
																	 self.trueminf, self.trueminf*self.newsrc_minmax_range, \
																	 self.pivot_dpl, self.alpha_1, self.alpha_2)
					elif self.gdat.flux_prior_type=='single_power_law':
						self.stars[self._F+b,0:self.n] **= -1./(self.truealpha - 1.)
						self.stars[self._F+b,0:self.n] *= self.trueminf
				else:
					new_colors = np.random.normal(loc=self.color_mus[b-1], scale=self.color_sigs[b-1], size=self.n)
					
					if self.linear_flux:
						self.stars[self._F+b,0:self.n] = self.stars[self._F,0:self.n]*new_colors
					else:
						self.stars[self._F+b,0:self.n] = self.stars[self._F,0:self.n]*10**(0.4*new_colors)
		else:
			# if loading in a previous catalog, make sure the bands of the catalog are in order
			print('Loading in catalog from run with timestr='+gdat.load_state_timestr+'...', file=gdat.flog)
			catpath = gdat.result_path+'/'+gdat.load_state_timestr+'/final_state.npz'
			catload = np.load(catpath)
			gdat_previous, _, _ = load_param_dict(gdat.load_state_timestr, result_path=gdat.result_path)
			previous_cat = np.load(catpath)['cat']
			self.n = np.count_nonzero(previous_cat[self._F,:])

			if self.gdat.float_background:
				for b in range(gdat_previous.nbands):
					self.bkg[b] = catload['bkg'][b]
			
			if self.gdat.float_templates:
				print('self template amplitudes is ', self.template_amplitudes)
				if gdat_previous.nbands == gdat.nbands:
					self.template_amplitudes=catload['templates']
				else:
					for t in range(self.n_templates):
						for b in range(gdat_previous.nbands):
							self.template_amplitudes[t, b] = catload['templates'][t,b]

			if self.gdat.float_fourier_comps:
				self.gdat.fourier_coeffs = catload['fourier_coeffs']

			if gdat_previous.nbands == gdat.nbands:
				print('same number of bands, set catalogs equal to each other')
				self.stars = previous_cat
			else:
				print('were gonna have to draw some colors babyy')
				self.stars[self._X,:] = previous_cat[self._X,:]
				self.stars[self._Y,:] = previous_cat[self._Y,:]
				for b in range(gdat.nbands):
					if gdat_previous.nbands > b:
						self.stars[self._F+b,:] = previous_cat[self._F+b,:]
					else:
						print('drawing colors on band ', b)
						new_colors = np.random.normal(loc=self.color_mus[b-1], scale=self.color_sigs[b-1], size=self.n)
						self.stars[self._F+b,0:self.n] = self.stars[self._F,0:self.n]*10**(0.4*new_colors)


			print('self.bkg is ', self.bkg, file=gdat.flog)
			print('self.template amplitudes is ', self.template_amplitudes, file=gdat.flog)


	def load_previous_model(self):
		#TODO wrap above code 
		pass 

	def update_moveweights(self, j):
		''' 
		During the burn in stage of sampling, this function gets used to update the proposals PCAT draws from with the specified weights. 
		'''
		moveweight_idx_dict = dict({'movestar':0, 'birth_death':1, 'merge_split':2, 'bkg':3, 'template':4, 'fourier_comp':5})
		sample_delays = [self.gdat.movestar_sample_delay, self.gdat.birth_death_sample_delay, self.gdat.merge_split_sample_delay, self.gdat.bkg_sample_delay, \
						self.gdat.temp_sample_delay, self.gdat.fc_sample_delay]
		moveweight_dict = dict({0:self.gdat.movestar_moveweight, 1:self.gdat.birth_death_moveweight, 2:self.gdat.merge_split_moveweight, 3:self.gdat.bkg_moveweight, \
								4:self.gdat.template_moveweight, 5:self.gdat.fourier_comp_moveweight})
		proposal_bools = dict({0:True, 1:True, 2:True, 3:self.gdat.float_background, 4:self.gdat.float_templates, 5:self.gdat.float_fourier_comps})

		key_list = list(moveweight_idx_dict.keys()) 
		val_list = list(moveweight_idx_dict.values()) 

		for moveidx, sample_delay in enumerate(sample_delays):
			if j == sample_delay and proposal_bools[moveidx]:
				print('starting '+str(key_list[val_list.index(moveidx)])+' proposals')
				self.moveweights[moveidx] = moveweight_dict[moveidx]
				self.moveweights[np.isnan(self.moveweights)] = 0.

				print('moveweights:', self.moveweights, file=self.gdat.flog)

	def normalize_weights(self, weights):
		''' 
		This gets used when updating proposal weights during burn-in.
		'''
		normalized_weights = weights / np.sum(weights)

		return normalized_weights
   
	def print_sample_status(self, dts, accept, outbounds, chi2, movetype, bkg_perturb_band_idxs=None, temp_perturb_band_idxs=None):  
		''' 
		This function prints out some information at the end of each thinned sample, 
		namely acceptance fractions for the different proposals and some time performance statistics as well. 
		'''  
		fmtstr = '\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f'
		print('Background '+str(np.round(self.bkg, 5)) + ', N_star '+str(self.n)+' chi^2 '+str(list(np.round(chi2, 2))), file=self.gdat.flog)
		print('Reduced chi^2 ', [np.round(chi2[b]/(self.dat.fracs[b]*self.dat.data_array[b].shape[0]*self.dat.data_array[b].shape[1]), 2) for b in range(self.gdat.nbands)])

		dts *= 1000
		accept_fracs = []
		timestat_array = np.zeros((6, 1+len(self.moveweights)), dtype=np.float32)
		statlabels = ['Acceptance', 'Out of Bounds', 'Proposal (s)', 'Likelihood (s)', 'Implement (s)', 'Coordinates (s)']
		statarrays = [accept, outbounds, dts[0,:], dts[1,:], dts[2,:], dts[3,:]]

		if bkg_perturb_band_idxs is not None:
			per_band_bkg_acpt = []
			bkg_all_acpts = np.array(statarrays[0][movetype==3])
			for b in range(self.gdat.nbands):
				per_band_bkg_acpt.append(np.mean(bkg_all_acpts[bkg_perturb_band_idxs==b]))
			print('Per band background accept : ', np.round(per_band_bkg_acpt, 3))

		if temp_perturb_band_idxs is not None:
			per_band_temp_acpt = []
			temp_all_acpts = np.array(statarrays[0][movetype==4])
			for b in range(self.gdat.nbands):
				per_band_temp_acpt.append(np.mean(temp_all_acpts[temp_perturb_band_idxs==b]))
			print('Per band SZ accept : ', np.round(per_band_temp_acpt, 3))


		for j in range(len(statlabels)):
			timestat_array[j][0] = np.sum(statarrays[j])/1000
			if j==0:
				accept_fracs.append(np.sum(statarrays[j])/1000)
			print(statlabels[j]+'\t(all) %0.3f' % (np.sum(statarrays[j])/1000), file=self.gdat.flog)
			for k in range(len(self.movetypes)):
				if j==0:
					accept_fracs.append(np.mean(statarrays[j][movetype==k]))
				timestat_array[j][1+k] = np.mean(statarrays[j][movetype==k])
				print('('+self.movetypes[k]+') %0.3f' % (np.mean(statarrays[j][movetype == k])), end=' ', file=self.gdat.flog)
			print(file=self.gdat.flog)
			if j == 1:
				print('-'*16, file=self.gdat.flog)
		print('-'*16, file=self.gdat.flog)
		print('Total (s): %0.3f' % (np.sum(statarrays[2:])/1000), file=self.gdat.flog)
		print('='*16, file=self.gdat.flog)

		return timestat_array, accept_fracs


	def pcat_multiband_eval(self, x, y, f, bkg, nc, cf, weights, ref, lib, beam_fac=1., margin_fac=1, dtemplate=None, rtype=None, dfc=None, idxvec=None, precomp_temps=None, fc_rel_amps=None, \
		perturb_band_idx=None):
		'''
		Wrapper for multiband likelihood evaluation given catalog model parameters.
		'''

		dmodels = []
		dt_transf = 0
		nb = 0

		for b in range(self.nbands):
			dtemp = None

			if perturb_band_idx is not None:
				if b != perturb_band_idx:
					dmodels.append(None)
					continue

			if dtemplate is not None:
				dtemp = []
				for i, temp in enumerate(self.dat.template_array[b]):
					verbprint(self.gdat.verbtype, 'dtemplate in multiband eval is '+str(dtemplate.shape), verbthresh=1)
					if temp is not None and dtemplate[i][b] != 0.:
						dtemp.append(dtemplate[i][b]*temp)
				if len(dtemp) > 0:
					dtemp = np.sum(np.array(dtemp), axis=0).astype(np.float32)
				else:
					dtemp = None

			if precomp_temps is not None:
				pc_temp = precomp_temps[b]

				# if passing fixed fourier comp template, fc_rel_amps should be model + d_rel_amps, if perturbing
				# relative amplitude, fc_rel_amps should be one hot vector with change in one of the bands
				if dtemp is None:
					dtemp = fc_rel_amps[b]*pc_temp
				else:
					dtemp += fc_rel_amps[b]*pc_temp

			elif dfc is not None:

				if idxvec is not None:
					pc_temp = self.fourier_templates[b][idxvec[0], idxvec[1], idxvec[2]]*dfc[idxvec[0], idxvec[1], idxvec[2]]

					if dtemp is None:
						dtemp = fc_rel_amps[b]*pc_temp
					else:
						dtemp += fc_rel_amps[b]*pc_temp

				else:
					pc_temp = np.sum([dfc[i,j,k]*self.fourier_templates[b][i,j,k] for i in range(self.n_fourier_terms) for j in range(self.n_fourier_terms) for k in range(4)], axis=0)

					if dtemp is None:
						dtemp = fc_rel_amps[b]*pc_temp
					else:
						dtemp += fc_rel_amps[b]*pc_temp

			if b>0:
				t4 = time.time()
				if self.gdat.bands[b] != self.gdat.bands[0]:
					xp, yp = self.dat.fast_astrom.transform_q(x, y, b-1)
				else:
					xp = x
					yp = y
				dt_transf += time.time()-t4


				dmodel, diff2 = image_model_eval(xp, yp, beam_fac[b]*nc[b]*f[b], bkg[b], self.imszs[b], \
												nc[b], np.array(cf[b]).astype(np.float32()), weights=self.dat.weights[b], \
												ref=ref[b], lib=lib, regsize=self.regsizes[b], \
												margin=self.margins[b]*margin_fac, offsetx=self.offsetxs[b], offsety=self.offsetys[b], template=dtemp)
				# diff2s += diff2
			else:    
				xp=x
				yp=y

				dmodel, diff2 = image_model_eval(xp, yp, beam_fac[b]*nc[b]*f[b], bkg[b], self.imszs[b], \
												nc[b], np.array(cf[b]).astype(np.float32()), weights=self.dat.weights[b], \
												ref=ref[b], lib=lib, regsize=self.regsizes[b], \
												margin=self.margins[b]*margin_fac, offsetx=self.offsetxs[b], offsety=self.offsetys[b], template=dtemp)
				# diff2s = diff2
			
			# if rtype==4:
			# 	plt.figure()
			# 	plt.title('bandidx = '+str(b))
			# 	plt.imshow(dmodel)
			# 	plt.colorbar()
			# 	plt.show()


			if nb==0:
				diff2s = diff2
				nb += 1
			else:
				diff2s += diff2

			dmodels.append(dmodel)

		return dmodels, diff2s, dt_transf 


	def run_sampler(self, sample_idx):
		''' 
		Main wrapper function for executing the calculation of a thinned sample in PCAT.
		run_sampler() completes nloop samples, so the function gets called 'nsamp' times in a full run.
	
		Parameters
		----------

		sample_idx : 'int'. Thinned sample index.

		Returns
		-------

		n : 'int'. Number of sources.
		chi2 : 'np.array' of type 'float' with shape (nbands,). Image model chi squared for each band.
		timestat_array : 'np.array' of type 'float'.
		accept_fracs : 'np.array' of type 'float'.
		diff2_list : 
		rtype_array : 'np.array' of type 'float' and shape (nloop,). Proposal types for nloop samples corresponding to thinned sample 'sample_idx'. 
		accept : 'np.array' of type 'bool' and shape (nloop,). Booleans indicate whether or not the proposals were accepted or not.
		resids : 'list' of 'np.arrays' of type 'float' and shape (nbands, dimx, dimy). Residual maps at end of thinned sample.
		models : 'list' of 'np.arrays' of type 'float' and shape (nbands, dimx, dimy). Model images at end of thinned sample.

		'''
		
		t0 = time.time()
		nmov = np.zeros(self.nloop)
		movetype = np.zeros(self.nloop)
		accept = np.zeros(self.nloop)
		outbounds = np.zeros(self.nloop)
		dts = np.zeros((4, self.nloop)) # array to store time spent on different proposals
		diff2_list = np.zeros(self.nloop) 


		''' I'm a bit concerned about setting the offsets for multiple observations with different sizes. 
		For now what I'll do is choose an offset for the pivot band and then compute scaled offsets for the other bands
		based on the relative sub region size, which hopefully shouldn't affect 
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

				verbprint(self.gdat.verbtype, str(self.offsetxs[b+1])+', '+str(self.offsetys[b+1])+', '+str(self.margins[b+1]), verbthresh=1)

		else:

			self.offsetxs = np.array([0 for b in range(self.gdat.nbands)])
			self.offsetys = np.array([0 for b in range(self.gdat.nbands)])


		self.nregx = int(self.imsz0[0] / self.regsizes[0] + 1)
		self.nregy = int(self.imsz0[1] / self.regsizes[0] + 1)

		resids = []

		for b in range(self.nbands):

			resid = self.dat.data_array[b].copy() # residual for zero image is data
			verbprint(self.gdat.verbtype, 'resid has shape '+str(resid.shape), verbthresh=1)

			resids.append(resid)

		evalx = self.stars[self._X,0:self.n]
		evaly = self.stars[self._Y,0:self.n]
		evalf = self.stars[self._F:,0:self.n]		
		n_phon = evalx.size

		verbprint(self.gdat.verbtype, 'Beginning of run sampler', verbthresh=1)
		verbprint(self.gdat.verbtype, 'self.n here is '+str(self.n), verbthresh=1)
		verbprint(self.gdat.verbtype, 'n_phon = '+str(n_phon), verbthresh=1)

		if self.gdat.cblas:
			lib = self.libmmult.pcat_model_eval
		else:
			lib = self.libmmult.clib_eval_modl

		dtemplate, fcoeff, running_temp = None, None, None

		if self.gdat.float_templates:
			dtemplate = self.template_amplitudes
		if self.gdat.float_fourier_comps:
			running_temp = []
			
			for b in range(self.nbands):

				running_temp.append(np.sum([self.fourier_coeffs[i,j,k]*self.fourier_templates[b][i,j,k] for i in range(self.n_fourier_terms) for j in range(self.n_fourier_terms) for k in range(4)], axis=0))
			
			running_temp = np.array(running_temp)

		models, diff2s, dt_transf = self.pcat_multiband_eval(evalx, evaly, evalf, self.bkg, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, ref=resids, lib=lib, beam_fac=self.pixel_per_beam,\
														 dtemplate=dtemplate, precomp_temps=running_temp, fc_rel_amps=self.fc_rel_amps)

		logL = -0.5*diff2s
	   
		for b in range(self.nbands):
			resids[b] -= models[b]

		'''the proposals here are: move_stars (P) which changes the parameters of existing model sources, 
		birth/death (BD) and merge/split (MS). Don't worry about perturb_astrometry. 
		The moveweights array, once normalized, determines the probability of choosing a given proposal. '''
		
		# fourier comp
		movefns = [self.move_stars, self.birth_death_stars, self.merge_split_stars, self.perturb_background, \
						self.perturb_template_amplitude, self.perturb_fourier_comp] 

		if self.gdat.nregion > 1:
			xparities = np.random.randint(2, size=self.nloop)
			yparities = np.random.randint(2, size=self.nloop)

		rtype_array = np.random.choice(self.moveweights.size, p=self.normalize_weights(self.moveweights), size=self.nloop)

		movetype = rtype_array

		bkg_perturb_band_idxs, temp_perturb_band_idxs = [], [] # used for per-band acceptance fractions

		for i in range(self.nloop):
			t1 = time.time()
			rtype = rtype_array[i]
			
			verbprint(self.verbtype, 'rtype = '+str(rtype), verbthresh=1)

			if self.nregion > 1:
				self.parity_x = xparities[i] # should regions be perturbed randomly or systematically?
				self.parity_y = yparities[i]
			else:
				self.parity_x = 0
				self.parity_y = 0

			#proposal types
			proposal = movefns[rtype]()

			dts[0,i] = time.time() - t1
			
			if proposal.goodmove:
				t2 = time.time()

				if self.gdat.cblas:
					lib = self.libmmult.pcat_model_eval
				else:
					lib = self.libmmult.clib_eval_modl

				dtemplate = None
				fcoeff = None
				bkg = None
				fc_rel_amps = None

				if self.gdat.float_templates:
					dtemplate = self.template_amplitudes+self.dtemplate
				if self.gdat.float_fourier_comps:
					fcoeff = self.fourier_coeffs+self.dfc
					fc_rel_amps=self.fc_rel_amps+self.dfc_rel_amps
				if self.gdat.float_background:
					bkg = self.bkg+self.dback

				else:
					bkg = self.bkg

				margin_fac = 1
				if rtype > 2:
					margin_fac = 0

				if rtype == 3: # background
					# recompute model likelihood with margins set to zero, use current values of star parameters and use background level equal to self.bkg (+self.dback up to this point)
					
					bkg_perturb_band_idxs.append(proposal.perturb_band_idx)

					mods, diff2s_nomargin, dt_transf = self.pcat_multiband_eval(self.stars[self._X,0:self.n], self.stars[self._Y,0:self.n], self.stars[self._F:,0:self.n], \
																bkg, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, ref=self.dat.data_array, lib=lib, \
																beam_fac=self.pixel_per_beam, margin_fac=margin_fac, rtype=rtype, dtemplate=dtemplate, precomp_temps=running_temp, fc_rel_amps=fc_rel_amps, \
																perturb_band_idx=proposal.perturb_band_idx)

					logL = -0.5*diff2s_nomargin

					dmodels, diff2s, dt_transf = self.pcat_multiband_eval(proposal.xphon, proposal.yphon, proposal.fphon, proposal.dback, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, \
													ref=resids, lib=lib, beam_fac=self.pixel_per_beam, margin_fac=margin_fac, rtype=rtype, perturb_band_idx=proposal.perturb_band_idx)
	
				
					# print('self.stars[self._F:0:self.n],', self.stars[self._F:,0:self.n])
					# print('proposal.fphon', proposal.fphon)

					# plt.figure(figsize=(12, 5))
					# plt.subplot(1,2,1)
					# plt.imshow(mods[0], cmap='Greys')
					# plt.colorbar()
					# # plt.subplot(1,3,2)
					# # plt.imshow(diff2s_nomargin, cmap='Greys')
					# # plt.colorbar()
					# plt.subplot(1,2,2)
					# plt.imshow(dmodels[0], cmap='Greys')
					# plt.colorbar()
					# plt.show()

				elif rtype == 4: # template

					temp_perturb_band_idxs.append(proposal.perturb_band_idx)
					mods, diff2s_nomargin, dt_transf = self.pcat_multiband_eval(self.stars[self._X,0:self.n], self.stars[self._Y,0:self.n], self.stars[self._F:,0:self.n], \
															bkg, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, ref=self.dat.data_array, lib=lib, \
															beam_fac=self.pixel_per_beam, margin_fac=margin_fac, dtemplate=dtemplate, rtype=rtype, precomp_temps=running_temp, fc_rel_amps=fc_rel_amps, \
															perturb_band_idx=proposal.perturb_band_idx)
					logL = -0.5*diff2s_nomargin

					dmodels, diff2s, dt_transf = self.pcat_multiband_eval(proposal.xphon, proposal.yphon, proposal.fphon, proposal.dback, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, \
													ref=resids, lib=lib, beam_fac=self.pixel_per_beam, margin_fac=margin_fac, dtemplate=proposal.dtemplate, rtype=rtype, \
													perturb_band_idx=proposal.perturb_band_idx)
	

				elif rtype == 5: # fourier comp

					mods, diff2s_nomargin, dt_transf = self.pcat_multiband_eval(self.stars[self._X,0:self.n], self.stars[self._Y,0:self.n], self.stars[self._F:,0:self.n], \
															bkg, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, ref=self.dat.data_array, lib=lib, \
															beam_fac=self.pixel_per_beam, margin_fac=margin_fac, dtemplate=dtemplate, rtype=rtype, precomp_temps=running_temp, fc_rel_amps=fc_rel_amps)
					
					logL = -0.5*diff2s_nomargin

					if proposal.fc_rel_amp_bool:

						dmodels, diff2s, dt_transf = self.pcat_multiband_eval(proposal.xphon, proposal.yphon, proposal.fphon, proposal.dback, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, \
														ref=resids, lib=lib, beam_fac=self.pixel_per_beam, margin_fac=margin_fac, rtype=rtype, precomp_temps=running_temp, fc_rel_amps=proposal.dfc_rel_amps)
		
					else:

						dmodels, diff2s, dt_transf = self.pcat_multiband_eval(proposal.xphon, proposal.yphon, proposal.fphon, proposal.dback, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, \
														ref=resids, lib=lib, beam_fac=self.pixel_per_beam, margin_fac=margin_fac, rtype=rtype, dfc=proposal.dfc, idxvec=[proposal.idx0, proposal.idx1, proposal.idxk], fc_rel_amps=fc_rel_amps)
		


				else: # movestar, birth/death, merge/split

					dmodels, diff2s, dt_transf = self.pcat_multiband_eval(proposal.xphon, proposal.yphon, proposal.fphon, proposal.dback, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, \
													ref=resids, lib=lib, beam_fac=self.pixel_per_beam, margin_fac=margin_fac, rtype=rtype)
	
				

				plogL = -0.5*diff2s  

				if rtype < 3:
					plogL[(1-self.parity_y)::2,:] = float('-inf') # don't accept off-parity regions
					plogL[:,(1-self.parity_x)::2] = float('-inf')
				
				dlogP = plogL - logL
				
				assert np.isnan(dlogP).any() == False
				
				dts[1,i] = time.time() - t2
				t3 = time.time()
				
				if rtype < 3:
					refx, refy = proposal.get_ref_xy()

					regionx = get_region(refx, self.offsetxs[0], self.regsizes[0])
					regiony = get_region(refy, self.offsetys[0], self.regsizes[0])

					verbprint(self.verbtype, 'Proposal factor has shape '+str(proposal.factor.shape), verbthresh=1)
					verbprint(self.verbtype, 'Proposal factor = '+str(proposal.factor), verbthresh=1)
					
					if proposal.factor is not None:
						dlogP[regiony, regionx] += proposal.factor
					else:
						print('proposal factor is None')

				# else:
					# is this taking the prior factor to the power nregion ^ 2 ? I think it might, TODO
					# if proposal.factor is not None:

						# if rtype == 3 and self.gdat.coupled_bkg_prop:
							# print('dlogP = ', dlogP)
							# print('proposal.factor is ', proposal.factor)
						# dlogP += (proposal.factor/(dlogP.shape[0]*dlogP.shape[1])) # dividing by the number of subregions 

				
				acceptreg = (np.log(np.random.uniform(size=(self.nregy, self.nregx))) < dlogP).astype(np.int32)

				if rtype < 3:

					acceptprop = acceptreg[regiony, regionx]
					numaccept = np.count_nonzero(acceptprop)

				else:
					# if background proposal:
					# sum up existing logL from subregions

					total_logL = np.sum(logL)
					total_dlogP = np.sum(dlogP)

					# print('total dlogP is ', total_dlogP)

					if proposal.factor is not None:
						if np.abs(proposal.factor) > 100:
							print('whoaaaaa proposal.factor is ', proposal.factor)
						total_dlogP += proposal.factor

					# compute dlogP over the full image
					# compute acceptance
					accept_or_not = (np.log(np.random.uniform()) < total_dlogP).astype(np.int32)

					if accept_or_not:
						# set all acceptreg for subregions to 1
						acceptreg = np.ones(shape=(self.nregy, self.nregx)).astype(np.int32)

						if total_dlogP < -10.:
							print('the chi squared degraded significantly in this proposal')
						# 	print('delta log likelihood:', total_dlogP-proposal.factor)
						# 	print('proposal.factor:', proposal.factor)
					else:
						acceptreg = np.zeros(shape=(self.nregy, self.nregx)).astype(np.int32)

				
				nb = 0 # index used for perturb_band_idx stuff

				''' for each band compute the delta log likelihood between states, then add these together'''
				for b in range(self.nbands):

					if proposal.perturb_band_idx is not None:
						if b != proposal.perturb_band_idx:
							continue

					dmodel_acpt = np.zeros_like(dmodels[b])
					diff2_acpt = np.zeros_like(diff2s)

					# if self.gdat.coupled_bkg_prop and rtype==3:
					# 	plt.figure(figsize=(8, 6))
					# 	plt.imshow(dmodels[b], origin='lower')
					# 	plt.colorbar()
					# 	plt.show()
	

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


					if nb==0:
						diff2_total1 = diff2_acpt
						nb += 1
					else:
						diff2_total1 += diff2_acpt

					# if rtype == 3:
					# 	print(diff2_total1)
					# 	if diff2_total1 > 500:
					# 		print('oooo diff2_total1 is ', diff2_total1, rtype, accept_or_not)

					# if b==0:
					# 	diff2_total1 = diff2_acpt
					# else:
					# 	diff2_total1 += diff2_acpt

				logL = -0.5*diff2_total1

				#implement accepted moves
				if proposal.idx_move is not None:

					if rtype==3:

						self.stars = proposal.starsp

					elif self.gdat.coupled_profile_temp_prop and rtype==4:
						# print('idx move here is ', proposal.idx_move)

						if accept_or_not: 
							acceptprop = np.zeros((self.stars.shape[1],))
							acceptprop[proposal.idx_move] = 1

							starsp = proposal.starsp.compress(acceptprop, axis=1)
							# print('aaand here starsp has shape', starsp.shape)

							# print('stars, starsp:', self.stars[:, proposal.idx_move], starsp)
							self.stars[:, proposal.idx_move] = starsp

					else:
						# print('while acceptprop has shape ', acceptprop.shape)
						starsp = proposal.starsp.compress(acceptprop, axis=1)
						# print("starsp has shape", starsp.shape)

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

				if proposal.change_bkg_bool:
					if np.sum(acceptreg) > 0:
						self.dback += proposal.dback

				if proposal.change_template_amp_bool:
					if np.sum(acceptreg) > 0:
						self.dtemplate += proposal.dtemplate

				if proposal.change_fourier_comp_bool:
					if np.sum(acceptreg) > 0:
						if proposal.fc_rel_amp_bool:
							self.dfc_rel_amps += proposal.dfc_rel_amps
						else:
							self.dfc += proposal.dfc

							for b in range(self.nbands):
								running_temp[b] += self.fourier_templates[b][proposal.idx0, proposal.idx1, proposal.idxk]*proposal.dfc[proposal.idx0, proposal.idx1, proposal.idxk]
	
				dts[2,i] = time.time() - t3

				if rtype < 3:
					if acceptprop.size > 0:
						accept[i] = np.count_nonzero(acceptprop) / float(acceptprop.size)
					else:
						accept[i] = 0
				else:
					if np.sum(acceptreg)>0:
						accept[i] = 1
					else:
						accept[i] = 0
			
			else:
				verbprint(self.verbtype, 'Out of bounds..', verbthresh=1)
				outbounds[i] = 1

			for b in range(self.nbands):
				diff2_list[i] += np.sum(self.dat.weights[b]*(self.dat.data_array[b]-models[b])*(self.dat.data_array[b]-models[b]))

			if i > 0:
				if diff2_list[i]-diff2_list[i-1] > 500: # testcoup
					print('diff2s_list = ', diff2_list[i]-diff2_list[i-1], rtype, proposal.factor)


			verbprint(self.verbtype, 'End of loop '+str(i), verbthresh=1)		
			verbprint(self.verbtype, 'self.n = '+str(self.n), verbthresh=1)					
			verbprint(self.verbtype, 'Diff2 = '+str(diff2_list[i]), verbthresh=1)					
			
		# this is after nloop iterations
		chi2 = np.zeros(self.nbands)
		for b in range(self.nbands):
			chi2[b] = np.sum(self.dat.weights[b]*(self.dat.data_array[b]-models[b])*(self.dat.data_array[b]-models[b]))
			
		verbprint(self.verbtype, 'End of sample. self.n = '+str(self.n), verbthresh=1)

		if self.gdat.float_templates:
			self.template_amplitudes += self.dtemplate 
			print('At the end of nloop, self.dtemplate is', self.dtemplate)
			print('so self.template_amplitudes are now ', self.template_amplitudes)
			self.dtemplate = np.zeros_like(self.template_amplitudes)

		if self.gdat.float_fourier_comps:
			self.fourier_coeffs += self.dfc 
			self.fc_rel_amps += self.dfc_rel_amps
			print('At the end of nloop, self.dfc_rel_amps is ', self.dfc_rel_amps)
			print('so self.fc_rel_amps is ', self.fc_rel_amps)
			self.dfc = np.zeros_like(self.fourier_coeffs)
			self.dfc_rel_amps = np.zeros_like(self.fc_rel_amps)


		self.bkg += self.dback
		print('At the end of nloop, self.dback is', np.round(self.dback, 4), 'so self.bkg is now ', np.round(self.bkg, 4))
		self.dback = np.zeros_like(self.bkg)

		timestat_array, accept_fracs = self.print_sample_status(dts, accept, outbounds, chi2, movetype, bkg_perturb_band_idxs=np.array(bkg_perturb_band_idxs), temp_perturb_band_idxs=np.array(temp_perturb_band_idxs))

		if self.gdat.visual:
			frame_dir_path = None

			if self.gdat.n_frames > 0:

				if sample_idx%(self.gdat.nsamp // self.gdat.n_frames)==0:
					frame_dir_path = self.gdat.frame_dir+'/sample_'+str(sample_idx)+'_of_'+str(self.gdat.nsamp)+'.png'


			fourier_any_bool = any(['fourier_bkg' in panel_name for panel_name in self.gdat.panel_list])
			fourier_bkg = None

			if self.gdat.float_fourier_comps and fourier_any_bool:
				fourier_bkg = [self.fc_rel_amps[b]*running_temp[b] for b in range(self.gdat.nbands)]

			if sample_idx < 50 or sample_idx%self.gdat.plot_sample_period==0:
				plot_custom_multiband_frame(self, resids, models, panels = self.gdat.panel_list, frame_dir_path = frame_dir_path, fourier_bkg = fourier_bkg)

		return self.n, chi2, timestat_array, accept_fracs, diff2_list, rtype_array, accept, resids, models


	def idx_parity_stars(self):
		return idx_parity(self.stars[self._X,:], self.stars[self._Y,:], self.n, self.offsetxs[0], self.offsetys[0], self.parity_x, self.parity_y, self.regsizes[0])

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


	def perturb_background(self):
		''' 
		Perturb mean background level according to Model.bkg_prop_sigs.

		Returns
		-------
		proposal : Proposal class object.
	
		'''

		proposal = Proposal(self.gdat)
		# I want this proposal to return the original dback + the proposed change. If the proposal gets approved later on
		# then model.dback will be set to the updated state
		bkg_idx = np.random.choice(self.nbands)
		dback = np.random.normal(0., scale=self.bkg_prop_sigs[bkg_idx])

		proposal.dback[bkg_idx] = dback

		# this is to do coupled proposal between mean background normalization and point sources. The idea here is to compensate a change in background level with a
		# change in point source fluxes. All (or maybe select?) sources across the FOV are perturbed by N_eff*dback. 
		factor = None

		if self.gdat.coupled_bkg_prop:
			# integrated Jy/beam over beam --> average change in flux density for sources
			dback_int = dback*self.gdat.N_eff 

			# choose stars at random 
 
			if self.gdat.couple_nfrac is not None:
				idx_move = np.random.choice(np.arange(self.n), int(self.n*self.gdat.couple_nfrac), replace=False) # want drawing without replacement
			else:
				idx_move = np.arange(self.n)

			stars0 = self.stars.copy()
			# change fluxes of stars in opposite direction to dback_int
			starsp = stars0.copy()
			starsp[self._X,:] = stars0[self._X,:]
			starsp[self._Y,:] = stars0[self._Y,:]
			starsp[self._F+bkg_idx,idx_move] -= dback_int/len(idx_move)

			if bkg_idx==0:
				fthr = self.gdat.trueminf
			else:
				fthr = 0.00001

			ltfmin_mask = (starsp[self._F+bkg_idx,:] < fthr)
			gtrfmin_mask = (starsp[self._F+bkg_idx,:] > fthr)
			starsp[self._F+bkg_idx, ltfmin_mask] = stars0[self._F+bkg_idx, ltfmin_mask]

			proposal.add_move_stars(idx_move, stars0, starsp)
			# flux_couple_factor = self.compute_flux_prior(stars0[self._F+bkg_idx,idx_move], starsp[self._F+bkg_idx,idx_move])
			flux_couple_factor = self.compute_flux_prior(stars0[self._F+bkg_idx,:], starsp[self._F+bkg_idx,:])
			if factor is None:
				factor = np.nansum(flux_couple_factor)
				# print('flux couple factor is ', factor)

		if self.gdat.dc_bkg_prior:
			bkg_factor = -(self.bkg[bkg_idx]+self.dback[bkg_idx]+proposal.dback[bkg_idx]- self.bkg_prior_mus[bkg_idx])**2/(2*self.bkg_prior_sig**2)
			bkg_factor += (self.bkg[bkg_idx]+self.dback[bkg_idx]-self.bkg_prior_mus[bkg_idx])**2/(2*self.bkg_prior_sig**2)
			if factor is not None:
				factor += bkg_factor
			else:
				factor = bkg_factor

		proposal.set_factor(factor)
		
		proposal.change_bkg(perturb_band_idx=bkg_idx)

		return proposal


	def perturb_fourier_comp(self): # fourier comp
		''' 
		Proposal to perturb amplitudes of Fourier component templates. The proposal width is determined by Model.temp_amplitude_sigs and can be
		scaled with the power law exponent of the assumed power law spectrum.
		'''
		
		proposal = Proposal(self.gdat)
		factor = None

		# set dfc_prob to zero if you only want to perturb the amplitudes
		if np.random.uniform() < self.gdat.dfc_prob:

			# choose a component
			proposal.idx0, proposal.idx1, proposal.idxk = np.random.randint(0, self.n_fourier_terms), np.random.randint(0, self.n_fourier_terms), np.random.randint(0, 4)
			
			fc_sig_fac = self.temp_amplitude_sigs['fc']
			
			if self.gdat.fc_prop_alpha is not None: 
				ellmag = np.sqrt((proposal.idx0+1)**2 + (proposal.idx1+1)**2)/np.sqrt(2.)
				fc_sig_fac *= ellmag**self.gdat.fc_prop_alpha

			coeff_pert = np.random.normal(0, fc_sig_fac)
			proposal.dfc[proposal.idx0, proposal.idx1, proposal.idxk] = coeff_pert

			if self.coupled_fc_prop:

				fcomp = self.fourier_templates[b][proposal.idx0, proposal.idx1, proposal.idxk]

				# choose stars at random 
				if self.gdat.couple_nfrac is not None:
					idx_move = np.random.choice(np.arange(self.n), int(self.n*self.gdat.couple_nfrac), replace=False) # want drawing without replacement
				else:
					idx_move = np.arange(self.n)

				stars0 = self.stars.copy()

				# change fluxes of stars in opposite direction to dback_int
				starsp = stars0.copy()
				starsp[self._X,:] = stars0[self._X,:]
				starsp[self._Y,:] = stars0[self._Y,:]

				# query the values of the fourier component template at the positions of selected sources, compute the integrated effective change in flux density
				xfloors = np.floor(starsp[self._X,idx_move])
				yfloors = np.floor(starsp[self._Y,idx_move])
				print('xfloors is ', xfloors)
				dflux_fc = -coeff_pert*self.gdat.N_eff*np.array([fcomp[xfloors[i],yfloors[i]] for i in range(len(idx_move))])
				print('dflux is ', dflux_fc)

				for band_idx in range(self.gdat.nbands):
					starsp[self._F+band_idx,idx_move] -= dflux_fc*self.fc_rel_amps[band_idx]/len(idx_move)/np.sqrt(self.gdat.nbands)

					if band_idx==0:
						fthr = self.gdat.trueminf
					else:
						fthr = 0.00001

					ltfmin_mask = (starsp[self._F+band_idx,:] < fthr)
					print('sum of ltfmin mask is ', np.sum(ltfmin_mask))
					starsp[self._F+band_idx, ltfmin_mask] = stars0[self._F+band_idx, ltfmin_mask]

				proposal.add_move_stars(idx_move, stars0, starsp)


				# flux_couple_factor = self.compute_flux_prior(stars0[self._F,idx_move], starsp[self._F,idx_move])
				# color_factors_orig, colors_orig = self.compute_color_prior(stars0[self._F:,idx_move])
				# color_factors_prop, colors_prop = self.compute_color_prior(starsp[self._F:,idx_move])

				flux_couple_factor = self.compute_flux_prior(stars0[self._F,:], starsp[self._F,:])
				color_factors_orig, colors_orig = self.compute_color_prior(stars0[self._F:,:])
				color_factors_prop, colors_prop = self.compute_color_prior(starsp[self._F:,:])
				color_factors = color_factors_prop - color_factors_orig

				if factor is None:
					factor = np.nansum(flux_couple_factor)
				factor += np.sum(color_factors)
				proposal.set_factor(factor)


		else:

			proposal.fc_rel_amp_bool=True
			# band_weights = []
			band_weights = get_band_weights(self.gdat.fourier_band_idxs)
			band_idx = int(np.random.choice(self.gdat.fourier_band_idxs, p=band_weights))

			d_amp = np.random.normal(0, scale=self.fourier_amp_sig)
			proposal.dfc_rel_amps[band_idx] = d_amp 

		proposal.change_fourier_comp()

		return proposal


	def perturb_template_amplitude(self):

		''' 
		Perturb (non-Fourier component) template amplitudes. These are being kept separate since it delineates between parametric/non-parametric models.
		For example, templates for the SZ effect are based on a parametric model, while cirrus or other diffuse emission can be fit with a non-parametric model, 
		with the note that you could have a full spatial model floated as one template (e.g., a Planck interpolated map of cirrus). 
		
		Returns
		-------

		proposal : Proposal class object.
			
		'''

		proposal = Proposal(self.gdat)
		proposal.dtemplate = np.zeros((self.gdat.n_templates, self.gdat.nbands))

		template_idx = np.random.choice(self.n_templates) # if multiple templates, choose one to change at a time
		temp_band_idxs = self.gdat.template_band_idxs[template_idx]
		factor = None

		# d_amp = np.random.normal(0., scale=self.temp_amplitude_sigs[self.gdat.template_order[template_idx]])

		if self.gdat.temp_prop_df is not None:
			d_amp = self.temp_amplitude_sigs[self.gdat.template_order[template_idx]]*np.random.standard_t(self.gdat.temp_prop_df)
		else:
			d_amp = np.random.normal(0., scale=self.temp_amplitude_sigs[self.gdat.template_order[template_idx]])

		if self.gdat.delta_cp_bool and self.gdat.template_order[template_idx] != 'sze':
			if self.gdat.template_order[template_idx] == 'planck' or self.gdat.template_order[template_idx]=='dust':
				proposal.dtemplate[template_idx,:] = d_amp

		else:
			band_weights = get_band_weights(temp_band_idxs) # this function returns normalized weights

			# uncomment to institute DELTA FN PRIOR SZE @ 250 micron
			if self.gdat.template_order[template_idx] == 'sze':
				band_weights[0] = 0.
				band_weights /= np.sum(band_weights)

			band_idx = int(np.random.choice(temp_band_idxs, p=band_weights))

			proposal.dtemplate[template_idx, band_idx] = d_amp*self.gdat.temp_prop_sig_fudge_facs[band_idx] # added fudge factor for more efficient sampling

			proposal.perturb_band_idx = band_idx

			if self.gdat.coupled_profile_temp_prop:
				stars0 = self.stars.copy()

				# change fluxes of stars in opposite direction to dback_int
				starsp = stars0.copy()
				starsp[self._X,:] = stars0[self._X,:]
				starsp[self._Y,:] = stars0[self._Y,:]

				xpf, ypf = self.dat.fast_astrom.transform_q(starsp[self._X,:], starsp[self._Y,:], band_idx-1)
				xpf = np.floor(xpf).astype(np.int)
				ypf = np.floor(ypf).astype(np.int)

				temp_vals = np.array([self.dat.template_array[band_idx][template_idx][xpf[i],ypf[i]] for i in range(self.n)])
				temp_vals_norm = temp_vals/np.sum(temp_vals)
				idx_move = np.random.choice(np.arange(self.n), self.gdat.coupled_profile_temp_nsrc, replace=False, p=temp_vals_norm) # want drawing without replacement

				# query the values of the fourier component template at the positions of selected sources, compute the integrated effective change in flux density
				# xp, yp = self.dat.fast_astrom.transform_q(starsp[self._X,idx_move], starsp[self._Y,idx_move], band_idx-1)
				# xp = np.floor(xp).astype(np.int)
				# yp = np.floor(yp).astype(np.int)

				# change in fluxes
				# print('proposal.dtemplate[template_idx,band_idx]:', proposal.dtemplate[template_idx,band_idx])
				# print('idx move is ', idx_move)
				# dflux_fc = proposal.dtemplate[template_idx,band_idx]*self.gdat.N_eff*np.array([self.dat.template_array[band_idx][template_idx][xpf[idx_move][i],ypf[idx_move][i]] for i in range(len(idx_move))])

				dflux_fc = proposal.dtemplate[template_idx,band_idx]*self.gdat.N_eff*temp_vals[idx_move]

				starsp[self._F+band_idx,idx_move] -= dflux_fc/len(idx_move)/np.sqrt(self.gdat.nbands) # divide instead by sqrt(len(idx_move))
				# starsp[self._F+band_idx,idx_move] -= dflux_fc/np.sqrt(self.gdat.nbands) # divide instead by sqrt(len(idx_move))?
				
				if band_idx==0:
					fthr = self.gdat.trueminf
				else:
					fthr = 0.00001
				ltfmin_mask = (starsp[self._F+band_idx,:] < fthr)
				# gtrfmin_mask = (starsp[self._F+bkg_idx,:] > fthr)
				starsp[self._F+band_idx, ltfmin_mask] = stars0[self._F+band_idx, ltfmin_mask]



				proposal.add_move_stars(idx_move, stars0, starsp)

				flux_couple_factor = self.compute_flux_prior(stars0[self._F+band_idx,idx_move], starsp[self._F+band_idx,idx_move])
				if factor is None:
					factor = np.nansum(flux_couple_factor)
				proposal.set_factor(factor)


		# the lines below are implementing a step function prior where the ln(prior) = -np.inf when the amplitude is negative
		if self.gdat.template_order[template_idx] == 'sze' and self.gdat.sz_positivity_prior:
			
			old_temp_amp = self.template_amplitudes[template_idx,band_idx] +self.dtemplate[template_idx, band_idx]
			new_temp_amp = old_temp_amp+proposal.dtemplate[template_idx,band_idx]
			
			if new_temp_amp < 0:

				proposal.goodmove = False

				return proposal

		proposal.change_template_amplitude()

		return proposal


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

	def eval_logp_dpl(self, fluxes):

		''' Evaluate the log-prior of the flux distribution, parameterized as a double power law.

		Inputs
		------
		fluxes : np.array of type 'float' and length N_src_max. 

		Class variables
		---------------
		self.pivot_dpl : 'float'. Pivot flux density of the double power law
		self.alpha_1/self.alpha_2 : variables of type 'float'.  Two assumed power law coefficients
		
		Returns
		-------

		logp_dpl : np.array of type 'float'. Log priors for each source


		'''

		logp_dpl = np.zeros_like(fluxes)
		logf = np.log(fluxes)
		piv_mask = (fluxes > self.pivot_dpl)

		logfac2 = (self.alpha_2-self.alpha_1)*np.log(self.pivot_dpl)
		logp_dpl[piv_mask] = logfac2-self.alpha_2*logf[piv_mask]
		logp_dpl[~piv_mask] = -self.alpha_1*logf[~piv_mask]

		return logp_dpl

	def compute_flux_prior(self, f0, pf):
		''' Function to compute the delta log prior of the flux distribution between model states.

		Parameters
		----------

		self : 'Model' class object. The flux distribution type is obtained from the class object.

		f0 : 'np.array'. Original flux densities
		pf : 'np.array'. Proposed flux densities

		Returns
		-------

		factor : 'np.array' of length len(f0). Delta log-prior for each source
		'''
		if self.gdat.flux_prior_type=='single_power_law':
			dlogf = np.log(pf/f0)
			factor = -self.truealpha*dlogf
		
		elif self.gdat.flux_prior_type=='double_power_law':
			log_prior_dpow_pf = np.log(pdfn_dpow(pf,  self.trueminf, self.trueminf*self.newsrc_minmax_range, self.pivot_dpl, self.alpha_1, self.alpha_2))
			log_prior_dpow_f0 = np.log(pdfn_dpow(f0,  self.trueminf, self.trueminf*self.newsrc_minmax_range, self.pivot_dpl, self.alpha_1, self.alpha_2))
			factor = log_prior_dpow_pf - log_prior_dpow_f0

		else:
			print("Need a valid flux prior type (either single_power_law or double_power_law")
			factor = None

		return factor

	def compute_color_prior(self, fluxes):
		all_colors = []
		color_factors = []
		for b in range(self.nbands-1):
			if self.linear_flux:
				colors = fluxes[0]/fluxes[b+1]
			else:
				colors = fluxes_to_color(fluxes[0], fluxes[b+1])
			colors[np.isnan(colors)] = self.color_mus[b]
			all_colors.append(colors)
			color_factors.append(-(colors - self.color_mus[b])**2/(2*self.color_sigs[b]**2))

		return np.array(color_factors), all_colors

	def move_stars(self): 

		''' 
		Proposal to perturb the positions/fluxes of model sources. This is done simultaneously by drawing a flux proposal and then 
		a position change that depends on the max flux of the source, i.e. max(current flux vs. proposed flux). 

		'''

		idx_move = self.idx_parity_stars()
		nw = idx_move.size
		stars0 = self.stars.take(idx_move, axis=1)
		starsp = np.empty_like(stars0)
		
		f0 = stars0[self._F:,:]
		pfs = []
		# color_factors = np.zeros((self.nbands-1, nw)).astype(np.float32)

		for b in range(self.nbands):
			if b==0:
				pf = self.flux_proposal(f0[b], nw)
			else:
				pf = self.flux_proposal(f0[b], nw, trueminf=0.00001) #place a minor minf to avoid negative fluxes in non-pivot bands
			pfs.append(pf)
 
		if (np.array(pfs)<0).any():
			print('negative flux!')
			print(np.array(pfs)[np.array(pfs)<0])


		verbprint(self.verbtype, 'Average flux difference : '+str(np.average(np.abs(f0[0]-pfs[0]))), verbthresh=1)

		factor = self.compute_flux_prior(f0[0], pfs[0])

		# if self.gdat.flux_prior_type=='single_power_law':
		# 	dlogf = np.log(pfs[0]/f0[0])
		# 	factor = -self.truealpha*dlogf
		# 	verbprint('factor at move_stars for single power law are ', factor)

		# elif self.gdat.flux_prior_type=='double_power_law':
		# 	log_prior_dpow_pf = np.log(pdfn_dpow(pfs[0],  self.trueminf, self.trueminf*self.newsrc_minmax_range, self.pivot_dpl, self.alpha_1, self.alpha_2))
		# 	log_prior_dpow_f0 = np.log(pdfn_dpow(f0[0],  self.trueminf, self.trueminf*self.newsrc_minmax_range, self.pivot_dpl, self.alpha_1, self.alpha_2))

		# 	verbprint('log_prior_dpow_pf:', log_prior_dpow_pf)
		# 	verbprint('log_prior_dpow_f0:', log_prior_dpow_f0)

		# 	factor = log_prior_dpow_pf - log_prior_dpow_f0

		# 	verbprint('factor at move_stars for dpl is ', factor)

		if np.isnan(factor).any():
			verbprint(self.verbtype,'Factor NaN from flux', verbthresh=1)
			verbprint(self.verbtype,'Number of f0 zero elements:'+str(len(f0[0])-np.count_nonzero(np.array(f0[0]))), verbthresh=1)
			verbprint(self.verbtype, 'prior factor = '+str(factor), verbthresh=1)

			factor[np.isnan(factor)]=0

		''' the loop over bands below computes colors and prior factors in color used when sampling the posterior
		come back to this later  '''
		modl_eval_colors = []

		color_factors_orig, colors_orig = self.compute_color_prior(f0)
		color_factors_prop, colors_prop = self.compute_color_prior(pfs)
		color_factors = color_factors_prop - color_factors_orig


		for b in range(self.nbands-1):
			modl_eval_colors.append(colors_prop[b])

			# colors = None
			# if self.linear_flux:
			# 	colors = pfs[0]/pfs[b+1]
			# 	orig_colors = f0[0]/f0[b+1]
			# else:
			# 	colors = fluxes_to_color(pfs[0], pfs[b+1])
			# 	orig_colors = fluxes_to_color(f0[0], f0[b+1])
			
			# colors[np.isnan(colors)] = self.color_mus[b] # make nan colors not affect color_factors
			# orig_colors[np.isnan(orig_colors)] = self.color_mus[b]

			# color_factors[b] -= (colors - self.color_mus[b])**2/(2*self.color_sigs[b]**2)
			# color_factors[b] += (orig_colors - self.color_mus[b])**2/(2*self.color_sigs[b]**2)
			# modl_eval_colors.append(colors)
	
		assert np.isnan(color_factors).any()==False       

		verbprint(self.verbtype,'Average absolute color factors : '+str(np.average(np.abs(color_factors))), verbthresh=1)
		verbprint(self.verbtype,'Average absolute flux factors : '+str(np.average(np.abs(factor))), verbthresh=1)

		factor = np.array(factor) + np.sum(color_factors, axis=0)
		
		dpos_rms = np.float32(np.sqrt(self.gdat.N_eff/(2*np.pi))*self.err_f/(np.sqrt(self.nominal_nsrc*self.regions_factor*(2+self.nbands))))/(np.maximum(f0[0],pfs[0]))

		verbprint(self.verbtype,'dpos_rms : '+str(dpos_rms), verbthresh=1)
		
		dpos_rms[dpos_rms < 1e-3] = 1e-3 #do we need this line? perhaps not
		dx = np.random.normal(size=nw).astype(np.float32)*dpos_rms
		dy = np.random.normal(size=nw).astype(np.float32)*dpos_rms
		starsp[self._X,:] = stars0[self._X,:] + dx
		starsp[self._Y,:] = stars0[self._Y,:] + dy
		
		verbprint(self.verbtype, 'dx : '+str(dx), verbthresh=1)
		verbprint(self.verbtype, 'dy : '+str(dy), verbthresh=1)
		verbprint(self.verbtype, 'Mean absolute dx and dy : '+str(np.mean(np.abs(dx)))+', '+str(np.mean(np.abs(dy))), verbthresh=1)

		for b in range(self.nbands):
			starsp[self._F+b,:] = pfs[b]
			if (pfs[b]<0).any():
				print('Proposal fluxes less than 0')
				print('band', b)
				print(pfs[b])
		self.bounce_off_edges(starsp)

		proposal = Proposal(self.gdat)
		proposal.add_move_stars(idx_move, stars0, starsp)
		
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
					if self.gdat.flux_prior_type=='single_power_law':
						starsb[self._F+b,:] = self.trueminf * np.exp(np.random.exponential(scale=1./(self.truealpha-1.),size=nbd))
					elif self.gdat.flux_prior_type=='double_power_law':
						starsb[self._F+b,:] = icdf_dpow(np.random.uniform(0, 1, nbd), self.trueminf, self.trueminf*self.newsrc_minmax_range,\
													 self.pivot_dpl, self.alpha_1, self.alpha_2)
				else:
					# draw new source colors from color prior
					new_colors = np.random.normal(loc=self.color_mus[b-1], scale=self.color_sigs[b-1], size=nbd)
					
					if self.gdat.linear_flux:
						starsb[self._F+b,:] = starsb[self._F,:]*new_colors
					else:
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
			
			# checking for what is in mask takes on average 50 us with scatter depending on how many sources there are being proposed
			not_in_mask = np.array([self.dat.weights[0][int(starsb[self._Y,k]), int(starsb[self._X, k])] > 0 for k in range(starsb.shape[1])])


			starsb = starsb.compress(not_in_mask, axis=1)
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

		''' PCAT proposal to merge/split model sources. '''

		splitsville = np.random.randint(2)
		idx_reg = self.idx_parity_stars()
		fracs, sum_fs = [], []
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
 
			verbprint(self.verbtype, 'stars0 at splitsville start: '+str(stars0), verbthresh=1)
			verbprint(self.verbtype, 'fminratio here is '+str(fminratio), verbthresh=1)
			verbprint(self.verbtype, 'dx = '+str(dx), verbthresh=1)
			verbprint(self.verbtype, 'dy = '+str(dy), verbthresh=1)
			verbprint(self.verbtype, 'idx_move : '+str(idx_move), verbthresh=1)
				
			fracs.append((1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32))
			
			for b in range(self.nbands-1):
				# changed to split similar fluxes
				d_color = np.random.normal(0,self.gdat.split_col_sig)
				# this frac_sim is what source 1 is multiplied by in its remaining bands, so source 2 is multiplied by (1-frac_sim)
				# print('dcolor is ', d_color)
				# F_b = F_1*(1 + [f_1*(1-F_1)*delta s/f_2])
				if self.linear_flux:
					frac_sim = fracs[0]*(1 + (stars0[self._F,:]*(1-fracs[0])*d_color)/stars0[self._F+b+1,:])
				else:
					frac_sim = np.exp(d_color/self.k)*fracs[0]/(1-fracs[0]+np.exp(d_color/self.k)*fracs[0])


				if (frac_sim < 0).any():
					print('negative fraction!!!!')
					goodmove = False

				fracs.append(frac_sim)

			starsp = np.empty_like(stars0)
			starsb = np.empty_like(stars0)

			# starsp is for source 1, starsb is for source 2

			starsp[self._X,:] = stars0[self._X,:] - ((1-fracs[0])*dx)
			starsp[self._Y,:] = stars0[self._Y,:] - ((1-fracs[0])*dy)
			starsb[self._X,:] = stars0[self._X,:] + fracs[0]*dx
			starsb[self._Y,:] = stars0[self._Y,:] + fracs[0]*dy


			for b in range(self.nbands):
				
				starsp[self._F+b,:] = stars0[self._F+b,:]*fracs[b]
				starsb[self._F+b,:] = stars0[self._F+b,:]*(1-fracs[b])

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

			goodmove = (nms > 0)*((np.array(fracs) > 0).all())
			if goodmove:
				proposal.add_move_stars(idx_move, stars0, starsp)
				proposal.add_birth_stars(starsb)
				# can this go nested in if statement? 
			invpairs = np.empty(nms)
			
			verbprint(self.verbtype, 'splitsville is happening', verbthresh=1)
			verbprint(self.verbtype, 'goodmove: '+str(goodmove), verbthresh=1)
			verbprint(self.verbtype, 'invpairs: '+str(invpairs), verbthresh=1)
			verbprint(self.verbtype, 'nms: '+str(nms), verbthresh=1)
			verbprint(self.verbtype, 'sum_fs: '+str(sum_fs), verbthresh=1)
			verbprint(self.verbtype, 'fminratio is '+str(fminratio), verbthresh=1)

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
			
			verbprint(self.verbtype, 'Merging two things!!', verbthresh=1)
			verbprint(self.verbtype, 'nms: '+str(nms), verbthresh=1)
			verbprint(self.verbtype, 'idx_move '+str(idx_move), verbthresh=1)
			verbprint(self.verbtype, 'idx_kill '+str(idx_kill), verbthresh=1)
				
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
			
			verbprint(self.verbtype, 'fminratio: '+str(fminratio)+', nms: '+str(nms), verbthresh=1)
			verbprint(self.verbtype, 'sum_fs[0] is '+str(sum_fs[0]), verbthresh=1)
			verbprint(self.verbtype, 'stars0: '+str(stars0), verbthresh=1)
			verbprint(self.verbtype, 'starsk: '+str(starsk), verbthresh=1)
			verbprint(self.verbtype, 'idx_move '+str(idx_move), verbthresh=1)
			verbprint(self.verbtype, 'idx_kill '+str(idx_kill), verbthresh=1)
				
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
			# the first three terms are the ratio of the flux priors, the next two come from the position terms when choosing sources to merge/split, 
			# the two terms after that capture the transition kernel since there are several combinations of sources that could be implemented, 
			# the last term is the Jacobian determinant f, which is the same for the single and multiband cases given the new proposals 
			

			# factor = np.log(self.truealpha-1) + (self.truealpha-1)*np.log(self.trueminf)-self.truealpha*np.log(fracs[0]*(1-fracs[0])*sum_fs[0]) \
			# 		+ np.log(2*np.pi*self.kickrange*self.kickrange) - np.log(self.imsz0[0]*self.imsz0[1]) \
			# 		+ np.log(bright_n) + np.log(invpairs)+ np.log(1. - 2./fminratio) + np.log(sum_fs[0])
			

			factor = np.log(2*np.pi*self.kickrange*self.kickrange) - np.log(self.imsz0[0]*self.imsz0[1]) \
					+ np.log(bright_n) + np.log(invpairs)+ np.log(1. - 2./fminratio) + np.log(sum_fs[0])
			
			if self.gdat.flux_prior_type=='single_power_law':

				fluxfac = np.log(self.truealpha-1) + (self.truealpha-1)*np.log(self.trueminf)-self.truealpha*np.log(fracs[0]*(1-fracs[0])*sum_fs[0])

			elif self.gdat.flux_prior_type=='double_power_law':

				log_prior_dpow_split1 = np.log(pdfn_dpow(fracs[0]*sum_fs[0],  self.trueminf,\
											 self.trueminf*self.newsrc_minmax_range,\
											  self.pivot_dpl, self.alpha_1, self.alpha_2))
				log_prior_dpow_split2 = np.log(pdfn_dpow((1-fracs[0])*sum_fs[0],  self.trueminf,\
											 self.trueminf*self.newsrc_minmax_range,\
											  self.pivot_dpl, self.alpha_1, self.alpha_2))
				log_prior_dpow_tot = np.log(pdfn_dpow(sum_fs[0],  self.trueminf,\
											 self.trueminf*self.newsrc_minmax_range,\
											  self.pivot_dpl, self.alpha_1, self.alpha_2))
				# print('sum logdiff:', log_prior_dpow_split1 + log_prior_dpow_split2 - log_prior_dpow_tot)

				fluxfac = log_prior_dpow_split1 + log_prior_dpow_split2 - log_prior_dpow_tot

				# print('fluxfac for dpl is ', fluxfac)
				# factor += self.eval_logp_dpl(fracs[0]*sum_fs[0]) \
				# 			+self.eval_logp_dpl((1-fracs[0])*sum_fs[0]) \
				# 			-self.eval_logp_dpl(sum_fs[0])

			factor += fluxfac

			for b in range(self.nbands-1):

				if self.linear_flux:
					stars0_color = stars0[self._F,:]/stars0[self._F+b+1,:]
					starsp_color = starsp[self._F,:]/starsp[self._F+b+1,:]
					# the difference in colors is
					dc = (sum_fs[b+1]/sum_fs[0])*((fracs[b+1]/fracs[0])-1)/(1-fracs[0])	
					# dc = (fracs[b+1]/fracs[0] - (sum_fs[b+1]/sum_fs[0]))/ fracs[0]
		
				else:
					stars0_color = fluxes_to_color(stars0[self._F,:], stars0[self._F+b+1,:])
					starsp_color = fluxes_to_color(starsp[self._F,:], starsp[self._F+b+1,:])
					dc = self.k*(np.log(fracs[b+1]/fracs[0]) - np.log((1-fracs[b+1])/(1-fracs[0])))

				# added_fac comes from the transition kernel of splitting colors in the manner that we do
				added_fac = 0.5*np.log(2*np.pi*self.gdat.split_col_sig**2)+(dc**2/(2*self.gdat.split_col_sig**2))
				factor += added_fac
				
				if splitsville:

					if self.linear_flux:
						starsb_color = starsb[self._F,:]/starsb[self._F+b+1,:]
					else:					
						starsb_color = fluxes_to_color(starsb[self._F,:], starsb[self._F+b+1,:])
					# colfac is ratio of color prior factors i.e. P(s_0)P(s_1)/P(s_merged), where 0 and 1 are original sources 
					color_fac = (stars0_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (starsp_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (starsb_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2)-0.5*np.log(2*np.pi*self.color_sigs[b]**2)
			 
				else:
					if self.linear_flux:
						starsk_color = starsk[self._F,:]/starsk[self._F+b+1,:]
					else:
						starsk_color = fluxes_to_color(starsk[self._F,:], starsk[self._F+b+1,:])
					
					# same as above but for merging sources
					color_fac = (starsp_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (stars0_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2) - (starsk_color - self.color_mus[b])**2/(2*self.color_sigs[b]**2)-0.5*np.log(2*np.pi*self.color_sigs[b]**2)

				factor += color_fac

			# this will penalize the model with extra parameters
			factor -= self.penalty

			# if we have a merge, we want to use the reciprocal acceptance factor, in this case the negative of log(factor)
			if not splitsville:
				factor *= -1

			proposal.set_factor(factor)
						
			if np.isnan(factor).any():
				verbprint(self.verbtype, 'There was a NaN factor in merge/split!', verbthresh=1)	

			verbprint(self.verbtype, 'kickrange factor: '+str(np.log(2*np.pi*self.kickrange*self.kickrange)), verbthresh=1)
			verbprint(self.verbtype, 'imsz factor: '+str(np.log(2*np.pi*self.kickrange*self.kickrange)), verbthresh=1)
			verbprint(self.verbtype, 'kickrange factor: '+str(np.log(self.imsz0[0]*self.imsz0[1])), verbthresh=1)
			verbprint(self.verbtype, 'fminratio: '+str(fminratio)+', fmin factor: '+str(np.log(1. - 2./fminratio)), verbthresh=1)
			verbprint(self.verbtype, 'factor after colors: '+str(factor), verbthresh=1)


		return proposal



class Samples():
	''' The Samples() class saves the parameter chains and other diagnostic statistics about the MCMC run. '''

	def __init__(self, gdat):

		self.nsample = np.zeros(gdat.nsamp, dtype=np.int32) # number of sources
		self.xsample = np.zeros((gdat.nsamp, gdat.max_nsrc), dtype=np.float32) # x positions of sample sources
		self.ysample = np.zeros((gdat.nsamp, gdat.max_nsrc), dtype=np.float32) # y positions of sample sources
		self.timestats = np.zeros((gdat.nsamp, 6, 7), dtype=np.float32) # contains information on computational performance for different parts of algorithm

		self.diff2_all = np.zeros((gdat.nsamp, gdat.nloop), dtype=np.float32) # saves log likelihoods of modoels
		self.accept_all = np.zeros((gdat.nsamp, gdat.nloop), dtype=np.float32) # accepted proposals
		self.rtypes = np.zeros((gdat.nsamp, gdat.nloop), dtype=np.float32) # proposal types at each step
		self.accept_stats = np.zeros((gdat.nsamp, 7), dtype=np.float32) # acceptance fractions for different types of proposals

		self.tq_times = np.zeros(gdat.nsamp, dtype=np.float32)
		self.fsample = [np.zeros((gdat.nsamp, gdat.max_nsrc), dtype=np.float32) for x in range(gdat.nbands)]
		
		self.bkg_sample = np.zeros((gdat.nsamp, gdat.nbands)) # thinned mean background levels
		self.template_amplitudes = np.zeros((gdat.nsamp, gdat.n_templates, gdat.nbands)) # amplitudes of templates used in fit 
		self.fourier_coeffs = np.zeros((gdat.nsamp, gdat.n_fourier_terms, gdat.n_fourier_terms, 4)) # amplitudes of Fourier templates

		self.fc_rel_amps = np.zeros((gdat.nsamp, gdat.nbands)) # relative amplitudes of diffuse Fourier component model across observing bands.

		self.colorsample = [[] for x in range(gdat.nbands-1)]
		self.residuals = [np.zeros((gdat.residual_samples, gdat.imszs[i][0], gdat.imszs[i][1])) for i in range(gdat.nbands)]
		self.model_images = [np.zeros((gdat.residual_samples, gdat.imszs[i][0], gdat.imszs[i][1])) for i in range(gdat.nbands)]

		self.chi2sample = np.zeros((gdat.nsamp, gdat.nbands), dtype=np.int32)
		self.nbands = gdat.nbands
		self.gdat = gdat

	def add_sample(self, j, model, diff2_list, accepts, rtype_array, accept_fracs, chi2_all, statarrays, resids, model_images):
		''' 
		For each thinned sample, adds model parameters to class variables. 
		This is an in place operation.

		Parameters
		----------

		j : 'int'. Index of thinned sample
		model : Class object 'Model'.
		diff2_list : 'list' of arrays.
		accepts :
		rtype_array : 
		accept_fracs : 
		chi2_all : 
		statarrays :
		resids : 
		model_images :

		'''
		self.nsample[j] = model.n
		self.xsample[j,:] = model.stars[Model._X, :]
		self.ysample[j,:] = model.stars[Model._Y, :]
		self.diff2_all[j,:] = diff2_list
		self.accept_all[j,:] = accepts
		self.rtypes[j,:] = rtype_array
		self.accept_stats[j,:] = accept_fracs
		self.chi2sample[j] = chi2_all
		self.timestats[j,:] = statarrays
		self.bkg_sample[j,:] = model.bkg
		self.template_amplitudes[j,:,:] = model.template_amplitudes 
		if self.gdat.float_fourier_comps:
			self.fourier_coeffs[j,:,:,:] = model.fourier_coeffs 
			self.fc_rel_amps[j,:] = model.fc_rel_amps

		for b in range(self.nbands):
			self.fsample[b][j,:] = model.stars[Model._F+b,:]
			if self.gdat.nsamp - j < self.gdat.residual_samples+1:
				self.residuals[b][-(self.gdat.nsamp-j),:,:] = resids[b] 
				self.model_images[b][-(self.gdat.nsamp-j),:,:] = model_images[b]

	def save_samples(self, result_path, timestr):

		''' 
		Save chain parameters/metadata with numpy compressed file. 
		This is an in place operation.
		
		Parameters
		----------
		
		result_path : 'str'. Path to result directory.
		timestr : 'str'. Timestring used to save the run (maybe make it possible to customize name?)

		'''
		# fourier comp, fourier comp colors
		if self.nbands < 3:
			residuals2, model_images2 = None, None
		else:
			residuals2, model_images2 = self.residuals[2], self.model_images[2]
		if self.nbands < 2:
			residuals1, model_images1 = None, None
		else:
			residuals1, model_images1 = self.residuals[1], self.model_images[1]

		residuals0, model_images0 = self.residuals[0], self.model_images[0]

		np.savez(result_path + '/' + str(timestr) + '/chain.npz', n=self.nsample, x=self.xsample, y=self.ysample, f=self.fsample, \
			chi2=self.chi2sample, times=self.timestats, accept=self.accept_stats, diff2s=self.diff2_all, rtypes=self.rtypes, \
			accepts=self.accept_all, residuals0=residuals0, residuals1=residuals1, residuals2=residuals2, model_images0=model_images0,\
			model_images1=model_images1, model_images2=model_images2, bkg=self.bkg_sample, template_amplitudes=self.template_amplitudes, \
			fourier_coeffs=self.fourier_coeffs, fc_rel_amps=self.fc_rel_amps)



class lion():

	''' 
	This is where the main lion() class is initialized and is the starting point for all PCAT runs.
	Below, the collection of configurable parameters in PCAT are presented, separated into relevant variable groups. 
	While there are many tunable variables in the implementation, in practice most of these can remain fixed. 
	As a note, there may be a better way of structuring this, or in storing variable initializations in dedicated parameter files.

	'''

	gdat = gdatstrt()


	def __init__(self, 
			
			# --------------------------------- IMAGE BANDS/SIZING --------------------------------

			# resizes images to largest square dimension modulo nregion
			auto_resize = True, \
			# don't use 'down' configuration yet, not implemented consistently in all data parsing routines
			round_up_or_down = 'up',\
			#specify these if you want to fix the dimension of incoming image
			width = 0, \
			height = 0, \
			# these set x/y coordinate of lower left corner if cropping image
			x0 = 0, \
			y0 = 0, \

			# if there is a shift offset in the astrometry but the images are themselves aligned, can use this to subtract a zero point off of the astrometric solution.
			# if True, the fast_astrom module this takes the coordinate (0, 0), transforms them into the other bands using the provided WCS header/s, and then subtracts these new coordinates
			# from the fast_astrom solution.

			correct_misaligned_shift = False, \

			bolocam_mask = False, \
			# change default to False
			use_mask = True, \
			mask_file = None, \

			#indices of bands used in fit, where 0->250um, 1->350um and 2->500um.
			band0 = 0, \
			band1 = None, \
			band2 = None, \

			# Full width at half maximum for the PSF of the instrument/observation. Currently assumed to be Gaussian, but other 
			# PCAT implementations have used a PSF template, so perhaps a more detailed PSF model could be added as another FITS header
			psf_pixel_fwhm = 3.0, \
			psf_fwhms = None, \

			# TBD
			psf_postage_stamp = None, \

			# if not None, then all pixels with a noise model above the preset values will be zero-weighted. should have one number for each band included in the fit
			noise_thresholds=None, \

	
			# ---------------------------------- BACKGROUND PARAMS --------------------------------

			# bias is used now for the initial background level for each band
			bias = None, \
			# mean offset can be used if one wants to subtract some initial level from the input map, but setting the bias to the value 
			# is functionally the same
			mean_offsets = None, \
			# boolean determining whether to use background proposals
			float_background = False, \
			# bkg_sig_fac scales the width of the background proposal distribution
			bkg_sig_fac = 5., \
			# bkg_moveweight sets what fraction of the MCMC proposals are dedicated to perturbing the background level, 
			# as opposed to changing source positions/births/deaths/splits/merges
			bkg_moveweight = 10., \

			# bkg_sample_delay determines how long Lion waits before sampling from the background. I figure it might be 
			# useful to have this slightly greater than zero so the chain can do a little burn in first.
			bkg_sample_delay = 0, \

			# background amplitude Gaussian prior width [in Jy/beam]
			bkg_prior_sig = 0.01, \

			# background amplitude Gaussian prior mean [in Jy/beam] 
			bkg_prior_mus = None, \

			# if set to True, includes Gaussian prior on background amplitude with mean bkg_mus[bkg_idx] and scale bkg_prior_sig
			dc_bkg_prior = False, \

			# if True, PCAT couples background proposals with change in point source fluxes
			coupled_bkg_prop = True, \

			couple_nfrac = 0.1, \

			# ---------------------------------- TEMPLATE PARAMS ----------------------------------------

			# this determines when templates start getting fit
			temp_sample_delay = 0, \
			# boolean determining whether to float emission template amplitudes, e.g. for SZ or lensing templates
			float_templates = False, \
			# names of templates to use in fit, I think there will be a separate template folder where the names specify which files to read in
			template_names = None, \
			# initial amplitudes for specified templates
			init_template_amplitude_dicts = None, \
			# if template file name is not None then it will grab the template from this path and replace PSW with appropriate band
			template_filename = None, \
			# same idea here as bkg_moveweight
			template_moveweight = 40., \
			# heavy tailed prroposal distribution for templates, df=number of degrees of freedom
			temp_prop_df = None, \
			# if injecting a signal, this fraction determines amplitude of injected signal w.r.t. fiducial values at 250/350/500 micron
			inject_sz_frac = None, \
			# if true, prior is renormalized with zero probability for amplitudes less than zero
			sz_positivity_prior = False, \

			# specifies proposal width for sz template sampling
			sz_amp_sig = None, \
			# if True, look for dust template in input data structure and inject directly to map once resized
			# with the dust, there is also a step that zero centers the template, since we are primarily concerned with the differential perturbation
			# to the image 
			inject_dust = False, \

			# boolean which when True results in a delta function color prior for dust templates 
			delta_cp_bool = False, \

			inject_diffuse_comp = False, \

			diffuse_comp_path = None, \

			# coupled proposals between template describing profile (e.g., SZ surface brightness profile) and point sources
			coupled_profile_temp_prop = False, \

			# number of sources to perturb each time
			coupled_profile_temp_nsrc = 1, \

			# ---------------------------------- FOURIER COMPONENT PARAMS ----------------------------------------

			# number of thinned samples before fourier components are included in the fit
			fc_sample_delay = 0, \

			# bool determining whether to fit fourier comps 
			float_fourier_comps = False, \

			# if there is some previous model component derived in terms of the 2D fourier expansion, they can be specified with this param
			init_fourier_coeffs = None, \

			# for multiple bands this sets the relative normalization
			fc_rel_amps = None, \

			fourier_comp_moveweight = 10., \

			# for a given proposal, this is the probability that the fourier coefficients are perturbed rather than 
			# the relative amplitude of the coefficients across bands
			dfc_prob = 0.5, \

			# this specifies the order of the fourier expansion. the number of fourier components that are fit for is equal to 
			# n_fourier_terms squared 
			n_fourier_terms = 5, \

			# this is for perturbing the relative amplitudes of a fixed Fourier comp model across bands
			fourier_amp_sig = 0.0005, \

			# power law slope of Fourier component proposal distribution. If set to None, constant proposal width used
			fc_prop_alpha = None, \

			# specifies the proposal distribution width for the largest spatial mode of the Fourier component model, or for all of them if fc_prop_alpha=None
			fc_amp_sig = None, \

			bkg_moore_penrose_inv = False, \

			MP_order = 4, \

			mean_sig = True, \

			ridge_fac = 10., \

			# if True, PCAT couples Fourier component proposals with change in point source fluxes
			coupled_fc_prop = True, \

			# --------------------------------- DATA CONFIGURATION ----------------------------------------

			# use if loading data from object and not from saved fits files in directories
			map_object = None, \
			# Configure these for individual directory structure
			base_path = '/Users/richardfeder/Documents/multiband_pcat/', \
			result_path = '/Users/richardfeder/Documents/multiband_pcat/spire_results',\
			data_path = None, \
			# the tail name can be configured when reading files from a specific dataset if the name space changes.
			# the default tail name should be for PSW, as this is picked up in a later routine and modified to the appropriate band.
			tail_name = None, \
			# file_path can be provided if only one image is desired with a specific path not consistent with the larger directory structure
			file_path = None, \

			im_fpath = None, \
			err_fpath = None, \
			# name of cluster being analyzed. If there is no map object, the location of files is assumed to be "data_repo/dataname/dataname_tailname.fits"
			dataname = 'a0370', \
			# mock dataset name
			mock_name = None, \
			# filepath for previous catalog if using as an initial state. loads in .npy files
			load_state_timestr = None, \
			# set flag to True if you want posterior plots/catalog samples/etc from run saved
			save = True, \

			image_extnames=['SIGNAL'], \

			error_extname='ERROR', \

			# if set to true, Gaussian noise realization of error model is added to signal image
			add_noise=False, \
			
			# if specified, error map assumed to be gaussian with variance scalar_noise_sigma**2
			scalar_noise_sigma=None, \

			use_errmap = True, \

			# if true catalog provided, passes on to posterior analysis
			truth_catalog = None, \

			# ---------------------------------- SAMPLER PARAMS ------------------------------------------

			# number of thinned samples
			nsamp = 500, \
			# factor by which the chain is thinned
			nloop = 1000, \
			# scalar factor in regularization prior, scales dlogL penalty when adding/subtracting a source
			alph = 1.0, \
			# if set to True, computes parsimony prior using F statistic, nominal_nsrc and the number of pixels in the immages
			F_statistic_alph = False, \
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
			split_col_sig = 0.2, \
			# set linear_flux to true in order to get color priors in terms of linear flux density ratios
			linear_flux = False, \
			# power law type
			flux_prior_type = 'single_power_law', \
			# number counts single power law slope for sources
			truealpha = 3.0, \
			# minimum flux allowed in fit for SPIRE sources (Jy)
			trueminf = 0.005, \
			# two parameters for double power law, one for pivot flux density
			alpha_1 = 1.01, \
			alpha_2 = 3.5, \
			pivot_dpl = 0.01, \

			# these two, if specified, should be dictionaries with the color prior mean and width (assuming Gaussian)
			color_mus = None, \
			color_sigs = None, \

			temp_prop_sig_fudge_facs = None, \

			# the scheduling within a chain does not work, use iter_fourier_comps() instead (10/13/20)
			trueminf_schedule_vals = [0.1, 0.05, 0.02, 0.01, 0.005],\
			trueminf_schedule_samp_idxs = [0, 50, 100, 200, 500],\
			schedule_trueminf=False, \

			# if specified, nsrc_init is the initial number of sources drawn from the model. otherwise a random integer between 1 and max_nsrc is drawn
			nsrc_init = None, \

			err_f_divfac = 2., \

			merge_split_sample_delay=0, \
			merge_split_moveweight = 60., \

			movestar_sample_delay = 0, \
			movestar_moveweight = 80., \

			birth_death_sample_delay=0, \
			birth_death_moveweight=60., \

			# if specified, delays all point source modeling until point_src_delay samples have passed. 
			point_src_delay = None, \

			# ----------------------------------- DIAGNOSTICS/POSTERIOR ANALYSIS -------------------------------------
			
			# interactive backend should be loaded before importing pyplot
			visual = False, \
			# used for visual mode
			weighted_residual = True, \
			# panel list controls what is shown in six panels plotted by PCAT intermittently when visual=True  
			panel_list = ['data0', 'model0', 'residual0', 'data_zoom0', 'dNdS0', 'residual_zoom0'], \
			# plots visual frames every "plot_sample_period" thinned samples
			plot_sample_period = 1, \
			# can have fully deterministic trials by specifying a random initial seed 
			init_seed = None, \
			# to show raw number counts set to True
			raw_counts = False, \
			# verbosity during program execution
			verbtype = 0, \
			# number of residual samples to average for final product
			residual_samples = 100, \
			# set to True to automatically make posterior/diagnostic plots after the run 
			make_post_plots = True, \
			# used for computing posteriors
			burn_in_frac = 0.75, \
			# save posterior plots
			bool_plot_save = True, \
			# return median model image from last 'residual_samples' samples 
			return_median_model = False, \
			# if PCAT run is part of larger ensemble of test realizations, a file with the associated run IDs (time strings) can be specified
			# and updated with the current run ID.
			timestr_list_file = None, \
			# print script output to log file for debugging
			print_log=False, \
			# this parameter can be set to true when validating the input data products are correct
			show_input_maps=False, \
			# when we have different SZ models to test from Bolocam, setting this to True will report integrated SZ contribution in Jy,
			# rather than the peak normalized amplitude.
			integrate_sz_prof=False, \

			n_frames = 0, \

			# ----------------------------------------- CONDENSED CATALOG --------------------------------------

			# if True, takes last n_condensed_samp catalog realizations and groups together samples to produce a marginalized 
			# catalog with reported uncertainties for each source coming from the several realizations
			generate_condensed_catalog = False, \
			# number of samples to construct condensed catalog from. Condensing the catalog can take a long time, so I usually choose like 100 for a "quick" answer
			# and closer to 300 for a science-grade catalog.
			n_condensed_samp = 100, \
			# cuts catalog sources that appear in less than {prevalence_cut} fraction of {n_condensed_samp} samples
			prevalence_cut = 0.1, \
			# removes sources within {mask_hwhm} pixels of image border, in case there are weird artifacts
			mask_hwhm = 2, \

			# used when grouping together catalog sources across realizations
			search_radius=0.75, \
			matching_dist = 0.75, \

			# ----------------------------------- COMPUTATIONAL ROUTINE OPTIONS -------------------------------
			
			# set to True if using CBLAS library
			cblas=False, \
			# set to True if using OpenBLAS library for non-Intel processors
			openblas=False):


		for attr, valu in locals().items():
			if '__' not in attr and attr != 'gdat' and attr != 'map_object':
				setattr(self.gdat, attr, valu)

		#if specified, use seed for random initialization
		if self.gdat.init_seed is not None:
			np.random.seed(self.gdat.init_seed)

		if self.gdat.point_src_delay is not None:
			self.gdat.movestar_sample_delay = self.gdat.point_src_delay
			self.gdat.birth_death_sample_delay = self.gdat.point_src_delay
			self.gdat.merge_split_sample_delay = self.gdat.point_src_delay

		self.gdat.band_dict = dict({0:'S',1:'M',2:'L'}) # for accessing different wavelength filenames
		self.gdat.lam_dict = dict({'S':250, 'M':350, 'L':500})
		self.gdat.pixsize_dict = dict({'S':6., 'M':8., 'L':12.})
		self.gdat.timestr = time.strftime("%Y%m%d-%H%M%S")

		# power law exponents equal to 1 in double power law will cause a numerical error.

		if self.gdat.alpha_1 == 1.0 and self.flux_prior_type=='double_power_law':
			self.gdat.alpha_1 += 0.01
		if self.gdat.alpha_2 == 1.0 and self.flux_prior_type=='double_power_law':
			self.gdat.alpha_2 += 0.01
		
		self.gdat.bands = [b for b in np.array([self.gdat.band0, self.gdat.band1, self.gdat.band2]) if b is not None]
		self.gdat.nbands = len(self.gdat.bands)

		if self.gdat.template_names is None:
			self.gdat.n_templates=0 
		else:
			self.gdat.n_templates=len(self.gdat.template_names)

		if self.gdat.mean_offsets is None:
			self.gdat.mean_offsets = np.zeros_like(np.array(self.gdat.bands))

		if type(self.gdat.bkg_sig_fac)==float: # if single number, make bkg_sig_fac an array length nbands where each band has same factor
			sigfacs = [self.gdat.bkg_sig_fac for b in range(self.gdat.nbands)]
			self.gdat.bkg_sig_fac = np.array(sigfacs).copy()

		if self.gdat.temp_prop_sig_fudge_facs is None:
			self.gdat.temp_prop_sig_fudge_facs = [1. for b in range(self.gdat.nbands)]

		template_band_idxs = dict({'sze':[0, 1, 2], 'sze':[0,1,2], 'lensing':[0, 1, 2], 'dust':[0, 1, 2], 'planck':[0,1,2]})

		# fourier comp colors
		fourier_band_idxs = [0, 1, 2]

		if self.gdat.psf_fwhms is None:
			self.gdat.psf_fwhms = [self.gdat.psf_pixel_fwhm for i in range(self.gdat.nbands)]
		
		self.gdat.template_order = []
		self.gdat.template_band_idxs = np.zeros(shape=(self.gdat.n_templates, self.gdat.nbands))
	
		if self.gdat.template_names is not None:
			for i, temp_name in enumerate(self.gdat.template_names):		
				for b, band in enumerate(self.gdat.bands):
					if band in template_band_idxs[temp_name]:
						self.gdat.template_band_idxs[i,b] = band
					else:
						self.gdat.template_band_idxs[i,b] = None
				self.gdat.template_order.append(temp_name)

		if self.gdat.data_path is None:
			self.gdat.data_path = self.gdat.base_path+'/Data/spire/'
		print('data path is ', self.gdat.data_path)

		self.data = pcat_data(self.gdat.auto_resize, self.gdat.nregion)
		self.data.load_in_data(self.gdat, map_object=map_object, show_input_maps=self.gdat.show_input_maps)

		# TODO add something that computes the nominal Nsrc down to min flux density threshold given flux prior and map size/resolution.


		if self.gdat.F_statistic_alph:
			alph = compute_Fstat_alph(self.gdat.imszs, self.gdat.nbands, self.gdat.nominal_nsrc)
			# npix = np.sum(np.array([self.gdat.imszs[b][0]*self.gdat.imszs[b][1] for b in range(self.gdat.nbands)]))
			# alph = 0.5*(2.+self.gdat.nbands)*npix/(npix - (2.+self.gdat.nbands)*self.gdat.nominal_nsrc)
			# alph /= 0.5*(2.+self.gdat.nbands) # regularization prior is normalized relative to limit with infinite data, per degree of freedom
			self.gdat.alph = alph
			print('Regularization prior (per degree of freedom) computed from the F-statistic with '+str(self.gdat.nominal_nsrc)+' sources is '+str(np.round(alph, 3)))

		# fourier comp
		if self.gdat.float_fourier_comps:
			print('WERE FLOATING FOURIER COMPS BABY')
			# if there are previous fourier components, use those
			if self.gdat.init_fourier_coeffs is not None:
				if self.gdat.n_fourier_terms != self.gdat.init_fourier_coeffs.shape[0]:
					self.gdat.n_fourier_terms = self.gdat.init_fourier_coeffs.shape[0]
			else:
				self.gdat.init_fourier_coeffs = np.zeros((self.gdat.n_fourier_terms, self.gdat.n_fourier_terms, 4))

			print('ATTENTION x_max_pivot_list IS', self.gdat.x_max_pivot_list)
			self.gdat.fc_templates = multiband_fourier_templates(self.gdat.imszs, self.gdat.n_fourier_terms, psf_fwhms=self.gdat.psf_fwhms, x_max_pivot_list=self.gdat.x_max_pivot_list)
			# self.gdat.fc_templates = multiband_fourier_templates(self.gdat.imszs, self.gdat.n_fourier_terms, show_templates=self.gdat.show_fc_temps, psf_fwhms=self.gdat.psf_fwhms)

			# fourier comp colors
			self.gdat.fourier_band_idxs = [None for b in range(self.gdat.nbands)]

			# if no fourier comp amplitudes specified set them all to unity
			if self.gdat.fc_rel_amps is None:
				self.gdat.fc_rel_amps = np.ones(shape=(self.gdat.nbands,))

			for b, band in enumerate(self.gdat.bands):
				if band in fourier_band_idxs:
					self.gdat.fourier_band_idxs[b] = band
				else:
					self.gdat.fourier_band_idxs[b] = None

		if self.gdat.bias is None:
			self.gdat.bias = np.zeros((self.gdat.nbands,))
			for b, band in enumerate(self.gdat.bands):
				median_val = np.median(self.data.data_array[b])
				self.gdat.bias[b] = median_val - 0.003 # subtract by 3 mJy/beam since background level is biased high by sources

			print('Initial background levels set to ', self.gdat.bias)

		else:
			for b, band in enumerate(self.gdat.bands):
				if band is None:
					if bias[b] is None:
						median_val = np.median(self.data.data_array[b])
						self.gdat.bias[b] = median_val - 0.003
					else:
						self.gdat.bias[b] = bias[b]

		if self.gdat.save:
			#create directory for results, save config file from run
			frame_dir, newdir, timestr = create_directories(self.gdat)
			self.gdat.timestr = timestr
			self.gdat.frame_dir = frame_dir
			self.gdat.newdir = newdir
			save_params(newdir, self.gdat)


	def initialize_libmmult(self):
		''' Initializes matrix multiplication used in PCAT.'''
		if self.gdat.cblas:
			print('Using CBLAS routines for Intel processors.. :-) ', file=self.gdat.flog)

			if sys.version_info[0] == 2:
				libmmult = npct.load_library('pcat-lion', '.')
			else:
				libmmult = npct.load_library('pcat-lion', '.')
				#libmmult = npct.load_library('pcat-lion.so', '.')

		elif self.gdat.openblas:
			print('Using OpenBLAS routines... :-/ ', file=self.gdat.flog)
			# libmmult = ctypes.cdll['pcat-lion-openblas.so']
			# libmmult = npct.load_library('pcat-lion-openblas', '.')
			if sys.version_info[0] == 2:
				libmmult = npct.load_library('blas-open', '.')
			else:
				libmmult = npct.load_library('blas-open.so', '.')

		else:
			print('Using slower BLAS routines.. :-( ', file=self.gdat.flog)
			libmmult = ctypes.cdll['./blas.so'] # not sure how stable this is, trying to find a good Python 3 fix to deal with path configuration
			# libmmult = npct.load_library('blas', '.')

		return libmmult


	def initialize_print_log(self):
		if self.gdat.print_log:
			self.gdat.flog = open(self.gdat.result_path+'/'+self.gdat.timestr+'/print_log.txt','w')
		else:
			self.gdat.flog = None		


	def main(self):
		''' 
		Here is where we initialize the C libraries and instantiate the arrays that will store our 
		thinned samples and other stats. We want the MKL routine if possible, then OpenBLAS, then regular C, with that order in priority.
		This is also where the MCMC sampler is initialized and run.

		Returns
		-------

		models (optional): 'list' of 'np.array' of type 'float' and shape (dimx, dimy). 
		If self.gdat.return_median_model is True, this is returned (default is False).

		'''

		self.initialize_print_log()
		
		libmmult = self.initialize_libmmult()

		initialize_c(self.gdat, libmmult, cblas=self.gdat.cblas)

		start_time = time.time()
		samps = Samples(self.gdat)

		verbprint(self.gdat.verbtype, 'Initializing model..', verbthresh=1)

		model = Model(self.gdat, self.data, libmmult)

		verbprint(self.gdat.verbtype, 'Done initializing model..', verbthresh=1)
		print('background sig facs are ', self.gdat.bkg_sig_fac)

		trueminf_schedule_counter = 0
		for j in range(self.gdat.nsamp): # run sampler for gdat.nsamp thinned states
			print('Sample', j, file=self.gdat.flog)

			if self.gdat.schedule_trueminf:

				if j==self.gdat.trueminf_schedule_samp_idxs[trueminf_schedule_counter]:
					self.gdat.trueminf = self.gdat.trueminf_schedule_vals[trueminf_schedule_counter]
					model.trueminf = self.gdat.trueminf_schedule_vals[trueminf_schedule_counter]

					trueminf_schedule_counter += 1
					print('Changing trueminf.. ', self.gdat.trueminf, model.trueminf)
			
			# once ready to sample, recompute proposal weights
			model.update_moveweights(j)

			_, chi2_all, statarrays,  accept_fracs, diff2_list, rtype_array, accepts, resids, model_images = model.run_sampler(j)
			samps.add_sample(j, model, diff2_list, accepts, rtype_array, accept_fracs, chi2_all, statarrays, resids, model_images)


		if self.gdat.save:
			print('Saving...', file=self.gdat.flog)

			# save catalog ensemble and other diagnostics
			samps.save_samples(self.gdat.result_path, self.gdat.timestr)

			# save final catalog state
			np.savez(self.gdat.result_path + '/'+str(self.gdat.timestr)+'/final_state.npz', cat=model.stars, bkg=model.bkg, templates=model.template_amplitudes, fourier_coeffs=model.fourier_coeffs)

		if self.gdat.timestr_list_file is not None:
			if path.exists(self.gdat.timestr_list_file):
				timestr_list = list(np.load(self.gdat.timestr_list_file)['timestr_list'])
				timestr_list.append(self.gdat.timestr)
			else:
				timestr_list = [self.gdat.timestr]
			np.savez(self.gdat.timestr_list_file, timestr_list=timestr_list)

		# if self.gdat.generate_condensed_catalog:

		# 	xmatch_roc = cross_match_roc(timestr=self.gdat.timestr, nsamp=self.gdat.n_condensed_samp)
		# 	xmatch_roc.load_gdat_params(gdat=self.gdat)
		# 	condensed_cat, seed_cat = xmatch_roc.condense_catalogs(prevalence_cut=self.gdat.prevalence_cut, save_cats=True, make_seed_bool=True,\
		# 															 mask_hwhm=self.gdat.mask_hwhm, search_radius=self.gdat.search_radius, matching_dist=self.gdat.matching_dist)

		if self.gdat.make_post_plots:
			result_plots(gdat = self.gdat, generate_condensed_cat=self.gdat.generate_condensed_catalog, n_condensed_samp=self.gdat.n_condensed_samp, prevalence_cut=self.gdat.prevalence_cut, mask_hwhm=self.gdat.mask_hwhm, condensed_catalog_plots=self.gdat.generate_condensed_catalog)

		dt_total = time.time() - start_time
		print('Full Run Time (s):', np.round(dt_total,3), file=self.gdat.flog)
		print('Time String:', str(self.gdat.timestr), file=self.gdat.flog)

		with open(self.gdat.newdir+'/time_elapsed.txt', 'w') as filet:
			filet.write('time elapsed: '+str(np.round(dt_total,3))+'\n')

		plt.close() # I think this is for result plots
			
		if self.gdat.print_log:
			self.gdat.flog.close()

		if self.gdat.return_median_model:
			models = []
			for b in range(self.gdat.nbands):
				model_samples = np.array([self.data.data_array[b]-samps.residuals[b][i] for i in range(self.gdat.residual_samples)])
				median_model = np.median(model_samples, axis=0)
				models.append(median_model)

			return models


