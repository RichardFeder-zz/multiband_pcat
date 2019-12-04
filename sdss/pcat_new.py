import numpy as np
import numpy.ctypeslib as npct
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
import random
from astropy.convolution import Gaussian2DKernel
import scipy.signal
from image_eval import psf_poly_fit, image_model_eval
from helpers import * 
from fast_astrom import *
import cPickle as pickle
np.seterr(divide='ignore', invalid='ignore')


#generate random seed for initialization
np.random.seed(20170502)

#-------------------------- this portion sets up base directories, probably will need to modify ---------------------

def get_parser_arguments(datatype='SDSS'):

	parser = argparse.ArgumentParser()
	''' These configure the input image dimensions '''
	parser.add_argument('--auto_resize', type=bool, default=False, help='resizes images to largest square dimension modulo opt.nregion')
	parser.add_argument('--width', type=int, default=0,  help='specify if you want to fix the dimension of incoming image, may be useful for subregion sampling')
	parser.add_argument('--height', type=int, default=0, help='same as width')
	parser.add_argument('--x0', type=int, default=0, help='sets x coordinate of lower left corner if cropping image')
	parser.add_argument('--y0', type=int, default=0, help='sets x coordinate of lower left corner if cropping image')
	parser.add_argument('--mock', type=bool, default=False, help='set to True if using mock data')
	parser.add_argument('--mock_name', default=None, help='specify name if using mock data, affects how the data is read in')
	parser.add_argument('--openblas', default=False, help='for use of OpenBLAS libraries')
	parser.add_argument('--cblas_mkl', default=True, help='for use of Intel MKL library')


	''' These indicate which bands to run on and some of their properties'''


	if datatype=='SDSS':
		parser.add_argument('--background', type=float, default=179.0, help='Background level for image')
		parser.add_argument('--truealpha', type=float, default=2.0, help='number counts power law slope for SPIRE sources')  
		parser.add_argument('--trueminf', type=float, default=256., help='minimum flux allowed in fit for sources') 
		parser.add_argument('--band0', type=str, default='r', help='indices of bands used in fit')
		parser.add_argument('--band1', type=str, default=None, help='indices of bands used in fit')
		parser.add_argument('--band2', type=str, default=None, help='indices of bands used in fit')
		parser.add_argument('--result_path', default='/Users/richardfeder/Documents/multiband_pcat/spire_results/')
		parser.add_argument('--run', type=int, default=2583)
		parser.add_argument('--camcol', type=int, default=2)
		parser.add_argument('--field', type=int, default=136)
		parser.add_argument('--dataname', default='idR-002583-2-0136', help='name of dataset being analyzed')


	elif datatype=='SPIRE':
		parser.add_argument('--mean_offset', type=float, default=0.0035, help='absolute level subtracted from SPIRE model image')
		parser.add_argument('--psf_pixel_fwhm', type=float, default=3.)
		parser.add_argument('--truealpha', type=float, default=3.0, help='number counts power law slope for SPIRE sources')    
		parser.add_argument('--trueminf', type=float, default=0.001, help='minimum flux allowed in fit for SPIRE sources (Jy)')
		parser.add_argument('--band0', type=int, default=0, help='indices of bands used in fit, where 0->250um, 1->350um and 2->500um.')
		parser.add_argument('--band1', type=int, default=None, help='indices of bands used in fit, where 0->250um, 1->350um and 2->500um.')
		parser.add_argument('--band2', type=int, default=None, help='indices of bands used in fit, where 0->250um, 1->350um and 2->500um.')
		parser.add_argument('--result_path', default='/Users/richardfeder/Documents/multiband_pcat/spire_results/')
		parser.add_argument('--dataname', default='a0370', help='name of dataset being analyzed')

	elif datatype=='lmc':
		parser.add_argument('--field', type=int, default=100, help='OGLE field to load in')
		parser.add_argument('--subfield', type=int, default=1)
		parser.add_argument('--band0', default='I')
		parser.add_argument('--band1', type=str, default=None, help='indices of bands used in fit')
		parser.add_argument('--band2', type=str, default=None, help='indices of bands used in fit')
		parser.add_argument('--result_path', default='/Users/richardfeder/Documents/multiband_pcat/lmc_results/')
		parser.add_argument('--truealpha', type=float, default=2.0, help='number counts power law slope for SPIRE sources')    
		parser.add_argument('--trueminf', type=float, default=500.0, help='minimum flux allowed in fit for OGLE sources (counts)')



	''' Configure these for individual directory structure '''
	parser.add_argument('--base_path', default='/Users/richardfeder/Documents/multiband_pcat/')
	parser.add_argument('--load_state_timestr', default=None, help='filepath for previous catalog if using as an initial state. loads in .npy files')
	
	''' Settings for sampler '''
	parser.add_argument('--nsamp', type=int, default=1000, help='Number of thinned samples')
	parser.add_argument('--nloop', type=int, default=1000, help='factor by which the chain is thinned')

	''' Settings for proposals '''
	parser.add_argument('--alph', type=float, default=1.0, help='used as scalar factor in regularization prior, which determines the penalty in dlogL when adding/subtracting a source')
	parser.add_argument('--kickrange', type=float, default=1.0, help='sets scale for merge proposal i.e. how far you look for neighbors to merge')
	parser.add_argument('--margin', type=int, default=10, help='used in model evaluation')
	parser.add_argument('--max_nsrc', type=int, default=2000, help='this sets the maximum number of sources allowed in thee code, can change depending on the image')
	parser.add_argument('--nominal_nsrc', type=int, default=1000, help='this is the nominal number of sources expected in a given image, helps set sample step sizes during MCMC')
	parser.add_argument('--nregion', type=int, default=1, help='splits up image into subregions to do proposals within')
	parser.add_argument('--split_col_sig', type=float, default=0.25, help='used when splitting sources and determining colors of resulting objects')
	parser.add_argument('--trueminf_beyond_pivot', type=float, default=0., help='minimum flux allowed in additional bands')

	''' Settings for user experience!! '''
	parser.add_argument('--visual', help='interactive backend should be loaded before importing pyplot')
	parser.add_argument('--weighted_residual', help='used for visual mode')
	parser.add_argument('--verbtype', type=int, default=0, help='verbosity during program execution')
	parser.add_argument('--residual_samples', type=int, default=100, help='number of residual samples to average for final product')
	opt = parser.parse_known_args()[0]

	return opt


def load_opt(datatype='SDSS'):
	opt = get_parser_arguments(datatype)
	opt.datatype=datatype
	opt.timestr = time.strftime("%Y%m%d-%H%M%S")
	opt.bands = []
	opt.bands.append(opt.band0)
	if opt.band1 is not None:
		opt.bands.append(opt.band1)
		if opt.band2 is not None:
			opt.bands.append(opt.band2)
	opt.nbands = len(opt.bands)

	return opt


def save_params(dir, opt):
	# save parameters as dictionary, then pickle them to txt file
	param_dict = vars(opt)
	print param_dict
	
	with open(dir+'/params.txt', 'w') as file:
		file.write(pickle.dumps(param_dict))
	file.close()
	
	with open(dir+'/params_read.txt', 'w') as file2:
		for key in param_dict:
			file2.write(key+': '+str(param_dict[key])+'\n')
	file2.close()

def fluxes_to_color(flux1, flux2):
	return 2.5*np.log10(flux1/flux2)


def get_gaussian_psf_template(pixel_fwhm=3., nbin=5):
	nc = 25
	print('pixel fwhm (scaled up):', nbin*pixel_fwhm/2.355)
	psfnew = Gaussian2DKernel((pixel_fwhm/2.355)*nbin, x_size=125, y_size=125).array.astype(np.float32)
	# psfnew = Gaussian2DKernel((pixel_fwhm/2.355)*nbin, x_size=125, y_size=125).array.astype(np.float32)
	psfnew *= nc
	cf = psf_poly_fit(psfnew, nbin=nbin)
	return psfnew, cf, nc, nbin


def load_in_lmc_map(opt):
	file_path = opt.base_path+'/Data/lmc/lmc'+str(opt.field)+'.i.'+str(opt.subfield)+'.fts'
	maps = fits.open(file_path)
	head = maps[0].header
	psf_fwhm = head['FWHM']
	print('FWHM:', psf_fwhm)
	sky_level = head['SKY']
	# gain = head['GAIN']
	gain = 1.3
	image = maps[0].data

	mask_centers = np.where(image > 60000)
	print('there are ', len(mask_centers[0]), ' pixels that are saturating')
	for c in xrange(len(mask_centers[0])):
		image[mask_centers[0][c]-10:mask_centers[0][c]+11, mask_centers[1][c]-10:mask_centers[1][c]+11] = 0.


	print('now the image max is ', np.max(image))
	print(np.median(image))
	return image, psf_fwhm, sky_level, gain

def load_in_spire_map(opt, band=0, astrom=None):
	band_dict = dict({0:'S',1:'M',2:'L'}) # for accessing different wavelength filenames
	file_path = opt.base_path+'/Data/spire/'+opt.dataname+'_P'+band_dict[band]+'W_nr_1.fits'
	print('file_path:', file_path)

	if astrom is not None:
		astrom.load_wcs_header_and_dim(opt.dataname+'_P'+band_dict[band]+'W_nr_1.fits', hdu_idx=3)

	spire_dat = fits.open(file_path)
	image = np.nan_to_num(spire_dat[1].data)
	error = np.nan_to_num(spire_dat[2].data)
	exposure = spire_dat[3].data
	mask = spire_dat[4].data

	return image, error, exposure, mask, band_dict

def load_in_sdss_data(opt, band=0, astrom=None):
	print('band:', band)
	band_dict=dict({0:'r', 1:'i', 2:'g'})
	paths = [opt.base_path+'/Data/'+opt.dataname+'/psfs/'+opt.dataname+'-psf.txt', \
				opt.base_path+'/Data/'+opt.dataname+'/pixs/'+opt.dataname+'-pix.txt', \
					opt.base_path+'/Data/'+opt.dataname+'/cts/'+opt.dataname+'-cts.txt']
	
	for p in xrange(len(paths)):
		paths[p] = paths[p][:-4]+str(band)+paths[p][-4:]

	psf, nc, cf = get_psf_and_vals(paths[0])

	print('sum of psf:', np.sum(psf))

	pix_file = open(paths[1])
	# w and h are the width and height of the image, might need to change a few things in the code that assume the image is a square
	w, h, nb = [np.int32(i) for i in pix_file.readline().split()]
	bias, gain = [np.float32(i) for i in pix_file.readline().split()]
	a = np.float32(pix_file.readline().split())
	print a, len(a)
	nmgy_per_count = a[0]
	print('nmgy_per_count:', nmgy_per_count)
	pix_file.close()

	data = np.loadtxt(paths[2]).astype(np.float32)

	return data, psf, nc, cf, nmgy_per_count, bias, gain, nb



# def initialize_c(opt, libmmult):
# 	if opt.verbtype > 1:
# 		print 'initializing c routines and data structs'
# 	if os.path.getmtime('pcat-lion.c') > os.path.getmtime('pcat-lion.so'):
# 		warnings.warn('pcat-lion.c modified after compiled pcat-lion.so', Warning)
# 	array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
# 	array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
# 	array_2d_double = npct.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
# 	libmmult.pcat_model_eval.restype = None
# 	libmmult.pcat_model_eval.argtypes = [c_int, c_int, c_int, c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_1d_int, array_1d_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]
# 	array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")
# 	libmmult.pcat_imag_acpt.restype = None
# 	libmmult.pcat_imag_acpt.argtypes = [c_int, c_int, array_2d_float, array_2d_float, array_2d_int, c_int, c_int, c_int, c_int]
# 	libmmult.pcat_like_eval.restype = None
# 	libmmult.pcat_like_eval.argtypes = [c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]


def initialize_c(opt, libmmult):

	if opt.verbtype > 1:
		print('initializing c routines and data structs')

	if opt.cblas_mkl or opt.openblas:
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


def create_directories(opt):
	new_dir_name = opt.result_path+'/'+opt.timestr
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

class Proposal:
	_X = 0
	_Y = 1
	_F = 2

	def __init__(self, opt):
		self.idx_move = None
		self.do_birth = False
		self.idx_kill = None
		self.factor = None
		self.goodmove = False
		self.dback = np.zeros(opt.nbands, dtype=np.float32)
		self.xphon = np.array([], dtype=np.float32)
		self.yphon = np.array([], dtype=np.float32)
		self.fphon = []
		self.modl_eval_colors = []
		for x in xrange(opt.nbands):
			self.fphon.append(np.array([], dtype=np.float32))
		self.opt = opt
	def set_factor(self, factor):
		self.factor = factor

	def in_bounds(self, catalogue):
		return np.logical_and(np.logical_and(catalogue[self._X,:] > 0, catalogue[self._X,:] < (self.opt.imsz[0] -1)), \
				np.logical_and(catalogue[self._Y,:] > 0, catalogue[self._Y,:] < self.opt.imsz[1] - 1))

	def assert_types(self):
		assert self.xphon.dtype == np.float32
		assert self.yphon.dtype == np.float32
		assert self.fphon[0].dtype == np.float32

	def __add_phonions_stars(self, stars, remove=False):
		fluxmult = -1 if remove else 1

		self.xphon = np.append(self.xphon, stars[self._X,:])
		self.yphon = np.append(self.yphon, stars[self._Y,:])

		for b in xrange(self.opt.nbands):
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
		elif self.eps_shift is not None:
			return self.stars0[self._X,:], self.stars0[self._Y,:]



class Model:
	# trueminf = np.float32(236) # minimum flux for source in ADU, might need to change
	#truealpha = np.float32(2) # power law flux slope parameter, might need to change
	
	_X = 0
	_Y = 1
	_F = 2

	k =2.5/np.log(10)

	trueback_dict_of_dicts = dict({'8151':dict({"r":135., "i":215., "g":234., "z":140.}), '0136':dict({"r":179., "i":310., "g":95.})})


	#sigs = dict({'r-i':0.5, 'r-g':0.5})
	color_mus, color_sigs = [], []
	
	''' the init function sets all of the data structures used for the catalog, 
	randomly initializes catalog source values drawing from catalog priors  '''
	def __init__(self, opt, dat, libmmult=None):
		self.back = np.zeros(opt.nbands, dtype=np.float32)
		self.err_f = opt.err_f
		self.imsz = opt.imsz
		self.kickrange = opt.kickrange
		self.margin = opt.margin
		self.max_nsrc = opt.max_nsrc
		self.moveweights = np.array([80., 40., 40.])
		self.movetypes = ['P *', 'BD *', 'MS *']
		self.n = np.random.randint(opt.max_nsrc)+1
		self.nbands = opt.nbands
		self.nloop = opt.nloop
		self.nregion = opt.nregion
		self.penalty = 1+0.5*opt.alph*opt.nbands
		self.regsize = opt.regsize
		self.stars = np.zeros((2+opt.nbands,opt.max_nsrc), dtype=np.float32)
		self.stars[:,0:self.n] = np.random.uniform(size=(2+opt.nbands,self.n))
		self.stars[self._X,0:self.n] *= opt.imsz[0]-1
		self.stars[self._Y,0:self.n] *= opt.imsz[1]-1
		self.verbtype = opt.verbtype
		self.nominal_nsrc = opt.nominal_nsrc
		self.regions_factor = opt.regions_factor
		self.truealpha = opt.truealpha
		self.trueminf = opt.trueminf
		self.opt = opt
		self.dat = dat
		self.libmmult = libmmult

		if self.opt.datatype=='SDSS':
			self.bkg = np.array([opt.background for b in xrange(opt.nbands)])
			mus = dict({'r-i':0.0, 'r-g':0.0}) # for same flux different noise tests, r, i, g are all different realizations of r band
			sigs = dict({'r-i':0.5, 'r-g':5.0, 'r-z':1.0, 'r-r':0.05})
		
		elif self.opt.datatype=='SPIRE':
			self.bkg = np.zeros(shape=(opt.nbands))
			mus = dict({'S-M':0., 'M-L':-0.3, 'L-S':0.0, 'M-S':0.0, 'S-L':0.5, 'L-M':0.2})
			sigs = dict({'S-M':0.5, 'M-L':0.5, 'L-S':0.5, 'M-S':0.5, 'S-L':0.5, 'L-M':0.5}) #very broad color prior

		elif self.opt.datatype=='lmc':
			self.bkg = np.array([opt.sky_level for b in xrange(opt.nbands)])

		print('self.bkg is ', self.bkg)

		for b in xrange(self.nbands-1):

			col_string = self.opt.band_dict[self.opt.bands[0]]+'-'+self.opt.band_dict[self.opt.bands[b+1]]
			print('col string is ', col_string)
			self.color_mus.append(self.mus[col_string])
			self.color_sigs.append(self.sigs[col_string])

		if opt.load_state_timestr is None:
			for b in xrange(opt.nbands):
				if b==0:
					print('self.truealpha here is ', self.truealpha)
					print('self.trueminf here is ', self.trueminf)
					self.stars[self._F+b,0:self.n] **= -1./(self.truealpha - 1.)
					self.stars[self._F+b,0:self.n] *= self.trueminf
				else:
					new_colors = np.random.normal(loc=self.color_mus[b-1], scale=self.color_sigs[b-1], size=self.n)
					if self.opt.datatype=='SDSS':
						self.stars[self._F+b,0:self.n] = self.stars[self._F,0:self.n]*10**(0.4*new_colors)
					elif self.opt.datatype=='SPIRE':
						self.stars[self._F+b,0:self.n] = self.stars[self._F,0:self.n]*10**(0.4*new_colors)

		else:
			print 'Loading in catalog from run with timestr='+opt.load_state_timestr+'...'
			catpath = opt.result_path+'/'+opt.load_state_timestr+'/final_state.npz'
			self.stars = np.load(catpath)['cat']
			self.n = np.count_nonzero(self.stars[self._F,:])

	''' this function prints out some information at the end of each thinned sample, 
	namely acceptance fractions for the different proposals and some time performance statistics as well. '''
   
	def print_sample_status(self, dts, accept, outbounds, chi2, movetype):    
		fmtstr = '\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f'
		print 'Background', self.bkg, 'N_star', self.n, 'chi^2', list(chi2)
		dts *= 1000

		accept_fracs = []
		timestat_array = np.zeros((6, 1+len(self.moveweights)), dtype=np.float32)
		statlabels = ['Acceptance', 'Out of Bounds', 'Proposal (s)', 'Likelihood (s)', 'Implement (s)', 'Coordinates (s)']
		statarrays = [accept, outbounds, dts[0,:], dts[1,:], dts[2,:], dts[3,:]]
		for j in xrange(len(statlabels)):
			timestat_array[j][0] = np.sum(statarrays[j])/1000
			if j==0:
				accept_fracs.append(np.sum(statarrays[j])/1000)
			print statlabels[j]+'\t(all) %0.3f' % (np.sum(statarrays[j])/1000),
			for k in xrange(len(self.movetypes)):
				if j==0:
					accept_fracs.append(np.mean(statarrays[j][movetype==k]))
				timestat_array[j][1+k] = np.mean(statarrays[j][movetype==k])
				print '('+self.movetypes[k]+') %0.3f' % (np.mean(statarrays[j][movetype == k])),
			print
			if j == 1:
				print '-'*16
		print '-'*16
		print 'Total (s): %0.3f' % (np.sum(statarrays[2:])/1000)
		print '='*16

		return timestat_array, accept_fracs


		''' the multiband model evaluation looks complicated because we tried out a bunch of things with the astrometry, but could probably rewrite this function. it uses image_model_eval which is written in both python and C (C is faster as you might expect)'''
	def pcat_multiband_eval(self, x, y, f, bkg, nc, cf, weights, ref, lib):
		dmodels = []
		dt_transf = 0
		for b in xrange(self.nbands):
			if self.opt.datatype=='SPIRE':
				fluxes = 25*f[b]
			elif self.opt.datatype=='SDSS' or self.opt.datatype=='lmc':
				fluxes = f[b]

			
			if b>0:
				t4 = time.clock()

				if self.opt.datatype=='SDSS':
					nmgys = [self.opt.nmgy_per_count[0], self.opt.nmgy_per_count[b]]
					colors = adus_to_color(np.abs(f[0]), np.abs(f[b]), nmgys) #need to take absolute value to get colors for negative flux phonions


				if self.opt.bands[b] != self.opt.bands[0]:
					xp, yp = self.dat.fast_astrom.transform_q(x, y, b-1)
				else:
					xp = x
					yp = y
				dt_transf += time.clock()-t4

				dmodel, diff2 = image_model_eval(xp, yp, fluxes, self.bkg[b], self.imsz, \
												nc[b], np.array(cf[b]).astype(np.float32()), bkg[b], weights=self.dat.weights[b], \
												ref=ref[b], lib=self.libmmult.pcat_model_eval, regsize=self.regsize, \
												margin=self.margin, offsetx=self.offsetx, offsety=self.offsety)
				diff2s += diff2
			else:    
				xp=x
				yp=y


				dmodel, diff2 = image_model_eval(xp, yp, fluxes, bkg[b], self.imsz, \
												nc[b], np.array(cf[b]).astype(np.float32()), weights=self.dat.weights[b], \
												ref=ref[b], lib=self.libmmult.pcat_model_eval, regsize=self.regsize, \
												margin=self.margin, offsetx=self.offsetx, offsety=self.offsety)
			
				# print('dmodel')
				# print dmodel
				diff2s = diff2

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

		if self.nregion > 1:
			self.offsetx = np.random.randint(self.regsize)
			self.offsety = np.random.randint(self.regsize)
		else:
			self.offsetx = 0
			self.offsety = 0
 
		self.nregx = self.imsz[0] / self.regsize + 1
		self.nregy = self.imsz[1] / self.regsize + 1

		resids = []
		for b in xrange(self.nbands):
			resid = self.dat.data_array[b].copy() # residual for zero image is data
			resids.append(resid)
			print np.mean(resids[b]), np.std(resids[b])

		evalx = self.stars[self._X,0:self.n]
		evaly = self.stars[self._Y,0:self.n]
		evalf = self.stars[self._F:,0:self.n]
		
		n_phon = evalx.size

		if self.opt.verbtype > 1:
			print 'beginning of run sampler'
			print 'self.n here'
			print self.n
			print 'n_phon'
			print n_phon

		if self.opt.cblas_mkl or self.opt.openblas:
			lib = self.libmmult.pcat_model_eval
		else:
			lib = self.libmmult.clib_eval_modl

		models, diff2s, dt_transf = self.pcat_multiband_eval(evalx, evaly, evalf, self.bkg, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, ref=resids, lib=lib)
		model = models[0]
		logL = -0.5*diff2s
	   
		for b in xrange(self.nbands):
			resids[b] -= models[b]
			print np.mean(resids[b]), np.std(resids[b])

		
		'''the proposals here are: move_stars (P) which changes the parameters of existing model sources, 
		birth/death (BD) and merge/split (MS). Don't worry about perturb_astrometry. 
		The moveweights array, once normalized, determines the probability of choosing a given proposal. '''

		movefns = [self.move_stars, self.birth_death_stars, self.merge_split_stars]
		self.moveweights /= np.sum(self.moveweights)
		if self.opt.nregion > 1:
			xparities = np.random.randint(2, size=self.nloop)
			yparities = np.random.randint(2, size=self.nloop)
		
		rtype_array = np.random.choice(self.moveweights.size, p=self.moveweights, size=self.nloop)
		movetype = rtype_array

		# if verbtype > 1:
  #           for b in xrange(self.opt.nbands):
  #               df2 = np.sum(self.dat.weights[b]*(self.dat.data_array[b]-models[b])*(self.dat.data_array[b]-self.dat.models[b]))
  #               print 'df2 for band right before nloop loop', b
  #               print df2

		for i in xrange(self.nloop):
			t1 = time.clock()
			rtype = rtype_array[i]
			
			if self.verbtype > 1:
				print 'rtype: ', rtype
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

				if self.opt.cblas_mkl or self.opt.openblas:
					lib = self.libmmult.pcat_model_eval
				else:
					lib = self.libmmult.clib_eval_modl

				dmodels, diff2s, dt_transf = self.pcat_multiband_eval(proposal.xphon, proposal.yphon, proposal.fphon, proposal.dback, self.dat.ncs, self.dat.cfs, weights=self.dat.weights, ref=resids, lib=lib)


				plogL = -0.5*diff2s                
				plogL[(1-self.parity_y)::2,:] = float('-inf') # don't accept off-parity regions
				plogL[:,(1-self.parity_x)::2] = float('-inf')
				dlogP = plogL - logL

				if self.verbtype > 1:
					print 'logL:'
					print logL
					print 'dlogP'
					print dlogP

				
				assert np.isnan(dlogP).any() == False
				
				dts[1,i] = time.clock() - t2
				t3 = time.clock()
				refx, refy = proposal.get_ref_xy()
				regionx = get_region(refx, self.offsetx, self.regsize)
				regiony = get_region(refy, self.offsety, self.regsize)
				
				if proposal.factor is not None:
					# print('proposal factor:', proposal.factor)
					dlogP[regiony, regionx] += proposal.factor
				else:
					print 'proposal factor is None'
				acceptreg = (np.log(np.random.uniform(size=(self.nregy, self.nregx))) < dlogP).astype(np.int32)
				acceptprop = acceptreg[regiony, regionx]
				numaccept = np.count_nonzero(acceptprop)
				''' for each band compute the delta log likelihood between states, theen add these together'''
				for b in xrange(self.nbands):
					dmodel_acpt = np.zeros_like(dmodels[b])
					diff2_acpt = np.zeros_like(diff2s)


					if self.opt.cblas_mkl or self.opt.openblas:

						self.libmmult.pcat_imag_acpt(self.imsz[0], self.imsz[1], dmodels[b], dmodel_acpt, acceptreg, self.regsize, self.margin, self.offsetx, self.offsety)
						# using this dmodel containing only accepted moves, update logL
						self.libmmult.pcat_like_eval(self.imsz[0], self.imsz[1], dmodel_acpt, resids[b], self.dat.weights[b], diff2_acpt, self.regsize, self.margin, self.offsetx, self.offsety)   
					else:

						self.libmmult.clib_updt_modl(self.imsz[0], self.imsz[1], dmodels[b], dmodel_acpt, acceptreg, self.regsize, self.margin, self.offsetx, self.offsety)
						# using this dmodel containing only accepted moves, update logL
						self.libmmult.clib_eval_llik(self.imsz[0], self.imsz[1], dmodel_acpt, resids[b], self.dat.weights[b], diff2_acpt, self.regsize, self.margin, self.offsetx, self.offsetys)   


					# self.libmmult.pcat_imag_acpt(self.imsz[0], self.imsz[1], dmodels[b], dmodel_acpt, acceptreg, self.regsize, self.margin, self.offsetx, self.offsety)
					# # using this dmodel containing only accepted moves, update logL
					# self.libmmult.pcat_like_eval(self.imsz[0], self.imsz[1], dmodel_acpt, resids[b], self.dat.weights[b], diff2_acpt, self.regsize, self.margin, self.offsetx, self.offsety)   

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
					print 'out of bounds'
				outbounds[i] = 1

			for b in xrange(self.nbands):
				diff2_list[i] += np.sum(self.dat.weights[b]*(self.dat.data_array[b]-models[b])*(self.dat.data_array[b]-models[b]))
					
			if self.verbtype > 1:
				print 'end of Loop', i
				print 'self.n'
				print self.n
				print 'diff2'
				print diff2_list[i]
			
		chi2 = np.zeros(self.nbands)
		for b in xrange(self.nbands):
			chi2[b] = np.sum(self.dat.weights[b]*(self.dat.data_array[b]-models[b])*(self.dat.data_array[b]-models[b]))
			
		if self.verbtype > 1:
			print 'end of sample'
			print 'self.n end'
			print self.n


		timestat_array, accept_fracs = self.print_sample_status(dts, accept, outbounds, chi2, movetype)

		if self.opt.visual:
			plt.gcf().clear()
			plt.figure(1, figsize=(9, 4))
			plt.clf()
			plt.subplot(2,3,1)
			plt.title('Data')
			plt.imshow(self.dat.data_array[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.percentile(self.dat.data_array[0], 5), vmax=np.percentile(self.dat.data_array[0], 95.))
			plt.colorbar()
			sizefac = 10.*136
			if self.opt.datatype=='SPIRE':
				s = self.stars[self._F, 0:self.n]*100
			elif self.opt.datatype=='lmc':
				s = 2.
			else:
				s = 10.
			plt.scatter(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], marker='x', s=s, color='r')
			plt.xlim(-0.5, self.imsz[0]-0.5)
			plt.ylim(-0.5, self.imsz[1]-0.5)
			plt.subplot(2,3,2)
			plt.title('Model')
			plt.imshow(models[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.percentile(self.dat.data_array[0], 5), vmax=np.percentile(self.dat.data_array[0], 95))
			# plt.imshow(models[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(models[0]), vmax=np.percentile(models[0], 99.9))
			plt.colorbar()
			plt.subplot(2,3,3)
			plt.title('Residual')
			# plt.imshow(models[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data_array[0]), vmax=np.percentile(data_array[0], 99.9))
			# plt.imshow(models[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(models[0]), vmax=np.percentile(models[0], 99.9))
			# plt.imshow(resids[0], origin='lower', interpolation='none', cmap='Greys',  vmin=np.min(data_array[0]), vmax=np.percentile(data_array[0], 99.9))
			# plt.imshow(resids[0], origin='lower', interpolation='none', cmap='Greys',  vmin = np.percentile(resids[0][weights[0] != 0.], 5), vmax=np.percentile(resids[0][weights[0] != 0.], 95))
			if self.opt.weighted_residual:
				plt.imshow(resids[0]*np.sqrt(self.dat.weights[0]), origin='lower', interpolation='none', cmap='Greys', vmin=-5, vmax=5)
			else:
				plt.imshow(resids[0], origin='lower', interpolation='none', cmap='Greys', vmin = np.percentile(resids[0][self.dat.weights[0] != 0.], 5), vmax=np.percentile(resids[0][self.dat.weights[0] != 0.], 95))

			plt.colorbar()
			plt.scatter(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], marker='x', s=s, color='r')

			plt.subplot(2,3,4)
			plt.title('Data (zoomed in)')
			plt.imshow(self.dat.data_array[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(self.dat.data_array[0]), vmax=np.percentile(self.dat.data_array[0], 99.9))
			plt.colorbar()
			plt.scatter(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], marker='x', s=s, color='r')
			plt.ylim(90, 140)
			plt.xlim(70, 120)
			plt.subplot(2,3,5)
			plt.title('Residual (zoomed in)')

			if self.opt.weighted_residual:
				plt.imshow(resids[0]*np.sqrt(self.dat.weights[0]), origin='lower', interpolation='none', cmap='Greys', vmin=-5, vmax=5)
			else:
				plt.imshow(resids[0], origin='lower', interpolation='none', cmap='Greys', vmin = np.percentile(resids[0][self.dat.weights[0] != 0.], 5), vmax=np.percentile(resids[0][self.dat.weights[0] != 0.], 95))
			plt.colorbar()
			plt.ylim(90, 140)
			plt.xlim(70, 120)
			plt.subplot(2,3,6)

			if self.opt.datatype=='SPIRE':
				binz = np.linspace(np.log10(self.trueminf)+3., np.ceil(np.log10(np.max(self.stars[self._F, 0:self.n]))+3.), 20)
				hist = np.histogram(np.log10(self.stars[self._F, 0:self.n])+3, bins=binz)
				logSv = 0.5*(hist[1][1:]+hist[1][:-1])-3
				binz_Sz = 10**(binz-3)
				dSz = binz_Sz[1:]-binz_Sz[:-1]
				dNdS = hist[0]
				n_steradian = 0.11/(180./np.pi)**2 # field covers 0.11 degrees, should change this though for different fields
				n_steradian *= self.opt.frac # a number of pixels in the image are not actually observing anything
				dNdS_S_twop5 = dNdS*(10**(logSv))**(2.5)
				
				plt.plot(logSv+3, dNdS_S_twop5/n_steradian/dSz, marker='.')
				plt.yscale('log')
				plt.legend()
				plt.xlabel('log($S_{\\nu}$) (mJy)')
				plt.ylabel('dN/dS.$S^{2.5}$ ($Jy^{1.5}/sr$)')
				plt.ylim(1e0, 1e5)
				plt.xlim(np.log10(self.trueminf)+3.-0.5, 2.5)
				plt.tight_layout()
			
			elif self.opt.datatype=='SDSS':
				print ' '

			plt.draw()
			# if savefig:
				# plt.savefig(frame_dir + '/frame_' + str(c) + '.png')
			plt.pause(1e-5)



		return self.n, chi2, timestat_array, accept_fracs, diff2_list, rtype_array, accept, resids


	def idx_parity_stars(self):
		# print self.n, self.offsetx, self.offsety, self.parity_x, self.parity_y, self.regsize
		return idx_parity(self.stars[self._X,:], self.stars[self._Y,:], self.n, self.offsetx, self.offsety, self.parity_x, self.parity_y, self.regsize)

	def bounce_off_edges(self, catalogue): # works on both stars and galaxies
		mask = catalogue[self._X,:] < 0
		catalogue[self._X, mask] *= -1
		mask = catalogue[self._X,:] > (self.imsz[0] - 1)
		catalogue[self._X, mask] *= -1
		catalogue[self._X, mask] += 2*(self.imsz[0] - 1)
		mask = catalogue[self._Y,:] < 0
		catalogue[self._Y, mask] *= -1
		mask = catalogue[self._Y,:] > (self.imsz[1] - 1)
		catalogue[self._Y, mask] *= -1
		catalogue[self._Y, mask] += 2*(self.imsz[1] - 1)
		# these are all inplace operations, so no return value

	def in_bounds(self, catalogue):
		return np.logical_and(np.logical_and(catalogue[self._X,:] > 0, catalogue[self._X,:] < (self.imsz[0] -1)), \
				np.logical_and(catalogue[self._Y,:] > 0, catalogue[self._Y,:] < self.imsz[1] - 1))


	def flux_proposal(self, f0, nw, trueminf=None):
		if trueminf is None:
			trueminf = self.trueminf
		lindf = np.float32(self.err_f/(self.regions_factor*np.sqrt(self.opt.nominal_nsrc*(2+self.nbands))))
		logdf = np.float32(0.01/np.sqrt(self.opt.nominal_nsrc))
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
		# print('were moving', nw, 'sources')
		stars0 = self.stars.take(idx_move, axis=1)
		starsp = np.empty_like(stars0)
		
		f0 = stars0[self._F:,:]
		pfs = []
		color_factors = np.zeros((self.nbands-1, nw)).astype(np.float32)

		for b in xrange(self.nbands):
			if b==0:
				pf = self.flux_proposal(f0[b], nw)
			else:
				pf = self.flux_proposal(f0[b], nw, trueminf=1) #place a minor minf to avoid negative fluxes in non-pivot bands
			pfs.append(pf)
 
		if (np.array(pfs)<0).any():
			print 'negative flux!'
			print np.array(pfs)[np.array(pfs)<0]

		dlogf = np.log(pfs[0]/f0[0])
		# print 'average flux difference'
		# print np.average(np.abs(f0[0]-pfs[0]))
		if self.verbtype > 1:
			print 'average flux difference'
			print np.average(np.abs(f0[0]-pfs[0]))
		# print 'average flux difference'
		# print np.median(np.abs(f0[0]-pfs[0]))
		factor = -self.truealpha*dlogf

		# print('factor:', factor)

		if np.isnan(factor).any():
			print 'factor nan from flux'
			print 'number of f0 zero elements:', len(f0[0])-np.count_nonzero(np.array(f0[0]))
			if self.verbtype > 1:
				print 'factor'
				print factor
			factor[np.isnan(factor)]=0

		''' the loop over bands below computes colors and prior factors in color used when sampling the posterior
		come back to this later  '''
		modl_eval_colors = []
		for b in xrange(self.nbands-1):
			colors = fluxes_to_color(pfs[0], pfs[b+1])
			orig_colors = fluxes_to_color(f0[0], f0[b+1])
			colors[np.isnan(colors)] = self.color_mus[b] # make nan colors not affect color_factors
			orig_colors[np.isnan(orig_colors)] = self.color_mus[b]
			color_factors[b] -= (colors - self.color_mus[b])**2/(2*self.color_sigs[b]**2)
			color_factors[b] += (orig_colors - self.color_mus[b])**2/(2*self.color_sigs[b]**2)
			modl_eval_colors.append(colors)
	
		assert np.isnan(color_factors).any()==False

		if np.isnan(color_factors).any():
			print 'color factors nan'                

		if self.verbtype > 1:
			print 'avg abs color_factors:', np.average(np.abs(color_factors))
			print 'avg abs flux factor:', np.average(np.abs(factor))

		factor = np.array(factor) + np.sum(color_factors, axis=0)
		
		dpos_rms = np.float32(np.sqrt(self.opt.N_eff/(2*np.pi))*self.err_f/(np.sqrt(self.nominal_nsrc*self.regions_factor*(2+self.nbands))))/(np.maximum(f0[0],pfs[0]))

		if self.verbtype > 1:
			print 'dpos_rms'
			print dpos_rms
		
		dpos_rms[dpos_rms < 1e-3] = 1e-3 #do we need this line? perhaps not
		dx = np.random.normal(size=nw).astype(np.float32)*dpos_rms
		dy = np.random.normal(size=nw).astype(np.float32)*dpos_rms
		starsp[self._X,:] = stars0[self._X,:] + dx
		starsp[self._Y,:] = stars0[self._Y,:] + dy
		
		if self.verbtype > 1:
			print 'dx'
			print dx
			print 'dy'
			print dy
			print 'mean dx and mean dy'
			print np.mean(np.abs(dx)), np.mean(np.abs(dy))

		for b in xrange(self.nbands):
			starsp[self._F+b,:] = pfs[b]
			if (pfs[b]<0).any():
				print 'proposal fluxes less than 0'
				print 'band', b
				print pfs[b]
		self.bounce_off_edges(starsp)

		proposal = Proposal(self.opt)
		proposal.add_move_stars(idx_move, stars0, starsp, modl_eval_colors)
		
		assert np.isinf(factor).any()==False
		assert np.isnan(factor).any()==False

		proposal.set_factor(factor)
		return proposal



	def birth_death_stars(self):
		lifeordeath = np.random.randint(2)
		nbd = (self.nregx * self.nregy) / 4
		proposal = Proposal(self.opt)
		# birth
		if lifeordeath and self.n < self.max_nsrc: # need room for at least one source
			nbd = min(nbd, self.max_nsrc-self.n) # add nbd sources, or just as many as will fit
			# mildly violates detailed balance when n close to nstar
			# want number of regions in each direction, divided by two, rounded up
			mregx = ((self.imsz[0] / self.regsize + 1) + 1) / 2 # assumes that imsz are multiples of regsize
			mregy = ((self.imsz[1] / self.regsize + 1) + 1) / 2
			starsb = np.empty((2+self.nbands, nbd), dtype=np.float32)
			starsb[self._X,:] = (np.random.randint(mregx, size=nbd)*2 + self.parity_x + np.random.uniform(size=nbd))*self.regsize - self.offsetx
			starsb[self._Y,:] = (np.random.randint(mregy, size=nbd)*2 + self.parity_y + np.random.uniform(size=nbd))*self.regsize - self.offsety
			
			for b in xrange(self.nbands):
				if b==0:
					starsb[self._F+b,:] = self.trueminf * np.exp(np.random.exponential(scale=1./(self.truealpha-1.),size=nbd))
				else:
					# draw new source colors from color prior
					new_colors = np.random.normal(loc=self.color_mus[b-1], scale=self.color_sigs[b-1], size=nbd)
					starsb[self._F+b,:] = starsb[self._F,:]*10**(0.4*new_colors)
			
					if (starsb[self._F+b,:]<0).any():
						print 'negative birth star fluxes'
						print 'new_colors'
						print new_colors
						print 'starsb fluxes:'
						print starsb[self._F+b,:]

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
			nbd = min(nbd, idx_reg.size) # kill nbd sources, or however many sources remain
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
		nms = (self.nregx * self.nregy) / 4
		goodmove = False
		proposal = Proposal(self.opt)
		# split
		if splitsville and self.n > 0 and self.n < self.max_nsrc and bright_n > 0: # need something to split, but don't exceed nstar
			nms = min(nms, bright_n, self.max_nsrc-self.n) # need bright source AND room for split source
			dx = (np.random.normal(size=nms)*self.kickrange).astype(np.float32)
			dy = (np.random.normal(size=nms)*self.kickrange).astype(np.float32)
			idx_move = np.random.choice(idx_bright, size=nms, replace=False)
			stars0 = self.stars.take(idx_move, axis=1)

			fminratio = stars0[self._F,:] / self.trueminf
 
			if self.verbtype > 1:
				print 'stars0 at splitsville start'
				print stars0
				print 'fminratio here'
				print fminratio
				print 'dx'
				print dx
				print 'dy'
				print dy
				print 'idx_move'
				print idx_move
				
			fracs.append((1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32))
			
			# color stuff, look at later
			for b in xrange(self.nbands-1):
				# changed to split similar fluxes
				d_color = np.random.normal(0,self.split_col_sig)
				frac_sim = np.exp(d_color/self.k)*fracs[0]/(1-fracs[0]+np.exp(d_color/self.k)*fracs[0])
				fracs.append(frac_sim)

				
			starsp = np.empty_like(stars0)
			starsb = np.empty_like(stars0)

			starsp[self._X,:] = stars0[self._X,:] + ((1-fracs[0])*dx)
			starsp[self._Y,:] = stars0[self._Y,:] + ((1-fracs[0])*dy)
			starsb[self._X,:] = stars0[self._X,:] - fracs[0]*dx
			starsb[self._Y,:] = stars0[self._Y,:] - fracs[0]*dy

			for b in xrange(self.nbands):
				
				starsp[self._F+b,:] = stars0[self._F+b,:]*fracs[b]
				starsb[self._F+b,:] = stars0[self._F+b,:]*(1-fracs[b])
				if (starsp[self._F+b,:]<0).any():
					print 'neg starsp in band', b
					print 'stars0'
					print stars0
					print 'fracs[b]'
					print fracs[b]
					print 'starsp[self._F+b,:]'
					print starsp[self._F+b,:]
				if (starsb[self._F+b,:]<0).any():
					print 'neg starsb in band', b
					print 'stars0'
					print stars0
					print '1-fracs[b]'
					print (1-fracs[b])
					print 'starsb[self._F+b,:]'
					print starsb[self._F+b,:]
			# don't want to think about how to bounce split-merge
			# don't need to check if above fmin, because of how frac is decided
			inbounds = np.logical_and(self.in_bounds(starsp), self.in_bounds(starsb))
			stars0 = stars0.compress(inbounds, axis=1)
			starsp = starsp.compress(inbounds, axis=1)
			starsb = starsb.compress(inbounds, axis=1)
			idx_move = idx_move.compress(inbounds)
			fminratio = fminratio.compress(inbounds)

			for b in xrange(self.nbands):
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
				print 'splitsville happening'
				print 'goodmove:', goodmove
				print 'invpairs'
				print invpairs
				print 'nms:', nms
				print 'sum_fs'
				print sum_fs
				print 'fminratio'
				print fminratio

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
			choosable = np.zeros(self.max_nsrc, dtype=np.bool)
			choosable[idx_reg] = True
			nchoosable = float(idx_reg.size)
			invpairs = np.empty(nms)
			
			if self.verbtype > 1:
				print 'merging two things!'
				print 'nms:', nms
				print 'idx_move', idx_move
				print 'idx_kill', idx_kill
				
			for k in xrange(nms):
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

			for b in xrange(self.nbands):
				sum_fs.append(f0[b,:] + fk[b,:])
				fracs.append(f0[b,:] / sum_fs[b])
			fminratio = sum_fs[0] / self.trueminf
			
			if self.verbtype > 1:
				print 'fminratio'
				print fminratio
				print 'nms is now', nms
				print 'sum_fs[0]', sum_fs[0]
				print 'all sum_fs:'
				print sum_fs
				print 'stars0'
				print stars0
				print 'starsk'
				print starsk
				print 'idx_move'
				print idx_move
				print 'idx_kill'
				print idx_kill
				
			starsp = np.empty_like(stars0)
			# place merged source at center of flux of previous two sources
			starsp[self._X,:] = fracs[0]*stars0[self._X,:] + (1-fracs[0])*starsk[self._X,:]
			starsp[self._Y,:] = fracs[0]*stars0[self._Y,:] + (1-fracs[0])*starsk[self._Y,:]
			
			for b in xrange(self.nbands):
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
			factor = np.log(self.truealpha-1) + (self.truealpha-1)*np.log(self.trueminf) - self.truealpha*np.log(fracs[0]*(1-fracs[0])*sum_fs[0]) + np.log(2*np.pi*self.kickrange*self.kickrange) - np.log(self.imsz[0]*self.imsz[1]) + np.log(1. - 2./fminratio) + np.log(bright_n) + np.log(invpairs) + np.log(sum_fs[0])

			for b in xrange(self.nbands-1):
				stars0_color = fluxes_to_color(stars0[self._F,:], stars0[self._F+b+1,:])
				starsp_color = fluxes_to_color(starsp[self._F,:], starsp[self._F+b+1,:])
				dc = self.k*(np.log(fracs[b+1]/fracs[0]) - np.log((1-fracs[b+1])/(1-fracs[0])))
				# added_fac comes from the transition kernel of splitting colors in the manner that we do
				added_fac = 0.5*np.log(2*np.pi*self.split_col_sig**2)+(dc**2/(2*self.split_col_sig**2))
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
				print 'kickrange factor', np.log(2*np.pi*self.kickrange*self.kickrange)
				print 'imsz factor', np.log(self.imsz[0]*self.imsz[1]) 
				print 'fminratio:', fminratio
				print 'fmin factor', np.log(1. - 2./fminratio)
				print 'kickrange factor', np.log(2*np.pi*self.kickrange*self.kickrange) - np.log(self.imsz[0]*self.imsz[1]) + np.log(1. - 2./fminratio)
				print 'factor after colors'
				print factor
		return proposal


class samples():

	def __init__(self, opt):
		self.nsample = np.zeros(opt.nsamp, dtype=np.int32)
		self.xsample = np.zeros((opt.nsamp, opt.max_nsrc), dtype=np.float32)
		self.ysample = np.zeros((opt.nsamp, opt.max_nsrc), dtype=np.float32)
		self.timestats = np.zeros((opt.nsamp, 6, 4), dtype=np.float32)
		self.diff2_all = np.zeros((opt.nsamp, opt.nloop), dtype=np.float32)
		self.accept_all = np.zeros((opt.nsamp, opt.nloop), dtype=np.float32)
		self.rtypes = np.zeros((opt.nsamp, opt.nloop), dtype=np.float32)
		self.accept_stats = np.zeros((opt.nsamp, 4), dtype=np.float32)
		self.tq_times = np.zeros(opt.nsamp, dtype=np.float32)
		self.fsample = [np.zeros((opt.nsamp, opt.max_nsrc), dtype=np.float32) for x in xrange(opt.nbands)]
		self.colorsample = [[] for x in xrange(opt.nbands-1)]
		self.residuals = np.zeros((opt.nbands, opt.residual_samples, opt.width, opt.height))
		self.chi2sample = np.zeros((opt.nsamp, opt.nbands), dtype=np.int32)
		self.nbands = opt.nbands
		self.opt = opt

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

		for b in xrange(self.nbands):
			self.fsample[b][j,:] = model.stars[Model._F+b,:]
			if self.opt.nsamp - j < self.opt.residual_samples+1:
				self.residuals[b,-(self.opt.nsamp-j),:,:] = resids[b] 

	def save_samples(self, result_path, timestr):

		np.savez(result_path + '/' + str(timestr) + '/chain.npz', n=self.nsample, x=self.xsample, y=self.ysample, f=self.fsample, \
				chi2=self.chi2sample, times=self.timestats, accept=self.accept_stats, diff2s=self.diff2_all, rtypes=self.rtypes, \
				accepts=self.accept_all, residuals=self.residuals)

''' This class sets up the data structures for data/data-related information. 
load_in_data() loads in data, generates the PSF template and computes weights from the noise model
'''
class pcat_data():

	trueback_dict_of_dicts = dict({'8151':dict({"r":135., "i":215., "g":234., "z":140.}), '136':dict({"r":179., "i":310., "g":95.})})


	def __init__(self, opt):
		self.ncs = []
		self.nbins = []
		self.psfs = []
		self.cfs = []
		self.data_array = []
		self.weights = []
		self.exposures = []
		self.errors = []
		self.opt = opt
		self.fast_astrom = wcs_astrometry()
		self.widths = []
		self.heights = []
		if opt.datatype=='SDSS':
			self.gains = []

	def find_lowest_mod(self, number, mod_number):
		''' this function finds largest dimension for a given image that is divisible by nregion, which is needed to do subregion mechanics'''
		while number > 0:
			if np.mod(number, mod_number) == 0:
				return number
			else:
				number -= 1
		return False

	def load_in_data(self, opt):

		for band in opt.bands:
			if opt.datatype=='SPIRE':
				if opt.mock_name is None:
					image, error, exposure, mask, band_dict = load_in_spire_map(opt, band, astrom=self.fast_astrom)
					opt.band_dict = band_dict
				else:
					image, error, exposure, mask = load_in_mock_map(opt.mock_name, band)
			elif opt.datatype=='SDSS':
				if opt.mock_name is None:
					image, psf, nc, cf, nmgy_per_count, bias, gain, nbin = load_in_sdss_data(opt, band=band)
					print('bias/gain/nbin:', bias, gain, nbin)
			elif opt.datatype=='lmc':
				image, psf_pixel_fwhm, sky_level, gain = load_in_lmc_map(opt)
				# image[image > 50000] = sky_level
				opt.psf_pixel_fwhm = psf_pixel_fwhm
				opt.sky_level = sky_level
				opt.gain = gain

			if opt.auto_resize:
				smaller_dim = np.min([image.shape[0], image.shape[1]])
				opt.width = self.find_lowest_mod(smaller_dim, opt.nregion)
				opt.height = opt.width
			
			if opt.width > 0:
				image = image[opt.x0:opt.x0+opt.width,opt.y0:opt.y0+opt.height]
				if opt.datatype=='SPIRE':
					error = error[opt.x0:opt.x0+opt.width,opt.y0:opt.y0+opt.height]
					exposure = exposure[opt.x0:opt.x0+opt.width,opt.y0:opt.y0+opt.height]
					mask = mask[opt.x0:opt.x0+opt.width,opt.y0:opt.y0+opt.height]
				self.opt.imsz = (opt.width, opt.height)
			else:
				self.opt.imsz = (image.shape[0], image.shape[1])

			if opt.datatype=='SDSS':
				image -= bias
				variance = image / gain
				weight = 1. / variance
				self.gains.append(gain)
				pixel_variance = self.trueback_dict_of_dicts[str(opt.field)][opt.bands[0]]/self.gains[0]

				print(' image has mean and std', np.mean(image), np.std(image))
				self.data_array.append(image.astype(np.float32)) 

			elif opt.datatype=='SPIRE':
				variance = error**2
				variance[variance==0.]=np.inf
				weight = 1. / variance
				opt.frac = np.count_nonzero(weight)/float(opt.width*opt.height)
				psf, cf, nc, nbin = get_gaussian_psf_template(pixel_fwhm=opt.psf_pixel_fwhm)
				self.data_array.append(image.astype(np.float32)+opt.mean_offset) # constant offset, may need to change
				self.errors.append(error.astype(np.float32))
				self.exposures.append(exposure.astype(np.float32))
				pixel_variance = np.median(self.errors[0]**2)

			elif opt.datatype=='lmc':
				variance = image / gain
				weight = 1. / variance
				print('weights:')
				print weight
				weight[image==0.]=0.
				print('min/max weight are ', np.min(weight), np.max(weight))
				psf, cf, nc, nbin = get_gaussian_psf_template(pixel_fwhm=psf_pixel_fwhm)
				self.data_array.append(image.astype(np.float32))
				pixel_variance = sky_level / gain

			print('sum of PSF is ', np.sum(psf))
			self.psfs.append(psf)
			self.cfs.append(cf)
			self.ncs.append(nc)
			self.nbins.append(nbin)
			self.weights.append(weight.astype(np.float32))

		opt.regsize = opt.imsz[0]/opt.nregion
		opt.regions_factor = 1./float(opt.nregion**2)
		print('regsize/regions_factor:', opt.regsize, opt.regions_factor)
		assert opt.imsz[0] % opt.regsize == 0 
		assert opt.imsz[1] % opt.regsize == 0 

		print('pixel_variance:', pixel_variance)
		if opt.datatype=='SPIRE' or opt.datatype=='lmc':
			opt.N_eff = 4*np.pi*(opt.psf_pixel_fwhm/2.355)**2
		elif opt.datatype=='SDSS':
			opt.N_eff = 17.5 # should just compute this automatically from the PSF template
		
		opt.err_f = np.sqrt(opt.N_eff * pixel_variance)

		print('opt.errf:', opt.err_f)
		print('opt.neff:', opt.N_eff)

		return opt



# -------------------- actually execute the thing ----------------

def pcat_main(datatype='lmc'):

	opt = load_opt(datatype)

	dat = pcat_data(opt)
	opt = dat.load_in_data(opt)

	print('width/height:', opt.width, opt.height)

	''' Here is where we initialize the C libraries and instantiate the arrays that will store our thinned samples and other stats '''
	# libmmult = npct.load_library('pcat-lion', '.')

	if opt.cblas_mkl:
		libmmult = npct.load_library('pcat-lion', '.')
	elif opt.openblas:
		# libmmult = ctypes.cdll['pcat-lion-openblas.so']
		libmmult = npct.load_library('pcat-lion-openblas', '.')
	else:
		libmmult = ctypes.cdll['blas.so'] # not sure how stable this is, trying to find a good Python 3 fix to deal with path configuration
		# libmmult = npct.load_library('blas', '.')


	initialize_c(opt, libmmult)

	#create directory for results, save config file from run
	frame_dir, newdir = create_directories(opt)
	opt.frame_dir = frame_dir
	opt.newdir = newdir
	save_params(newdir, opt)

	start_time = time.clock()

	samps = samples(opt)

	model = Model(opt, dat, libmmult)

	# run sampler for opt.nsamp thinned states

	for j in xrange(opt.nsamp):
		print 'Sample', j

		_, chi2_all, statarrays,  accept_fracs, diff2_list, rtype_array, accepts, resids = model.run_sampler()
		samps.add_sample(j, model, diff2_list, accepts, rtype_array, accept_fracs, chi2_all, statarrays, resids)

	print 'saving...'

	# save catalog ensemble and other diagnostics
	samps.save_samples(opt.result_path, opt.timestr)

	# save final catalog state
	np.savez(opt.result_path + '/'+str(opt.timestr)+'/final_state.npz', cat=model.stars)

	dt_total = time.clock() - start_time
	print 'Full Run Time (s):', np.round(dt_total,3)
	print 'Time String:', str(opt.timestr)


# run PCAT!

pcat_main(datatype='SDSS')
