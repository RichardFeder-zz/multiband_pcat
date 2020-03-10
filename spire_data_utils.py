from fast_astrom import *
import numpy as np
from astropy.convolution import Gaussian2DKernel
from image_eval import psf_poly_fit, image_model_eval


def get_gaussian_psf_template_3_5_20(pixel_fwhm = 3., nbin=5):
	nc = nbin**2
	psfnew = Gaussian2DKernel((pixel_fwhm/2.355)*nbin, x_size=125, y_size=125).array.astype(np.float32)
	cf = psf_poly_fit(psfnew, nbin=nbin)
	return psfnew, cf, nc, nbin

def get_gaussian_psf_template(pixel_fwhm=3., nbin=5, normalization='max'):
	nc = 25
	psfnew = Gaussian2DKernel((pixel_fwhm/2.355)*nbin, x_size=125, y_size=125).array.astype(np.float32)
	print('psfmax is ', np.max(psfnew))

	if normalization == 'max':
		print('Normalizing PSF by kernel maximum')
		psfnew /= np.max(psfnew)
		psfnew /= 4*np.pi*(pixel_fwhm/2.355)**2
	else:
		print('Normalizing PSF by kernel sum')
		psfnew *= nc
	cf = psf_poly_fit(psfnew, nbin=nbin)
	return psfnew, cf, nc, nbin


def load_in_map(gdat, band=0, astrom=None):

	if gdat.file_path is None:
		file_path = gdat.data_path+gdat.dataname+'/'+gdat.dataname+'_'+gdat.tail_name+'.fits'
	else:
		file_path = gdat.file_path

	print('band is ', gdat.band_dict[band])
	file_path = file_path.replace('PSW', 'P'+str(gdat.band_dict[band])+'W')


	print('file_path:', file_path)

	if astrom is not None:
		print('loading from ', gdat.band_dict[band])
		astrom.load_wcs_header_and_dim(file_path)

	spire_dat = fits.open(file_path)
	image = np.nan_to_num(spire_dat[1].data)
	error = np.nan_to_num(spire_dat[2].data)
	exposure = spire_dat[3].data
	mask = spire_dat[4].data

	return image, error, exposure, mask


def load_param_dict(timestr, result_path='/Users/richardfeder/Documents/multiband_pcat/spire_results/'):
	
	filepath = result_path + timestr
	filen = open(filepath+'/params.txt','rb')
	print(filen)
	pdict = pickle.load(filen)
	print(pdict)
	opt = objectview(pdict)

	print('param dict load')
	return opt, filepath, result_path



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

	def load_in_data(self, gdat, map_object=None, tail_name=None):

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
				if gdat.round_up_or_down=='up':
					gdat.width = self.find_nearest_upper_mod(larger_dim, gdat.nregion)
				else:
					gdat.width = self.find_lowest_mod(smaller_dim, gdat.nregion)
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
				self.data_array.append(padded_image.astype(np.float32)-gdat.mean_offsets[i]) # constant offset, will need to change
				self.exposures.append(padded_exposure.astype(np.float32))


			elif gdat.width > 0:
				print('were here now')
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
				self.data_array.append(image.astype(np.float32)-gdat.mean_offsets[i]) # constant offset, will need to change
				print('image maximum is ', np.max(image))

				self.exposures.append(exposure.astype(np.float32))

			else:
				image_size = (image.shape[0], image.shape[1])
				variance = error**2
				variance[variance==0.]=np.inf
				weight = 1. / variance

				self.weights.append(weight.astype(np.float32))
				self.errors.append(error.astype(np.float32))
				#self.data_array.append(image.astype(np.float32)) # constant offset, will need to change
				# self.data_array.append(image.astype(np.float32)+gdat.mean_offset) 
				self.data_array.append(image.astype(np.float32)-gdat.mean_offsets[i]) 
				# self.exposures.append(exposure.astype(np.float32))


			if i==0:
				gdat.imsz0 = image_size
			gdat.imszs.append(image_size)
			gdat.regsizes.append(image_size[0]/gdat.nregion)



			print('image maximum is ', np.max(self.data_array[0]))


			gdat.frac = np.count_nonzero(weight)/float(gdat.width*gdat.height)
			print('gdat.frac is ', gdat.frac)
			# psf, cf, nc, nbin = get_gaussian_psf_template(pixel_fwhm=gdat.psf_pixel_fwhm, normalization=gdat.normalization)
			psf, cf, nc, nbin = get_gaussian_psf_template_3_5_20(pixel_fwhm=gdat.psf_pixel_fwhm)

			print('sum of PSF is ', np.sum(psf))
			self.psfs.append(psf)
			self.cfs.append(cf)
			self.ncs.append(nc)
			self.nbins.append(nbin)
			self.biases.append(gdat.bias)

		gdat.regions_factor = 1./float(gdat.nregion**2)
		print(gdat.imsz0[0], gdat.regsizes[0], gdat.regions_factor)
		assert gdat.imsz0[0] % gdat.regsizes[0] == 0 
		assert gdat.imsz0[1] % gdat.regsizes[0] == 0 

		pixel_variance = np.median(self.errors[0]**2)
		print('pixel_variance:', pixel_variance)
		gdat.N_eff = 4*np.pi*(gdat.psf_pixel_fwhm/2.355)**2 # 2 instead of 4 for spire beam size
		gdat.err_f = np.sqrt(gdat.N_eff * pixel_variance)/10


		# return gdat