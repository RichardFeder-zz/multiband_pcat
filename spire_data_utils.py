from fast_astrom import *
import numpy as np
from astropy.convolution import Gaussian2DKernel
from image_eval import psf_poly_fit, image_model_eval
import pickle
import matplotlib
import matplotlib.pyplot as plt
from astropy.wcs import WCS


class objectview(object):
	def __init__(self, d):
		self.__dict__ = d

def get_gaussian_psf_template_3_5_20(pixel_fwhm = 3., nbin=5):
	nc = nbin**2
	psfnew = Gaussian2DKernel((pixel_fwhm/2.355)*nbin, x_size=125, y_size=125).array.astype(np.float32)
	cf = psf_poly_fit(psfnew, nbin=nbin)
	return psfnew, cf, nc, nbin

def get_gaussian_psf_template(pixel_fwhm=3., nbin=5, normalization='max'):
	nc = 25
	psfnew = Gaussian2DKernel((pixel_fwhm/2.355)*nbin, x_size=125, y_size=125).array.astype(np.float32)

	if normalization == 'max':
		print('Normalizing PSF by kernel maximum')
		psfnew /= np.max(psfnew)
		psfnew /= 4*np.pi*(pixel_fwhm/2.355)**2
	else:
		print('Normalizing PSF by kernel sum')
		psfnew *= nc
	cf = psf_poly_fit(psfnew, nbin=nbin)
	return psfnew, cf, nc, nbin


def load_in_map(gdat, band=0, astrom=None, show_input_maps=False):

	if gdat.file_path is None:
		file_path = gdat.data_path+gdat.dataname+'/'+gdat.tail_name+'.fits'
	else:
		file_path = gdat.file_path

	file_path = file_path.replace('PSW', 'P'+str(gdat.band_dict[band])+'W')


	if gdat.verbtype > 1:
		print('band is ', gdat.band_dict[band])
		print('file_path:', file_path)

	if astrom is not None:
		print('loading from ', gdat.band_dict[band])
		astrom.load_wcs_header_and_dim(file_path)

	spire_dat = fits.open(file_path)

	image = np.nan_to_num(spire_dat['SIGNAL'].data)
	error = np.nan_to_num(spire_dat['ERROR'].data)

	if gdat.use_mask:
		if gdat.bolocam_mask:
			mask = fits.open(gdat.data_path+'bolocam_mask_P'+str(gdat.band_dict[band])+'W.fits')[0].data
		else:
			mask = spire_dat['MASK'].data
	else:
		mask = np.ones_like(image)

	# image = np.nan_to_num(spire_dat[1].data)
	# error = np.nan_to_num(spire_dat[2].data)
	# exposure = spire_dat[3].data
	# mask = spire_dat[3].data
	
	# error = np.nan_to_num(spire_dat[2].data) # temporary for sims_for_richard dataset
	# exposure = spire_dat[3].data
	# mask = spire_dat[4].data

	if show_input_maps:

		plt.figure(figsize=(12, 4))
		plt.subplot(1,3,1)
		plt.title('image map')
		plt.imshow(image, vmin=np.percentile(image, 5), vmax=np.percentile(image, 95), origin='lower')
		plt.colorbar()
		plt.subplot(1,3,2)
		plt.title('err map')
		plt.imshow(error, vmin=np.percentile(error, 5), vmax=np.percentile(error, 95), origin='lower')
		plt.colorbar()
		plt.subplot(1,3,3)
		plt.title('mask')
		plt.imshow(mask, origin='lower')

		plt.show()

	return image, error, mask, file_path


def load_param_dict(timestr, result_path='/Users/luminatech/Documents/multiband_pcat/spire_results/'):
	
	filepath = result_path + timestr
	filen = open(filepath+'/params.txt','rb')
	pdict = pickle.load(filen)
	opt = objectview(pdict)

	return opt, filepath, result_path


def get_rect_mask_bounds(mask):

	''' this function assumes the mask is rectangular in shape, with ones in the desired region and zero otherwise. '''

	idxs = np.argwhere(mask == 1.0)
	bounds = np.array([[np.min(idxs[:,0]), np.max(idxs[:,0])], [np.min(idxs[:,1]), np.max(idxs[:,1])]])

	return bounds


''' This class sets up the data structures for data/data-related information. 
load_in_data() loads in data, generates the PSF template and computes weights from the noise model
'''
class pcat_data():

	template_bands = dict({'sze':['S', 'M', 'L'], 'lensing':['S', 'M', 'L'], 'dust':['S', 'M', 'L'], 'planck':['S', 'M', 'L']}) # should just integrate with the same thing in Lion main

	def __init__(self, auto_resize=False, nregion=1):
		self.ncs, self.nbins, self.psfs, self.cfs, self.biases, self.data_array, self.weights, self.masks, self.errors, \
			self.widths, self.heights, self.fracs, self.template_array = [[] for x in range(13)]
		self.fast_astrom = wcs_astrometry(auto_resize, nregion=nregion)

	def load_in_data(self, gdat, map_object=None, tail_name=None, show_input_maps=False):

		gdat.imszs, gdat.regsizes, gdat.margins, gdat.bounds = [[] for x in range(4)]

		for i, band in enumerate(gdat.bands):


			if map_object is not None:

				obj = map_object[band]
				image = np.nan_to_num(obj['signal'])
				error = np.nan_to_num(obj['error'])

				exposure = obj['exp'].data
				mask = obj['mask']
				gdat.psf_pixel_fwhm = obj['widtha']/obj['pixsize']# gives it in arcseconds and neet to convert to pixels
				self.fast_astrom.load_wcs_header_and_dim(head=obj['shead'], round_up_or_down=gdat.round_up_or_down)
				gdat.dataname = obj['name']
				if i > 0:
					self.fast_astrom.fit_astrom_arrays(0, i)


			elif gdat.mock_name is None:

				image, error, mask, file_name = load_in_map(gdat, band, astrom=self.fast_astrom, show_input_maps=show_input_maps)

				# bounds = get_rect_mask_bounds(mask) if gdat.bolocam_mask else None
				bounds = get_rect_mask_bounds(mask)

				print('bounds are ', bounds)

				if gdat.verbtype > 1:
					print('bounds for band ', i, 'are ', bounds)

				if bounds is not None:
					if gdat.round_up_or_down == 'up':
						big_dim = np.maximum(find_nearest_upper_mod(bounds[0,1]-bounds[0,0], gdat.nregion), find_nearest_upper_mod(bounds[1,1]-bounds[1,0], gdat.nregion))
					else:
						big_dim = np.maximum(find_lowest_mod(bounds[0,1]-bounds[0,0], gdat.nregion), find_lowest_mod(bounds[1,1]-bounds[1,0], gdat.nregion))

					self.fast_astrom.dims[i] = (big_dim, big_dim)


				gdat.bounds.append(bounds)

				template_list = [] 

				sed_cirr = dict({100:1.7, 250:3.5, 350:1.6, 500:0.85}) # MJy/sr
				relative_dust_sed_dict = dict({'S':sed_cirr[250]/sed_cirr[100], 'M':sed_cirr[350]/sed_cirr[100], 'L':sed_cirr[500]/sed_cirr[100]}) # relative to 100 micron
				temp_mock_amps_dict = dict({'S':0.0111, 'M': 0.1249, 'L': 0.6912})
				temp_mock_amps = [0.0111, 0.1249, 0.6912] # MJy/sr

				flux_density_conversion_dict = dict({'S': 86.29e-4, 'M':16.65e-3, 'L':34.52e-3})
				flux_density_conversion_facs = [86.29e-4, 16.65e-3, 34.52e-3]

				if gdat.n_templates > 0:

					for t, template_name in enumerate(gdat.template_names):
						print('template name is ', template_name)
						if gdat.band_dict[band] in self.template_bands[template_name]:

							if gdat.verbtype > 1:
								print('were in business, ', gdat.band_dict[band], self.template_bands[template_name], gdat.lam_dict[gdat.band_dict[band]])

							# if gdat.template_filename is not None:
							# 	temp_name = gdat.template_filename[t]
							# 	template_file_name = temp_name.replace('PSW', 'P'+str(gdat.band_dict[band])+'W')
							# else:
							# 	template_file_name = file_name.replace('.fits', '_'+template_name+'.fits')
							
							# print('template file name is ', template_file_name)

							if template_name=='planck':
								print('file name here is ', file_name)
								template = fits.open(file_name)[template_name +'_500'].data # temporary
								# template = fits.open(file_name)[template_name +'_'+str(gdat.lam_dict[gdat.band_dict[band]])].data
							else:
								template = fits.open(file_name)[template_name].data

							if show_input_maps:
								plt.figure()
								plt.title(template_name)
								plt.imshow(template, origin='lower')
								plt.colorbar()
								plt.show()

							# if template_name=='sze':
							# 	template = fits.open(template_file_name)[0].data
							# elif template_name=='dust':
							# 	template = fits.open(file_name)[5].data
							# 	# template = np.load(template_file_name)['iris_map']


							# if gdat.inject_dust and template_name=='planck':
							# 	print('we are putting in dust now')

							# 	image += template

							# 	if show_input_maps:
							# 		plt.figure()
							# 		plt.subplot(1,2,1)
							# 		plt.title('injected dust')
							# 		plt.imshow(template, origin='lower')
							# 		plt.colorbar()
							# 		plt.subplot(1,2,2)
							# 		plt.title('image + dust')
							# 		plt.imshow(image, origin='lower')
							# 		plt.colorbar()
							# 		plt.tight_layout()
							# 		plt.show()

							if gdat.inject_sz_frac > 0. and template_name=='sze':
								print('injecting SZ frac of ', gdat.inject_sz_frac)
								
								template_inject = gdat.inject_sz_frac*template*temp_mock_amps_dict[gdat.band_dict[band]]*flux_density_conversion_dict[gdat.band_dict[band]]

								image += template_inject

								if show_input_maps:
									plt.figure(figsize=(8, 4))
									plt.subplot(1,2,1)
									plt.title('inj. amp is '+str(np.round(gdat.inject_sz_frac*temp_mock_amps_dict[gdat.band_dict[band]]*flux_density_conversion_dict[gdat.band_dict[band]], 4)))
									plt.imshow(template_inject, origin='lower')
									plt.colorbar()
									plt.subplot(1,2,2)
									plt.title('image + sz')
									plt.imshow(image, origin='lower')
									plt.colorbar()
									plt.tight_layout()
									plt.show()


						else:

							template = None

						template_list.append(template)


				if i > 0:
					if gdat.verbtype > 1:
						print('we have more than one band:', gdat.bands[0], band)
					self.fast_astrom.fit_astrom_arrays(0, i, bounds0=gdat.bounds[0], bounds1=gdat.bounds[i])


				if gdat.noise_thresholds is not None:
					error[error > gdat.noise_thresholds[i]] = 0 # this equates to downweighting the pixels


			else:
				image, error, mask = load_in_mock_map(gdat.mock_name, band)
			
			if gdat.auto_resize:

				# if gdat.bolocam_mask:

				error = error[bounds[0,0]:bounds[0,1], bounds[1,0]:bounds[1,1]]
				image = image[bounds[0,0]:bounds[0,1], bounds[1,0]:bounds[1,1]]
				smaller_dim = np.min(image.shape)
				larger_dim = np.max(image.shape)

				# else:

				# 	smaller_dim = np.min([image.shape[0]-gdat.x0, image.shape[1]-gdat.y0]) # option to include lower left corner
				# 	larger_dim = np.max([image.shape[0]-gdat.x0, image.shape[1]-gdat.y0])

				if gdat.verbtype > 1:
					print('smaller dim is', smaller_dim)
					print('larger dim is ', larger_dim)
				
				if gdat.round_up_or_down=='up':
					gdat.width = find_nearest_upper_mod(larger_dim, gdat.nregion)
				else:
					gdat.width = find_lowest_mod(smaller_dim, gdat.nregion)
				
				gdat.height = gdat.width
				image_size = (gdat.width, gdat.height)

				resized_image = np.zeros(shape=(gdat.width, gdat.height))
				resized_error = np.zeros(shape=(gdat.width, gdat.height))
				resized_mask = np.zeros(shape=(gdat.width, gdat.height))

				crop_size_x = np.minimum(gdat.width, image.shape[0])
				crop_size_y = np.minimum(gdat.height, image.shape[1])

				resized_image[:image.shape[0]-gdat.x0, : image.shape[1]-gdat.y0] = image[gdat.x0:crop_size_x, gdat.y0:crop_size_y]
				resized_error[:image.shape[0]-gdat.x0, : image.shape[1]-gdat.y0] = error[gdat.x0:crop_size_x, gdat.y0:crop_size_y]

				resized_template_list = []
				for t, template in enumerate(template_list):
					if template is not None:

						resized_template = np.zeros(shape=(gdat.width, gdat.height))

						# if gdat.bolocam_mask:
						template = template[bounds[0,0]:bounds[0,1], bounds[1,0]:bounds[1,1]]


						resized_template[:image.shape[0]-gdat.x0, : image.shape[1]-gdat.y0] = template[gdat.x0:crop_size_x, gdat.y0:crop_size_y]

						if show_input_maps:
							plt.figure()
							plt.title('resized template -- '+gdat.template_order[t])
							plt.imshow(resized_template, cmap='Greys', origin='lower')
							plt.colorbar()
							plt.show()

						if gdat.template_order[t] == 'dust' or gdat.template_order[t] == 'planck':

							resized_template -= np.mean(resized_template)

							if show_input_maps:
								plt.figure()
								plt.title(gdat.template_order[t]+', '+gdat.tail_name)
								# plt.title('zero-centered template -- '+gdat.template_order[t]+', '+gdat.tail_name)
								plt.imshow(resized_template, cmap='Greys', origin='lower', vmin=np.percentile(resized_template, 5), vmax=np.percentile(resized_template, 95))
								plt.colorbar()
								# plt.savefig('../zc_dust_temps/zc_dust_'+gdat.tail_name+'_band'+str(i)+'.png', bbox_inches='tight')
								plt.show()


							if gdat.inject_dust and template_name=='planck':
								print('we are putting in dust now')

								resized_image += resized_template

								if show_input_maps:
									plt.figure()
									plt.subplot(1,2,1)
									plt.title('injected dust')
									plt.imshow(resized_template, origin='lower')
									plt.colorbar()
									plt.subplot(1,2,2)
									plt.title('image + dust')
									plt.imshow(resized_image, origin='lower')
									plt.colorbar()
									plt.tight_layout()
									plt.show()

						resized_template_list.append(resized_template.astype(np.float32))
					else:
						resized_template_list.append(None)


				variance = resized_error**2

				variance[variance==0.]=np.inf
				weight = 1. / variance

				print('GDAT.MEAN OFFSET[i] is ', gdat.mean_offsets[i])

				self.weights.append(weight.astype(np.float32))
				self.errors.append(resized_error.astype(np.float32))
				self.data_array.append(resized_image.astype(np.float32)-gdat.mean_offsets[i]) # constant offset, will need to change
				self.template_array.append(resized_template_list)


			elif gdat.width > 0:
				image = image[gdat.x0:gdat.x0+gdat.width,gdat.y0:gdat.y0+gdat.height]
				error = error[gdat.x0:gdat.x0+gdat.width,gdat.y0:gdat.y0+gdat.height]
				cropped_template_list = []
				for template in template_list:
					if template is not None:
						cropped_template = template[gdat.x0:gdat.x0+gdat.width, gdat.y0:gdat.y0+gdat.height]
						cropped_template /= np.max(cropped_template)

						print('cropped_template here has sum ', np.max(cropped_template))
						cropped_template_list.append(cropped_template.astype(np.float32))
					else:
						cropped_template_list.append(None)
				
				image_size = (gdat.width, gdat.height)
				variance = error**2
				variance[variance==0.]=np.inf
				weight = 1. / variance

				
				self.weights.append(weight.astype(np.float32))
				self.errors.append(error.astype(np.float32))
				self.data_array.append(image.astype(np.float32)-gdat.mean_offsets[i]) # constant offset, will need to change
				self.template_array.append(cropped_template_list)

			else:
				image_size = (image.shape[0], image.shape[1])
				variance = error**2
				variance[variance==0.]=np.inf
				weight = 1. / variance

				self.weights.append(weight.astype(np.float32))
				self.errors.append(error.astype(np.float32))
				self.data_array.append(image.astype(np.float32)-gdat.mean_offsets[i]) 
				self.template_array.append(template_list)



			if i==0:
				gdat.imsz0 = image_size

			
			if show_input_maps:
				plt.figure()
				plt.title('data, '+gdat.tail_name)
				plt.imshow(self.data_array[i], vmin=np.percentile(self.data_array[i], 5), vmax=np.percentile(self.data_array[i], 95), cmap='Greys', origin=[0,0])
				plt.colorbar()
				if i==2:
					print('saving this one boyyyy')
					plt.savefig('../data_sims/'+gdat.tail_name+'_data_500micron.png', bbox_inches='tight')
				plt.show()

				plt.figure()
				plt.title('errors')
				plt.imshow(self.errors[i], vmin=np.percentile(self.errors[i], 5), vmax=np.percentile(self.errors[i], 95), cmap='Greys', origin=[0,0])
				plt.colorbar()
				plt.show()

			gdat.imszs.append(image_size)
			gdat.regsizes.append(image_size[0]/gdat.nregion)


			gdat.frac = np.count_nonzero(weight)/float(gdat.width*gdat.height)
			# psf, cf, nc, nbin = get_gaussian_psf_template(pixel_fwhm=gdat.psf_pixel_fwhm, normalization=gdat.normalization)
			psf, cf, nc, nbin = get_gaussian_psf_template_3_5_20(pixel_fwhm=gdat.psf_pixel_fwhm)

			if gdat.verbtype > 1:
				print('image maximum is ', np.max(self.data_array[0]))
				print('gdat.frac is ', gdat.frac)
				print('sum of PSF is ', np.sum(psf))

			self.psfs.append(psf)
			self.cfs.append(cf)
			self.ncs.append(nc)
			self.nbins.append(nbin)
			# self.biases.append(gdat.bias)
			self.fracs.append(gdat.frac)


		gdat.regions_factor = 1./float(gdat.nregion**2)

		assert gdat.imsz0[0] % gdat.regsizes[0] == 0 
		assert gdat.imsz0[1] % gdat.regsizes[0] == 0 

		pixel_variance = np.median(self.errors[0]**2)
		print('pixel_variance:', pixel_variance)
		print('self.dat.fracs is ', self.fracs)
		gdat.N_eff = 4*np.pi*(gdat.psf_pixel_fwhm/2.355)**2 # 2 instead of 4 for spire beam size
		gdat.err_f = np.sqrt(gdat.N_eff * pixel_variance)/10



class spire_data():
    
    image = None
    wcs = None
    
    def __init__(self, filename, base_path='/Users/richardfeder/Documents/multiband_pcat/Data/spire/'):
        self.base_path = base_path
        self.file_path = self.base_path+filename
        
    def get_wcs_header(self):
        
        self.wcs = WCS(self.spire_dat[1].header)
        
    
    def load_in_maps(self, zero_nans=True):
        self.spire_dat = fits.open(self.file_path)
        self.image = self.spire_dat[1].data
        self.error = self.spire_dat[2].data
        self.exposure = self.spire_dat[3].data
        self.mask = self.spire_dat[4].data
        self.imsz = self.image.shape
        print('self imsz is ', self.imsz)
        
        if zero_nans:
            self.image[np.isnan(self.image)] = 0.0
            self.error[np.isnan(self.error)] = 0.0
            self.exposure[np.isnan(self.exposure)] = 0.0
            self.mask[np.isnan(self.mask)] = 0.0
        
    
        
    def show_hist(self, median_plot=True):
        plt.figure(figsize=(10, 5))
        plt.subplot(1,2,1)
        plt.hist(self.image.ravel(), bins=100)
        if median_plot:
            plt.axvline(np.median(self.image), label='median='+str(np.round(np.median(self.image), 3)), linestyle='dashed')
        plt.subplot(1,2,2)
        plt.hist(self.error, bins=100)
        if median_plot:
            plt.axvline(np.median(self.error), label='median='+str(np.round(np.median(self.error), 3)), linestyle='dashed')
        plt.show()
        
        
    def show_maps(self, noise_max = 0.002, err_vrange=[0., 5e-3], im_vrange_percentiles=[5, 95], wcs_project=True):
        
        if wcs_project:
            if self.wcs is None:
                self.get_wcs_header()
            project = self.wcs
        else:
            project = None
            
        if self.image is None:
            self.load_in_maps()
            
        err = self.error.copy()
        
        mask = (err < noise_max)*(err != 0.)
        im = self.image.copy()
        im[~mask] = 0
        
        mindim, maxdim = np.min(np.nonzero(mask)[1]), np.max(np.nonzero(mask)[1])
        print('min/max dim:', mindim, maxdim)
        f = plt.figure(figsize=(15,15))
        
        plt.subplot(2,2,1, projection=project)
        plt.title('Raw Image')
        plt.imshow(im[mindim:maxdim,mindim:maxdim], cmap='Greys', origin=(0,0), vmin=np.percentile(im[mindim:maxdim,mindim:maxdim], im_vrange_percentiles[0]), vmax=np.percentile(im[mindim:maxdim,mindim:maxdim], im_vrange_percentiles[1]))
        plt.colorbar()
        
        plt.subplot(2,2,2, projection=project)
        plt.title('Error')
        err[~mask] = 0
        plt.imshow(err[mindim:maxdim, mindim:maxdim], origin=(0,0),cmap='Greys', vmin=err_vrange[0], vmax=err_vrange[1])
        plt.colorbar()
        
        plt.subplot(2,2,3, projection=project)
        plt.title('Exposure')
        plt.imshow(self.exposure[mindim:maxdim,mindim:maxdim],origin=(0,0), cmap='Greys')
        plt.colorbar()
        
        plt.subplot(2,2,4, projection=project)
        plt.title('Mask')
        plt.imshow(self.mask,origin=(0,0), cmap='Greys')
        plt.colorbar()
        plt.savefig('maps.pdf', bbox_inches='tight')
        plt.show()
        
        return f


