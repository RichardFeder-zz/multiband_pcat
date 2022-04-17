from fast_astrom import *
import numpy as np
from astropy.convolution import Gaussian2DKernel
from image_eval import psf_poly_fit
import pickle
import matplotlib
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy import units as u
from astropy.io import fits


class objectview(object):
	def __init__(self, d):
		self.__dict__ = d

def verbprint(verbose, text, file=None, verbthresh=None):
	if verbthresh is not None:
		if verbose > verbthresh:
			print(text)
	else:
		if verbose:
			print(text)

def get_gaussian_psf_template_3_5_20(pixel_fwhm = 3., nbin=5):
	''' 
	Computes Gaussian PSF kernel for fast model evaluation with lion

	Parameters
	----------
	pixel_fwhm : float
		Full width at half-maximum (FWHM) of PSF in units of pixels.
		Default is 3 (for SPIRE analysis).

	nbin : int
		Upsampling factor for sub-pixel interpolation method in lion
		Default is 5.

	Returns
	-------

	psfnew : '~numpy.ndarray'
		Sum normalized PSF template up sampled by factor nbin in both x and y

	cf : '~numpy.ndarray'
		Coefficients of polynomial fit to upsampled PSF. These are used by lion

	nbin : int
		Upsampling factor, same as before.. 

	'''
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

def make_pcat_fits_file_simp(images, card_names, new_wcs=None, header=None, janscalefac=None):
    ''' Simple function for making FITS file from set of images and desired HDU card names. 

	Parameters
	----------

	images : 
	card_names :
	new_wcs (optional) :
		Default is 'None'.
	header (optional) : 
		Default is 'None'.
	janscalefac (optional) :
		Default is 'None'.

	Returns
	-------

	hdulist : 'list' of 'astropy.fits.HDU' objects. FITS file just absolutely ready to be saved
    '''
    hdu = fits.PrimaryHDU(None)
    
    if header is not None:
        hdu.header = header
    temphdu = None

    cards = [hdu]
    for e, card_name in enumerate(card_names):
        card_hdu = fits.ImageHDU(images[e], name=card_name)
        
        if janscalefac is not None:
        	card_hdu.header['JANSCALE'] = janscalefac

        cards.append(card_hdu)

    if new_wcs is not None:
        
        for card_hdu in cards:
            card_hdu.header.update(new_wcs.to_header())

    hdulist = fits.HDUList(cards)
    
    return hdulist


def make_pcat_fits_file_janscale(sig_image, unc_image, image_name='IMAGE', unc_name='ERROR', extra_images=None, extra_card_names=None, template_image=None, template_name='SZE', new_wcs=None, header=None, x0=None, y0=None, janscalefac=None):
    
    hdu = fits.PrimaryHDU(None)
    
    if header is not None:
        hdu.header = header
    temphdu = None
    sighdu = fits.ImageHDU(sig_image, name=image_name)
    unchdu = fits.ImageHDU(unc_image, name=unc_name)
    if template_image is not None:
        temphdu = fits.ImageHDU(template_image, name=template_name)
    if extra_card_names is not None:
        extra_cards = []
        for e, extra_card_name in enumerate(extra_card_names):
            card_hdu = fits.ImageHDU(extra_images[e], name=extra_card_name)
            extra_cards.append(card_hdu)
            
    if new_wcs is not None:
        sighdu.header.update(new_wcs.to_header())
        unchdu.header.update(new_wcs.to_header())
        if template_image is not None:
            temphdu.header.update(new_wcs.to_header())
        if extra_card_names is not None:
            for card_hdu in extra_cards:
                card_hdu.header.update(new_wcs.to_header())
                
        
    if x0 is not None and y0 is not None:
        
        sighdu.header['x0'] = x0
        sighdu.header['y0'] = y0
        unchdu.header['x0'] = x0
        unchdu.header['y0'] = y0
        
    if janscalefac is not None:
        sighdu.header['JANSCALE'] = janscalefac
        unchdu.header['JANSCALE'] = janscalefac

    hdul = [hdu, sighdu, unchdu]
    

    if temphdu is not None:
        hdul.append(temphdu)
    
    if extra_card_names is not None:
        for card_hdu in extra_cards:
            hdul.append(card_hdu)
            
    hdulist = fits.HDUList(hdul)
    
    return hdulist

def multiband_cut_up_image(psw_path, psw_unc_path, psw_xbound, psw_ybound=None, pmw_path=None, pmw_unc_path=None, plw_path=None, plw_unc_path=None,\
                            imkey='IMAGE', unckey='ERROR'):
    
    # assumes we want PSW, this was for looking at LMC/HELMS data, might update

    if psw_ybound is None:
        psw_ybound = psw_xbound

    big_psw = fits.open(psw_path)[imkey]
    unc_psw = fits.open(psw_unc_path)[unckey]
    
    wcs_psw = WCS(big_psw.header)
    cut_psw = big_psw.data[psw_xbound[0]:psw_xbound[1], psw_ybound[0]:psw_ybound[1]]/big_psw.header['JANSCALE']
    cut_unc_psw = unc_psw.data[psw_xbound[0]:psw_xbound[1], psw_ybound[0]:psw_ybound[1]]/big_psw.header['JANSCALE']

    hdul_psw = make_pcat_fits_file(cut_psw, cut_unc_psw, header=big_psw.header, x0=psw_xbound[0], y0=psw_ybound[0], janscalefac=big_psw.header['JANSCALE'])
    hdul_psw.writeto('Data/spire/LMC_HERITAGE/cutouts/test_lmc_PSW_100_2.fits', clobber=True)
    
    show_im(cut_psw, title='PSW')
    print('cut_psw has shape ', cut_psw.shape)

    if pmw_path is not None:
        big_pmw = fits.open(pmw_path)[imkey]
        unc_pmw = fits.open(pmw_unc_path)[unckey]
        
        wcs_pmw = WCS(big_pmw.header)

    if plw_path is not None:
        big_plw = fits.open(plw_path)[imkey]
        unc_plw = fits.open(plw_unc_path)[unckey]

        wcs_plw = WCS(big_plw.header)
    
    ra, dec = wcs_psw.all_pix2world(psw_xbound[0], psw_ybound[0], 0)
    rahi, dechi = wcs_psw.all_pix2world(psw_xbound[1], psw_ybound[1], 0)

    if pmw_path is not None:
        pmw_lowx, pmw_lowy = wcs_pmw.all_world2pix(ra, dec, 0)
        pmw_hix, pmw_hiy = wcs_pmw.all_world2pix(rahi, dechi, 0)
        pmw_xbound = [int(np.floor(pmw_lowx)), int(np.floor(pmw_hix))]
        pmw_ybound = [int(np.floor(pmw_lowy)), int(np.floor(pmw_hiy))]
        
        cut_pmw = big_pmw.data[pmw_xbound[0]:pmw_xbound[1], pmw_ybound[0]:pmw_ybound[1]]/big_pmw.header['JANSCALE']
        print('cut_pmw has shape ', cut_pmw.shape)
        cut_unc_pmw = unc_pmw.data[pmw_xbound[0]:pmw_xbound[1], pmw_ybound[0]:pmw_ybound[1]]/big_pmw.header['JANSCALE']

        hdul_pmw = make_pcat_fits_file(cut_pmw, cut_unc_pmw, header=big_pmw.header, x0=pmw_xbound[0], y0=pmw_ybound[0], janscalefac=big_pmw.header['JANSCALE'])
        hdul_pmw.writeto('Data/spire/LMC_HERITAGE/cutouts/test_lmc_PMW_100_2.fits', clobber=True)

        show_im(cut_pmw, title='PMW')

    if plw_path is not None:
        plw_lowx, plw_lowy = wcs_plw.all_world2pix(ra, dec, 0)
        plw_hix, plw_hiy = wcs_plw.all_world2pix(rahi, dechi, 0)
        plw_xbound = [int(np.floor(plw_lowx)), int(np.floor(plw_hix))]
        plw_ybound = [int(np.floor(plw_lowy)), int(np.floor(plw_hiy))]

        cut_plw = big_plw.data[plw_xbound[0]:plw_xbound[1], plw_ybound[0]:plw_ybound[1]]/big_plw.header['JANSCALE']
        print('cut_plw has shape ', cut_plw.shape)
        cut_unc_plw = unc_plw.data[plw_xbound[0]:plw_xbound[1], plw_ybound[0]:plw_ybound[1]]/big_plw.header['JANSCALE']


        hdul_plw = make_pcat_fits_file(cut_plw, cut_unc_plw, header=big_plw.header, x0=plw_xbound[0], y0=plw_ybound[0], janscalefac=big_plw.header['JANSCALE'])
        hdul_plw.writeto('Data/spire/LMC_HERITAGE/cutouts/test_lmc_PLW_100_2.fits', clobber=True)

        show_im(cut_plw, title='PLW')
    
    return None

def multiband_cutout_obs(filenames, n_cut_arcsec, ra, dec, tail_names, bandstrs=['PSW', 'PMW', 'PLW'], sigkey='SIGNAL', \
                        diff_comp_path=None, show=False, savedir='Data/spire/', save=True, \
                        im_headers = ['SIGNAL', 'ERROR', 'SZE']):
    sznormfacs = [0.00026, 0.003, 0.018]
    for b, bandstr in enumerate(bandstrs):
        
        obs = fits.open(filenames[b])
        
        wcs_obs = WCS(obs[sigkey].header)
        
        if diff_comp_path is not None:
            diff_comp = np.load(diff_comp_path)[bandstr[1]]
        
        if b==0 and show:
            plt.figure()
            plt.imshow(obs[sigkey].data, vmax=0.05)
            plt.colorbar()
            plt.show()
            
        xpix, ypix = wcs_obs.all_world2pix(ra, dec, 0)
        print(xpix, ypix)
        
        cutouts = []
        
        hduheaders = []
        
        for h, headkey in enumerate(im_headers):
            
            print('h = ', h, 'headkey = ', headkey)
            if h==0:
                cutout_obj = Cutout2D(obs[headkey].data, (xpix, ypix), (n_cut_arcsec*u.arcsec, n_cut_arcsec*u.arcsec), wcs=wcs_obs, mode='partial', fill_value=np.nan, copy=True)
                cutout = cutout_obj.data
            else:
                cutout = Cutout2D(obs[headkey].data, (xpix, ypix), (n_cut_arcsec*u.arcsec, n_cut_arcsec*u.arcsec), wcs=wcs_obs, mode='partial', fill_value=np.nan, copy=True).data
            
            
            if headkey=='SZETEMP':
                print('weere here')
                cutout[np.isnan(cutout)] = 0.
                cutout /= sznormfacs[b]
                hduheaders.append('SZE')
            else:
                hduheaders.append(headkey)
            
            cutouts.append(cutout)
            
            if show:

                plt.figure()
                plt.title(headkey, fontsize=16)
                plt.imshow(cutout, origin='lower', vmin=np.nanpercentile(cutout, 5), vmax=np.nanpercentile(cutout, 95))
                plt.colorbar()
                plt.show()
                
        if diff_comp_path is not None:
            dcomp_cutout = diff_comp[:im_cutout.shape[0],:im_cutout.shape[1]]
            
            cutouts.append(dcomp_cutout)
            im_headers.append('DUST')
            
            if show:

                plt.figure()
                plt.title('diffuse comp', fontsize=16)
                plt.imshow(dcomp_cutout, origin='lower', vmin=np.nanpercentile(dcomp_cutout, 5), vmax=np.nanpercentile(dcomp_cutout, 95))
                plt.colorbar()
                plt.show()
                
        hdul = make_pcat_fits_file_simp(images=cutouts, card_names=hduheaders, new_wcs=cutout_obj.wcs)

                
        if save:
            print('writing to ', savedir+'/'+tail_names[b]+'.fits')
            hdul.writeto(savedir+'/'+tail_names[b]+'.fits', clobber=True)
        



def load_in_map(gdat, band=0, astrom=None, show_input_maps=False, image_extnames=['SIGNAL'], err_hdu_idx=1):

	''' 
	This function does some of the initial data parsing needed in constructing the pcat_data object.

	Parameters
	----------

	gdat : global object
		This is a super object which is used throughout different parts of PCAT. 
		Contains data configuration information

	band : int, optional
		Index of bandpass. Convention is 0->250um, 1->350um and 2->500um.
		Default is 0.

	astrom : object of type 'fast_astrom', optional
		astrometry object that can be loaded along with other data products
		Default is 'None'.

	show_input_maps : bool, optional

		Default is 'False'.

	image_extnames : list of strings, optional
		If list of extensions is specified, observed data will be combination of extension images.
		For example, one can test mock unlensed data with noise ['UNLENSED', 'NOISE'] 
		or mock lensed data with noise ['LENSED', 'NOISE']. Default is ['SIGNAL'].

	Returns
	-------

	image : '~numpy.ndarray'
		Numpy array containing observed data for PCAT's Bayesian analysis. Can be composed of
		several mock components, or it can be real observed data.

	error : '~numpy.ndarray'
		Numpy array containing noise model estimates over field to be analyzed. Masked pixels will have
		NaN values, which get modified to zeros in this script.

	mask : '~numpy.ndarray'
		Numpy array which contains mask used on data. 
		Masked regions are cropped out/down-weighted when preparing data for PCAT.

	file_path : str
		File path for map that is loaded in.

	'''
	if gdat.file_path is None:
		file_path = gdat.data_path+gdat.dataname+'/'+gdat.tail_name+'.fits'
	else:
		file_path = gdat.file_path

	file_path = file_path.replace('PSW', 'P'+str(gdat.band_dict[band])+'W')
	verbprint(gdat.verbtype, 'Band is '+str(gdat.band_dict[band]), verbthresh=1)
	verbprint(gdat.verbtype, 'file path is '+str(file_path), verbthresh=1)

	if astrom is not None:
		print('ATTENTION loading from ', gdat.band_dict[band])
		astrom.load_wcs_header_and_dim(file_path, round_up_or_down=gdat.round_up_or_down)

	#by loading in the image this way, we can compose maps from several components, e.g. noiseless CIB + noise realization
	
	if gdat.im_fpath is None:
		spire_dat = fits.open(file_path)
	else:
		spire_dat = fits.open(gdat.im_fpath)

	for e, extname in enumerate(image_extnames):
		if e==0:
			image = np.nan_to_num(spire_dat[extname].data)
		else:
			image += np.nan_to_num(spire_dat[extname].data)

		if gdat.show_input_maps:
			plt.figure()
			plt.title(extname)
			plt.imshow(image, origin='lower', vmin=np.nanpercentile(image, 5), vmax=np.nanpercentile(image, 95))
			plt.colorbar()
			plt.show()

	# if gdat.im_fpath is None:
	# 	spire_dat = fits.open(file_path)

	# 	for e, extname in enumerate(image_extnames):
	# 		if e==0:
	# 			image = np.nan_to_num(spire_dat[extname].data)
	# 		else:
	# 			image += np.nan_to_num(spire_dat[extname].data)
	# 		if gdat.show_input_maps:
	# 			plt.figure()
	# 			plt.title(extname)
	# 			plt.imshow(image, origin='lower')
	# 			plt.colorbar()
	# 			plt.show()
	# else:
	# 	spire_dat = fits.open(gdat.im_fpath)
	# 	# hduim = spire_dat[0]
	# 	# image = np.nan_to_num(hduim.data/hduim.header['JANSCALE'])

	# 	for e, extname in enumerate(image_extnames):
	# 		if e==0:
	# 			image = np.nan_to_num(spire_dat[extname].data)
	# 		else:
	# 			image += np.nan_to_num(spire_dat[extname].data)

	x0 = None
	y0 = None

	if not gdat.use_errmap:
		error = np.zeros_like(image)

	elif gdat.err_fpath is None:

		error = np.nan_to_num(spire_dat[gdat.error_extname].data)
	else:
		hdu = fits.open(gdat.err_fpath)[err_hdu_idx]
		# error = np.nan_to_num(hdu.data/hdu.header['JANSCALE'])
		error = np.nan_to_num(hdu.data)

	# main functionality is following eight lines
	if gdat.use_mask:

		if gdat.bolocam_mask:
			mask = fits.open(gdat.data_path+'bolocam_mask_P'+str(gdat.band_dict[band])+'W.fits')[0].data
		
		elif gdat.mask_file is not None:
			mask_fpath = gdat.mask_file.replace('PSW', 'P'+str(gdat.band_dict[band])+'W')

			mask = fits.open(mask_fpath)[0].data
		else:
			mask = spire_dat['MASK'].data
	else:
		print('Not using mask..')
		mask = np.ones_like(image)


	# this is for Ben's masks which are interpolated oddly
	# mask[mask > 0.4] = 1
	# mask[mask <= 0.4] = 0


	# this is for gen_2_sims, which don't have same FITS structure as new sims
	# image = np.nan_to_num(spire_dat[1].data)
	# error = np.nan_to_num(spire_dat[2].data)

	if gdat.add_noise:

		if gdat.scalar_noise_sigma is not None:
			if type(gdat.scalar_noise_sigma)==float:
				noise_sig = gdat.scalar_noise_sigma
			else:
				noise_sig = gdat.scalar_noise_sigma[band]

			if gdat.noise_fpath is not None:
				noise_realization = np.load(gdat.noise_fpath)['noise_realization']
			else:
				noise_realization = np.random.normal(0, noise_sig, (image.shape[0], image.shape[1]))
			image += noise_realization

			if not gdat.use_errmap:
				error = noise_sig*np.ones((image.shape[0], image.shape[1]))
			else:
				old_error = error.copy()
				old_variance = old_error**2
				old_variance[error != 0] += noise_sig**2
				error = np.sqrt(old_variance)

			if gdat.show_input_maps:
				plt.figure(figsize=(15, 5))
				plt.subplot(1,3,1)
				plt.title('noise realization')

				plt.imshow(noise_realization)
				plt.colorbar()
				plt.subplot(1,3,2)
				plt.title('image + gaussian noise')

				showim = image.copy()
				# showim[error != 0] += noise_realization[error != 0]
				plt.imshow(showim)
				plt.colorbar()
				plt.subplot(1,3,3)
				plt.title('error map')
				plt.imshow(error, vmin=np.percentile(error, 5), vmax=np.percentile(error, 95))
				plt.colorbar()
				plt.show()


		else:
			print('using error map to generate noise realization..')
			noise_realization = np.zeros_like(error)
			for rowidx in range(error.shape[0]):
				for colidx in range(error.shape[1]):
					if not np.isnan(error[rowidx,colidx]):
						noise_realization[rowidx,colidx] = np.random.normal(0, error[rowidx,colidx])

			if show_input_maps:
				plt.figure()
				plt.imshow(noise_realization, vmin=np.nanpercentile(noise_realization, 5), vmax=np.nanpercentile(noise_realization, 95))
				plt.title('noise raelization')
				plt.colorbar()
				plt.show()

			image += noise_realization
	# mask = fits.open(gdat.base_path+'/data/spire/GOODSN/GOODSN_P'+str(gdat.band_dict[band])+'W_mask.fits')[0].data
	# mask = fits.open(gdat.base_path+'/data/spire/gps_0/gps_0_P'+str(gdat.band_dict[band])+'W_mask.fits')[0].data
	# mask = fits.open(gdat.base_path+'/data/spire/rxj1347_831/rxj1347_P'+str(gdat.band_dict[band])+'W_nr_1_ext.fits')['MASK'].data
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

	return image, error, mask, file_path, x0, y0


def load_param_dict(timestr=None, result_path='/Users/luminatech/Documents/multiband_pcat/spire_results/', encoding=None):
	
	''' 
	Loads dictionary of configuration parameters from prior run of PCAT.

	Parameters
	----------

	timestr : string
		time string associated with desired PCAT run.

	result_path : string, optional
		file location of PCAT run results.
		Default is '/Users/luminatech/Documents/multiband_pcat/spire_results/'.

	Returns
	-------

	opt : object containing parameter dictionary

	filepath : string
		file path of parameter file

	result_path : string
		file location of PCAT run results. not sure why its here

	'''
	filepath = result_path
	if timestr is not None:
		filepath += timestr
	# Python version thing here
	filen = open(filepath+'/params.txt','rb')
	if encoding is not None:
		pdict = pickle.load(filen, encoding=encoding)
	else:
		pdict = pickle.load(filen)
	opt = objectview(pdict)

	return opt, filepath, result_path


def get_rect_mask_bounds(mask):

	''' 
	This function assumes the mask is rectangular in shape, with ones in the desired region and zero otherwise.
	
	Parameters
	----------

	mask : 'np.array' of type 'int'. Mask

	Returns
	-------

	bounds : 'np.array' of type 'float' and shape (2, 2). x and y bounds for mask.

	'''

	idxs = np.argwhere(mask == 1.0)
	bounds = np.array([[np.min(idxs[:,0]), np.max(idxs[:,0])], [np.min(idxs[:,1]), np.max(idxs[:,1])]])

	return bounds



class pcat_data():

	''' 
	This class sets up the data structures for data/data-related information. 
	
	- load_in_data() loads in data, generates the PSF template and computes weights from the noise model
	'''

	template_bands = dict({'sze':['M', 'L'], 'szetemp':['S', 'M', 'L'], 'lensing':['S', 'M', 'L'], 'dust':['S', 'M', 'L'], 'planck':['S', 'M', 'L']}) # should just integrate with the same thing in Lion main

	def __init__(self, auto_resize=False, nregion=1):
		self.ncs, self.nbins, self.psfs, self.cfs, self.biases, self.data_array, self.weights, self.masks, self.errors, \
			self.widths, self.heights, self.fracs, self.template_array, self.injected_diffuse_comp = [[] for x in range(14)]
		self.fast_astrom = wcs_astrometry(auto_resize, nregion=nregion)

	def load_in_data(self, gdat, map_object=None, tail_name=None, show_input_maps=False, \
		temp_mock_amps_dict=None, flux_density_conversion_dict=None, sed_cirrus=None):

		'''
		This function does the heavy lifting for parsing input data products and setting up variables in pcat_data class. At some point, template section should be cleaned up.
		This is an in place operation. 

		Parameters
		----------

		gdat : Global data object. 

		map_object (optional) : 
				Default is 'None'.
		tail_name (optional) : 
				Default is 'None'.
		show_input_maps (optional) : 'boolean'.
				Default is 'False'.
		temp_mock_amps_dict (optional) :  'dictionary' of floats.
				Default is 'None'.
		flux_density_conversion_dict (optional) : 'dictionary' of floats.
				Default is 'None'. 
		sed_cirrus (optional) : 

		'''

		if flux_density_conversion_dict is None:
			flux_density_conversion_dict = dict({'S': 86.29e-4, 'M':16.65e-3, 'L':34.52e-3})

		if temp_mock_amps_dict is None:
			# temp_mock_amps_dict = dict({'S':0.4, 'M': 0.2, 'L': 0.8}) # MJy/sr, this was to test how adding a signal at 250 um (like dust) would affect the fit if not modeled

			temp_mock_amps_dict = dict({'S':0.03, 'M': 0.2, 'L': 0.8}) # MJy/sr

		if sed_cirrus is None:
			sed_cirrus = dict({100:1.7, 250:3.5, 350:1.6, 500:0.85}) # MJy/sr

		relative_dust_sed_dict = dict({'S':sed_cirrus[250]/sed_cirrus[100], 'M':sed_cirrus[350]/sed_cirrus[100], 'L':sed_cirrus[500]/sed_cirrus[100]}) # relative to 100 micron

		gdat.imszs, gdat.regsizes, gdat.margins, gdat.bounds, gdat.x_max_pivot_list, gdat.y_max_pivot_list, gdat.imszs_orig = [[] for x in range(7)]

		for i, band in enumerate(gdat.bands):

			if gdat.mock_name is None:

				image, error, mask, file_name, x0, y0 = load_in_map(gdat, band, astrom=self.fast_astrom, show_input_maps=show_input_maps, image_extnames=gdat.image_extnames)

				gdat.imszs_orig.append([image.shape[0], image.shape[1]])
				print(gdat.imszs_orig)
				# bounds = get_rect_mask_bounds(mask) if gdat.bolocam_mask else None
				bounds = get_rect_mask_bounds(mask)

				print('Bounds are ', bounds)

				if bounds is not None:

					big_dim = np.maximum(find_nearest_mod(bounds[0,1]-bounds[0,0]+1, gdat.nregion, mode=gdat.round_up_or_down),\
											 find_nearest_mod(bounds[1,1]-bounds[1,0]+1, gdat.nregion, mode=gdat.round_up_or_down))

					self.fast_astrom.dims[i] = (big_dim, big_dim)

				gdat.bounds.append(bounds)

				template_list = [] 

				print('gdat.template_name is ', gdat.template_filename)
				if gdat.n_templates > 0:

					for t, template_name in enumerate(gdat.template_names):
						print('template name is ', template_name)
						if gdat.band_dict[band] in self.template_bands[template_name]:

							verbprint(gdat.verbtype, 'Were in business, '+str(gdat.band_dict[band])+', '+str(self.template_bands[template_name])+', '+str(gdat.lam_dict[gdat.band_dict[band]]), verbthresh=1)

							# if template_name=='sze':
							# 	template = fits.open(gdat.base_path+'/data/spire/rxj1347_sz_templates/rxj1347_P'+str(gdat.band_dict[band])+'W_nr_sze.fits')[0].data

							if gdat.template_filename is not None and template_name=='sze':
								print('we want to load in a template!!')
								
								temp_name = gdat.template_filename[template_name]

								# template_file_name = temp_name.replace()
								print('template file name ehere is ', temp_name)
								for loopband in ['S', 'M', 'L']:
								    if 'P'+loopband+'W' in temp_name:
								    	print('P'+loopband+'W'+' is in '+temp_name)
								    	print('gdat.band_dict[band] is', gdat.band_dict[band])
								        template_file_name = temp_name.replace('P'+loopband+'W',  'P'+str(gdat.band_dict[band])+'W')

								# template_file_name = temp_name.replace('PSW', 'P'+str(gdat.band_dict[band])+'W')

								print('template file name is ', template_file_name)

								template = fits.open(template_file_name)[1].data 

								# for new templates, divide by conversion factor
								# template /= flux_density_conversion_dict[gdat.band_dict[band]]
								# template /= temp_mock_amps_dict[gdat.band_dict[band]]

								if show_input_maps:
									plt.figure()
									plt.title(template_file_name)
									plt.imshow(template, origin='lower')
									plt.colorbar()
									plt.show()

							else:
								template = fits.open(file_name)[template_name].data

								if show_input_maps:
									plt.figure()
									plt.title('loaded directly from fits extension')
									plt.imshow(template, origin='lower')
									plt.colorbar()
									plt.show()

							if show_input_maps:
								plt.figure()
								plt.title(template_name)
								plt.imshow(template, origin='lower')
								plt.colorbar()
								plt.show()


							if gdat.inject_sz_frac is not None and template_name=='sze':
								print('injecting SZ frac of ', gdat.inject_sz_frac)
								
								
								template_inject = gdat.inject_sz_frac*template*temp_mock_amps_dict[gdat.band_dict[band]]*flux_density_conversion_dict[gdat.band_dict[band]]

								image += template_inject

								if show_input_maps:
									plt.figure(figsize=(8, 4))
									plt.subplot(1,2,1)
									plt.title('injected amp is '+str(np.round(gdat.inject_sz_frac*temp_mock_amps_dict[gdat.band_dict[band]]*flux_density_conversion_dict[gdat.band_dict[band]], 4)))
									plt.imshow(template_inject, origin='lower', cmap='Greys')
									plt.colorbar()
									plt.subplot(1,2,2)
									plt.title('image + sz')
									plt.imshow(image, origin='lower', cmap='Greys')
									plt.colorbar()
									plt.tight_layout()
									plt.show()


						else:
							print('no band in self.template_bands')
							template = None

						template_list.append(template)


				if i > 0:
					verbprint(gdat.verbtype, 'We have more than one band! Handling band '+str(band))

					self.fast_astrom.fit_astrom_arrays(0, i, bounds0=gdat.bounds[0], bounds1=gdat.bounds[i], correct_misaligned_shift=gdat.correct_misaligned_shift)

					x_max_pivot, y_max_pivot = self.fast_astrom.transform_q(np.array([gdat.imsz0[0]]), np.array([gdat.imsz0[1]]), i-1)
					# x_max_pivot, y_max_pivot = self.fast_astrom.obs_to_obs(0, i, gdat.imsz0[0], gdat.imsz0[1])

					print('xmaxpivot, ymaxpivot for band ', i, ' are ', x_max_pivot, y_max_pivot)
					gdat.x_max_pivot_list.append(x_max_pivot)
					gdat.y_max_pivot_list.append(y_max_pivot)

				else:
					gdat.x_max_pivot_list.append(big_dim)
					gdat.y_max_pivot_list.append(big_dim)

				if gdat.noise_thresholds is not None:
					error[error > gdat.noise_thresholds[i]] = 0 # this equates to downweighting the pixels


			else:
				image, error, mask = load_in_mock_map(gdat.mock_name, band)
			
			if gdat.auto_resize:
				# error = error[bounds[0,0]:bounds[0,1], bounds[1,0]:bounds[1,1]]
				# image = image[bounds[0,0]:bounds[0,1], bounds[1,0]:bounds[1,1]]
				error = error[bounds[0,0]:bounds[0,1]+1, bounds[1,0]:bounds[1,1]+1] # bounds obtained from mask, error map copied from within those bounds
				image = image[bounds[0,0]:bounds[0,1]+1, bounds[1,0]:bounds[1,1]+1] # same with image
				smaller_dim = np.min(image.shape)
				larger_dim = np.max(image.shape)

				print('smaller dim for band ', band, ' is', smaller_dim, ' larger dim is ', larger_dim)

				gdat.width = find_nearest_mod(larger_dim, gdat.nregion, mode=gdat.round_up_or_down)
				gdat.height = gdat.width
				image_size = (gdat.width, gdat.height)

				print('image size is ', image_size)

				resized_image = np.zeros(shape=(gdat.width, gdat.height))
				resized_error = np.zeros(shape=(gdat.width, gdat.height))
				resized_mask = np.zeros(shape=(gdat.width, gdat.height))

				crop_size_x = np.minimum(gdat.width, image.shape[0])
				crop_size_y = np.minimum(gdat.height, image.shape[1])

				resized_image[:image.shape[0]-gdat.x0, : image.shape[1]-gdat.y0] = image[gdat.x0:crop_size_x, gdat.y0:crop_size_y]
				resized_error[:image.shape[0]-gdat.x0, : image.shape[1]-gdat.y0] = error[gdat.x0:crop_size_x, gdat.y0:crop_size_y]
				resized_mask[:image.shape[0]-gdat.x0, : image.shape[1]-gdat.y0] = mask[gdat.x0:crop_size_x, gdat.y0:crop_size_y]


				if i > 0:
					if int(x_max_pivot) < resized_image.shape[0]:
						print('Setting pixels in band '+str(i)+' not in band 0 FOV to zero..')
						resized_image[int(x_max_pivot):,:] = 0.
						resized_image[:,int(x_max_pivot):] = 0.
						resized_error[int(x_max_pivot):,:] = 0.
						resized_error[:,int(x_max_pivot):] = 0.
						resized_mask[int(x_max_pivot):,:] = 0.
						resized_mask[:,int(x_max_pivot):] = 0.

				resized_template_list = []

				for t, template in enumerate(template_list):

					if template is not None:

						resized_template = np.zeros(shape=(gdat.width, gdat.height))
						# template = template[bounds[0,0]:bounds[0,1], bounds[1,0]:bounds[1,1]]
						template = template[bounds[0,0]:bounds[0,1]+1, bounds[1,0]:bounds[1,1]+1]

						resized_template[:image.shape[0]-gdat.x0, : image.shape[1]-gdat.y0] = template[gdat.x0:crop_size_x, gdat.y0:crop_size_y]

						if show_input_maps:
							plt.figure()
							plt.subplot(1,2,1)
							plt.title('resized template -- '+gdat.template_order[t])
							plt.imshow(resized_template, cmap='Greys', origin='lower')
							plt.colorbar()
							if i > 0:
								plt.axhline(x_max_pivot, color='r', linestyle='solid')
								plt.axvline(y_max_pivot, color='r', linestyle='solid')
							plt.subplot(1,2,2)
							plt.title('image + '+gdat.template_order[t])
							plt.imshow(resized_image, origin='lower', cmap='Greys')
							plt.colorbar()
							if i > 0:
								plt.axhline(x_max_pivot, color='r', linestyle='solid')
								plt.axvline(y_max_pivot, color='r', linestyle='solid')
							plt.tight_layout()
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
								print('Injecting dust template into image..')

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


				if gdat.inject_diffuse_comp and gdat.diffuse_comp_path is not None:

					bands = ['S', 'M', 'L']
					diffuse_comp = np.load(gdat.diffuse_comp_path)[bands[i]]

					if show_input_maps:
						plt.figure()
						plt.title(diffuse_comp.shape)
						plt.imshow(diffuse_comp)
						plt.colorbar()
						plt.show()

					cropped_diffuse_comp = diffuse_comp[:gdat.width, :gdat.height]

					if show_input_maps:
						plt.figure()
						plt.title(cropped_diffuse_comp.shape)
						plt.imshow(cropped_diffuse_comp)
						plt.colorbar()
						plt.show()

					resized_image += cropped_diffuse_comp

					if show_input_maps:
						plt.figure()
						plt.title('resized image with cropped diffuse comp')
						plt.imshow(resized_image)
						plt.colorbar()
						plt.show()

					self.injected_diffuse_comp.append(cropped_diffuse_comp.astype(np.float32))


				variance = resized_error**2
				variance[variance==0.]=np.inf
				weight = 1. / variance


				print('GDAT.MEAN OFFSET[i] is ', gdat.mean_offsets[i])

				self.weights.append(weight.astype(np.float32))
				self.errors.append(resized_error.astype(np.float32))
				resized_image[weight==0] = 0.
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
				self.data_array.append(image.astype(np.float32)-gdat.mean_offsets[i]) # gdat.mean_offsets largely deprecated
				self.template_array.append(cropped_template_list)

			else:
				image_size = (image.shape[0], image.shape[1])
				variance = error**2
				variance[variance==0.]=np.inf
				weight = 1. / variance

				self.weights.append(weight.astype(np.float32))
				self.errors.append(error.astype(np.float32))
				image[weight==0] = 0.
				self.data_array.append(image.astype(np.float32)-gdat.mean_offsets[i]) # gdat.mean_offsets largely deprecated
				self.template_array.append(template_list)

			if i==0:
				gdat.imsz0 = image_size

			if show_input_maps:
				plt.figure()
				plt.title('data, '+gdat.tail_name)
				plt.imshow(self.data_array[i], vmin=np.percentile(self.data_array[i], 5), vmax=np.percentile(self.data_array[i], 95), cmap='Greys', origin=[0,0])
				plt.colorbar()

				if i > 0:
					plt.axhline(x_max_pivot, color='r', linestyle='solid')
					plt.axvline(y_max_pivot, color='r', linestyle='solid')

				plt.xlim(0, self.data_array[i].shape[0])
				plt.ylim(0, self.data_array[i].shape[1])
				# if i==2:
					# print('saving this one boyyyy')
					# plt.savefig('../data_sims/'+gdat.tail_name+'_data_500micron.png', bbox_inches='tight')
				plt.show()

				plt.figure()
				plt.title('errors')
				plt.imshow(self.errors[i], vmin=np.percentile(self.errors[i], 5), vmax=np.percentile(self.errors[i], 95), cmap='Greys', origin=[0,0])
				plt.colorbar()
				if i > 0:
					plt.axhline(x_max_pivot, color='r', linestyle='solid')
					plt.axvline(y_max_pivot, color='r', linestyle='solid')
				plt.xlim(0, self.errors[i].shape[0])
				plt.ylim(0, self.errors[i].shape[1])
				plt.show()

			gdat.imszs.append(image_size)
			gdat.regsizes.append(image_size[0]/gdat.nregion)


			gdat.frac = np.count_nonzero(weight)/float(gdat.width*gdat.height)
			# psf, cf, nc, nbin = get_gaussian_psf_template(pixel_fwhm=gdat.psf_pixel_fwhm, normalization=gdat.normalization)
			# psf, cf, nc, nbin = get_gaussian_psf_template_3_5_20(pixel_fwhm=gdat.psf_pixel_fwhm)
			psf, cf, nc, nbin = get_gaussian_psf_template_3_5_20(pixel_fwhm=gdat.psf_fwhms[i]) # variable psf pixel fwhm

			verbprint(gdat.verbtype, 'Image maximum is '+str(np.max(self.data_array[0]))+', gdat.frac = '+str(gdat.frac)+', sum of PSF is '+str(np.sum(psf)), verbthresh=1)

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
		# gdat.N_eff = 4*np.pi*(gdat.psf_pixel_fwhm/2.355)**2 # 2 instead of 4 for spire beam size
		gdat.N_eff = 4*np.pi*(gdat.psf_fwhms[0]/2.355)**2 # variable psf pixel fwhm, use pivot band fwhm

		gdat.err_f = np.sqrt(gdat.N_eff * pixel_variance)/gdat.err_f_divfac
		# gdat.err_f = np.sqrt(gdat.N_eff * pixel_variance)



