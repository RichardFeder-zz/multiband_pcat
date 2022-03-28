import numpy as np
import matplotlib 
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
import PIL.Image as Image

def generate_subregion_cib_templates(dimx, dimy, nregion, cib_rel_amps = [1.0, 1.41, 1.17], \
                                    bands=None, conv_facs=None, verbose=False):
    
    ''' 

    This function generates a template set designed to model the unresolved CIB. These templates are 2D tophats in map space 

	Parameters
	----------
	
	dimx : 'int', or 'list' of ints. This and dimy specify image dimensions for potentially several maps
	dimy : 'int', or 'list' of ints.
	nregion : 'int'. Number of subregions along each image axis.

	Returns
	-------

	coarse_template_list : 'list' of `np.array' of type 'float', shape (nregion**2, dimx, dimy) for each band.


    Notes
    -----

    This works for 100/72/50 (i.e., SPIRE 10x10 arcmin maps), but with 51 two of the pixels are errant, fine for now but in future need to 
    get this right. 
    
    '''
    if type(dimx)==int or type(dimx)==float:
        dimx = [dimx]
        dimy = [dimy]
        
    # make sure you can divide properly. not sure how this should be for several bands.. 
    assert dimx[0]%nregion==0
    assert dimy[0]%nregion==0
    
    nbands = len(dimx)
    
    if cib_rel_amps is None:
        cib_rel_amps = np.array([1. for x in range(nbands)])
    
    if conv_facs is not None:
        cib_rel_amps *= conv_facs # MJy/sr to mJy/beam

    subwidths = [dimx[n]//nregion for n in range(nbands)]
    
    # I should make these overlapping for non-integer positions, weighted by decimal contribution to given pixel
    subwidths_exact = [float(dimx[n])/nregion for n in range(nbands)]
    ntemp = nregion**2

    if verbose:
        print('subwidths exact:', subwidths_exact)
        print('ntemp is ', ntemp)
        print('subwidths are ', subwidths)
    
    coarse_template_list = []
    
    for n in range(nbands):
        is_divisible = (subwidths[n]==subwidths_exact[n])
        
        if verbose:
            print(subwidths[n], subwidths_exact[n])
            print('is divisible is ', is_divisible)
        
        init_x = 0.
        running_x = 0.
        
        templates = np.zeros((ntemp, dimx[n], dimy[n]))
        
        for i in range(nregion):
            init_y = 0.
            running_y = 0.
            
            running_x += subwidths_exact[n]
            x_remainder = running_x - np.floor(running_x)
            init_x_remainder = init_x - np.floor(init_x)
            
            for j in range(nregion):
                                
                running_y += subwidths_exact[n]
                y_remainder = running_y - np.floor(running_y)
                init_y_remainder = init_y - np.floor(init_y)

                if verbose:
                    print(running_x, running_y, int(np.floor(init_x)), int(np.floor(running_x)), int(np.ceil(init_y)), int(np.floor(running_y)))
                
                templates[i*nregion + j, int(np.ceil(init_x)):int(np.floor(running_x)), int(np.ceil(init_y)):int(np.floor(running_y))] = 1.0
                                
                if not is_divisible:
                
                    if np.ceil(running_x) > np.floor(running_x): # right edge
                        templates[i*nregion + j, int(np.floor(running_x)),  int(np.ceil(init_y)):int(np.floor(running_y))] = x_remainder
                        if np.floor(running_x) < dimx[n] and np.floor(running_y) < dimy[n]: # top right corner
                            templates[i*nregion + j, int(np.floor(running_x)), int(np.floor(running_y))] = (y_remainder+x_remainder)/4.

                    if np.ceil(running_y) > np.floor(running_y): # top edge
                        templates[i*nregion + j, int(np.ceil(init_x)):int(np.floor(running_x)), int(np.floor(running_y))] = y_remainder
                        if init_x > 0 and np.floor(running_y) < dimy[n]: # top left corner
                            templates[i*nregion + j, int(np.floor(init_x)), int(np.floor(running_y))] = (y_remainder+np.ceil(init_x)-init_x)/4.

                    if init_x > np.floor(init_x): # left edge
                        templates[i*nregion + j, int(np.floor(init_x)),  int(np.ceil(init_y)):int(np.floor(running_y))] = np.ceil(init_x)-init_x
                        if init_x > 0 and init_y > 0: # bottom left corner
                            templates[i*nregion + j, int(np.floor(init_x)), int(np.floor(init_y))] = (np.ceil(init_x)-init_x+np.ceil(init_y)-init_y)/4.

                    if init_y > np.floor(init_y): # bottom edge
                        templates[i*nregion + j, int(np.ceil(init_x)):int(np.floor(running_x)), int(np.floor(init_y))] = np.ceil(init_y)-init_y
                        
                        if init_y > 0 and np.floor(running_x) < dimx[n]: # bottom right corner
                            templates[i*nregion + j, int(np.floor(running_x)), int(np.floor(init_y))] = (x_remainder+np.ceil(init_y)-init_y)/4.


                init_y = running_y
                
                
            init_x = running_x

 
        coarse_template_list.append(cib_rel_amps[n]*templates)
        
    return coarse_template_list




def generate_diffuse_realization(N, M, power_law_idx=-2.7):
	'''
	Given image dimensions, generates Gaussian random field diffuse realization, assuming a power law power spectrum.

	Parameters
	----------
	
	N : 'int'
		Width of image in pixels.
	M : 'int'
		Height of image in pixels.
	power_law_idx : 'float', optional
		Power law index for power spectrum of diffuse realization. Default is -2.7.
	
	Returns
	-------
	
	ell_map : '~numpy.ndarray' of shape (N, M)
		2D map of Fourier frequencies used to generate diffuse realization.
	ps : '~numpy.ndarray' of shape (N, M)
		2D power spectrum used to generate diffuse realization.
	diffuse_realiz : '~numpy.ndarray' of shape (N, M)
		Diffuse realization produced as gaussian random field.

	'''
	
	freq_x = fftshift(np.fft.fftfreq(N, d=1.0))
	freq_y = fftshift(np.fft.fftfreq(M, d=1.0))
	
	ell_x,ell_y = np.meshgrid(freq_x,freq_y)
	ell_x = ifftshift(ell_x)
	ell_y = ifftshift(ell_y)
	
	ell_map = np.sqrt(ell_x**2 + ell_y**2)
	
	ps = ell_map**power_law_idx
	ps[0,0] = 0.
	
	diffuse_realiz = ifft2(np.sqrt(ps)*(np.random.normal(0, 1, size=(N, M)) + 1j*np.random.normal(0, 1, size=(N, M))))
			
	return ell_map, ps, diffuse_realiz.real

def generate_spire_cirrus_realizations(n_realizations, planck_template, imdims, power_law_idx=-2.6, psf_fwhms=[3., 3., 3.],\
									   show=False, vmin=-0.003, vmax=0.003, psf_smooth=True, rms_scale_fac=2.2):
	
	'''
	Given a power spectrum and multiband specifications, this function generates an arbitrary number of galactic cirrus FIR realizations.

	Parameters
	----------

	n_realizations : 'int'
		Number of cirrus realizations.
	planck_template : `~numpy.ndarray' of shape (imdims[0][0], imdims[0][1])
		Planck template used for normalization of cirrus power spectrum. This is useful when one wants to generate synthetic cirrus realizations
		for a cluster with previous large scale dust measurements.
	imdims : list of lists, or `~numpy.ndarray' of shape (Nbands, 2)
		Image dimensions for each of the modeled observations.
	power_law_idx : 'float', optional
		Power law index for power spectrum of diffuse realization. Default is -2.6.
	psf_fwhms : 'list' or `~numpy.ndarray' of shape (Nbands,)
		Beam FWHMs for observations, which are used to convolve cirrus realizations with instrument response.
		Default is [3., 3., 3.] (PSF FWHMs for SPIRE maps)
	show : bool
		If True, show the cirrus realizations. Default is 'False'.
	vmin, vmax : 'float'
		minimum and maximum stretch for plots showing cirrus realizations. 
		Defaults are -0.003 and 0.003, respectively.

	Returns
	-------

	all_realizations : list of objects of type `~numpy.ndarray' with shape (n_realizations, N, M)
		List of cirrus realizations in each band.

	fs : list of objects with class `matplotlib.figure.Figure', optional
		Figures showing cirrus realizations. Only returned if 'show' set to True.

	'''
	all_realizations = []
	fs = []
	for k in range(n_realizations):
		
		multiband_dust_realiz = multiband_diffuse_realization(imdims, power_law_idx=power_law_idx)

		norms = get_spire_diffuse_norms(planck_template, rms_scale_fac=rms_scale_fac)

		if psf_smooth:
			multiband_dust_realiz = psf_smooth_templates(multiband_dust_realiz, psf_sigmas=np.array(psf_fwhms)/2.355)

		final_ts = [norms[i]*multiband_dust_realiz[i] for i in range(len(imdims))]

		if show:
			f = show_diffuse_temps(final_ts, titles=['250 micron [Jy/beam]', '350 micron [Jy/beam]', '500 micron [Jy/beam]'], vmin=vmin, vmax=vmax)
			fs.append(f)
		all_realizations.append(final_ts)
		
	if show:
		return all_realizations, fs
	else:
		return all_realizations

def get_spire_diffuse_norms(planck_template, bands=[250., 350., 500.], rms_scale_fac=2.2):

	'''
	Given a coarse Planck template, computes power spectrum normalization for a cirrus-like spectrum across bands.

	[This is a crude way of doing it currently, should make cleaner at some point.]

	Parameters
	----------
	
	planck_template : `numpy.ndarray' of shape (N, M)
		Coarse template used to derive power spectrum normalization.
	bands : 'list' of 'floats', optional
		wavelengths of SPIRE bands used. Default is [250., 350., 500.].
	rms_scale_fac : 'float', optional
		This is a multiplicative scaling factor to make sure PS normalization is correct. Default is 2.2.


	Returns
	-------

	norms : 'list' of 'floats'
		List of normalizations to scale cirrus template realizations by.

	'''
	
	flux_density_conversion_dict = dict({'S': 86.29e-4, 'M':16.65e-3, 'L':34.52e-3})
	
	dust_I_lams = [3e6, 1.6e6, 8e5] # in Jy/sr
	band_dict = dict({0:'S', 1:'M', 2:'L'})
	planck_rms_250 = np.std(planck_template)

	norms = []
	norms.append(rms_scale_fac*planck_rms_250)

	for i in range(len(bands)-1):
		norm = norms[0]/flux_density_conversion_dict[band_dict[0]]
		norm *= dust_I_lams[i+1]/dust_I_lams[0]
		norm *= flux_density_conversion_dict[band_dict[i+1]]
		norms.append(norm)
	return norms

def multiband_diffuse_realization(N_vals, M_vals=None, power_law_idx=-2.7, psf_sigmas=None, relative_amplitudes=None, normalize=True, show=False):
	
	''' 
	Generates multiple band diffuse realization with gaussian random fields, given some image dimensions and fluctuation information.

	Parameters
	----------
	
	N_vals : 'list' of floats with length (Nbands,)
		Widths of image in pixels for each observation. 
	M_vals : 'list' of floats with length (Nbands,), optional
		Heights of image in pixels for each observation. If left unspecified, M_vals set equal to N_vals. 
		Default is 'None'.
	power_law_idx : 'float', optional
		Power law index for power spectrum of diffuse realization. Default is -2.7.
	normalize : bool, optional
		If True, normalizes each template to have peak equal to unity. Default is True.
	psf_sigmas : 'list' of floats, optional
		Assuming Gaussian beams, list specifies PSF sigmas in pixels for each observation.
		Default is 'None'.
	show : bool, optional
		If True, makes plot of diffuse templates. Default is 'False'.

	Returns
	-------

	templates : list of `~numpy.ndarrays' of shape (N_vals[b][0], M_vals[b][1]) for b in Nbands
		Generated diffuse templates.
	
	'''
	
	if M_vals is None:
		M_vals = N_vals

	nbands = len(N_vals)
		
	templates = []
	_, _, diffuse_realization = generate_diffuse_realization(N_vals[0], M_vals[0], power_law_idx=power_law_idx)
	
	diff_real = diffuse_realization.copy()
	for i in range(nbands):

		if relative_amplitudes is not None:

			diffuse_rel = diffuse_realization*relative_amplitudes[b]
			resized_realiz = np.array(Image.fromarray(diffuse_rel).resize((N_vals[i], M_vals[i]),resample=Image.BICUBIC))

		else:
			resized_realiz = np.array(Image.fromarray(diffuse_realization).resize((N_vals[i], M_vals[i]),resample=Image.BICUBIC))
		
		if psf_sigmas is not None:
			resized_realiz = gaussian_filter(resized_realiz, sigma=psf_sigmas[i])
			
		if normalize:
			resized_realiz /= np.max(np.abs(resized_realiz))
		templates.append(resized_realiz)

	if show:
		f = show_diffuse_temps(templates)
		
	return templates

def psf_smooth_templates(templates, psf_sigmas=[1.27, 1.27, 1.27]):
	smoothed_ts = []
	for i, template in enumerate(templates):
		smoothed_ts.append(gaussian_filter(template, sigma=psf_sigmas[i]))
	return smoothed_ts

def show_diffuse_temps(templates, titles=None, return_fig=True, vmin=-0.003, vmax=0.003):
	f = plt.figure(figsize=(4*len(templates), 4))
	for i, temp in enumerate(templates):
		plt.subplot(1,len(templates), i+1)
		if titles is not None:
			plt.title(titles[i])
		plt.imshow(templates[i], vmin=vmin, vmax=vmax)
		plt.colorbar()
	plt.tight_layout()
	plt.show()

	if return_fig:
		return f


def azim_average_cl2d(ps2d, l2d, nbins=29, lbinedges=None, lbins=None, weights=None, logbin=False):
    
    if lbinedges is None:
        lmin = np.min(l2d[l2d!=0])
        lmax = np.max(l2d[l2d!=0])
        if logbin:
            lbinedges = np.logspace(np.log10(lmin), np.log10(lmax), nbins)
            lbins = np.sqrt(lbinedges[:-1] * lbinedges[1:])
        else:
            lbinedges = np.linspace(lmin, lmax, nbins)
            lbins = (lbinedges[:-1] + lbinedges[1:]) / 2

        lbinedges[-1] = lbinedges[-1]*(1.01)
        
    if weights is None:
        weights = np.ones(ps2d.shape)
        
    Cl = np.zeros(len(lbins))
    Clerr = np.zeros(len(lbins))
    Nmodes = np.zeros(len(lbins),dtype=int)
    Neffs = np.zeros(len(lbins))
    for i,(lmin, lmax) in enumerate(zip(lbinedges[:-1], lbinedges[1:])):
        sp = np.where((l2d>=lmin) & (l2d<lmax))
        p = ps2d[sp]
        w = weights[sp]

        Neff = compute_Neff(w)

        Cl[i] = np.sum(p*w) / np.sum(w)
        Clerr[i] = np.std(p) / np.sqrt(len(p))
        Nmodes[i] = len(p)
        Neffs[i] = Neff
        
    return lbins, Cl, Clerr

def get_power_spec(map_a, map_b=None, mask=None, pixsize=7., 
                   lbinedges=None, lbins=None, nbins=29, 
                   logbin=True, weights=None, return_full=False, return_Dl=False):
    '''
    calculate 1d cross power spectrum Cl
    
    Inputs:
    =======
    map_a: 
    map_b:map_b=map_a if no input, i.e. map_a auto
    mask: common mask for both map
    pixsize:[arcsec]
    lbinedges: predefined lbinedges
    lbins: predefined lbinedges
    nbins: number of ell bins
    logbin: use log or linear ell bin
    weights: Fourier weight
    return_full: return full output or not
    return_Dl: return Dl=Cl*l*(l+1)/2pi or Cl
    
    Outputs:
    ========
    lbins: 1d ell bins
    ps2d: 2D Cl
    Clerr: Cl error, calculate from std(Cl2d(bins))/sqrt(Nmode)
    Nmodes: # of ell modes per ell bin
    lbinedges: 1d ell binedges
    l2d: 2D ell modes
    ps2d: 2D Cl before radial binning
    '''

    if map_b is None:
        map_b = map_a.copy()

    if mask is not None:
        map_a = map_a*mask - np.mean(map_a[mask==1])
        map_b = map_b*mask - np.mean(map_b[mask==1])
    else:
        map_a = map_a - np.mean(map_a)
        map_b = map_b - np.mean(map_b)
        
    l2d, ps2d = get_power_spectrum_2d(map_a, map_b=map_b, pixsize=pixsize)
            
    lbins, Cl, Clerr = azim_average_cl2d(ps2d, l2d, nbins=nbins, lbinedges=lbinedges, lbins=lbins, weights=weights, logbin=logbin)
    
    if return_Dl:
        Cl = Cl * lbins * (lbins+1) / 2 / np.pi
        
    if return_full:
        return lbins, Cl, Clerr, Nmodes, lbinedges, l2d, ps2d
    else:
        return lbins, Cl, Clerr


def get_power_spectrum_2d(map_a, map_b=None, pixsize=7.):
    '''
    calculate 2d cross power spectrum Cl
    
    Inputs:
    =======
    map_a: 
    map_b:map_b=map_a if no input, i.e. map_a auto
    pixsize:[arcsec]
    
    Outputs:
    ========
    l2d: corresponding ell modes
    ps2d: 2D Cl
    '''
    
    if map_b is None:
        map_b = map_a.copy()
        
    dimx, dimy = map_a.shape
    sterad_per_pix = (pixsize/3600/180*np.pi)**2
    V = dimx * dimy * sterad_per_pix
    
    ffta = np.fft.fftn(map_a*sterad_per_pix)
    fftb = np.fft.fftn(map_b*sterad_per_pix)
    ps2d = np.real(ffta * np.conj(fftb)) / V 
    ps2d = np.fft.ifftshift(ps2d)
    
    l2d = get_l2d(dimx, dimy, pixsize)

    return l2d, ps2d

def get_l2d(dimx, dimy, pixsize):
    lx = np.fft.fftfreq(dimx)*2
    ly = np.fft.fftfreq(dimy)*2
    lx = np.fft.ifftshift(lx)*(180*3600./pixsize)
    ly = np.fft.ifftshift(ly)*(180*3600./pixsize)
    ly, lx = np.meshgrid(ly, lx)
    l2d = np.sqrt(lx**2 + ly**2)
    
    return l2d

def compute_Neff(weights):
    N_eff = np.sum(weights)**2/np.sum(weights**2)

    return N_eff



def grab_rms_diffuse_gen_model(timestr_list, dirname, nsamp=100):
    
    nfc_terms_list = np.zeros((11,))
    chi2_diffs = np.zeros((11,))
    median_diffuse_bkgs = np.zeros((11, 100, 100))
    for timestr in timestr_list:

        chain = np.load('../../Downloads/'+dirname+'/'+timestr+'/chain.npz')
        gdat, filepath, result_path = load_param_dict(timestr, result_path='/Users/luminatech/Downloads/'+dirname+'/')

        diffuse_realiz = 1e3*np.load('Data/spire/cirrus_gen/051821/cirrus_sim_idx'+str(int(gdat.tail_name[-3:])-300)+'_051821_4x_planck_cleansky.npz')['S']

        nfc_terms = gdat.n_fourier_terms        
        nfc_terms_list[nfc_terms-5] = nfc_terms
        residz = chain['residuals0']
        bkgs = chain['bkg']


        fourier_coeffs = chain['fourier_coeffs']
        fourier_templates = make_fourier_templates(dimx, dimy, nfc_terms, psf_fwhm=psf_fwhm)              
        burnin_fourier_coeff = np.median(fourier_coeffs[:50,:,:,:], axis=0)

        av_fourier_coeff = np.median(fourier_coeffs[-nsamp:,:,:,:], axis=0)
        std_fourier_coeff = np.std(fourier_coeffs[-nsamp:,:,:,:], axis=0)
        
        burnin_diffuse_bkg = generate_template(burnin_fourier_coeff, nfc_terms, fourier_templates=fourier_templates, N=dimx, M=dimy)
        burnin_diffuse_bkg *= 1e3
        
        median_diffuse_bkg = generate_template(av_fourier_coeff, nfc_terms, fourier_templates=fourier_templates, N=dimx, M=dimy)
        median_diffuse_bkg *= 1e3

        chi2_difference = np.sum((median_diffuse_bkg-diffuse_realiz)**2)
        chi2_diffs[nfc_terms-5] = chi2_difference
        
        median_diffuse_bkgs[nfc_terms-5] = median_diffuse_bkg        
    
        if plot:
            plt.figure(figsize=(15, 5))
            plt.subplot(1,3,1)
            plt.title('burn in')
            plt.imshow(burnin_diffuse_bkg, vmin=np.percentile(diffuse_realiz, 1), vmax=np.percentile(diffuse_realiz, 99), origin='lower', cmap='Greys')
            plt.colorbar()
            plt.subplot(1,3,2)
            plt.title('end of chain')
            plt.imshow(median_diffuse_bkg, vmin=np.percentile(diffuse_realiz, 1), vmax=np.percentile(diffuse_realiz, 99), origin='lower', cmap='Greys')
            plt.colorbar()
            plt.subplot(1,3,3)
            plt.title('difference')
            plt.imshow(burnin_diffuse_bkg-median_diffuse_bkg, vmin=-4, vmax=4, origin='lower', cmap='Greys')
            plt.colorbar()
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(15,5))
            plt.suptitle('Order of Fourier component model = '+str(nfc_terms), fontsize=24)
            plt.subplot(1,3,1)
            plt.imshow(median_diffuse_bkg, vmin=np.percentile(diffuse_realiz, 1), vmax=np.percentile(diffuse_realiz, 99), origin='lower', cmap='Greys')
            cbar = plt.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('mJy/beam')
            plt.subplot(1,3,2)
            plt.imshow(diffuse_realiz, vmin=np.percentile(diffuse_realiz, 1), vmax=np.percentile(diffuse_realiz, 99), origin='lower', cmap='Greys')
            cbar = plt.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('mJy/beam')
            plt.subplot(1,3,3)
            vmin_resid = -4
            vmax_resid = 4
            plt.imshow((median_diffuse_bkg-diffuse_realiz), vmin=vmin_resid, vmax=vmax_resid, origin='lower', cmap='Greys')
            cbar = plt.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('mJy/beam', fontsize=16)
            plt.tight_layout()
    #         plt.savefig('/Users/luminatech/Downloads/conley_dustonly_10arcmin_1band_1mJy_beam_060121_4x_Planck_simidx_301/figures/threepan_nfc='+str(nfc_terms)+'.png', bbox_inches='tight', dpi=200)
    #         plt.savefig('/Users/luminatech/Downloads/conley_dustonly_10arcmin_1band_1mJy_beam_060121_4x_Planck_simidx_301_wptsrcmodl/figures/threepan_nfc='+str(nfc_terms)+'.png', bbox_inches='tight', dpi=200)
            plt.show()
        
    return chi2_diffs, nfc_terms_list, median_diffuse_bkgs, diffuse_realiz, av_fourier_coeff, std_fourier_coeff



