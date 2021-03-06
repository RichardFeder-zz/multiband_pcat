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



