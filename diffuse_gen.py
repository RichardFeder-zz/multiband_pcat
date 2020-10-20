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

def generate_diffuse_realization(N=None, M=None, power_law_idx=-2.7):
	if N is None:
		N = 100
		M = 100
	
	freq_x = fftshift(np.fft.fftfreq(N, d=1.0))
	freq_y = fftshift(np.fft.fftfreq(M, d=1.0))
	
	ell_x,ell_y = np.meshgrid(freq_x,freq_y)
	ell_x = ifftshift(ell_x)
	ell_y = ifftshift(ell_y)
	
	ell_map = np.sqrt(ell_x**2 + ell_y**2)
	
	ps = ell_map**power_law_idx
	ps[0,0] = 0.
	
	diffuse_realiz = ifft2(ps*(np.random.normal(0, 1, size=(N, M)) + 1j*np.random.normal(0, 1, size=(N, M))))
			
	return ell_map, ps, diffuse_realiz.real

def generate_spire_cirrus_realizations(n_realizations, planck_template, imdims, power_law_idx=-2.6, psf_widths=[3., 3., 3.],\
                                       show=False, vmin=-0.003, vmax=0.003):
    all_realizations = []
    fs = []
    for k in range(n_realizations):
        
        multiband_dust_realiz = multiband_diffuse_realization(imdims, power_law_idx=power_law_idx)

        norms = get_spire_diffuse_norms(planck_template)

        smoothed_ts = psf_smooth_templates(multiband_dust_realiz, psf_widths=psf_widths)

        final_ts = [norms[i]*smoothed_ts[i] for i in range(len(imdims))]
        
        
        if show:
            f = show_diffuse_temps(final_ts, titles=['250 micron [Jy/beam]', '350 micron [Jy/beam]', '500 micron [Jy/beam]'], vmin=vmin, vmax=vmax)
            fs.append(f)
        all_realizations.append(final_ts)
        
    if show:
        return all_realizations, fs
    else:
        return all_realizations

def get_spire_diffuse_norms(planck_template, bands=[250., 350., 500.], rms_scale_fac=2.2):
	
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

def multiband_diffuse_realization(N_vals, M_vals=None, power_law_idx=-2.7, psf_widths=None, normalize=True, show=False):
	
	if M_vals is None:
		M_vals = N_vals
		
	templates = []
	_, _, diffuse_realization = generate_diffuse_realization(N_vals[0], M_vals[0], power_law_idx=power_law_idx)
	
	diff_real = diffuse_realization.copy()
	for i in range(len(N_vals)):

		resized_realiz = np.array(Image.fromarray(diffuse_realization).resize((N_vals[i], M_vals[i]),resample=Image.BICUBIC))
		
		if psf_widths is not None:
			resized_realiz = gaussian_filter(resized_realiz, sigma=psf_widths[i])
			
		if normalize:
			resized_realiz /= np.max(np.abs(resized_realiz))
		templates.append(resized_realiz)
	if show:
		f = show_diffuse_temps(templates)
		
	return templates

def psf_smooth_templates(templates, psf_widths=[3., 3., 3.]):
	smoothed_ts = []
	for i, template in enumerate(templates):
		smoothed_ts.append(gaussian_filter(template, sigma=psf_widths[i]))
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



