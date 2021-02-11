import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fftpack import fft, ifft
import networkx as nx
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
from PIL import Image
import sys
import pandas as pd
from pcat_spire import *
from spire_roc_condensed_cat import *

if sys.version_info[0] == 3:
	from celluloid import Camera

from fourier_bkg_modl import *

def add_directory(dirpath):
	if not os.path.isdir(dirpath):
		os.makedirs(dirpath)
	return dirpath

def convert_pngs_to_gif(filenames, gifdir='/Users/richardfeder/Documents/multiband_pcat/', name='', duration=1000, loop=0):

	# Create the frames
	frames = []
	for i in range(len(filenames)):
		new_frame = Image.open(gifdir+filenames[i])
		frames.append(new_frame)

	# Save into a GIF file that loops forever
	frames[0].save(gifdir+name+'.gif', format='GIF',
				   append_images=frames[1:],
				   save_all=True,
				   duration=duration, loop=loop)

def compute_dNdS(trueminf, stars, nsrc, _F=2):

	''' function for computing number counts '''

	binz = np.linspace(np.log10(trueminf)+3., np.ceil(np.log10(np.max(stars[_F, 0:nsrc]))+3.), 20)
	hist = np.histogram(np.log10(stars[_F, 0:nsrc])+3., bins=binz)
	logSv = 0.5*(hist[1][1:]+hist[1][:-1])-3.
	binz_Sz = 10**(binz-3)
	dSz = binz_Sz[1:]-binz_Sz[:-1]
	dNdS = hist[0]

	return logSv, dSz, dNdS


def compute_degradation_fac(condensed_cat, err, flux_err_idx, smooth_fac=5, xidx=0, yidx=2, psf_fwhm=3.):

	''' Given a condensed catalog and underlying noise model, compute the departure of PCAT estimated source uncertainties from 
	what one would expect for an isolated source.

	Parameters
	----------

	condensed_cat : `~numpy.ndarray' of shape (nsrc, n_features)
		Input condensed catalog

	err : `~numpy.ndarray'
		Noise model for field being cataloged. 

	smooth_fac : float, optional
		Because there may be significant pixel variation in the noise model, smooth_fac sets the scale for smoothing the error map, 
		such that when the error is quoted based on a sources position, it represents something closer to the effective (beam averaged) noise.
		Default is 5 pixels. 

	psf_fwhm : float, optional
		Full width at half maximum for point spread function. Default is 3 pixels. 

	Returns
	-------

	flux_deg_fac : `~numpy.ndarray' of length (N_src)
		array of flux degradation factors for each source in the condensed catalog.

	'''

	optimal_ferr_map = np.sqrt(err**2/(4*np.pi*(psf_fwhm/2.355)**2))
	smoothed_optimal_ferr_map = gaussian_filter(optimal_ferr_map, smooth_fac)
	flux_deg_fac = np.zeros_like(condensed_cat[:,0])

	for s, src in enumerate(condensed_cat):
		flux_deg_fac[s] = src[flux_err_idx]/smoothed_optimal_ferr_map[int(src[xidx]), int(src[yidx])]
	
	return flux_deg_fac

def plot_atcr(listsamp, title):

	numbsamp = listsamp.shape[0]
	four = fft(listsamp - np.mean(listsamp, axis=0), axis=0)
	atcr = ifft(four * np.conjugate(four), axis=0).real
	atcr /= np.amax(atcr, 0)

	autocorr = atcr[:int(numbsamp/2), ...]
	indxatcr = np.where(autocorr > 0.2)
	timeatcr = np.argmax(indxatcr[0], axis=0)

	numbsampatcr = autocorr.size

	figr, axis = plt.subplots(figsize=(6,4))
	plt.title(title, fontsize=16)
	axis.plot(np.arange(numbsampatcr), autocorr)
	axis.set_xlabel(r'$\tau$', fontsize=16)
	axis.set_ylabel(r'$\xi(\tau)$', fontsize=16)
	axis.text(0.8, 0.8, r'$\tau_{exp} = %.3g$' % timeatcr, ha='center', va='center', transform=axis.transAxes, fontsize=16)
	axis.axhline(0., ls='--', alpha=0.5)
	plt.tight_layout()

	return figr


def plot_custom_multiband_frame(obj, resids, models, panels=['data0','model0', 'residual0','residual1','residual2','residual2zoom'], \
							zoomlims=[[[0, 40], [0, 40]],[[70, 110], [70, 110]], [[50, 70], [50, 70]]], \
							ndeg=0.11, panel0=None, panel1=None, panel2=None, panel3=None, panel4=None, panel5=None, fourier_bkg=None, sz=None, frame_dir_path=None, smooth_fac=4):

	
	if panel0 is not None:
		panels[0] = panel0
	if panel1 is not None:
		panels[1] = panel1
	if panel2 is not None:
		panels[2] = panel2
	if panel3 is not None:
		panels[3] = panel3
	if panel4 is not None:
		panels[4] = panel4
	if panel5 is not None:
		panels[5] = panel5

	plt.gcf().clear()
	plt.figure(1, figsize=(15, 10))
	plt.clf()

	scatter_sizefac = 300

	for i in range(6):

		plt.subplot(2,3,i+1)

		band_idx = int(panels[i][-1])


		if 'data' in panels[i]:

			plt.imshow(obj.dat.data_array[band_idx], origin='lower', interpolation='none', cmap='Greys', vmin=np.percentile(obj.dat.data_array[band_idx], 5.), vmax=np.percentile(obj.dat.data_array[band_idx], 95.0))
			plt.colorbar()

			if band_idx > 0:
				xp, yp = obj.dat.fast_astrom.transform_q(obj.stars[obj._X, 0:obj.n], obj.stars[obj._Y, 0:obj.n], band_idx-1)
				plt.scatter(xp, yp, marker='x', s=obj.stars[obj._F+1, 0:obj.n]*scatter_sizefac, color='r')
			else:

				plt.scatter(obj.stars[obj._X, 0:obj.n], obj.stars[obj._Y, 0:obj.n], marker='x', s=obj.stars[obj._F, 0:obj.n]*scatter_sizefac, color='r', alpha=0.8)

				# plt.scatter(obj.stars[obj._X, 0:obj.n], obj.stars[obj._Y, 0:obj.n], marker='x', s=obj.stars[obj._F, 0:obj.n]*100, color='r')
			if 'zoom' in panels[i]:
				plt.title('Data (band '+str(band_idx)+', zoomed in)')
				plt.xlim(zoomlims[band_idx][0][0], zoomlims[band_idx][0][1])
				plt.ylim(zoomlims[band_idx][1][0], zoomlims[band_idx][1][1])
			else:
				plt.title('Data (band '+str(band_idx)+')')
				plt.xlim(-0.5, obj.imszs[band_idx][0]-0.5)
				plt.ylim(-0.5, obj.imszs[band_idx][1]-0.5)


		elif 'model' in panels[i]:

			plt.imshow(models[band_idx], origin='lower', interpolation='none', cmap='Greys', vmin=np.percentile(models[band_idx], 5.), vmax=np.percentile(models[band_idx], 95.0))
			plt.colorbar()

			if 'zoom' in panels[i]:
				plt.title('Model (band '+str(band_idx)+', zoomed in)')
				plt.xlim(zoomlims[band_idx][0][0], zoomlims[band_idx][0][1])
				plt.ylim(zoomlims[band_idx][1][0], zoomlims[band_idx][1][1])
			else:
				plt.title('Model (band '+str(band_idx)+')')
				plt.xlim(-0.5, obj.imszs[band_idx][0]-0.5)
				plt.ylim(-0.5, obj.imszs[band_idx][1]-0.5)

		elif 'injected_diffuse_comp' in panels[i]:
			plt.imshow(obj.dat.injected_diffuse_comp[band_idx], origin='lower', interpolation='none', cmap='Greys', vmin=np.percentile(obj.dat.injected_diffuse_comp[band_idx], 5.), vmax=np.percentile(obj.dat.injected_diffuse_comp[band_idx], 95.))
			plt.colorbar()
		
			plt.title('Injected cirrus (band '+str(band_idx)+')')
			plt.xlim(-0.5, obj.imszs[band_idx][0]-0.5)
			plt.ylim(-0.5, obj.imszs[band_idx][1]-0.5)

		elif 'fourier_bkg' in panels[i]:
			fbkg = fourier_bkg[band_idx]

			fbkg[obj.dat.weights[band_idx]==0] = 0.

			plt.imshow(fbkg, origin='lower', interpolation='none', cmap='Greys', vmin = np.percentile(fbkg , 5), vmax=np.percentile(fbkg, 95))
			plt.colorbar()
		
			plt.title('Sum of FCs (band '+str(band_idx)+')')
			plt.xlim(-0.5, obj.imszs[band_idx][0]-0.5)
			plt.ylim(-0.5, obj.imszs[band_idx][1]-0.5)


		elif 'residual' in panels[i]:

			if obj.gdat.weighted_residual:
				plt.imshow(resids[band_idx]*np.sqrt(obj.dat.weights[band_idx]), origin='lower', interpolation='none', cmap='Greys', vmin=-5, vmax=5)
			else:
				plt.imshow(resids[band_idx], origin='lower', interpolation='none', cmap='Greys', vmin = np.percentile(resids[band_idx][obj.dat.weights[band_idx] != 0.], 5), vmax=np.percentile(resids[band_idx][obj.dat.weights[band_idx] != 0.], 95))
			plt.colorbar()

			if band_idx > 0:
				xp, yp = obj.dat.fast_astrom.transform_q(obj.stars[obj._X, 0:obj.n], obj.stars[obj._Y, 0:obj.n], band_idx-1)
				plt.scatter(xp, yp, marker='x', s=obj.stars[obj._F+1, 0:obj.n]*scatter_sizefac, color='r')
			else:
				plt.scatter(obj.stars[obj._X, 0:obj.n], obj.stars[obj._Y, 0:obj.n], marker='x', s=obj.stars[obj._F, 0:obj.n]*scatter_sizefac, color='r', alpha=0.8)

			if 'zoom' in panels[i]:
				plt.title('Residual (band '+str(band_idx)+', zoomed in)')
				plt.xlim(zoomlims[band_idx][0][0], zoomlims[band_idx][0][1])
				plt.ylim(zoomlims[band_idx][1][0], zoomlims[band_idx][1][1])
			else:           
				plt.title('Residual (band '+str(band_idx)+')')
				plt.xlim(-0.5, obj.imszs[band_idx][0]-0.5)
				plt.ylim(-0.5, obj.imszs[band_idx][1]-0.5)		

		elif 'sz' in panels[i]:


			plt.imshow(sz[band_idx], origin='lower', interpolation='none', cmap='Greys', vmin = np.percentile(sz[band_idx], 1), vmax=np.percentile(sz[band_idx], 99))
			plt.colorbar()

			if band_idx > 0:
				xp, yp = obj.dat.fast_astrom.transform_q(obj.stars[obj._X, 0:obj.n], obj.stars[obj._Y, 0:obj.n], band_idx-1)
				plt.scatter(xp, yp, marker='x', s=obj.stars[obj._F+1, 0:obj.n]*100, color='r')
			else:
				plt.scatter(obj.stars[obj._X, 0:obj.n], obj.stars[obj._Y, 0:obj.n], marker='x', s=obj.stars[obj._F, 0:obj.n]*scatter_sizefac, color='r')

			if 'zoom' in panels[i]:
				plt.title('SZ (band '+str(band_idx)+', zoomed in)')
				plt.xlim(zoomlims[band_idx][0][0], zoomlims[band_idx][0][1])
				plt.ylim(zoomlims[band_idx][1][0], zoomlims[band_idx][1][1])
			else:           
				plt.title('SZ (band '+str(band_idx)+')')
				plt.xlim(-0.5, obj.imszs[band_idx][0]-0.5)
				plt.ylim(-0.5, obj.imszs[band_idx][1]-0.5)		


		elif 'dNdS' in panels[i]:

			if obj.n > 0:
				logSv, dSz, dNdS = compute_dNdS(obj.trueminf, obj.stars, obj.n, _F=2+band_idx)

				if obj.gdat.raw_counts:
					plt.plot(logSv+3, dNdS, marker='.')
					plt.ylabel('dN/dS')
					plt.ylim(5e-1, 3e3)

				else:
					n_steradian = ndeg/(180./np.pi)**2 # field covers 0.11 degrees, should change this though for different fields
					n_steradian *= obj.gdat.frac # a number of pixels in the image are not actually observing anything
					dNdS_S_twop5 = dNdS*(10**(logSv))**(2.5)
					plt.plot(logSv+3, dNdS_S_twop5/n_steradian/dSz, marker='.')
					plt.ylabel('dN/dS.$S^{2.5}$ ($Jy^{1.5}/sr$)')
					plt.ylim(1e0, 1e5)

				plt.yscale('log')
				plt.legend()
				plt.xlabel('log($S_{\\nu}$) (mJy)')
				plt.xlim(np.log10(obj.trueminf)+3.-0.5, 2.5)


	if frame_dir_path is not None:
		plt.savefig(frame_dir_path, bbox_inches='tight', dpi=200)
	plt.draw()
	plt.pause(1e-5)



def scotts_rule_bins(samples):
	'''
	Computes binning for a collection of samples using Scott's rule, which minimizes the integrated MSE of the density estimate.

	Parameters
	----------

	samples : 'list' or '~numpy.ndarray' of shape (Nsamples,)

	Returns
	-------

	bins : `~numpy.ndarray' of shape (Nsamples,)
		bin edges

	'''
	n = len(samples)
	bin_width = 3.5*np.std(samples)/n**(1./3.)
	k = np.ceil((np.max(samples)-np.min(samples))/bin_width)
	bins = np.linspace(np.min(samples), np.max(samples), int(k))
	return bins


def make_pcat_sample_gif(timestr, im_path, image_extension='SIGNAL', gif_path=None, cat_xy=True, resids=True, color_color=True, number_cts=True, result_path='/Users/luminatech/Documents/multiband_pcat/spire_results/', \
						gif_fpr = 5):

	chain = np.load('spire_results/'+timestr+'/chain.npz')
	residz = chain['residuals0']
	if number_cts:
		fsrcs = chain['f']
		nsrcs = chain['n']

	# gdat, filepath, result_path = load_param_dict(timestr, result_path='spire_results/')


	n_panels = np.sum(np.array([cat_xy, resids, color_color, number_cts]).astype(np.int))
	print('n_panels is ', n_panels)
	mask = fits.open('data/spire/GOODSN/GOODSN_PSW_mask.fits')[0].data
	bounds = get_rect_mask_bounds(mask)

	im = fits.open(im_path)[image_extension].data

	im = im[bounds[0,0]:bounds[0,1], bounds[1,0]:bounds[1,1]]

	# im = im[gdat.bounds[0][0,0]:gdat.bounds[0][0,1], gdat.bounds[0][1,0]:gdat.bounds[0][1,1]]

	minshap = min(np.array(residz).shape[1], np.array(residz).shape[2])

	print(minshap)

	im = im[:minshap, :minshap].copy()

	fig = plt.figure(figsize=(6*n_panels,7))
	camera = Camera(fig)

	nresid = int(len(residz))
	# print('nresid is ', nresid)

	for k in np.arange(0, len(residz), 10):
		tick = 1

		if cat_xy:
			ax1 = plt.subplot(1,n_panels, tick)
			plt.imshow(im-np.median(im), vmin=-0.007, vmax=0.01, cmap='Greys')
			plt.scatter(chain['x'][-nresid+k,:], chain['y'][-nresid+k,:], s=4e3*chain['f'][0,-nresid+k,:], marker='+', color='r')
			plt.text(25, -2, 'Nsamp = '+str(chain['x'].shape[0]-nresid+k)+', Nsrc='+str(chain['n'][-nresid+k]), fontsize=20)
			# ax1.set_title('Nsamp = '+str(chain['x'].shape[0]-300+k)+', Nsrc='+str(chain['n'][-300+k]), fontsize=20)

			# plt.xlim(0, min(im.shape[0], im.shape[1]))
			# plt.ylim(0, min(im.shape[0], im.shape[1]))

			tick += 1

		if resids:
			ax2 = plt.subplot(1,n_panels, tick)
			ax2.set_title('Data - Model', fontsize=20)
			plt.imshow(residz[-nresid+k,:,:], cmap='Greys', vmax=0.008, vmin=-0.005)
			tick += 1

		if color_color:
			ax3 = plt.subplot(1,n_panels, tick)
			asp = np.diff(ax3.get_xlim())[0] / np.diff(ax3.get_ylim())[0]
			ax3.set_aspect(asp)
			plt.scatter(chain['f'][1,-nresid+k,:]/chain['f'][0,-nresid+k,:], chain['f'][2,-nresid+k,:]/chain['f'][1,-nresid+k,:], s=1.5e4*chain['f'][0,-nresid+k,:], marker='x', c='k')
			plt.ylim(-0.5, 10.5)
			plt.xlim(-0.5, 10.5)

			plt.xlabel('$S_{350}/S_{250}$', fontsize=18)
			plt.ylabel('$S_{500}/S_{350}$', fontsize=18)

		if number_cts:
			nbins = 20
			binz = np.linspace(np.log10(0.005)+3.-1., 3., nbins)

			fsrcs_in_fov = np.array([fsrcs[0][-nresid+k][i] for i in range(nsrcs[-nresid+k])])

			hist = np.histogram(np.log10(fsrcs_in_fov)+3, bins=binz)
			logSv = 0.5*(hist[1][1:]+hist[1][:-1])-3
			binz_Sz = 10**(binz-3)
			dSz = binz_Sz[1:]-binz_Sz[:-1]
			dNdS = hist[0]
			npix = im.shape[0]

			pixel_sizes_nc = dict({0:6, 1:8, 2:12}) # arcseconds

			nsidedeg = npix*6./3600.
			n_steradian = nsidedeg**2/(180./np.pi)**2
			
			dNdS_S_twop5 = dNdS*(10**(logSv))**(2.5)
			mean_number_cts = dNdS_S_twop5/n_steradian/dSz

			ax3 = plt.subplot(1,n_panels, tick)
			ax3.set_title('Number counts - 250 micron', fontsize=20)
			plt.xlabel('$S_{\\nu}$ [mJy]', fontsize=16)
			plt.ylabel('log dN/dS.$S^{2.5}$ [$Jy^{1.5}/sr$]', fontsize=16)


			import pandas as pd
			
			ncounts_bethermin = pd.read_csv('~/Downloads/Bethermin_et_al_PSW.csv', header=None)
			plt.plot(np.array(ncounts_bethermin[0]), np.array(ncounts_bethermin[1]), marker='.', markersize=10, label='Bethermin et al. (2012a)', color='cyan')

			ncounts_oliver = pd.read_csv('~/Downloads/Oliver_et_al_PSW.csv', header=None)
			plt.plot(np.array(ncounts_oliver[0]), np.array(ncounts_oliver[1]), marker='3', markersize=10, label='Oliver et al. (2010)', color='b')

			ncounts_Glenn = pd.read_csv('~/Downloads/Glenn_et_al_PSW.csv', header=None)
			plt.plot(np.array(ncounts_Glenn[0]), np.array(ncounts_Glenn[1]), marker='+', markersize=10, label='Glenn et al. (2010)', color='limegreen')

			ncounts_SIDES_srcextract = pd.read_csv('~/Downloads/SIDES_srcextract_PSW.csv', header=None)
			plt.plot(np.array(ncounts_SIDES_srcextract[0]), np.array(ncounts_SIDES_srcextract[1]), marker='^', markersize=10, label='SIDES Source Extraction (2017)', color='r')

			ncounts_XIDp_srcextract = pd.read_csv('~/Downloads/Wang_2019_XIDplus_PSW.csv', header=None)
			plt.plot(np.array(ncounts_XIDp_srcextract[0]), np.array(ncounts_XIDp_srcextract[1]), marker='4', markersize=10, label='XID+ (Wang et al. 2019)', color='y')

			ncounts_SIDES_model = pd.read_csv('~/Downloads/SIDES_PSW.csv', header=None)
			plt.plot(np.array(ncounts_SIDES_model[0]), np.array(ncounts_SIDES_model[1]), marker='o', markersize=5, label='SIDES empirical model', color='royalblue')


			plt.plot(10**(logSv+3), np.log10(mean_number_cts), marker='x', markersize=10,  color='k', linewidth=2, label='PCAT')


			plt.xscale('log')
			if k==0:
				print('making legend')
				plt.legend(fontsize=14)
			plt.ylim(1.8, 5)
			plt.xlim(0.5, 200)



		plt.tight_layout()
		camera.snap()

	an = camera.animate()
	if gif_path is None:
		gif_path = 'pcat_sample_gif_'+timestr+'.gif'
	an.save(gif_path, writer='PillowWriter', fps=gif_fpr)


def plot_bkg_sample_chain(bkg_samples, band='250 micron', title=True, show=False, convert_to_MJy_sr_fac=None, smooth_fac=None):

	''' This function takes a chain of background samples from PCAT and makes a trace plot. '''

	if convert_to_MJy_sr_fac is None:
		convert_to_MJy_sr_fac = 1.
		ylabel_unit = ' [Jy/beam]'
	else:
		ylabel_unit = ' [MJy/sr]'

	f = plt.figure()
	if title:
		plt.title('Uniform background level - '+str(band))

	if smooth_fac is not None:
		bkg_samples = np.convolve(bkg_samples, np.ones((smooth_fac,))/smooth_fac, mode='valid')

	plt.plot(np.arange(len(bkg_samples)), bkg_samples/convert_to_MJy_sr_fac, label=band)
	plt.xlabel('Sample index')
	plt.ylabel('Amplitude'+ylabel_unit)
	plt.legend()
	
	if show:
		plt.show()

	return f

def plot_template_amplitude_sample_chain(template_samples, band='250 micron', template_name='sze', title=True, show=False, xlabel='Sample index', ylabel='Amplitude',\
									 convert_to_MJy_sr_fac=None, smooth_fac = None):

	''' This function takes a chain of template amplitude samples from PCAT and makes a trace plot. '''

	ylabel_unit = None
	if convert_to_MJy_sr_fac is None:
		convert_to_MJy_sr_fac = 1.
		ylabel_unit = ' [Jy/beam]'
	else:
		ylabel_unit = ' [MJy/sr]'

	if template_name=='dust' or template_name == 'planck':
		ylabel_unit = None

	if smooth_fac is not None:
		template_samples = np.convolve(template_samples, np.ones((smooth_fac,))/smooth_fac, mode='valid')

	f = plt.figure()
	if title:
		plt.title(template_name +' template level - '+str(band))

	plt.plot(np.arange(len(template_samples)), template_samples, label=band)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()
	
	if show:
		plt.show()

	return f

def plot_template_median_std(template, template_samples, band='250 micron', template_name='cirrus dust', title=True, show=False, convert_to_MJy_sr_fac=None):

	''' This function takes a template and chain of template amplitudes samples from PCAT. 
	These are used to compute the median template estimate as well the pixel-wise standard deviation on the template.
	'''

	if convert_to_MJy_sr_fac is None:
		convert_to_MJy_sr_fac = 1.
		xlabel_unit = '[Jy/beam]'
	else:
		xlabel_unit = '[MJy/sr]'

	f = plt.figure(figsize=(10, 5))

	if title:
		plt.suptitle(template_name)

	mean_t, std_t = np.mean(template_samples), np.std(template_samples)

	plt.subplot(1,2,1)
	plt.title('Median')
	plt.imshow(mean_t*template/convert_to_MJy_sr_fac, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(mean_t*template, 5)/convert_to_MJy_sr_fac, vmax=np.percentile(mean_t*template, 95)/convert_to_MJy_sr_fac)
	cbar = plt.colorbar(fraction=0.046, pad=0.04)
	cbar.set_label(xlabel_unit, fontsize=14)
	plt.subplot(1,2,2)
	plt.title('Standard deviation'+xlabel_unit)
	plt.imshow(std_t*np.abs(template)/convert_to_MJy_sr_fac, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(std_t*np.abs(template), 5)/convert_to_MJy_sr_fac, vmax=np.percentile(std_t*np.abs(template), 95)/convert_to_MJy_sr_fac)
	cbar = plt.colorbar(fraction=0.046, pad=0.04)
	cbar.set_label(xlabel_unit, fontsize=14)

	if show:
		plt.show()
	return f

# fourier comps

def plot_fc_median_std(fourier_coeffs, imsz, ref_img=None, bkg_samples=None, fourier_templates=None, title=True, show=False, convert_to_MJy_sr_fac=None, psf_fwhm=None):
	
	if convert_to_MJy_sr_fac is None:
		convert_to_MJy_sr_fac = 1.
		xlabel_unit = ' [Jy/beam]'
	else:
		xlabel_unit = ' [MJy/sr]'

	n_terms = fourier_coeffs.shape[-2]

	all_temps = np.zeros((fourier_coeffs.shape[0], imsz[0], imsz[1]))
	if fourier_templates is None:
		fourier_templates = make_fourier_templates(imsz[0], imsz[1], n_terms, psf_fwhm=psf_fwhm)

	for i, fourier_coeff_state in enumerate(fourier_coeffs):
		all_temps[i] = generate_template(fourier_coeff_state, n_terms, fourier_templates=fourier_templates, N=imsz[0], M=imsz[1])
		if bkg_samples is not None:
			all_temps[i] += bkg_samples[i]

	mean_fc_temp = np.median(all_temps, axis=0)
	std_fc_temp = np.std(all_temps, axis=0)

	if ref_img is not None:
		f = plt.figure(figsize=(15, 5))
		# if title:
			# plt.suptitle('Best fit Fourier component model', y=1.02)

		plt.subplot(1,3,1)
		plt.title('Data', fontsize=14)
		plt.imshow(ref_img/convert_to_MJy_sr_fac, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(ref_img, 5)/convert_to_MJy_sr_fac, vmax=np.percentile(ref_img, 99)/convert_to_MJy_sr_fac)
		cb = plt.colorbar(orientation='vertical', pad=0.04, fraction=0.046)
		cb.set_label(xlabel_unit)
		plt.subplot(1,3,2)
		plt.title('Median background model', fontsize=14)
		plt.imshow(mean_fc_temp/convert_to_MJy_sr_fac, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(mean_fc_temp, 5)/convert_to_MJy_sr_fac, vmax=np.percentile(mean_fc_temp, 99)/convert_to_MJy_sr_fac)
		cb = plt.colorbar(orientation='vertical', pad=0.04, fraction=0.046)
		cb.set_label(xlabel_unit)		
		plt.subplot(1,3,3)
		plt.title('Data - median background model', fontsize=14)
		plt.imshow((ref_img - mean_fc_temp)/convert_to_MJy_sr_fac, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(ref_img, 5)/convert_to_MJy_sr_fac, vmax=np.percentile(ref_img, 99)/convert_to_MJy_sr_fac)
		# plt.imshow((ref_img - mean_fc_temp)/convert_to_MJy_sr_fac, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(ref_img-mean_fc_temp, 5)/convert_to_MJy_sr_fac, vmax=np.percentile(ref_img-mean_fc_temp, 95)/convert_to_MJy_sr_fac)
		cb = plt.colorbar(orientation='vertical', pad=0.04, fraction=0.046)
		cb.set_label(xlabel_unit)
		# plt.subplot(2,2,4)
		# plt.title('Std. dev. of background model')
		# plt.imshow(std_fc_temp/convert_to_MJy_sr_fac, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(std_fc_temp, 5)/convert_to_MJy_sr_fac, vmax=np.percentile(std_fc_temp, 95)/convert_to_MJy_sr_fac)
		# cb = plt.colorbar(orientation='horizontal', pad=0.04, fraction=0.046)
		# cb.set_label(xlabel_unit, fontsize='small')

	else:

		plt.subplot(1,2,1)
		plt.title('Median'+xlabel_unit)
		plt.imshow(mean_fc_temp/convert_to_MJy_sr_fac, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(mean_fc_temp, 5)/convert_to_MJy_sr_fac, vmax=np.percentile(mean_fc_temp, 95)/convert_to_MJy_sr_fac)
		plt.colorbar(pad=0.04)
		plt.subplot(1,2,2)
		plt.title('Standard deviation'+xlabel_unit)
		plt.imshow(std_fc_temp/convert_to_MJy_sr_fac, origin='lower', cmap='Greys', interpolation=None, vmin=np.percentile(std_fc_temp, 5)/convert_to_MJy_sr_fac, vmax=np.percentile(std_fc_temp, 95)/convert_to_MJy_sr_fac)
		plt.colorbar(pad=0.04)

	plt.tight_layout()
	if show:
		plt.show()
	
	return f

def plot_last_fc_map(fourier_coeffs, imsz, ref_img=None, fourier_templates=None, title=True, show=False, convert_to_MJy_sr_fac=None, titlefontsize=16):
	if convert_to_MJy_sr_fac is None:
		convert_to_MJy_sr_fac = 1.
		xlabel_unit = ' [Jy/beam]'
	else:
		xlabel_unit = ' [MJy/sr]'

	n_terms = fourier_coeffs.shape[-2]


	if fourier_templates is None:
		fourier_templates = make_fourier_templates(imsz[0], imsz[1], n_terms, psf_fwhm=3.)
	last_temp_bc = generate_template(fourier_coeffs, n_terms, fourier_templates=fourier_templates, N=imsz[0], M=imsz[1])


	fourier_templates_unconv = make_fourier_templates(imsz[0], imsz[1], n_terms, psf_fwhm=None)
	last_temp = generate_template(fourier_coeffs, n_terms, fourier_templates=fourier_templates_unconv, N=imsz[0], M=imsz[1])	

	if ref_img is not None:
		f = plt.figure(figsize=(15, 5))
		plt.subplot(1,3,1)
		plt.title('Data', fontsize=titlefontsize)
		plt.imshow(ref_img, cmap='Greys', interpolation=None, vmin=np.percentile(ref_img, 5), vmax=np.percentile(ref_img, 95), origin='lower')
		cbar = plt.colorbar(orientation='horizontal')
		cbar.set_label('[Jy/beam]', fontsize=16)
		plt.subplot(1,3,2)
		plt.title('Last Fourier model realization (+PSF)', fontsize=titlefontsize)
		plt.imshow(last_temp_bc/convert_to_MJy_sr_fac, cmap='Greys', interpolation=None, origin='lower', vmin=np.percentile(ref_img, 5), vmax=np.percentile(ref_img, 95))
		cbar = plt.colorbar(orientation='horizontal')
		cbar.set_label('[Jy/beam]', fontsize=16)
		plt.subplot(1,3,3)
		plt.title('Last Fourier model realization', fontsize=titlefontsize)
		plt.imshow(last_temp/convert_to_MJy_sr_fac, cmap='Greys', interpolation=None, origin='lower', vmin=np.percentile(ref_img, 5), vmax=np.percentile(ref_img, 95))
		cbar = plt.colorbar(orientation='horizontal')
		cbar.set_label('[Jy/beam]', fontsize=16)

	else:
		f = plt.figure()
		plt.title('Last fourier model realization', fontsize=titlefontsize)
		plt.imshow(last_temp/convert_to_MJy_sr_fac, cmap='Greys', interpolation=None, origin='lower')
		plt.colorbar()

	plt.tight_layout()
	if show:
		plt.show()

	return f

def plot_flux_vs_fluxerr(fluxes, flux_errs, show=False, alpha=0.1, snr_levels = [2., 5., 10., 20., 50.], xlim=[1, 1e3], ylim=[0.1, 2e2]):

	''' This takes a list of fluxes and flux uncertainties and plots them against each other, along with selection of SNR levels for reference. '''
	
	f = plt.figure()
	plt.title('Flux errors', fontsize=16)
	plt.scatter(fluxes, flux_errs, alpha=alpha, color='k', label='Condensed catalog')
	plt.xlabel('F [mJy]', fontsize=16)
	plt.ylabel('$\\sigma_F$ [mJy]')

	xspace = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 100)
	plt.xscale('log')
	plt.xlim(xlim)
	plt.yscale('log')
	plt.ylim(ylim)
	for s, snr in enumerate(snr_levels):
		plt.plot(xspace, xspace/snr, label='SNR = '+str(np.round(snr)), color='C'+str(s), linestyle='dashed')

	plt.legend(fontsize=14)
	plt.tight_layout()
	if show:
		plt.show()

	return f



def plot_degradation_factor_vs_flux(fluxes, deg_fac, show=False, deg_fac_mode='Flux', alpha=0.1, xlim=[1, 1e3], ylim=[0.5, 60]):

	f = plt.figure()
	plt.title(deg_fac_mode+' degradation factor', fontsize=16)
	plt.scatter(fluxes, deg_fac, alpha=alpha, color='k', label='Condensed catalog')
	plt.xlabel('F [mJy]', fontsize=16)
	if deg_fac_mode=='Flux':
		plt.ylabel('DF = $\\sigma_F^{obs.}/\\sigma_F^{opt.}$', fontsize=16)
	elif deg_fac_mode=='Position':
		plt.ylabel('DF = $\\sigma_x^{obs.}/\\sigma_x^{opt.}$', fontsize=16)

	plt.xscale('log')
	plt.xlim(xlim)
	plt.yscale('log')
	plt.ylim(ylim)
	plt.axhline(1., linestyle='dashed', color='k', label='Optimal '+deg_fac_mode.lower()+' error \n (instrument noise only)')
	plt.legend(fontsize=14, loc=1)
	plt.tight_layout()
	if show:
		plt.show()

	return f



def plot_fourier_coeffs_covariance_matrix(fourier_coeffs, show=False):


	perkmode_data_matrix = np.mean(fourier_coeffs, axis=3)
	data_matrix = np.array([perkmode_data_matrix[i].ravel() for i in range(perkmode_data_matrix.shape[0])]).transpose()
	fc_covariance_matrix = np.cov(data_matrix)
	fc_corrcoef_matrix = np.corrcoef(data_matrix)
	f = plt.figure(figsize=(10,5))
	plt.subplot(1,2,1)
	plt.title('Covariance')
	plt.imshow(fc_covariance_matrix, vmin=np.percentile(fc_covariance_matrix, 5), vmax=np.percentile(fc_covariance_matrix, 95))
	plt.colorbar()
	plt.subplot(1,2,2)
	plt.title('Correlation coefficient')
	plt.imshow(fc_corrcoef_matrix, vmin=np.percentile(fc_corrcoef_matrix, 5), vmax=np.percentile(fc_corrcoef_matrix, 95))
	plt.colorbar()

	if show:
		plt.show()


	return f

def plot_fourier_coeffs_sample_chains(fourier_coeffs, show=False):
	
	norm = matplotlib.colors.Normalize(vmin=0, vmax=np.sqrt(2)*fourier_coeffs.shape[1])
	colormap = matplotlib.cm.ScalarMappable(norm=norm, cmap='jet')

	ravel_ks = [np.sqrt(i**2+j**2) for i in range(fourier_coeffs.shape[1]) for j in range(fourier_coeffs.shape[2])]

	colormap.set_array(ravel_ks/(np.sqrt(2)*fourier_coeffs.shape[1]))

	f, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
	ax1, ax2, ax3, ax4 = axes.flatten()
	xvals = np.arange(fourier_coeffs.shape[0])

	for i in range(fourier_coeffs.shape[1]):
		for j in range(fourier_coeffs.shape[2]):
			ax1.plot(xvals, fourier_coeffs[:,i,j,0], alpha=0.5, linewidth=2, color=plt.cm.jet(np.sqrt(i**2+j**2)/(np.sqrt(2)*fourier_coeffs.shape[1])))
	ax1.set_xlabel('Thinned (post burn-in) samples')
	ax1.set_ylabel('$B_{ij,1}$', fontsize=18)

	for i in range(fourier_coeffs.shape[1]):
		for j in range(fourier_coeffs.shape[2]):
			ax2.plot(xvals, fourier_coeffs[:,i,j,1], alpha=0.5, linewidth=2, color=plt.cm.jet(np.sqrt(i**2+j**2)/(np.sqrt(2)*fourier_coeffs.shape[1])))
	ax2.set_xlabel('Thinned (post burn-in) samples')
	ax2.set_ylabel('$B_{ij,2}$', fontsize=18)

	for i in range(fourier_coeffs.shape[1]):
		for j in range(fourier_coeffs.shape[2]):
			ax3.plot(xvals, fourier_coeffs[:,i,j,2], alpha=0.5, linewidth=2, color=plt.cm.jet(np.sqrt(i**2+j**2)/(np.sqrt(2)*fourier_coeffs.shape[1])))
	ax3.set_xlabel('Thinned (post burn-in) samples')
	ax3.set_ylabel('$B_{ij,3}$', fontsize=18)

	for i in range(fourier_coeffs.shape[1]):
		for j in range(fourier_coeffs.shape[2]):
			ax4.plot(xvals, fourier_coeffs[:,i,j,3], alpha=0.5, linewidth=2, color=plt.cm.jet(np.sqrt(i**2+j**2)/(np.sqrt(2)*fourier_coeffs.shape[1])))
	ax4.set_xlabel('Thinned (post burn-in) samples')
	ax4.set_ylabel('$B_{ij,4}$', fontsize=18)

	f.colorbar(colormap, orientation='vertical', ax=ax1).set_label('$|k|$', fontsize=14)
	f.colorbar(colormap, orientation='vertical', ax=ax2).set_label('$|k|$', fontsize=14)
	f.colorbar(colormap, orientation='vertical', ax=ax3).set_label('$|k|$', fontsize=14)
	f.colorbar(colormap, orientation='vertical', ax=ax4).set_label('$|k|$', fontsize=14)


	plt.tight_layout()

	if show:
		plt.show()


	return f



def plot_posterior_fc_power_spectrum(fourier_coeffs, N, pixsize=6., show=False):

	n_terms = fourier_coeffs.shape[1]
	mesha, meshb = np.meshgrid(np.arange(1, n_terms+1) , np.arange(1, n_terms+1))
	kmags = np.sqrt(mesha**2+meshb**2)
	ps_bins = np.logspace(0, np.log10(n_terms+2), 6)

	twod_power_spectrum_realiz = []
	oned_ps_realiz = []
	kbin_masks = []
	for i in range(len(ps_bins)-1):
		kbinmask = (kmags >= ps_bins[i])*(kmags < ps_bins[i+1])
		kbin_masks.append(kbinmask)

	for i, fourier_coeff_state in enumerate(fourier_coeffs):
		av_2d_ps = np.mean(fourier_coeff_state**2, axis=2)
		oned_ps_realiz.append(np.array([np.mean(av_2d_ps[mask]) for mask in kbin_masks]))
		twod_power_spectrum_realiz.append(av_2d_ps)

	power_spectrum_realiz = np.array(twod_power_spectrum_realiz)
	oned_ps_realiz = np.array(oned_ps_realiz)


	f = plt.figure(figsize=(10, 5))
	plt.subplot(1,2,1)
	plt.imshow(np.abs(np.median(power_spectrum_realiz, axis=0)), norm=matplotlib.colors.LogNorm())
	plt.colorbar()
	plt.subplot(1,2,2)
	fov_in_rad = N*(pixsize/3600.)*(np.pi/180.)

	plt.errorbar(2*np.pi/(fov_in_rad/np.sqrt(ps_bins[1:]*ps_bins[:-1])), np.median(oned_ps_realiz,axis=0), yerr=np.std(oned_ps_realiz, axis=0))
	plt.xlabel('$2\\pi/\\theta$ [rad$^{-1}$]', fontsize=16)
	plt.ylabel('$C_{\\ell}$', fontsize=16)
	plt.yscale('log')
	plt.xscale('log')
	plt.tight_layout()
	if show:
		plt.show()

	return f

# TODO -- this will show whether the posteriors are converged on the coefficients
# def plot_fourier_coeff_posteriors(fourier_coeffs):
# 	f = plt.figure()

# 	return f


def plot_posterior_bkg_amplitude(bkg_samples, band='250 micron', title=True, show=False, xlabel='Amplitude', convert_to_MJy_sr_fac=None):
	
	if convert_to_MJy_sr_fac is None:
		convert_to_MJy_sr_fac = 1.
		xlabel_unit = ' [Jy/beam]'
	else:
		xlabel_unit = ' [MJy/sr]'

	f = plt.figure()
	if title:
		plt.title('Uniform background level - '+str(band))
		
	if len(bkg_samples)>50:
		binz = scotts_rule_bins(bkg_samples/convert_to_MJy_sr_fac)
	else:
		binz = 10

	plt.hist(np.array(bkg_samples)/convert_to_MJy_sr_fac, label=band, histtype='step', bins=binz)
	plt.xlabel(xlabel+xlabel_unit)
	plt.ylabel('$N_{samp}$')
	
	if show:
		plt.show()

	return f    

def plot_posterior_template_amplitude(template_samples, band='250 micron', template_name='sze', title=True, show=False, xlabel='Amplitude', convert_to_MJy_sr_fac=None, \
									mock_truth=None, xlabel_unit=None):
	
	if convert_to_MJy_sr_fac is None:
		convert_to_MJy_sr_fac = 1.

		if xlabel_unit is None:
			xlabel_unit = ' [Jy/beam]'
	else:
		if xlabel_unit is None:
			xlabel_unit = ' [MJy/sr]'

	if template_name=='dust' or template_name == 'planck':
		xlabel_unit = ''

	f = plt.figure()
	if title:
		plt.title(template_name +' template level - '+str(band))

	if len(template_samples)>50:
		binz = scotts_rule_bins(template_samples/convert_to_MJy_sr_fac)
	else:
		binz = 10
	plt.hist(np.array(template_samples)/convert_to_MJy_sr_fac, label=band, histtype='step', bins=binz)
	if mock_truth is not None:
		plt.axvline(mock_truth, linestyle='dashdot', color='r', label='Mock truth')
		plt.legend()
	plt.xlabel(xlabel+xlabel_unit)
	plt.ylabel('$N_{samp}$')
	
	if show:
		plt.show()

	return f


def plot_posterior_flux_dist(logSv, raw_number_counts, band='250 micron', title=True, show=False):

	mean_number_cts = np.mean(raw_number_counts, axis=0)
	lower = np.percentile(raw_number_counts, 16, axis=0)
	upper = np.percentile(raw_number_counts, 84, axis=0)
	f = plt.figure()
	if title:
		plt.title('Posterior Flux Density Distribution - ' +str(band))

	plt.errorbar(logSv+3, mean_number_cts, yerr=np.array([np.abs(mean_number_cts-lower), np.abs(upper - mean_number_cts)]),fmt='.', label='Posterior')
	# plt.xscale('log')
	plt.legend()
	plt.yscale('log', nonposy='clip')

	plt.xlabel('$S_{\\nu}$ - ' + str(band) + ' [mJy]')
	plt.ylim(5e-1, 5e2)
	# plt.ylim(1.8, 5)
	# plt.xlim(0.5, 300)

	if show:
		plt.show()

	return f


def plot_posterior_number_counts(logSv, lit_number_counts, trueminf=0.001, nsamp=None, band='250 micron', title=True, show=False):

	mean_number_cts = np.median(lit_number_counts, axis=0)
	lower = np.percentile(lit_number_counts, 16, axis=0)
	upper = np.percentile(lit_number_counts, 84, axis=0)
	f = plt.figure(figsize=(7,6))
	if title:
		plt.title('Posterior Flux Density Distribution \n ' +str(band), fontsize=20)

	print('mean number counts:' , mean_number_cts)
	print('lower is ', lower)
	yerrs = np.array([np.log10(mean_number_cts)-np.log10(lower), np.log10(upper) - np.log10(mean_number_cts)])

	pl_key = dict({'250 micron':'PSW', '350 micron':'PMW', '500 micron':'PLW'})


	import pandas as pd
	
	ncounts_bethermin = pd.read_csv('number_counts_spire/Bethermin_et_al_'+pl_key[band]+'.csv', header=None)
	plt.plot(np.array(ncounts_bethermin[0]), np.array(ncounts_bethermin[1]), marker='.', markersize=10, label='Bethermin et al. (2012a)', color='cyan')

	ncounts_oliver = pd.read_csv('number_counts_spire/Oliver_et_al_'+pl_key[band]+'.csv', header=None)
	plt.plot(np.array(ncounts_oliver[0]), np.array(ncounts_oliver[1]), marker='3', markersize=10, label='Oliver et al. (2010)', color='b')

	ncounts_Glenn = pd.read_csv('number_counts_spire/Glenn_et_al_'+pl_key[band]+'.csv', header=None)
	plt.plot(np.array(ncounts_Glenn[0]), np.array(ncounts_Glenn[1]), marker='+', markersize=10, label='Glenn et al. (2010)', color='limegreen')

	ncounts_SIDES_srcextract = pd.read_csv('number_counts_spire/SIDES_srcextract_'+pl_key[band]+'.csv', header=None)
	plt.plot(np.array(ncounts_SIDES_srcextract[0]), np.array(ncounts_SIDES_srcextract[1]), marker='^', markersize=10, label='SIDES Source Extraction (2017)', color='r')

	# if pl_key[band]=='PSW':
	ncounts_XIDp_srcextract = pd.read_csv('number_counts_spire/Wang_2019_XIDplus_'+pl_key[band]+'.csv', header=None)
	plt.plot(np.array(ncounts_XIDp_srcextract[0]), np.array(ncounts_XIDp_srcextract[1]), marker='4', markersize=10, label='XID+ (Wang et al. 2019)', color='y')

	ncounts_SIDES_model = pd.read_csv('number_counts_spire/SIDES_'+pl_key[band]+'.csv', header=None)
	plt.plot(np.array(ncounts_SIDES_model[0]), np.array(ncounts_SIDES_model[1]), marker='o', markersize=5, label='SIDES empirical model', color='royalblue')



	# plt.text(0.7, 4.6, 'PRELIMINARY', fontsize=20)

	print('yerrs is ', yerrs)
	yerrs[np.isinf(yerrs)] = 0
	print('yerrs is nowww', yerrs)
	if nsamp is not None:
		print('nsamp is ', nsamp, ', computing standard error')
		yerrs /= np.sqrt(nsamp)
		print('yerrs standard err is ', yerrs)
	plt.errorbar(10**(logSv+3), np.log10(mean_number_cts), yerr=yerrs, marker='x', markersize=10, capsize=5, label='PCAT posterior $\\mathcal{P}(S_{\\nu}|D)$ (uncorrected)', color='k', linewidth=2)

	# plt.yscale('log')
	plt.legend()
	plt.xlabel('$S_{\\nu}$ [mJy]', fontsize=16)
	plt.ylabel('log dN/dS.$S^{2.5}$ ($Jy^{1.5}/sr$)', fontsize=16)
	# plt.ylim(1e-1, 1e5)
	# plt.xlim(np.log10(trueminf)+3.-0.5-1.0, 2.5)


	plt.ylim(1.8, 5)
	plt.xlim(0.5, 300)
	plt.xscale('log')


	plt.tight_layout()

	if show:
		plt.show()


	return f



def plot_flux_color_posterior(fsrcs, colors, band_strs, title='Posterior Flux-density Color Distribution', titlefontsize=14, show=False, xmin=0.005, xmax=4e-1, ymin=1e-2, ymax=40, fmin=0.005, colormax=40, \
								flux_sizes=None):

	print('fmin here is', fmin)
	
	xlims=[xmin, xmax]
	ylims=[ymin, ymax]

	f_str = band_strs[0]
	col_str = band_strs[1]

	nanmask = ~np.isnan(colors)

	if flux_sizes is not None:
		zeromask = (flux_sizes > fmin)
	else:
		zeromask = (fsrcs > fmin)

	colormask = (colors < colormax)

	print('lennanmask:', len(nanmask))
	print('zeromask:', zeromask)

	print('sums are ', np.sum(nanmask), np.sum(zeromask), np.sum(colormask))

	f = plt.figure()
	if title:
		plt.title(title, fontsize=titlefontsize)

	if flux_sizes is not None:
		pt_sizes = (2e2*flux_sizes[nanmask*zeromask*colormask])**2
		print('pt sizes heere is ', pt_sizes)
		c = np.log10(flux_sizes[nanmask*zeromask*colormask])

		print('minimum c is ', np.min(flux_sizes[nanmask*zeromask*colormask]))
		# c = flux_sizes[nanmask*zeromask*colormask]

		alpha=0.1

		ylims = [0, 3.0]
		xlims = [0, 3.0]

	else:
		pt_sizes = 10
		c = 'k'
		alpha=0.01


	# bright_src_mask = (flux_sizes > 0.05)
	# not_bright_src_mask = ~bright_src_mask

	if flux_sizes is not None:
		# plt.scatter(colors[nanmask*zeromask*colormask*not_bright_src_mask], fsrcs[nanmask*zeromask*colormask*not_bright_src_mask], alpha=alpha, c=np.log10(flux_sizes[nanmask*zeromask*colormask*not_bright_src_mask]), s=(2e2*flux_sizes[nanmask*zeromask*colormask*not_bright_src_mask])**2, marker='+', label='PCAT ($F_{250} > 5$ mJy')
		# plt.scatter(colors[nanmask*zeromask*colormask*bright_src_mask], fsrcs[nanmask*zeromask*colormask*bright_src_mask], alpha=0.5, c=np.log10(flux_sizes[nanmask*zeromask*colormask*bright_src_mask]), s=(2e2*flux_sizes[nanmask*zeromask*colormask*bright_src_mask])**2, marker='+')
		plt.scatter(colors[nanmask*zeromask*colormask], fsrcs[nanmask*zeromask*colormask], alpha=alpha, c=c, s=pt_sizes, marker='.', label='PCAT ($F_{250} > 5$ mJy)')

	else:
		plt.scatter(colors[nanmask*zeromask*colormask], fsrcs[nanmask*zeromask*colormask], alpha=alpha, c=c, cmap='viridis_r', s=pt_sizes, marker='+', label='PCAT ($F_{250} > 5$ mJy)')


	if flux_sizes is not None:

		plt.scatter(0.*colors[nanmask*zeromask*colormask], fsrcs[nanmask*zeromask*colormask], alpha=1.0, c=c, s=pt_sizes, marker='+')

		cbar = plt.colorbar()
		cbar.ax.set_ylabel("$\\log_{10} S_{250}\\mu m$")
	plt.ylabel(f_str, fontsize=14)
	plt.xlabel(col_str, fontsize=14)

	plt.ylim(ylims)
	plt.xlim(xlims)

	if flux_sizes is None:
		plt.yscale('log')
		plt.xscale('log')
		plt.xlim(0, 5)

	plt.legend()

	if show:
		plt.show()

	return f

def plot_color_posterior(fsrcs, band0, band1, lam_dict, mock_truth_fluxes=None, title=True, titlefontsize=14, show=False):

	f = plt.figure()
	if title:
		plt.title('Posterior Color Distribution', fontsize=titlefontsize)

	_, bins, _ = plt.hist(fsrcs[band0].ravel()/fsrcs[band1].ravel(), histtype='step', label='Posterior', bins=np.linspace(0.01, 5, 50), density=True)
	if mock_truth_fluxes is not None:
		plt.hist(mock_truth_fluxes[band0,:].ravel()/mock_truth_fluxes[band1,:].ravel(), bins=bins, density=True, histtype='step', label='Mock Truth')

	plt.legend()
	plt.ylabel('PDF')
	plt.xlabel('$S_{'+str(lam_dict[band0])+'}\\mu m/S_{'+str(lam_dict[band1])+'} \\mu m$', fontsize=14)

	if show:
		plt.show()


	return f


def plot_residual_map(resid, mode='median', band='S', titlefontsize=14, smooth=True, smooth_sigma=3, \
					minmax_smooth=None, minmax=None, show=False, plot_refcat=False, convert_to_MJy_sr_fac=None):


	# TODO - overplot reference catalog on image

	if minmax_smooth is None:
		minmax_smooth = [-0.005, 0.005]
		minmax = [-0.005, 0.005]

	if convert_to_MJy_sr_fac is None:
		convert_to_MJy_sr_fac = 1.
		title_unit = ' [Jy/beam]'
	else:
		title_unit = ' [MJy/sr]'

	minmax_smooth = np.array(minmax_smooth)/convert_to_MJy_sr_fac
	minmax = np.array(minmax)/convert_to_MJy_sr_fac

	if mode=='median':
		title_mode = 'Median residual'
	elif mode=='last':
		title_mode = 'Last residual'

	if smooth:
		f = plt.figure(figsize=(10, 5))
	else:
		f = plt.figure(figsize=(8,8))
	
	if smooth:
		plt.subplot(1,2,1)

	plt.title(title_mode+' -- '+band+title_unit, fontsize=titlefontsize)
	plt.imshow(resid/convert_to_MJy_sr_fac, cmap='Greys', interpolation=None, vmin=minmax[0], vmax=minmax[1], origin='lower')
	plt.colorbar()

	if smooth:
		plt.subplot(1,2,2)
		plt.title('Smoothed Residual'+title_unit, fontsize=titlefontsize)
		plt.imshow(gaussian_filter(resid, sigma=smooth_sigma)/convert_to_MJy_sr_fac, interpolation=None, cmap='Greys', vmin=minmax_smooth[0], vmax=minmax_smooth[1], origin='lower')
		plt.colorbar()

	if show:
		plt.show()


	return f

def plot_residual_1pt_function(resid, mode='median', band='S', noise_model=None, show=False, binmin=-0.02, binmax=0.02, nbin=50, convert_to_MJy_sr_fac=None):

	if noise_model is not None:
		noise_model *= np.random.normal(0, 1, noise_model.shape)
		density=True
	else:
		density=False



	if convert_to_MJy_sr_fac is None:
		convert_to_MJy_sr_fac = 1.
		xlabel_unit = '[Jy/beam]'
	else:
		xlabel_unit = '[MJy/sr]'
		binmin /= convert_to_MJy_sr_fac
		binmax /= convert_to_MJy_sr_fac

	if len(resid.shape) > 1:
		median_resid_rav = resid.ravel()/convert_to_MJy_sr_fac
	else:
		median_resid_rav = resid/convert_to_MJy_sr_fac

	if mode=='median':
		title_mode = 'Median residual'
	elif mode=='last':
		title_mode = 'Last residual'
	
	f = plt.figure()
	plt.title(title_mode+' 1pt function -- '+band)
	plt.hist(median_resid_rav, bins=np.linspace(binmin, binmax, nbin), histtype='step', density=density)

	if noise_model is not None:
		plt.hist(noise_model.ravel(), bins=np.linspace(binmin, binmax, nbin), histtype='step', color='r', label='Noise model (Gaussian draw)', density=density)

	plt.axvline(np.median(median_resid_rav), label='Median='+str(np.round(np.median(median_resid_rav), 5))+'\n $\\sigma=$'+str(np.round(np.std(median_resid_rav), 5)))
	plt.legend(frameon=False)

	if density:
		plt.ylabel('PDF')
	else:
		plt.ylabel('$N_{pix}$')

	plt.xlabel('data - model '+xlabel_unit)

	if show:
		plt.show()

	return f


def plot_chi_squared(chi2, sample_number, band='S', show=False):

	burn_in = sample_number[0]
	f = plt.figure()
	plt.plot(sample_number, chi2[burn_in:], label=band)
	plt.axhline(np.min(chi2[burn_in:]), linestyle='dashed',alpha=0.5, label=str(np.min(chi2[burn_in:]))+' (' + str(band) + ')')
	plt.xlabel('Sample')
	plt.ylabel('Chi-Squared')
	plt.legend()
	
	if show:
		plt.show()

	return f


def plot_comp_resources(timestats, nsamp, labels=['Proposal', 'Likelihood', 'Implement'], show=False):
	time_array = np.zeros(3, dtype=np.float32)
	
	for samp in range(nsamp):
		time_array += np.array([timestats[samp][2][0], timestats[samp][3][0], timestats[samp][4][0]])
	
	f = plt.figure()
	plt.title('Computational Resources')
	plt.pie(time_array, labels=labels, autopct='%1.1f%%', shadow=True)
	
	if show:
		plt.show()
	
	return f

def plot_acceptance_fractions(accept_stats, proposal_types=['All', 'Move', 'Birth/Death', 'Merge/Split', 'Templates', 'Fourier Comps'], show=False, smooth_fac=None, bad_idxs=None):

	f = plt.figure()
	
	samp_range = np.arange(accept_stats.shape[0])
	for x in range(len(proposal_types)):
		if bad_idxs is not None:
			if x in bad_idxs:
				continue
		print(accept_stats[0,x])
		accept_stats[:,x][np.isnan(accept_stats[:,x])] = 0.

	# if smooth_fac is not None:
		# bkg_samples = np.convolve(bkg_samples, np.ones((smooth_fac,))/smooth_fac, mode='valid')
		if smooth_fac is not None:
			accept_stat_chain = np.convolve(accept_stats[:,x], np.ones((smooth_fac,))/smooth_fac, mode='valid')
		else:
			accept_stat_chain = accept_stats[:,x]
		# plt.plot(samp_range, accept_stats[:,x], label=proposal_types[x])

		plt.plot(np.arange(len(accept_stat_chain)), accept_stat_chain, label=proposal_types[x])
	plt.legend()
	plt.xlabel('Sample number')
	plt.ylabel('Acceptance fraction')
	if show:
		plt.show()

	return f

def plot_fluxbias_vs_flux(mean_frac_flux_error_binned, pct16_frac_flux_error_binned, pct84_frac_flux_error_binned, fluxbins,\
						 band=0, nsrc_perfbin=None, xlim = [2, 700], ylim=[-1.5, 2.5], title=None, titlefontsize=18, verbose=True, load_jank_txts=True, fractional_bias=False):



	g = plt.figure(figsize=(9, 6))
	if title is not None:
		plt.title(title, fontsize=titlefontsize)

	plt.axhline(0.0, linestyle='solid', color='grey', alpha=0.4, linewidth=1.5, zorder=-10)


	geom_mean = np.sqrt(fluxbins[1:]*fluxbins[:-1])
	xerrs = [[1e3*(geom_mean[f] - fluxbins[f]) for f in range(len(geom_mean))], [1e3*(fluxbins[f+1] - geom_mean[f]) for f in range(len(geom_mean))]]

	sqrt_nsrc_perfbin = np.sqrt(np.array(nsrc_perfbin))
	if verbose:
		print('for band '+str(band)+', nsrc_perfbin is ')
		print(nsrc_perfbin)
		print('sqrt nsrc perfbin :', sqrt_nsrc_perfbin)
	yerr = [(mean_frac_flux_error_binned-pct16_frac_flux_error_binned)/sqrt_nsrc_perfbin, (pct84_frac_flux_error_binned-mean_frac_flux_error_binned)/sqrt_nsrc_perfbin]


	plt.errorbar(geom_mean*1e3, mean_frac_flux_error_binned, xerr=xerrs, \
				 yerr=yerr, fmt='.', color='C3', \
				 linewidth=3, capsize=5, alpha=1., capthick=2, label='PCAT (GOODS-N) three-band fit \n mean, error on mean', markersize=15)


	plt.fill_between(geom_mean*1e3, pct16_frac_flux_error_binned, pct84_frac_flux_error_binned, color='C3', \
				 alpha=0.3, label='PCAT (GOODS-N) three-band fit \n 1$\\sigma$ scatter')


	lamstrs = ['250', '350', '500']

	if load_jank_txts:
		roseboom_medianx = np.array(pd.read_csv('~/Downloads/median_roseboom_'+lamstrs[band]+'.csv', header=None)[0])
		roseboom_mediany = np.array(pd.read_csv('~/Downloads/median_roseboom_'+lamstrs[band]+'.csv', header=None)[1])
		rb_pct16x = np.array(pd.read_csv('~/Downloads/roseboom_16_'+lamstrs[band]+'.csv', header=None)[0])
		rb_pct16y = np.array(pd.read_csv('~/Downloads/roseboom_16_'+lamstrs[band]+'.csv', header=None)[1])
		rb_pct84x = np.array(pd.read_csv('~/Downloads/roseboom_84_'+lamstrs[band]+'.csv', header=None)[0])
		rb_pct84y = np.array(pd.read_csv('~/Downloads/roseboom_84_'+lamstrs[band]+'.csv', header=None)[1])
		xid_medianx = np.array(pd.read_csv('~/Downloads/xid_median_'+lamstrs[band]+'.csv', header=None)[0])
		xid_mediany = np.array(pd.read_csv('~/Downloads/xid_median_'+lamstrs[band]+'.csv', header=None)[1])
		xid_pct16x = np.array(pd.read_csv('~/Downloads/xid_16_'+lamstrs[band]+'.csv', header=None)[0])
		xid_pct16y = np.array(pd.read_csv('~/Downloads/xid_16_'+lamstrs[band]+'.csv', header=None)[1])
		xid_pct84x = np.array(pd.read_csv('~/Downloads/xid_84_'+lamstrs[band]+'.csv', header=None)[0])
		xid_pct84y = np.array(pd.read_csv('~/Downloads/xid_84_'+lamstrs[band]+'.csv', header=None)[1])

		roseboom_y = roseboom_mediany.copy()

		plt.plot(roseboom_medianx, roseboom_y, label='XID (Deep) \n Roseboom et al. 2010', color='C2', marker='.', linewidth=3, markersize=15)
		plt.fill_between(rb_pct16x, rb_pct16y/roseboom_medianx, rb_pct84y/roseboom_medianx, color='C2', alpha=0.3)
		plt.plot(xid_medianx, xid_mediany, label='XID+ (COSMOS) \n Hurley et al. 2016', color='b', marker='.', linewidth=3, markersize=15)
		plt.fill_between(xid_pct16x, xid_pct16y, xid_pct84y, color='b', alpha=0.3)
		plt.legend(fontsize=13)

	if band==0:
		plt.xlabel('$S_{250}^{True}$ [mJy]', fontsize=18)
		if fractional_bias:
			plt.ylabel('$(S_{250}^{Obs} - S_{250}^{True})/S_{250}^{True}$', fontsize=16)
		else:
			plt.ylabel('$S_{250}^{Obs} - S_{250}^{True}$ [mJy]', fontsize=16)

	elif band==1:
		plt.xlabel('$S_{350}^{True}$ [mJy]', fontsize=18)
		if fractional_bias:
			plt.ylabel('$(S_{350}^{Obs} - S_{350}^{True})/S_{350}^{True}$', fontsize=16)		
		else:
			plt.ylabel('$S_{350}^{Obs} - S_{350}^{True}$ [mJy]', fontsize=16)		

	elif band==2:
		plt.xlabel('$S_{500}^{True}$ [mJy]', fontsize=18)
		if fractional_bias:
			plt.ylabel('$(S_{500}^{Obs} - S_{500}^{True})/S_{500}^{True}$', fontsize=16)	
		else:
			plt.ylabel('$S_{500}^{Obs} - S_{500}^{True}$ [mJy]', fontsize=16)	

	plt.xscale('log')
	plt.ylim(ylim)
	plt.xlim(xlim)

	plt.tight_layout()

	plt.show()


	return g, mean_frac_flux_error_binned, yerr, geom_mean
	# g.savefig(filepath+'/fluxerr_vs_fluxdensity_dx='+str(pos_thresh)+'_dS-S='+str(frac_flux_thresh)+'_mindist.pdf', bbox_inches='tight')


def plot_completeness_vs_flux(pos_thresh, frac_flux_thresh, fluxbins, completeness_vs_flux, cvf_stderr, image=None, catalog_inject=None, xstack=None, ystack=None, fstack=None, multiband=True, band=None, nbands=3, \
								title=None, titlefontsize=16, maxp=95, minp=5, colorbar=True, cbar_label='MJy/sr', show=True):

	if image is not None:
		f = plt.figure(figsize=(10, 5.5))

		plt.subplot(1,2,1)
		if title is not None:
			plt.title(title, fontsize=titlefontsize)

		plt.imshow(image, cmap='Greys', vmax=np.percentile(image, maxp), vmin=np.percentile(image, minp), origin='lower')
		plt.xlabel('x [pix]', fontsize=14)
		plt.ylabel('y [pix]', fontsize=14)

		if colorbar:
			cbar = plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.12)
			cbar.set_label(cbar_label, fontsize=14)

		if catalog_inject is not None:
			plt.scatter(catalog_inject[:,0], catalog_inject[:, 1], marker='+', color='b', s=5e2*catalog_inject[:,2], alpha=0.5, label='Injected sources')

		if xstack is not None and ystack is not None:
			plt.scatter(xstack, ystack, marker='x', color='r', s=5e2*fstack, alpha=0.05)
			plt.scatter([0], [0], marker='x', color='r', label='PCAT')


		plt.legend(loc=1)
		plt.xlim(0, image.shape[0]-5)
		plt.ylim(0, image.shape[0]-5)

		plt.subplot(1,2,2)

	else:

		f = plt.figure()

	plt.title('$|\\delta \\vec{x}| < $'+str(np.round(pos_thresh, 2))+', $|\\delta S/S| < $'+str(np.round(frac_flux_thresh, 2)), fontsize=16)

	lamstrs = ['250', '350', '500']

	if multiband:
		for b in range(nbands):

			plt.errorbar(1e3*np.sqrt(fluxbins[:-1]*fluxbins[1:]), completeness_vs_flux[b], yerr=cvf_stderr[b], color='C'+str(b), marker='x', capsize=5, label='PCAT ($\\lambda = $'+lamstrs[b]+'$\\mu m$)')



	plt.plot([10, 20, 30, 40, 50, 60, 200, 500], [0.14, 0.3, 0.67, 0.91,0.99, 0.995, 1.0, 1.0], marker='.', markersize=10, label='SPIRE PSC (COSMOS) \n250 $\\mu m$', color='k')
	plt.plot([10, 20, 30, 40, 50, 60, 200, 500], [0.25, 0.5, 0.81, 0.95, 0.99, 0.995, 1.0, 1.0], marker='.', markersize=10, label='350 $\\mu m$', color='k', linestyle='dashdot')
	plt.plot([10, 20, 30, 40, 50, 60, 200, 500], [0.15, 0.35, 0.66, 0.87, 0.96, 0.99, 1.0, 1.0], marker='.', markersize=10, label='500 $\\mu m$', color='k', linestyle='dashed')
	


	plt.xscale('log')
	plt.xlabel('$S$ [mJy]', fontsize=14)
	# plt.xlabel('$S_{250}$ [mJy]', fontsize=14)
	plt.ylabel('Completeness', fontsize=14)
	plt.xlim(2, 1e3)
	# plt.xlim(15, 1e3)
	plt.legend()
	plt.ylim(-0.05, 1.05)
	plt.tight_layout()

	if show:
		plt.show()

	return f

def plot_src_number_posterior(nsrc_fov, show=False, title=False, nsrc_truth=None, fmin=4.0, units='mJy'):

	f = plt.figure()
	
	if title:
		plt.title('Posterior Source Number Histogram')
	
	plt.hist(nsrc_fov, histtype='step', label='Posterior', color='b', bins=15)
	plt.axvline(np.median(nsrc_fov), label='Median=' + str(np.median(nsrc_fov)), color='b', linestyle='dashed')
	if nsrc_truth is not None:
		plt.axvline(nsrc_truth, label='N (F > '+str(fmin)+' mJy) = '+str(nsrc_truth), linestyle='dashed', color='k', linewidth=1.5)
	plt.xlabel('$N_{src}$', fontsize=16)
	plt.ylabel('Number of samples', fontsize=16)
	plt.legend()
		
	if show:
		plt.show()
	
	return f


def plot_src_number_trace(nsrc_fov, show=False, title=False):

	f = plt.figure()
	
	if title:
		plt.title('Source number trace plot (post burn-in)')
	
	plt.plot(np.arange(len(nsrc_fov)), nsrc_fov)

	plt.xlabel('Sample index', fontsize=16)
	plt.ylabel('$N_{src}$', fontsize=16)
	plt.legend()
	
	if show:
		plt.show()

	return f


def plot_grap():

	'''
	Makes plot of probabilistic graphical model for SPIRE
	'''
		
	figr, axis = plt.subplots(figsize=(6, 6))

	grap = nx.DiGraph()   
	grap.add_edges_from([('muS', 'svec'), ('sigS', 'svec'), ('alpha', 'f0'), ('beta', 'nsrc')])
	grap.add_edges_from([('back', 'modl'), ('xvec', 'modl'), ('f0', 'modl'), ('svec', 'modl'), ('PSF', 'modl'), ('ASZ', 'modl')])
	grap.add_edges_from([('modl', 'data')])
	listcolr = ['black' for i in range(7)]
	
	labl = {}

	nameelem = r'\rm{pts}'


	labl['beta'] = r'$\beta$'
	labl['alpha'] = r'$\alpha$'
	labl['muS'] = r'$\vec{\mu}_S$'
	labl['sigS'] = r'$\vec{\sigma}_S$'
	labl['xvec'] = r'$\vec{x}$'
	labl['f0'] = r'$F_0$'
	labl['svec'] = r'$\vec{s}$'
	labl['PSF'] = r'PSF'
	labl['modl'] = r'$M_D$'
	labl['data'] = r'$D$'
	labl['back'] = r'$\vec{A_{sky}}$'
	labl['nsrc'] = r'$N_{src}$'
	labl['ASZ'] = r'$\vec{A_{SZ}}$'
	
	
	posi = nx.circular_layout(grap)
	posi['alpha'] = np.array([-0.025, 0.15])
	posi['muS'] = np.array([0.025, 0.15])
	posi['sigS'] = np.array([0.075, 0.15])
	posi['beta'] = np.array([0.12, 0.15])

	posi['xvec'] = np.array([-0.075, 0.05])
	posi['f0'] = np.array([-0.025, 0.05])
	posi['svec'] = np.array([0.025, 0.05])
	posi['PSF'] = np.array([0.07, 0.05])
	posi['back'] = np.array([-0.125, 0.05])
	
	posi['modl'] = np.array([-0.05, -0.05])
	posi['data'] = np.array([-0.05, -0.1])
	posi['nsrc'] = np.array([0.08, 0.01])
	
	posi['ASZ'] = np.array([-0.175, 0.05])


	rect = patches.Rectangle((-0.10,0.105),0.2,-0.11,linewidth=2, facecolor='none', edgecolor='k')

	axis.add_patch(rect)

	size = 1000
	nx.draw(grap, posi, labels=labl, ax=axis, edgelist=grap.edges())
	nx.draw_networkx_nodes(grap, posi, ax=axis, labels=labl, nodelist=['nsrc'], node_color='white', node_size=500)

	nx.draw_networkx_nodes(grap, posi, ax=axis, labels=labl, nodelist=['modl'], node_color='xkcd:sky blue', node_size=1000)
	nx.draw_networkx_nodes(grap, posi, ax=axis, labels=labl, nodelist=['beta'], node_shape='d', node_color='y', node_size=size)
	
	nx.draw_networkx_nodes(grap, posi, ax=axis, labels=labl, nodelist=['data'],  node_color='grey', node_shape='s', node_size=size)
	nx.draw_networkx_nodes(grap, posi, ax=axis, labels=labl, nodelist=['xvec', 'f0', 'svec'], node_color='orange', node_size=size)
	nx.draw_networkx_nodes(grap, posi, ax=axis, labels=labl, nodelist=['PSF'], node_shape='d', node_color='orange', node_size=size)
	
	nx.draw_networkx_nodes(grap, posi, ax=axis, labels=labl, nodelist=['back'], node_color='orange', node_size=size)
	
	nx.draw_networkx_nodes(grap, posi, ax=axis, labels=labl, nodelist=['ASZ'], node_color='violet', node_size=size)

	nx.draw_networkx_nodes(grap, posi, ax=axis, labels=labl, nodelist=['alpha', 'muS', 'sigS'], node_shape='d', node_color='y', node_size=size)

	
	plt.tight_layout()
	plt.show()
	
	return figr

def grab_atcr(timestr, paramstr='template_amplitudes', band=0, result_dir=None, nsamp=500, template_idx=0, return_fig=True):
	
	band_dict = dict({0:'250 micron', 1:'350 micron', 2:'500 micron'})
	lam_dict = dict({0:250, 1:350, 2:500})

	if result_dir is None:
		result_dir = '/Users/richardfeder/Documents/multiband_pcat/spire_results/'
		print('Result directory assumed to be '+result_dir)
		
	chain = np.load(result_dir+str(timestr)+'/chain.npz')
	if paramstr=='template_amplitudes':
		listsamp = chain[paramstr][-nsamp:, band, template_idx]
	else:
		listsamp = chain[paramstr][-nsamp:, band]
		
	f = plot_atcr(listsamp, title=paramstr+', '+band_dict[band])

	if return_fig:
		return f



def result_plots(timestr=None, burn_in_frac=0.8, boolplotsave=True, boolplotshow=False, \
				plttype='png', gdat=None, cattype='SIDES', min_flux_refcat=1e-4, dpi=150, flux_density_unit='MJy/sr', \
				accept_fraction_plots=True, chi2_plots=True, dc_background_plots=True, fourier_comp_plots=True, \
				template_plots=True, flux_dist_plots=True, flux_color_plots=False, flux_color_color_plots=False, \
				comp_resources_plot=True, source_number_plots=True, residual_plots=True, condensed_catalog_plots=False, condensed_catalog_fpath=None, generate_condensed_cat=False, \
				n_condensed_samp=None, prevalence_cut=None, mask_hwhm=None, search_radius=None, matching_dist=None, truth_catalog=None):

	
	title_band_dict = dict({0:'250 micron', 1:'350 micron', 2:'500 micron'})
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
		gdat.filepath = gdat.result_path + gdat.timestr

	if truth_catalog is None:
		if gdat.truth_catalog is not None:
			truth_catalog = gdat.truth_catalog

	if matching_dist is not None:
		gdat.matching_dist = matching_dist
	if n_condensed_samp is not None:
		gdat.n_condensed_samp = n_condensed_samp
	if prevalence_cut is not None:
		gdat.prevalence_cut = prevalence_cut
	if mask_hwhm is not None:
		gdat.mask_hwhm = mask_hwhm
	if search_radius is not None:
		gdat.search_radius = search_radius

	condensed_cat = None

	if condensed_catalog_plots:

		condensed_cat_dir = add_directory(gdat.filepath+'/condensed_catalog')


		if generate_condensed_cat:
			print('Generating condensed catalog from last '+str(gdat.n_condensed_samp)+' samples of catalog ensemble..')
			print('prevalence_cut = '+str(gdat.prevalence_cut))
			print('search_radius = '+str(gdat.search_radius))
			print('mask_hwhm = '+str(mask_hwhm))

			xmatch_roc = cross_match_roc(timestr=gdat.timestr, nsamp=gdat.n_condensed_samp)
			xmatch_roc.load_chain(result_path+'/'+timestr+'/chain.npz')
			xmatch_roc.load_gdat_params(gdat=gdat)
			condensed_cat, seed_cat = xmatch_roc.condense_catalogs(prevalence_cut=gdat.prevalence_cut, save_cats=True, make_seed_bool=True, mask_hwhm=gdat.mask_hwhm, search_radius=gdat.search_radius)

			np.savetxt(gdat.result_path+'/'+gdat.timestr+'/condensed_catalog_nsamp='+str(gdat.n_condensed_samp)+'_prevcut='+str(gdat.prevalence_cut)+'_searchradius='+str(gdat.search_radius)+'_maskhwhm='+str(gdat.mask_hwhm)+'.txt', condensed_cat)
			np.savetxt(gdat.result_path+'/'+gdat.timestr+'/raw_seed_catalog_nsamp='+str(gdat.n_condensed_samp)+'_matching_dist='+str(gdat.matching_dist)+'_maskhwhm='+str(gdat.mask_hwhm)+'.txt', seed_cat)

		else:
			if condensed_catalog_fpath is None:
				condensed_catalog_fpath = gdat.result_path+'/'+gdat.timestr+'/condensed_catalog_nsamp='+str(gdat.n_condensed_samp)+'_prevcut='+str(gdat.prevalence_cut)+'_searchradius='+str(gdat.search_radius)+'_maskhwhm='+str(gdat.mask_hwhm)+'.txt'

			condensed_cat = np.loadtxt(condensed_catalog_fpath)

			print('condensed_cat has shape ', condensed_cat.shape)
			print(condensed_cat[:, 5])
			print(condensed_cat[:, 9])
			print(condensed_cat[:, 13])
			print(condensed_cat[:, 6])

	gdat.show_input_maps=False
	datapath = gdat.base_path+'/Data/spire/'+gdat.dataname+'/'

	for i, band in enumerate(gdat.bands):

		if gdat.mock_name is not None:

			if cattype=='SIDES':
				ref_path = datapath+'sides_cat_P'+gdat.band_dict[band]+'W_20.npy'
				roc.load_cat(path=ref_path)
				if i==0:
					cat_fluxes = np.zeros(shape=(gdat.nbands, len(roc.mock_cat['flux'])))
				cat_fluxes[i,:] = roc.mock_cat['flux']
		else:
			cat_fluxes=None


	dat = pcat_data(gdat.auto_resize, nregion=gdat.nregion)
	dat.load_in_data(gdat)

	chain = np.load(gdat.filepath+'/chain.npz')

	flux_density_conversion_dict = dict({'S': 86.29e-4, 'M':16.65e-3, 'L':34.52e-3})

	fd_conv_fac = None # if units are MJy/sr this changes to a number, otherwise default flux density units are mJy/beam
	nsrcs = chain['n']
	xsrcs = chain['x']
	ysrcs = chain['y']
	fsrcs = chain['f']
	chi2 = chain['chi2']
	timestats = chain['times']
	accept_stats = chain['accept']
	diff2s = chain['diff2s']

	burn_in = int(gdat.nsamp*burn_in_frac)
	bands = gdat.bands
	print('Bands are ', bands)

	if gdat.float_background is not None:
		bkgs = chain['bkg']

	if gdat.float_templates is not None:
		template_amplitudes = chain['template_amplitudes']

	if gdat.float_fourier_comps is not None: # fourier comps
		fourier_coeffs = chain['fourier_coeffs']

	# ------------------- mean residual ---------------------------

	if residual_plots:

		for b in range(gdat.nbands):
			residz = chain['residuals'+str(b)]

			median_resid = np.median(residz, axis=0)

			print('Median residual has shape '+str(median_resid.shape))
			smoothed_resid = gaussian_filter(median_resid, sigma=3)

			minpct = np.percentile(median_resid[dat.weights[b] != 0.], 5.)
			maxpct = np.percentile(median_resid[dat.weights[b] != 0.], 95.)

			# minpct_smooth = np.percentile(smoothed_resid[dat.weights[b] != 0.], 5.)
			# maxpct_smooth = np.percentile(smoothed_resid[dat.weights[b] != 0.], 99.)
			maxpct_smooth = 0.002
			minpct_smooth = -0.002

			if b==0:
				resid_map_dir = add_directory(gdat.filepath+'/residual_maps')
				onept_dir = add_directory(gdat.filepath+'/residual_1pt')

			if flux_density_unit=='MJy/sr':
				fd_conv_fac = flux_density_conversion_dict[gdat.band_dict[bands[b]]]
				print('fd conv fac is ', fd_conv_fac)
			

			f_last = plot_residual_map(residz[-1], mode='last', band=title_band_dict[bands[b]], minmax_smooth=[minpct_smooth, maxpct_smooth], minmax=[minpct, maxpct], show=boolplotshow, convert_to_MJy_sr_fac=None)
			f_last.savefig(resid_map_dir +'/last_residual_and_smoothed_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)


			f_median = plot_residual_map(median_resid, mode='median', band=title_band_dict[bands[b]], minmax_smooth=[minpct_smooth, maxpct_smooth], minmax=[minpct, maxpct], show=boolplotshow, convert_to_MJy_sr_fac=fd_conv_fac)
			f_median.savefig(resid_map_dir +'/median_residual_and_smoothed_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)

			median_resid_rav = median_resid[dat.weights[b] != 0.].ravel()

			noise_mod = dat.errors[b]

			f_1pt_resid = plot_residual_1pt_function(median_resid_rav, mode='median', noise_model=noise_mod, band=title_band_dict[bands[b]], show=False, convert_to_MJy_sr_fac=None)
			f_1pt_resid.savefig(onept_dir +'/median_residual_1pt_function_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)

			plt.close()	

	# -------------------- CHI2 ------------------------------------

	if chi2_plots:
		sample_number = np.arange(burn_in, gdat.nsamp)
		full_sample = range(gdat.nsamp)

		chi2_dir = add_directory(gdat.filepath+'/chi2')

		
		for b in range(gdat.nbands):

			fchi = plot_chi_squared(chi2[:,b], sample_number, band=title_band_dict[bands[b]], show=False)
			fchi.savefig(chi2_dir + '/chi2_sample_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)

			plt.close()

	# ------------------------- BACKGROUND AMPLITUDE ---------------------

	if gdat.float_background and dc_background_plots:

		bkg_dir = add_directory(gdat.filepath+'/bkg')

		for b in range(gdat.nbands):

			if flux_density_unit=='MJy/sr':
				fd_conv_fac = flux_density_conversion_dict[gdat.band_dict[bands[b]]]
				print('fd conv fac is ', fd_conv_fac)

			f_bkg_chain = plot_bkg_sample_chain(bkgs[:,b], band=title_band_dict[bands[b]], show=False, convert_to_MJy_sr_fac=fd_conv_fac)
			f_bkg_chain.savefig(bkg_dir+'/bkg_amp_chain_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)
			
			if gdat.nsamp > 50:
				f_bkg_atcr = plot_atcr(bkgs[burn_in:, b], title='Background level, '+title_band_dict[bands[b]])
				f_bkg_atcr.savefig(bkg_dir+'/bkg_amp_autocorr_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)

			f_bkg_post = plot_posterior_bkg_amplitude(bkgs[burn_in:,b], band=title_band_dict[bands[b]], show=False, convert_to_MJy_sr_fac=None)
			f_bkg_post.savefig(bkg_dir+'/bkg_amp_posterior_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)


	# ------------------------- FOURIER COMPONENTS ----------------------

	if gdat.float_fourier_comps and fourier_comp_plots:

		fc_dir = add_directory(gdat.filepath+'/fourier_comps')

		# median and variance of fourier component model posterior
		print('Computing Fourier component posterior..')
		f_fc_median_std = plot_fc_median_std(fourier_coeffs[burn_in:], gdat.imszs[0], ref_img=dat.data_array[0], convert_to_MJy_sr_fac=flux_density_conversion_dict['S'], psf_fwhm=3.)
		f_fc_median_std.savefig(fc_dir+'/fourier_comp_model_median_std.'+plttype, bbox_inches='tight', dpi=dpi)

		f_fc_last = plot_last_fc_map(fourier_coeffs[-1], gdat.imszs[0],ref_img=dat.data_array[0])
		f_fc_last.savefig(fc_dir+'/last_sample_fourier_comp_model.'+plttype, bbox_inches='tight', dpi=dpi)

		# covariance matrix of fourier components

		f_fc_covariance = plot_fourier_coeffs_covariance_matrix(fourier_coeffs[burn_in:])
		f_fc_covariance.savefig(fc_dir+'/fourier_coeffs_covariance_matrix.'+plttype, bbox_inches='tight', dpi=dpi)

		# sample chain for fourier coeffs

		f_fc_amp_chain = plot_fourier_coeffs_sample_chains(fourier_coeffs)
		f_fc_amp_chain.savefig(fc_dir+'/fourier_coeffs_sample_chains.'+plttype, bbox_inches='tight', dpi=dpi)

		# posterior power spectrum of fourier component model
		f_fc_ps = plot_posterior_fc_power_spectrum(fourier_coeffs[burn_in:], gdat.imszs[0][0])
		f_fc_ps.savefig(fc_dir+'/posterior_bkg_power_spectrum.'+plttype, bbox_inches='tight', dpi=dpi)

	
	# ------------------------- TEMPLATE AMPLITUDES ---------------------

	if gdat.float_templates and template_plots:

		template_dir = add_directory(gdat.filepath+'/templates')

		for t in range(gdat.n_templates):
			print('looking at template with name ', gdat.template_order[t])
			for b in range(gdat.nbands):

				if flux_density_unit=='MJy/sr':
					fd_conv_fac = flux_density_conversion_dict[gdat.band_dict[bands[b]]]
					print('fd conv fac is ', fd_conv_fac)

				if not np.isnan(gdat.template_band_idxs[t,b]):

					if gdat.template_order[t]=='dust' or gdat.template_order[t]=='planck':
						f_temp_amp_chain = plot_template_amplitude_sample_chain(template_amplitudes[:, t, b], template_name=gdat.template_order[t], band=title_band_dict[bands[b]], ylabel='Relative amplitude', convert_to_MJy_sr_fac=None) # newt
						f_temp_amp_post = plot_posterior_template_amplitude(template_amplitudes[burn_in:, t, b], template_name=gdat.template_order[t], band=title_band_dict[bands[b]], xlabel='Relative amplitude', convert_to_MJy_sr_fac=None) # newt
					
						f_temp_median_and_variance = plot_template_median_std(dat.template_array[b][t], template_amplitudes[burn_in:, t, b], template_name=gdat.template_order[t], band=title_band_dict[bands[b]], show=False, convert_to_MJy_sr_fac=fd_conv_fac)
						
						f_temp_median_and_variance.savefig(template_dir+'/'+gdat.template_order[t]+'_template_median_std_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)

					else:
						mock_truth = None
						if gdat.template_order[t]=='sze':
							temp_mock_amps_dict = dict({'S':0.0111, 'M': 0.1249, 'L': 0.6912})
							mock_truth = None
							if gdat.inject_sz_frac is not None:
								mock_truth = temp_mock_amps_dict[gdat.band_dict[bands[b]]]*gdat.inject_sz_frac
								print('mock truth is ', mock_truth)


						if gdat.integrate_sz_prof:

							pixel_sizes = dict({'S':6, 'M':8, 'L':12}) # arcseconds

							npix = dat.template_array[b][t].shape[0]*dat.template_array[b][t].shape[1]
							geom_fac = (np.pi*pixel_sizes[gdat.band_dict[bands[b]]]/(180.*3600.))**2
							print('geometric factor is ', geom_fac)
							print('integrating sz profiles..')

							template_flux_densities = np.array([np.sum(amp*dat.template_array[b][t]) for amp in template_amplitudes[burn_in:, t, b]])

							print('template_flux densities:', template_flux_densities)

							if fd_conv_fac is not None:
								template_flux_densities /= fd_conv_fac

							template_flux_densities *= geom_fac

							template_flux_densities *= 1e6 # MJy to Jy
							
							if gdat.template_moveweight > 0:
								f_temp_amp_chain = plot_template_amplitude_sample_chain(template_amplitudes[:, t, b], template_name=gdat.template_order[t], band=title_band_dict[bands[b]], convert_to_MJy_sr_fac=fd_conv_fac) 
						
								f_temp_amp_post = plot_posterior_template_amplitude(template_flux_densities, mock_truth=mock_truth,  template_name=gdat.template_order[t], band=title_band_dict[bands[b]], xlabel_unit='[Jy]') 

						else:
							if gdat.template_moveweight > 0:
								f_temp_amp_chain = plot_template_amplitude_sample_chain(template_amplitudes[:, t, b], template_name=gdat.template_order[t], band=title_band_dict[bands[b]], convert_to_MJy_sr_fac=fd_conv_fac)
								f_temp_amp_post = plot_posterior_template_amplitude(template_amplitudes[burn_in:, t, b],mock_truth=mock_truth,	template_name=gdat.template_order[t], band=title_band_dict[bands[b]], convert_to_MJy_sr_fac=fd_conv_fac)


					if gdat.template_order != 'sze' or b > 0: # test specific
						f_temp_amp_chain.savefig(template_dir+'/'+gdat.template_order[t]+'_template_amp_chain_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)
						f_temp_amp_post.savefig(template_dir+'/'+gdat.template_order[t]+'_template_amp_posterior_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)
					if gdat.nsamp > 50:
						f_temp_amp_atcr = plot_atcr(template_amplitudes[burn_in:, t, b], title='Template amplitude, '+gdat.template_order[t]+', '+title_band_dict[bands[b]]) # newt
						f_temp_amp_atcr.savefig(template_dir+'/'+gdat.template_order[t]+'_template_amp_autocorr_band'+str(b)+'.'+plttype, bbox_inches='tight', dpi=dpi)


	# ---------------------------- COMPUTATIONAL RESOURCES --------------------------------

	if comp_resources_plot:
		labels = ['Proposal', 'Likelihood', 'Implement']

		f_comp = plot_comp_resources(timestats, gdat.nsamp, labels=labels)
		f_comp.savefig(gdat.filepath+ '/time_resource_statistics.'+plttype, bbox_inches='tight', dpi=dpi)
		plt.close()

	# ------------------------------ ACCEPTANCE FRACTION -----------------------------------------
	
	if accept_fraction_plots:	

		proposal_types = ['All', 'Move', 'Birth/Death', 'Merge/Split', 'Background', 'Templates', 'Fourier comps']

		bad_idxs = []
		if not gdat.float_background:
			bad_idxs.append(4)
		if not gdat.float_templates:
			bad_idxs.append(5)
		if not gdat.float_fourier_comps:
			bad_idxs.append(6)

		print('proposal types:', proposal_types)
		print('accept_stats is ', accept_stats)
		f_proposal_acceptance = plot_acceptance_fractions(accept_stats, proposal_types=proposal_types, smooth_fac=10, bad_idxs=bad_idxs)
		f_proposal_acceptance.savefig(gdat.filepath+'/acceptance_fraction.'+plttype, bbox_inches='tight', dpi=dpi)


	# -------------------------------- ITERATE OVER BANDS -------------------------------------

	nsrc_fov = []
	color_lin_post_bins = np.linspace(0.0, 5.0, 30)

	flux_color_dir = add_directory(gdat.filepath+'/fluxes_and_colors')

	pairs = []

	fov_sources = [[] for x in range(gdat.nbands)]

	ndeg = None

	for b in range(gdat.nbands):

		color_lin_post = []
		residz = chain['residuals'+str(b)]
		median_resid = np.median(residz, axis=0)

		nbins = 20
		lit_number_counts = np.zeros((gdat.nsamp - burn_in, nbins-1)).astype(np.float32)
		raw_number_counts = np.zeros((gdat.nsamp - burn_in, nbins-1)).astype(np.float32)
		binz = np.linspace(np.log10(gdat.trueminf)+3.-1., 3., nbins)
		weight = dat.weights[b]
		
		pixel_sizes_nc = dict({0:6, 1:8, 2:12}) # arcseconds
		ratio = pixel_sizes_nc[b]/pixel_sizes_nc[0]

		


		if condensed_catalog_plots:

			if b > 0:
				# xp, yp = obj.dat.fast_astrom.transform_q(obj.stars[obj._X, 0:obj.n], obj.stars[obj._Y, 0:obj.n], band_idx-1)
				dat.fast_astrom.fit_astrom_arrays(0, b, bounds0=gdat.bounds[0], bounds1=gdat.bounds[b])

				xp, yp = dat.fast_astrom.transform_q(condensed_cat[:,0], condensed_cat[:,2], b-1)

				xp[xp > dat.errors[b].shape[0]] = dat.errors[b].shape[0]-1.
				yp[yp > dat.errors[b].shape[1]] = dat.errors[b].shape[1]-1.

				print(xp)
				print(yp)


			psf_fwhm = 3.
			optimal_ferr_map = np.sqrt(dat.errors[b]**2/(4*np.pi*(psf_fwhm/2.355)**2))
			smoothed_optimal_ferr_map = gaussian_filter(optimal_ferr_map, 5)
			flux_err_idx = 6+4*b
			flux_deg_fac = np.zeros_like(condensed_cat[:,0])
			for s, src in enumerate(condensed_cat):
				if b > 0:

					flux_deg_fac[s] = src[flux_err_idx]/smoothed_optimal_ferr_map[int(np.floor(xp[s])), int(np.floor(yp[s]))]
				else:
					flux_deg_fac[s] = src[flux_err_idx]/smoothed_optimal_ferr_map[int(src[0]), int(src[1])]
				

			# plt.figure()
			# plt.imshow(dat.errors[b])
			# plt.colorbar()
			# plt.title('band '+str(b))
			# plt.show()
			# flux_deg_fac = compute_degradation_fac(condensed_cat, dat.errors[b], flux_err_idx = 5+4*b)
			fdf_plot = plot_degradation_factor_vs_flux(1e3*condensed_cat[:,5+4*b], flux_deg_fac, deg_fac_mode='Flux')
			fdf_plot.savefig(condensed_cat_dir + '/flux_deg_fac_vs_flux_'+str(title_band_dict[bands[b]]) + '.'+plttype, bbox_inches='tight', dpi=dpi)

			flux_vs_fluxerr_plot = plot_flux_vs_fluxerr(1e3*condensed_cat[:,5+4*b], 1e3*condensed_cat[:,6+4*b])
			flux_vs_fluxerr_plot.savefig(condensed_cat_dir+'/flux_vs_fluxerr_'+str(title_band_dict[bands[b]])+'.'+plttype, bbox_inches='tight', dpi=dpi)




		if flux_dist_plots:
			for i, j in enumerate(np.arange(burn_in, gdat.nsamp)):
		
				# fsrcs_in_fov = np.array([fsrcs[b][j][k] for k in range(nsrcs[j]) if dat.weights[0][int(ysrcs[j][k]),int(xsrcs[j][k])] != 0. and xsrcs[j][k] < dat.weights[0].shape[0]-10. and ysrcs[j][k] < dat.weights[0].shape[1]-10.])
				fsrcs_in_fov = np.array([fsrcs[b][j][k] for k in range(nsrcs[j]) if dat.weights[0][int(ysrcs[j][k]),int(xsrcs[j][k])] != 0. and xsrcs[j][k] < dat.weights[0].shape[0] and ysrcs[j][k] < dat.weights[0].shape[1]])

				fov_sources[b].extend(fsrcs_in_fov)

				if b==0:
					nsrc_fov.append(len(fsrcs_in_fov))

				hist = np.histogram(np.log10(fsrcs_in_fov)+3, bins=binz)
				logSv = 0.5*(hist[1][1:]+hist[1][:-1])-3
				binz_Sz = 10**(binz-3)
				dSz = binz_Sz[1:]-binz_Sz[:-1]
				dNdS = hist[0]
				raw_number_counts[i,:] = hist[0]
				npix = median_resid.shape[0]-(10.*ratio)
				# assert npix==median_resid.shape[1]


				nsidedeg = npix*pixel_sizes_nc[b]/3600.
				# n_steradian = nsidedeg**2/(180./np.pi)**2
				
				n_steradian = 0.1/(180./np.pi)**2 # field covers 0.11 degrees, should change this though for different fields
				n_steradian *= gdat.frac # a number of pixels in the image are not actually observing anything
				dNdS_S_twop5 = dNdS*(10**(logSv))**(2.5)
				lit_number_counts[i,:] = dNdS_S_twop5/n_steradian/dSz

			f_post_number_cts = plot_posterior_number_counts(logSv, lit_number_counts, trueminf=gdat.trueminf, band=title_band_dict[bands[b]])
			# # f_post_number_cts = plot_posterior_number_counts(logSv, lit_number_counts, nsamp=gdat.nsamp-burn_in, trueminf=gdat.trueminf, band=title_band_dict[bands[b]])
			f_post_number_cts.savefig(flux_color_dir+'/posterior_number_counts_histogram_'+str(title_band_dict[bands[b]])+'.'+plttype, bbox_inches='tight', dpi=dpi)

			f_post_flux_dist = plot_posterior_flux_dist(logSv, raw_number_counts, band=title_band_dict[bands[b]])
			f_post_flux_dist.savefig(flux_color_dir+'/posterior_flux_histogram_'+str(title_band_dict[bands[b]])+'.'+plttype, bbox_inches='tight', dpi=dpi)



		if b > 0 and flux_color_plots:

			for sub_b in range(b):
				print('sub_b, b = ', sub_b, b)
				pairs.append([sub_b, b])

				print('fov srclengths are', len(fov_sources[sub_b]), len(fov_sources[b]))

				color_lin_post.append(fsrcs[sub_b].ravel()/fsrcs[b].ravel())

				if sub_b==1 and b==2:
					ymax = 0.4
				else:
					ymax = 0.1

				# f_flux_color = plot_flux_color_posterior(np.array(fov_sources[sub_b]), np.array(fov_sources[sub_b])/np.array(fov_sources[b]), [title_band_dict[sub_b], title_band_dict[sub_b]+' / '+title_band_dict[b]], xmin=1e-2, xmax=40, ymin=0.005, ymax=ymax)
				# f_flux_color.savefig(flux_color_dir+'/posterior_flux_color_diagram_'+gdat.band_dict[sub_b]+'_'+gdat.band_dict[b]+'_nonlogx.'+plttype, bbox_inches='tight', dpi=dpi)

			# f_color_post = plot_color_posterior(fsrcs, b-1, b, lam_dict, mock_truth_fluxes=cat_fluxes)
			# f_color_post.savefig(flux_color_dir +'/posterior_color_dist_'+str(lam_dict[bands[b-1]])+'_'+str(lam_dict[bands[b]])+'.'+plttype, bbox_inches='tight', dpi=dpi)

	if gdat.nbands == 3 and flux_color_color_plots:


		if condensed_catalog_plots:
			color_color_flux_cond = plt.figure(figsize=(8, 6))
			plt.scatter(condensed_cat[:,5]/condensed_cat[:,9], condensed_cat[:,13]/condensed_cat[:,9], s=10*condensed_cat[:, 4], c=1e3*condensed_cat[:,5], alpha=0.5, label='Condensed catalog \n prevalence > 0.5')
			cbar = plt.colorbar(fraction=0.046, pad=0.04)
			cbar.set_label('$F_{250}$ [mJy]', fontsize=18)
			plt.xlabel('$F_{250}/F_{350}$', fontsize=18)
			plt.ylabel('$F_{500}/F_{350}$', fontsize=18)
			plt.legend()
			# plt.xscale('log')
			# plt.yscale('log')
			# plt.xlim(3e-1, 50)
			# plt.ylim(8e-2, 50)

			plt.xlim(0, 3)
			plt.ylim(0, 3)

			# plt.show()

			color_color_flux_cond.savefig(condensed_cat_dir+'/color_color_flux_diagram_SM_LM_S.'+plttype, bbox_inches='tight', dpi=dpi)


			color_flux_cond = plt.figure(figsize=(8, 6))
			plt.scatter(1e3*condensed_cat[:, 5], condensed_cat[:,5]/condensed_cat[:,9], alpha=0.5, label='Condensed catalog \n prevalence > 0.5')
			# cbar = plt.colorbar(fraction=0.046, pad=0.04)
			# cbar.set_label('$F_{250}$ [mJy]')
			plt.xlabel('$F_{250}$ [mJy]', fontsize=18)
			plt.ylabel('$F_{250}/F_{350}$', fontsize=18)
			plt.legend()
			plt.xscale('log')
			# plt.yscale('log')
			plt.xlim(5, 500)
			# plt.ylim(8e-2, 50)
			plt.ylim(0, 3.5)
			# plt.show()

			color_flux_cond.savefig(condensed_cat_dir+'/color_color_flux_diagram_SM_S.'+plttype, bbox_inches='tight', dpi=dpi)


		# f_color_color = plot_flux_color_posterior(np.array(fov_sources[0])/np.array(fov_sources[1]), np.array(fov_sources[1])/np.array(fov_sources[2]), [title_band_dict[0]+' / '+title_band_dict[1], title_band_dict[1]+' / '+title_band_dict[2]], colormax=60, xmin=1e-2, xmax=60, ymin=1e-2, ymax=80, fmin=0.005, title='Posterior Color-Color Distribution', flux_sizes=np.array(fov_sources[0]))
		# f_color_color.savefig(flux_color_dir+'/posterior_color_color_diagram_'+gdat.band_dict[0]+'-'+gdat.band_dict[1]+'_'+gdat.band_dict[1]+'-'+gdat.band_dict[2]+'_5mJy_band0_linear.'+plttype, bbox_inches='tight', dpi=dpi)

		# f_color_color2 = plot_flux_color_posterior(np.array(fov_sources[2])/np.array(fov_sources[1]), np.array(fov_sources[0])/np.array(fov_sources[1]), [title_band_dict[2]+' / '+title_band_dict[1], title_band_dict[0]+' / '+title_band_dict[1]], colormax=60, xmin=1e-2, xmax=60, ymin=1e-2, ymax=80, fmin=0.005, title='Posterior Color-Color Distribution', flux_sizes=np.array(fov_sources[0]))
		# f_color_color2.savefig(flux_color_dir+'/posterior_color_color_diagram_'+gdat.band_dict[2]+'-'+gdat.band_dict[1]+'_'+gdat.band_dict[0]+'-'+gdat.band_dict[1]+'_5mJy_band0_linear.'+plttype, bbox_inches='tight', dpi=dpi)

		# f_color_color3 = plot_flux_color_posterior(np.array(fov_sources[2])/np.array(fov_sources[0]), np.array(fov_sources[0])/np.array(fov_sources[1]), [title_band_dict[2]+' / '+title_band_dict[0], title_band_dict[0]+' / '+title_band_dict[1]], colormax=60, xmin=1e-2, xmax=60, ymin=1e-2, ymax=80, fmin=0.005, title='Posterior Color-Color Distribution', flux_sizes=np.array(fov_sources[0]))
		# f_color_color3.savefig(flux_color_dir+'/posterior_color_color_diagram_'+gdat.band_dict[2]+'-'+gdat.band_dict[0]+'_'+gdat.band_dict[0]+'-'+gdat.band_dict[1]+'_5mJy_band0_linear.'+plttype, bbox_inches='tight', dpi=dpi)


	# ------------------- SOURCE NUMBER ---------------------------

	if source_number_plots:

		nsrc_fov_truth = None
		if truth_catalog is not None:
			nsrc_fov_truth = len(truth_catalog)

		f_nsrc = plot_src_number_posterior(nsrc_fov, nsrc_truth=nsrc_fov_truth, fmin=1e3*gdat.trueminf, units='mJy')
		f_nsrc.savefig(gdat.filepath +'/posterior_histogram_nstar.'+plttype, bbox_inches='tight', dpi=dpi)

		f_nsrc_trace = plot_src_number_trace(nsrc_fov)
		f_nsrc_trace.savefig(gdat.filepath +'/nstar_traceplot.'+plttype, bbox_inches='tight', dpi=dpi)


		nsrc_full = []
		for i, j in enumerate(np.arange(0, gdat.nsamp)):
		
			fsrcs_full = np.array([fsrcs[0][j][k] for k in range(nsrcs[j]) if dat.weights[0][int(ysrcs[j][k]),int(xsrcs[j][k])] != 0.])

			nsrc_full.append(len(fsrcs_full))

		f_nsrc_trace_full = plot_src_number_trace(nsrc_full)
		f_nsrc_trace_full.savefig(gdat.filepath +'/nstar_traceplot_full.'+plttype, bbox_inches='tight', dpi=dpi)

		


	


