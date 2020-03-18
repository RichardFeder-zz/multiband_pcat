import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def scotts_rule_bins(samples):
	n = len(samples)
	print('n:', n)
	bin_width = 3.5*np.std(samples)/n**(1./3.)
	print(bin_width)
	k = np.ceil((np.max(samples)-np.min(samples))/bin_width)
	print('number of bins:', k)


	bins = np.linspace(np.min(samples), np.max(samples), k)
	return bins

def plot_bkg_sample_chain(bkg_samples, band='250 micron', title=True, show=False):

	f = plt.figure()
	if title:
		plt.title('Uniform background level - '+str(band))

	plt.plot(np.arange(len(bkg_samples)), bkg_samples, label=band)
	plt.xlabel('Sample index')
	plt.ylabel('Background amplitude [Jy/beam]')
	plt.legend()
	
	if show:
		plt.show()

	return f

def plot_posterior_bkg_amplitude(bkg_samples, band='250 micron', title=True, show=False):
	f = plt.figure()
	if title:
		plt.title('Uniform background level - '+str(band))

	plt.hist(bkg_samples, label=band, histtype='step', bins=scotts_rule_bins(bkg_samples))
	plt.xlabel('Amplitude [Jy/beam]')
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
		plt.title('Posterior Flux Distribution - ' +str(band))

	plt.errorbar(logSv+3, mean_number_cts, yerr=np.array([np.abs(mean_number_cts-lower), np.abs(upper - mean_number_cts)]), fmt='o', label='Posterior')
	
	plt.legend()
	plt.yscale('log', nonposy='clip')
	plt.xlabel('log10(Flux) - ' + str(band))

	if show:
		plt.show()

	return f


def plot_posterior_number_counts(logSv, lit_number_counts, trueminf=0.001, band='250 micron', title=True, show=False):

	mean_number_cts = np.mean(lit_number_counts, axis=0)
	lower = np.percentile(lit_number_counts, 16, axis=0)
	upper = np.percentile(lit_number_counts, 84, axis=0)
	f = plt.figure()
	if title:
		plt.title('Posterior Flux Distribution - ' +str(band))

	plt.errorbar(logSv+3, mean_number_cts, yerr=np.array([np.abs(mean_number_cts-lower), np.abs(upper - mean_number_cts)]), marker='.', label='Posterior')
	
	plt.yscale('log')
	plt.legend()
	plt.xlabel('log($S_{\\nu}$) (mJy)')
	plt.ylabel('dN/dS.$S^{2.5}$ ($Jy^{1.5}/sr$)')
	plt.ylim(1e-1, 1e5)
	plt.xlim(np.log10(trueminf)+3.-0.5-1.0, 2.5)
	plt.tight_layout()

	if show:
		plt.show()


	return f


def plot_color_posterior(fsrcs, band0, band1, lam_dict, mock_truth_fluxes=None, title=True, titlefontsize=14, show=False, \
	):

	f = plt.figure()
	if title:
		plt.title('Posterior Color Distribution', fontsize=titlefontsize)

	_, bins, _ = plt.hist(fsrcs[band0].ravel()/fsrcs[band1].ravel(), histtype='step', label='Posterior', bins=np.linspace(0.01, 5, 50), density=True)
	if mock_truth_fluxes is not None:
		plt.hist(mock_truth_fluxes[band0,:].ravel()/mock_truth_fluxes[band1,:].ravel(), bins=bins, density=True, histtype='step', label='Mock Truth')

	plt.legend()
	plt.ylabel('PDF')
	plt.xlabel('$F_{'+str(lam_dict[band0])+'}/F_{'+str(lam_dict[band1])+'}$', fontsize=14)

	if show:
		plt.show()


	return f


def plot_residual_map(resid, mode='median', band='S', titlefontsize=14, smooth=True, smooth_sigma=3, \
					minmax_smooth=None, minmax=None, show=False, plot_refcat=False):


	# TODO - overplot reference catalog on image

	if minmax_smooth is None:
		minmax_smooth = [-0.005, 0.005]
		minmax = [-0.005, 0.005]

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

	plt.title(title_mode+' -- '+band, fontsize=titlefontsize)
	plt.imshow(resid, cmap='Greys', interpolation=None, vmin=minmax[0], vmax=minmax[1], origin='lower')
	plt.colorbar()

	if smooth:
		plt.subplot(1,2,2)
		plt.title('Smoothed Residual', fontsize=titlefontsize)
		plt.imshow(gaussian_filter(resid, sigma=smooth_sigma), interpolation=None, cmap='Greys', vmin=minmax_smooth[0], vmax=minmax_smooth[1], origin='lower')
		plt.colorbar()

	if show:
		plt.show()


	return f

def plot_residual_1pt_function(resid, mode='median', band='S', show=False, binmin=-0.02, binmax=0.02, nbin=50):

	if len(resid.shape) > 1:
		median_resid_rav = resid.ravel()
	else:
		median_resid_rav = resid

	if mode=='median':
		title_mode = 'Median residual'
	elif mode=='last':
		title_mode = 'Last residual'
	
	f = plt.figure()
	plt.title(title_mode+' 1pt function -- '+band)
	plt.hist(median_resid_rav, bins=np.linspace(binmin, binmax, nbin), histtype='step')
	plt.axvline(np.median(median_resid_rav), label='Median='+str(np.round(np.median(median_resid_rav), 5))+'\n $\\sigma=$'+str(np.round(np.std(median_resid_rav), 5)))
	plt.legend(frameon=False)
	plt.ylabel('$N_{pix}$')
	plt.xlabel('data - model [Jy/beam]')

	if show:
		plt.show()

	return f


def plot_chi_squared(chi2, sample_number, band='S', show=False):

	burn_in = sample_number[0]
	f = plt.figure()
	plt.plot(sample_number, chi2[burn_in:], label=band)
	plt.axhline(np.min(chi2[burn_in:]), linestyle='dashed',alpha=0.5, label=str(np.min(chi2[burn_in:]))+' (' + str(band) + ')')
	plt.xlabel('Sample')
	plt.ylabel('Reduced Chi-Squared')
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

def plot_acceptance_fractions(accept_stats, proposal_types=['All', 'Move', 'Birth/Death', 'Merge/Split'], show=False):

	f = plt.figure()
	
	samp_range = np.arange(accept_stats.shape[0])
	for x in range(len(proposal_types)):
		if not np.isnan(accept_stats[0,x]):
			plt.plot(samp_range, accept_stats[:,x], label=proposal_types[x])
	plt.legend()
	plt.xlabel('Sample number')
	plt.ylabel('Acceptance fraction')
	if show:
		plt.show()

	return f


def plot_src_number_posterior(nsrc_fov, show=False, title=False):

	f = plt.figure()
	
	if title:
		plt.title('Posterior Source Number Histogram')
	
	plt.hist(nsrc_fov, histtype='step', label='Posterior', color='b', bins=15)
	plt.axvline(np.median(nsrc_fov), label='Median=' + str(np.median(nsrc_fov)), color='b', linestyle='dashed')
	plt.xlabel('$N_{src}$', fontsize=16)
	plt.ylabel('Number of samples', fontsize=16)
	plt.legend()
		
	if show:
		plt.show()
	
	return f








