from pcat_spire import *
from diffuse_gen import *
from spire_plotting_fns import *
import pandas as pd

''' This script contains the pcat_test_suite module, which is meant to be a container for various tests run with PCAT. Because different tests require
different PCAT parameter configurations, and because there are now many tunable knobs incorporated into the code, separating into different tests should 
help reduce the confusion in fine-tuning runs of PCAT. Over time it would be nice to include more validation tests for input data like validate_astrometry(), 
since much of the work is properly feeding in the various data products. 

Maybe I could construct sub-classes dedicated to different aspects of PCAT, to help minimize redundant variable specifications and to make the fine tuning less clunky. 
I think this can be a barrier to using the tool most effectively if certain running configurations are not clear. 
 '''

def compute_gelman_rubin_diagnostic(list_of_chains, i0=0):
    
    list_of_chains = np.array(list_of_chains)
    print('list of chains has shape ', list_of_chains.shape)
    m = len(list_of_chains)
    n = len(list_of_chains[0])-i0
    
    print('n=',n,' m=',m)
    
    B = (n/(m-1))*np.sum((np.mean(list_of_chains[:,i0:], axis=1)-np.mean(list_of_chains[:,i0:]))**2)
    
    W = 0.
    for j in range(m):
        sumsq = np.sum((list_of_chains[j,i0:]-np.mean(list_of_chains[j,i0:]))**2)
                
        W += (1./m)*(1./(n-1.))*sumsq
    
    var_th = ((n-1.)/n)*W + (B/n)
    
    Rhat = np.sqrt(var_th/W)
    
    print("rhat = ", Rhat)
    
    return Rhat, m, n


def compute_chain_rhats(all_chains, labels=[''], i0=0, nthin=1):
    
    rhats = []
    for chains in all_chains:
        chains = np.array(chains)
        print(chains.shape)
        rhat, m, n = compute_gelman_rubin_diagnostic(chains[:,::nthin], i0=i0//nthin)
                    
        rhats.append(rhat)
        
    f = plt.figure()
    plt.title('Gelman Rubin statistic $\\hat{R}$ ($N_{c}=$'+str(m)+', $N_s=$'+str(n)+')', fontsize=14)
    barlabel = None
    if nthin > 1:
        barlabel = '$N_{thin}=$'+str(nthin)
    plt.bar(labels, rhats, width=0.5, alpha=0.4, label=barlabel)
    plt.axhline(1.2, linestyle='dashed', label='$\\hat{R}$=1.2')
    plt.axhline(1.1, linestyle='dashed', label='$\\hat{R}$=1.1')

    plt.legend()
    plt.xticks(fontsize=16)
    plt.ylabel('$\\hat{R}$', fontsize=16)
    plt.show()
    
    return f, rhats

def spec(x, order=2):
    from statsmodels.regression.linear_model import yule_walker
    beta, sigma = yule_walker(x, order)
    return sigma**2 / (1. - np.sum(beta))**2
    
def geweke_test(chain, first=0.1, last=0.5, intervals=20):
    ''' Adapted from pymc's diagnostics.py script '''
    
    assert first+last <= 1.0
    zscores = [None] * intervals
    starts = np.linspace(0, int(len(chain)*(1.-last)), intervals).astype(int)

    # Loop over start indices
    for i,s in enumerate(starts):

        # Size of remaining array
        x_trunc = chain[s:]
        n = len(x_trunc)

        # Calculate slices
        first_slice = x_trunc[:int(first * n)]
        last_slice = x_trunc[int(last * n):]

        z = (first_slice.mean() - last_slice.mean())
        z /= np.sqrt(spec(first_slice)/len(first_slice) +
                     spec(last_slice)/len(last_slice))
        zscores[i] = len(chain) - n, z

    return zscores  
    

class pcat_test_suite():


	flux_density_conversion_dict = dict({'S': 86.29e-4, 'M':16.65e-3, 'L':34.52e-3})
	# in Jy/sr 
	dust_I_lams=dict({'S':3e6, 'M': 1.6e6,'L':8e5})

	# this is the band convention I use almost exclusively throughout the code when there are stray indices
	band_dict = dict({0:'S', 1:'M', 2:'L'})
	
	def __init__(self,\
				base_path='/Users/luminatech/Documents/multiband_pcat/',\
				result_path='/Users/luminatech/Documents/multiband_pcat/spire_results/', \
				cluster_name='rxj1347',\
				sz_tail_name='rxj1347_PSW_nr_sze', \
				cblas=True, \
				openblas=False):
	
		self.base_path = base_path
		self.result_path = result_path
		self.cluster_name=cluster_name
		self.sz_tail_name = sz_tail_name
		self.cblas=cblas
		self.openblas=openblas
		self.sz_filename = self.base_path+'Data/spire/'+cluster_name+'_sz_templates/'+sz_tail_name+'.fits'


	def procure_cirrus_realizations(self, nsims, cirrus_tail_name='rxj1347_cirrus_sim_idx_', dataname='new_gen3_sims', \
					tail_name='rxj1347_PSW_sim0300'):

		''' 
		Given cluster data and associated Planck template, generates and saves a set of cirrus dust realizations.

		Parameters
		----------

		nsims : 'int'
			Number of cirrus realizations to make.
		cirrus_tail_name : 'str', optional
			tail name for saving cirrus realization. 
		dataname : 'str', optional
			Folder name for data. Default is 'new_gen3_sims'.
		tail_name : 'str', optional
			tail name for loading cluster image data.

		Returns
		-------

		Nothing! But cirrus realizations are saved to predefined repository.

		'''

		
		tail_name_M = tail_name.replace('PSW', 'PMW')
		tail_name_L = tail_name.replace('PSW', 'PLW')

		planck_s = fits.open(self.base_path+'/Data/spire/'+dataname+'/'+tail_name+'.fits')['PLANCK'].data
		planck_m = fits.open(self.base_path+'/Data/spire/'+dataname+'/'+tail_name_M+'.fits')['PLANCK'].data
		planck_l = fits.open(self.base_path+'/Data/spire/'+dataname+'/'+tail_name_L+'.fits')['PLANCK'].data

		shapes = [planck_s.shape, planck_m.shape, planck_l.shape]
		imdims_max = [np.max(planck_s.shape), np.max(planck_m.shape), np.max(planck_l.shape)]
		imdims_min = [np.min(planck_s.shape), np.min(planck_m.shape), np.min(planck_l.shape)]

		allo = generate_spire_cirrus_realizations(nsims, planck_s, imdims=imdims_max, show=False)
		allo = [[allo[i][b][:shapes[b][0],:shapes[b][1]] for b in range(3)] for i in range(len(allo))]

		for i, al in enumerate(allo):
			print(np.std(al[0]), np.std(al[1]), np.std(al[2]))
			if not os.path.exists(self.base_path+'/Data/spire/cirrus_gen'):
				os.makedirs(self.base_path+'/Data/spire/cirrus_gen')
			np.savez(self.base_path+'/Data/spire/cirrus_gen/'+cirrus_tail_name+str(i)+'.npz', S=al[0], M=al[1], L=al[2])
		print('we are done now!')

	
	def validate_astrometry(self, band0=0, band1=1, band2=2, tail_name='rxj1347_PSW_nr_1_ext', dataname='rxj1347_831', use_mask=False, nregion=5, auto_resize=True,\
						ngrid=20, return_validation_figs=True, image_extnames=['IMAGE'], save=False, error_extname='ERROR'):

		''' 

		This function validates that the fast astrometric transformations approximated by PCAT are consistent with transformations of the WCS header,
		and also can be used to visually confirm that the WCS header is properly aligned with the input images. 

		Parameters
		----------

		band0, band1, band2 : integers
			indices of bands to test astrometry for

		use_mask : boolean, optional
			If True, applies mask to observations first. Default is 'False'.

		ngrid : integer, optional
			Specifies the gridding of test points that get transformed using both WCS and fast astrometric transforms. 

		Returns
		-------

		validation_figs : list of matplotlib Figures

		'''

		ob = lion(band0=band0, band1=band1, band2=band2, base_path=self.base_path, result_path=self.result_path, cblas=self.cblas, openblas=self.openblas, \
			  tail_name=tail_name, dataname=dataname, use_mask=use_mask, nregion=nregion, image_extnames=image_extnames, error_extname=error_extname)

		if ob.gdat.nbands > 1:
			for b in range(ob.gdat.nbands - 1):
				pos0_pivot = None
				pos0 = None

				print('BOUNDS[b+1] is ', ob.gdat.bounds[b+1])
				print('BOUNDS[0] is ', ob.gdat.bounds[0])

				ob.data.fast_astrom.fit_astrom_arrays(b+1, 0, bounds0=ob.gdat.bounds[b+1], bounds1=ob.gdat.bounds[0], pos0_pivot=pos0_pivot, pos0=pos0, correct_misaligned_shift=True)

		xr = np.linspace(0, ob.gdat.imszs[0][0], ngrid)
		yr = np.linspace(0, ob.gdat.imszs[0][1], ngrid)
		xv, yv = np.meshgrid(xr, yr)

		validation_figs = []

		f = plt.figure(figsize=(6*ob.gdat.nbands, 5))
		plt.subplot(1,ob.gdat.nbands,1)
		plt.title('band 0')
		plt.imshow(ob.data.data_array[0]-np.median(ob.data.data_array[0]), cmap='Greys', vmin=-0.005, vmax=0.02, origin='lower')
		cbar = plt.colorbar(fraction=0.046, pad=0.04)
		cbar.set_label('[Jy/beam]', fontsize=14)
		plt.scatter(xv, yv, marker='x', color='r')
		plt.xlabel('x [pix]', fontsize=16)
		plt.ylabel('y [pix]', fontsize=16)

		for b in range(ob.gdat.nbands - 1):

			xnew, ynew = ob.data.fast_astrom.transform_q(xv, yv, b)

			x0, y0 = ob.data.fast_astrom.transform_q(np.array([0]), np.array([0]), b)
			print(x0, y0)

			# xnew -= np.min(xnew)
			# ynew -= np.min(ynew)
			# print(np.min(xnew), np.min(ynew))

			xv_rt, yv_rt = ob.data.fast_astrom.transform_q(xnew, ynew, ob.gdat.nbands-1+b)
			xnew_wcs, ynew_wcs = ob.data.fast_astrom.obs_to_obs(0, b+1, xv, yv)

			# f = plt.figure(figsize=(12,5))
			# plt.subplot(1,2,1)
			# plt.title('band 0')
			# plt.imshow(ob.data.data_array[0]-np.median(ob.data.data_array[0]), cmap='Greys', vmin=-0.005, vmax=0.02, origin='lower')
			# plt.colorbar()
			# plt.scatter(xv, yv, marker='x', color='r')
			plt.subplot(1,ob.gdat.nbands,b+2)
			plt.title('band '+str(b+1))
			plt.imshow(ob.data.data_array[b+1]-np.median(ob.data.data_array[b+1]), cmap='Greys', vmin=-0.005, vmax=0.02, origin='lower')
			cbar = plt.colorbar(fraction=0.046, pad=0.04)
			cbar.set_label('[Jy/beam]', fontsize=14)
			plt.xlabel('x [pix]', fontsize=16)
			plt.ylabel('y [pix]', fontsize=16)
			plt.scatter(xnew, ynew, marker='+', color='g', label='Fast astrom')
			plt.scatter(xnew_wcs, ynew_wcs, marker='x', color='r', label='WCS')
			plt.legend()
		plt.tight_layout()
		plt.show()

		validation_figs.append(f)

		if save:
			f.savefig(self.base_path+'/Data/spire/'+dataname+'/astrometry_validation.pdf', bbox_inches='tight')

		if return_validation_figs:
			return validation_figs


	def artificial_star_test(self, n_src_perbin=10, inject_fmin=0.01, inject_fmax=0.2, nbins=20, fluxbins=None, frac_flux_thresh=2., pos_thresh=0.5,\
		band0=0, band1=None, band2=None, fmin=0.01, nsamp=500, load_timestr=None, dataname='SMC_HERITAGE', tail_name='SMC_HERITAGE_mask2_PSW', \
		bias = None, max_nsrc=1000, visual=False, alph=1.0, show_input_maps=False, make_post_plots=True, \
		residual_samples=50, float_background=True, timestr_list_file=None, \
		nbands=None, mask_file=None, use_mask=True, image_extnames=['IMAGE'], float_fourier_comps=False, n_fc_terms=10, fc_sample_delay=0, \
		point_src_delay=0, nsrc_init=None, fc_prop_alpha=None, fourier_comp_moveweight=200., fc_amp_sig=0.001, n_frames=10, color_mus=None, color_sigs=None, im_fpath=None, err_fpath=None, \
		bkg_moore_penrose_inv=True, MP_order=5, ridge_fac=2., inject_catalog_path=None, save=True, inject_color_means=[1.0, 0.7], inject_color_sigs=[0.25, 0.25], cond_cat_fpath=None):

		'''
		This function injects a population of artificial sources into a given image and provides diagnostics on what artificial sources PCAT recovers and to what accuracy fluxes are estimated.

		Unique parameters
		-----------------
		
		n_src_perbin : int, optional
			Number of injected sources per flux bin. Default is 10.

		inject_fmin : float, optional
			Minimum flux density of injected sources. Default is 0.01 [Jy].

		inject_fmax : float, optional
			Maximum flux density of injected sources. Default is 0.2 [Jy].

		nbins : int, optional
			Number of flux bins to create between inject_fmin and inject_fmax. Default is 20.

		fluxbins : list or `~numpy.ndarray', optional
			fluxbins can be specified if custom flux bins are desired. Default is None.

		frac_flux_thresh : float, optional
			When matching PCAT sources to injected sources, frac_flux_thresh defines part of the cross-matching criteria, 
			in which the fractional error of the PCAT source w.r.t. the injected source must be less than frac_flux_thresh. 
			Default is 2.0.

		pos_thresh : float, optional
			Same idea as frac_flux_thresh, but applies a position cross matching criterion. 
			Default is 0.5 [pixels].

		inject_color_means, inject_color_sigs : lists, optional
			Mean colors for injected sources. The colors of sources are drawn from a Gaussian distribution with mean injected_color_means[b] and  
			Default assumes bands are ordered as 250, 350, 500 micron, colors of S350/S250 = 1.0, S500/S200 = 0.7 and scatters of [0.25, 0.25]. 

		cond_cat_fpath : string, optional
			If PCAT has already been run and one wishes to use results from a condensed catalog, it can be specified here to bypass running PCAT again. 
			Default is 'None'.


		Returns 
		-------

		Nothing! 

		'''

		if nbands is None:
			nbands = 0
			if band0 is not None:
				nbands += 1
			if band1 is not None:
				nbands += 1
			if band2 is not None:
				nbands += 1

		if load_timestr is not None:
			# if we are analyzing a run of Lion that is already finished, don't bother making a new directory structure when initializing PCAT
			save = False

		ob = lion(band0=band0, band1=band1, band2=band2, base_path=self.base_path, result_path=self.result_path, round_up_or_down='down', \
						float_background=float_background, burn_in_frac=0.75, \
		 				cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps, \
		 				tail_name=tail_name, dataname=dataname, bias=bias, max_nsrc=max_nsrc,\
		 				trueminf=fmin, nregion=5, make_post_plots=make_post_plots, nsamp=nsamp, use_mask=use_mask,\
		 				residual_samples=residual_samples, float_fourier_comps=float_fourier_comps, \
		 				n_fourier_terms=n_fc_terms, fc_sample_delay=fc_sample_delay, fourier_comp_moveweight=fourier_comp_moveweight,\
		 				alph=alph, dfc_prob=1.0, nsrc_init=nsrc_init, mask_file=mask_file, \
		 				point_src_delay=point_src_delay, n_frames=n_frames, image_extnames=image_extnames, fc_prop_alpha=fc_prop_alpha, \
		 				im_fpath=im_fpath, err_fpath=err_fpath, fc_amp_sig=fc_amp_sig, MP_order=MP_order, bkg_moore_penrose_inv=bkg_moore_penrose_inv, ridge_fac=ridge_fac, \
		 				save=save)


		if fluxbins is None:
			fluxbins = np.logspace(np.log10(inject_fmin), np.log10(inject_fmax), nbins)

		if load_timestr is None:

			if inject_catalog_path is not None:
				print('loading catalog from ', inject_catalog_path)
				catalog_inject = np.load(inject_catalog_path)['catalog_inject']

			else:
				print('FLUXBINS ARE ', fluxbins)

				if type(n_src_perbin)==list:
					catalog_inject = np.zeros((np.sum(np.array(n_src_perbin)), 2+nbands)).astype(np.float32())
				else:
					catalog_inject = np.zeros((n_src_perbin*(nbins-1), 2+nbands)).astype(np.float32())

				print('catalog inject has shape ', catalog_inject.shape)

				catalog_inject[:, :2] = np.array(np.random.uniform(5., ob.data.data_array[0].shape[0]-5., (catalog_inject.shape[0], 2)), dtype=np.float32)

				idxctr = 0
				for f in range(len(fluxbins)-1):

					if type(n_src_perbin)==list:
						print('here')
						pivot_fluxes = np.array(np.random.uniform(fluxbins[f], fluxbins[f+1], n_src_perbin[f]), dtype=np.float32)
						catalog_inject[idxctr:idxctr+int(n_src_perbin[f]), 2] = pivot_fluxes
						
						# if multiband, draw colors from prior and multiply pivot band fluxes
						if nbands > 0:
							for b in range(nbands - 1):
								colors = np.random.normal(inject_color_means[b], inject_color_sigs[b], n_src_perbin[f])
								
								print('injected sources in band ', b+1, 'are ')
								print(pivot_fluxes*colors)
								catalog_inject[idxctr:idxctr+int(n_src_perbin[f]), 3+b] = pivot_fluxes*colors

						idxctr += int(n_src_perbin[f])
					else:
						pivot_fluxes = np.array(np.random.uniform(fluxbins[f], fluxbins[f+1], n_src_perbin), dtype=np.float32)

						catalog_inject[f*n_src_perbin:(f+1)*n_src_perbin, 2] = pivot_fluxes
						if nbands > 0:
							for b in range(nbands - 1):
								colors = np.random.normal(color_means[b], color_sigs[b], n_src_perbin)
								catalog_inject[f*n_src_perbin:(f+1)*n_src_perbin, 3+b] = pivot_fluxes*colors


			ob.gdat.catalog_inject = catalog_inject.copy()


		else:
			catalog_inject = np.load(self.result_path+load_timestr+'/inject_catalog.npz')['catalog_inject']
			ob.gdat.timestr = load_timestr
			print('ob.gdat.timestr is ', ob.gdat.timestr)


		flux_bin_idxs = [[np.where((catalog_inject[:,2+b] > fluxbins[i])&(catalog_inject[:,2+b] < fluxbins[i+1]))[0] for i in range(len(fluxbins)-1)] for b in range(nbands)]

		if load_timestr is None:
			inject_src_image = np.zeros_like(ob.data.data_array[0])

			ob.initialize_print_log()

			libmmult = ob.initialize_libmmult()

			initialize_c(ob.gdat, libmmult, cblas=ob.gdat.cblas)

			if ob.gdat.cblas:
				lib = libmmult.pcat_model_eval
			else:
				lib = libmmult.clib_eval_modl

			model = Model(ob.gdat, ob.data, libmmult)

			resid = ob.data.data_array[0].copy()

			resids = []

			for b in range(nbands):

				resid = ob.data.data_array[b].copy() # residual for zero image is data
				resids.append(resid)

			print('fluxes have shape ', np.array(catalog_inject[:,2:]).astype(np.float32()).shape)

			print('model pixel per beam:', model.pixel_per_beam)
			inject_src_images, diff2s, dt_transf = model.pcat_multiband_eval(catalog_inject[:,0].astype(np.float32()), catalog_inject[:,1].astype(np.float32()), np.array(catalog_inject[:,2:]).astype(np.float32()).transpose(),\
																 np.array([0. for x in range(nbands)]), ob.data.ncs, ob.data.cfs, weights=ob.data.weights, ref=resids, lib=libmmult.pcat_model_eval, beam_fac=model.pixel_per_beam)


			for b in range(nbands):
				print('models look like')
				plt.figure()
				plt.imshow(inject_src_images[b])
				plt.colorbar()
				plt.show()


			# inject_src_image, diff2 = image_model_eval(catalog_inject[:,0].astype(np.float32()), catalog_inject[:,1].astype(np.float32()), np.array(model.pixel_per_beam*ob.data.ncs[0]*catalog_inject[:,2]).astype(np.float32()), 0., model.imszs[0], \
			# 										ob.data.ncs[0], np.array(ob.data.cfs[0]).astype(np.float32()), weights=np.array(ob.data.weights[0]).astype(np.float32()), \
			# 										ref=resid, lib=libmmult.pcat_model_eval, regsize=model.regsizes[0], \
			# 										margin=0, offsetx=0, offsety=0, template=None)

			if show_input_maps:

				plt.figure()
				plt.subplot(1,2,1)
				plt.title('Original image')
				plt.imshow(ob.data.data_array[0], cmap='Greys', origin='lower', vmax=0.1)
				plt.colorbar()
				plt.subplot(1,2,2)
				plt.title('Injected source image')
				plt.imshow(inject_src_image, cmap='Greys', origin='lower')
				plt.colorbar()
				plt.show()

			# add the injected mdoel image into the real data
			for b in range(nbands):
				ob.data.data_array[b] += inject_src_images[b]

			if show_input_maps:
				plt.figure()
				plt.title('original + injected image')
				plt.imshow(ob.data.data_array[0], cmap='Greys', origin='lower', vmax=0.1)
				plt.colorbar()
				plt.show()

			# run the thing
			ob.main()
			# save the injected catalog to the result directory
			print(self.result_path+ob.gdat.timestr+'/inject_catalog.npz')
			np.savez(self.result_path+ob.gdat.timestr+'/inject_catalog.npz', catalog_inject=catalog_inject)

		# load the run and compute the completeness for each artificial source

		if load_timestr is not None:
			_, filepath, _ = load_param_dict(load_timestr, result_path=self.result_path)
			timestr = load_timestr
		else:
			_, filepath, _ = load_param_dict(ob.gdat.timestr, result_path=self.result_path)
			timestr = ob.gdat.timestr

		chain = np.load(filepath+'/chain.npz')

		xsrcs = chain['x']
		ysrcs = chain['y']
		fsrcs = chain['f']

		completeness_ensemble = np.zeros((residual_samples, nbands, catalog_inject.shape[0]))
		fluxerror_ensemble = np.zeros((residual_samples, nbands, catalog_inject.shape[0]))

		recovered_flux_ensemble = np.zeros((residual_samples, nbands, catalog_inject.shape[0]))

		for b in range(nbands):

			for i in range(residual_samples):

				for j in range(catalog_inject.shape[0]):

					# make position cut 
					idx_pos = np.where(np.sqrt((xsrcs[-i] - catalog_inject[j,0])**2 +(ysrcs[-i] - catalog_inject[j,1])**2)  < pos_thresh)[0]
					
					fluxes_poscutpass = fsrcs[b][-i][idx_pos]
					# fluxes_poscutpass = fsrcs[0][-i][idx_pos]
					
					# make flux cut
					# mask_flux = np.where(np.abs(fluxes_poscutpass - catalog_inject[j,2])/catalog_inject[j,2] < frac_flux_thresh)[0]
					mask_flux = np.where(np.abs(fluxes_poscutpass - catalog_inject[j,2+b])/catalog_inject[j,2+b] < frac_flux_thresh)[0]

					if len(mask_flux) >= 1:

						# print('we got one! true source is ', catalog_inject[j])
						# print('while PCAT source is ', xsrcs[-i][idx_pos][mask_flux], ysrcs[i][idx_pos][mask_flux], fluxes_poscutpass[mask_flux])

						# completeness_ensemble[i,j] = 1.
						completeness_ensemble[i,b,j] = 1.

						# compute the relative difference in flux densities between the true and PCAT source and add to list specific for each source 
						# (in practice, a numpy.ndarray with zeros truncated after the fact). 
						# For a given injected source, PCAT may or may not have a detection, so one is probing the probability distribution P(S_{Truth, i} - S_{PCAT} | N_{PCAT, i} == 1)
						# where N_{PCAT, i} is an indicator variable for whether PCAT has a source within the desired cross match criteria. 

						# if there is more than one source satisfying the cross-match criteria, choose the brighter source

						flux_candidates = fluxes_poscutpass[mask_flux]

						brighter_flux = np.max(flux_candidates)
						dists = np.sqrt((xsrcs[-i][mask_flux] - catalog_inject[j,0])**2 + (ysrcs[-i][mask_flux] - catalog_inject[j,1])**2)


						mindist_idx = np.argmin(dists)					
						mindist_flux = flux_candidates[mindist_idx]

						recovered_flux_ensemble[i,b,j] = mindist_flux
						# fluxerror_ensemble[i,j] = brighter_flux/catalog_inject[j,2]
						# fluxerror_ensemble[i,j] = (brighter_flux-catalog_inject[j,2])/catalog_inject[j,2]
						# fluxerror_ensemble[i,j] = (mindist_flux-catalog_inject[j,2])/catalog_inject[j,2]
						
						# fluxerror_ensemble[i,b,j] = (mindist_flux-catalog_inject[j,2+b])/catalog_inject[j,2+b]

						fluxerror_ensemble[i,b,j] = (mindist_flux-catalog_inject[j,2+b])


		mean_frac_flux_error = np.zeros((catalog_inject.shape[0],nbands))
		pct_16_fracflux = np.zeros((catalog_inject.shape[0],nbands))
		pct_84_fracflux = np.zeros((catalog_inject.shape[0],nbands))


		mean_recover_flux = np.zeros((catalog_inject.shape[0],nbands))
		pct_16_recover_flux = np.zeros((catalog_inject.shape[0],nbands))
		pct_84_recover_flux = np.zeros((catalog_inject.shape[0],nbands))


		prevalences = [[] for x in range(nbands)]

		
		for b in range(nbands):
			for j in range(catalog_inject.shape[0]):
				nonzero_fidx = np.where(fluxerror_ensemble[:,b,j] != 0)[0]
				prevalences[b].append(float(len(nonzero_fidx))/float(residual_samples))
				if len(nonzero_fidx) > 0:
					mean_frac_flux_error[j,b] = np.median(fluxerror_ensemble[nonzero_fidx,b, j])
					pct_16_fracflux[j,b] = np.percentile(fluxerror_ensemble[nonzero_fidx,b, j], 16)
					pct_84_fracflux[j,b] = np.percentile(fluxerror_ensemble[nonzero_fidx,b, j], 84)

					mean_recover_flux[j,b] = np.median(recovered_flux_ensemble[nonzero_fidx, b, j])
					pct_16_recover_flux[j,b] = np.percentile(recovered_flux_ensemble[nonzero_fidx, b, j], 16)
					pct_84_recover_flux[j,b] = np.percentile(recovered_flux_ensemble[nonzero_fidx, b, j], 84)


		# nonzero_ferridx = np.where(mean_frac_flux_error != 0)[0]
		nonzero_ferridx = np.where(mean_frac_flux_error[:,0] != 0)[0]
		lamtitlestrs = ['PSW', 'PMW', 'PLW']

		plt_colors = ['b', 'g', 'r']
		plt.figure(figsize=(5*nbands, 5))

		for b in range(nbands):
			plt.subplot(1,nbands, b+1)
			yerr_recover = [1e3*mean_recover_flux[nonzero_ferridx, b] - 1e3*pct_16_recover_flux[nonzero_ferridx, b], 1e3*pct_84_recover_flux[nonzero_ferridx, b]-1e3*mean_recover_flux[nonzero_ferridx, b]]

			plt.title(lamtitlestrs[b], fontsize=18)
			print(1e3*mean_recover_flux[nonzero_ferridx, b])
			print(1e3*catalog_inject[nonzero_ferridx, 2+b])
			print(yerr_recover)

			plt.errorbar(1e3*catalog_inject[nonzero_ferridx, 2+b], 1e3*mean_recover_flux[nonzero_ferridx, b], yerr=yerr_recover, capsize=5, fmt='.', linewidth=2, markersize=10, alpha=0.25, color=plt_colors[b])

			plt.xscale('log')
			plt.yscale('log')

			if b==0:
				plt.xlabel('$S_{True}^{250}$ [mJy]', fontsize=16)
				plt.ylabel('$S_{Recover}^{250}$ [mJy]', fontsize=16)
			elif b==1:
				plt.xlabel('$S_{True}^{350}$ [mJy]', fontsize=16)
				plt.ylabel('$S_{Recover}^{350}$ [mJy]', fontsize=16)
			elif b==2:
				plt.xlabel('$S_{True}^{500}$ [mJy]', fontsize=16)
				plt.ylabel('$S_{Recover}^{500}$ [mJy]', fontsize=16)

			plt.plot(np.logspace(-1, 3, 100), np.logspace(-1, 3, 100), linestyle='dashed', color='k', linewidth=3)
			plt.xlim(1e-1, 1e3)
			plt.ylim(1e-1, 1e3)

			# plt.xlim(5, 700)
			# plt.ylim(5, 700)
			# plt.savefig(filepath+'/injected_vs_recovered_flux_band'+str(b)+'_PCAT_dx='+str(pos_thresh)+'_dS-S='+str(frac_flux_thresh)+'_mindist.pdf', bbox_inches='tight')
		plt.tight_layout()
		# plt.savefig(filepath+'/injected_vs_recovered_flux_threeband_PCAT_dx='+str(pos_thresh)+'_dS-S='+str(frac_flux_thresh)+'_mindist_4.pdf', bbox_inches='tight')
		# plt.savefig(filepath+'/injected_vs_recovered_flux_threeband_PCAT_dx='+str(pos_thresh)+'_dS-S='+str(frac_flux_thresh)+'_mindist_3.png', bbox_inches='tight', dpi=300)
			
		plt.show()


		prevalences = np.array(prevalences)


		mean_ferr_nonzero_ferridx = mean_frac_flux_error[nonzero_ferridx,:]
		pct_16_nonzero_ferridx = pct_16_fracflux[nonzero_ferridx,:]
		pct_84_nonzero_ferridx = pct_84_fracflux[nonzero_ferridx,:]

		yerrs = [mean_ferr_nonzero_ferridx - pct_16_nonzero_ferridx, pct_84_nonzero_ferridx-mean_ferr_nonzero_ferridx]

		mean_frac_flux_error_binned = np.zeros((len(fluxbins)-1, nbands))
		pct16_frac_flux_error_binned = np.zeros((len(fluxbins)-1, nbands))
		pct84_frac_flux_error_binned = np.zeros((len(fluxbins)-1, nbands))

		print('flux bins in function are ', fluxbins)

		for b in range(nbands):
			for f in range(len(fluxbins)-1):

				finbin = np.where((catalog_inject[nonzero_ferridx, 2+b] >= fluxbins[f])&(catalog_inject[nonzero_ferridx, 2+b] < fluxbins[f+1]))[0]
				
				if len(finbin)>0:
					mean_frac_flux_error_binned[f,b] = np.median(mean_ferr_nonzero_ferridx[finbin,b])

					print(mean_ferr_nonzero_ferridx.shape)
					print(mean_ferr_nonzero_ferridx[finbin,b])
					pct16_frac_flux_error_binned[f,b] = np.percentile(mean_ferr_nonzero_ferridx[finbin,b], 16)

					pct84_frac_flux_error_binned[f,b] = np.percentile(mean_ferr_nonzero_ferridx[finbin,b], 84)

		for b in range(nbands):
			nsrc_perfbin = [len(fbin_idxs) for fbin_idxs in flux_bin_idxs[b]]

			g, _, yerr, geom_mean = plot_fluxbias_vs_flux(1e3*mean_frac_flux_error_binned[:,b], 1e3*pct16_frac_flux_error_binned[:,b], 1e3*pct84_frac_flux_error_binned[:,b], fluxbins, \
				band=b, nsrc_perfbin=nsrc_perfbin, ylim=[-20, 20], load_jank_txts=False)
			
			g.savefig(filepath+'/fluxerr_vs_fluxdensity_band'+str(b)+'_dx='+str(pos_thresh)+'_dS-S='+str(frac_flux_thresh)+'_mindist_nonfrac_012021.png', bbox_inches='tight', dpi=300)

			# np.savez('goodsn_band'+str(b)+'_fluxbias_vs_flux_nbands='+str(nbands)+'_012021.npz', mean_frac_flux_error_binned=mean_frac_flux_error_binned[:,b], yerr=yerr, geom_mean=geom_mean)

	
		# completeness_vs_flux = None
		xstack = xsrcs[-20:,:].ravel()
		ystack = ysrcs[-20:,:].ravel()
		fstack = fsrcs[0][-20:,:].ravel()

		lamstrs = ['250', '350', '500']

		nsrc_perfbin_bands, cvf_stderr_bands, completeness_vs_flux_bands = [], [], []

		for b in range(nbands):

			avg_completeness = np.mean(completeness_ensemble[:,b,:], axis=0)
			std_completeness = np.std(completeness_ensemble[:,b,:], axis=0)

			nsrc_perfbin = [len(fbin_idxs) for fbin_idxs in flux_bin_idxs[b]]

			nsrc_perfbin_bands.append(nsrc_perfbin)
			completeness_vs_flux = [np.mean(avg_completeness[fbin_idxs]) for fbin_idxs in flux_bin_idxs[b]]

			print('completeness vs flux? its')
			print(completeness_vs_flux)
			cvf_std = [np.mean(std_completeness[fbin_idxs]) for fbin_idxs in flux_bin_idxs[b]]



			completeness_vs_flux_bands.append(completeness_vs_flux)

			cvf_stderr_bands.append(np.array(cvf_std)/np.sqrt(np.array(nsrc_perfbin)))

			print(fluxbins)
			print(completeness_vs_flux)

			colors = ['b', 'g', 'r']

		# psc_spire_cosmos_comp = pd.read_csv('~/Downloads/PSW_completeness.csv', header=None)
		# plt.plot(np.array(psc_spire_cosmos_comp[0]), np.array(psc_spire_cosmos_comp[1]), marker='.', markersize=10, label='SPIRE Point Source Catalog (2017) \n 250 $\\mu m$', color='k')
		
		f = plot_completeness_vs_flux(pos_thresh, frac_flux_thresh, fluxbins, completeness_vs_flux_bands, cvf_stderr=cvf_stderr_bands,\
										 image=91.*ob.data.data_array[0][:-5,:-5] - 91.*np.nanmean(ob.data.data_array[0][:-5,:-5]), xstack=xstack, ystack=ystack, fstack=fstack, \
										 catalog_inject = catalog_inject)


		f.savefig(filepath+'/completeness_vs_fluxdensity_dx='+str(pos_thresh)+'_dS-S='+str(frac_flux_thresh)+'_multiband_012321.png', bbox_inches='tight', dpi=300)


		return fluxbins, completeness_vs_flux, f


	def real_dat_run(self, band0=0, band1=None, band2=None, fmin=0.007, nsamp=500, template_names=None, dataname='rxj1347_831', tail_name='rxj1347_PSW_nr_1_ext', \
		bias = [-0.004, -0.007, -0.008], max_nsrc=1000, visual=False, alph=1.0, show_input_maps=False, make_post_plots=True, \
		inject_sz_frac=0.0, residual_samples=50, float_background=True, timestr_list_file=None, \
		nbands=None, mask_file=None, weighted_residual=False, float_templates=False, use_mask=True, image_extnames=['SIGNAL'], error_extname='ERROR', \
		float_fourier_comps=False, n_fc_terms=10, fc_sample_delay=0, fourier_comp_moveweight=200., \
		template_moveweight=40., template_filename=None, psf_fwhms=None, \
		bkg_sample_delay=0, birth_death_sample_delay=0, movestar_sample_delay=0, merge_split_sample_delay=0, temp_sample_delay=30, \
		movestar_moveweight=80., birth_death_moveweight=60., merge_split_moveweight=60., \
		load_state_timestr=None, nsrc_init=None, fc_prop_alpha=None, fc_amp_sig=0.0001, n_frames=10, color_mus=None, color_sigs=None, im_fpath=None, err_fpath=None, \
		bkg_moore_penrose_inv=False, MP_order=5., ridge_fac=None, point_src_delay=0, nregion=5, fc_rel_amps=None, correct_misaligned_shift=False, \
		inject_diffuse_comp=False, diffuse_comp_path=None, panel_list = None, F_statistic_alph=False, raw_counts=False, generate_condensed_catalog=False, \
		err_f_divfac=1., bkg_sig_fac=5.0, n_condensed_samp=50, prevalence_cut=0.5, burn_in_frac=0.7, \
		temp_prop_sig_fudge_facs=None, estimate_dust_first=False, nominal_nsrc=1000, nsamp_dustestimate=100, initial_template_amplitude_dicts=None, init_fourier_coeffs=None):

		''' General function for running PCAT on real (or mock, despite the name) data. '''
		if nbands is None:
			nbands = 0
			if band0 is not None:
				nbands += 1
			if band1 is not None:
				nbands += 1
			if band2 is not None:
				nbands += 1

		if nbands > 1 and float_fourier_comps:
			print("WERE USING THE DUST")
			dust_rel_SED = [self.dust_I_lams[self.band_dict[i]]/self.dust_I_lams[self.band_dict[0]] for i in range(nbands)]
			fc_rel_amps = [dust_rel_SED[i]*self.flux_density_conversion_dict[self.band_dict[i]]/self.flux_density_conversion_dict[self.band_dict[0]] for i in range(nbands)]

		if estimate_dust_first:
			dfc_prob_main = 0.
			fourier_comp_moveweight_main=0.
			point_src_delay_main = 0.
			float_fourier_comps = True
		else:
			dfc_prob_main = 1.0
			fourier_comp_moveweight_main = fourier_comp_moveweight
			point_src_delay_main = point_src_delay


		if estimate_dust_first:
			
			pan_list = ['data0', 'model0', 'residual0', 'fourier_bkg0', 'residual_zoom0', 'dNdS0']

			# start with 250 micron image only

			ob = lion(band0=band0, base_path=self.base_path, result_path=self.result_path, burn_in_frac=burn_in_frac, float_background=float_background, \
			  bkg_sample_delay=0, cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps, \
			  tail_name=tail_name, dataname=dataname, bias=bias, use_mask=use_mask, mask_file=mask_file, max_nsrc=max_nsrc, trueminf=fmin, nregion=nregion, \
			  make_post_plots=False, nsamp=nsamp_dustestimate, residual_samples=5, template_moveweight=template_moveweight, float_templates=False, \
			  image_extnames=image_extnames, error_extname=error_extname, panel_list=pan_list, err_f_divfac=err_f_divfac, bkg_sig_fac=bkg_sig_fac, \
			  movestar_moveweight=movestar_moveweight, nominal_nsrc=nominal_nsrc, birth_death_moveweight=birth_death_moveweight, merge_split_moveweight=merge_split_moveweight, \
			  float_fourier_comps=True, n_fourier_terms=n_fc_terms, fc_sample_delay=0, fourier_comp_moveweight=200., \
			  dfc_prob=1.0, nsrc_init=0, point_src_delay=point_src_delay, fc_amp_sig=fc_amp_sig)

			ob.main()
			_, filepath, _ = load_param_dict(ob.gdat.timestr, result_path=self.result_path)
			timestr = ob.gdat.timestr


			# use filepath from the last iteration to load estiamte of median background estimate
			chain = np.load(filepath+'/chain.npz')
			init_fourier_coeffs = np.median(chain['fourier_coeffs'][10:], axis=0)
			last_bkg_sample_250 = chain['bkg'][-1,0]


		
		# panel_list = ['data0', 'data1', 'data2', 'fourier_bkg0', 'residual1', 'residual2']

		# panel_list = ['data0', 'data1', 'data2', 'residual0', 'residual1', 'residual2']

		ob = lion(band0=band0, band1=band1, band2=band2, base_path=self.base_path, result_path=self.result_path, \
					float_background=float_background, burn_in_frac=burn_in_frac, bkg_sample_delay=bkg_sample_delay, float_templates=float_templates, template_moveweight=template_moveweight, \
	 				cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps, \
	 				template_names=template_names, temp_sample_delay=temp_sample_delay, template_filename=template_filename, tail_name=tail_name, dataname=dataname, bias=bias, load_state_timestr=load_state_timestr, max_nsrc=max_nsrc,\
	 				trueminf=fmin, nregion=nregion, make_post_plots=make_post_plots, nsamp=nsamp, use_mask=use_mask,\
	 				residual_samples=residual_samples, float_fourier_comps=float_fourier_comps, fc_rel_amps=fc_rel_amps,\
	 				n_fourier_terms=n_fc_terms, fc_sample_delay=fc_sample_delay, fourier_comp_moveweight=fourier_comp_moveweight_main,\
	 				alph=alph, dfc_prob=dfc_prob_main, nsrc_init=nsrc_init, mask_file=mask_file, birth_death_sample_delay=birth_death_sample_delay, movestar_sample_delay=movestar_sample_delay,\
	 				 merge_split_sample_delay=merge_split_sample_delay, color_mus=color_mus, color_sigs=color_sigs, n_frames=n_frames, weighted_residual=weighted_residual, image_extnames=image_extnames, error_extname=error_extname, fc_prop_alpha=fc_prop_alpha, \
	 				 im_fpath=im_fpath, err_fpath=err_fpath, init_fourier_coeffs=init_fourier_coeffs, psf_fwhms=psf_fwhms, point_src_delay=point_src_delay_main, fc_amp_sig=fc_amp_sig, MP_order=MP_order, bkg_moore_penrose_inv=bkg_moore_penrose_inv, ridge_fac=ridge_fac, \
	 				 correct_misaligned_shift=correct_misaligned_shift, inject_diffuse_comp=inject_diffuse_comp, diffuse_comp_path=diffuse_comp_path, panel_list=panel_list, \
	 				 F_statistic_alph=F_statistic_alph, movestar_moveweight=movestar_moveweight, nominal_nsrc=nominal_nsrc, birth_death_moveweight=birth_death_moveweight, merge_split_moveweight=merge_split_moveweight, raw_counts=raw_counts, generate_condensed_catalog=generate_condensed_catalog, err_f_divfac=err_f_divfac, \
	 				 bkg_sig_fac=bkg_sig_fac, n_condensed_samp=n_condensed_samp, prevalence_cut=prevalence_cut, init_template_amplitude_dicts=initial_template_amplitude_dicts)

		ob.main()

	def run_sims_with_injected_sz(self, visual=False, show_input_maps=False, fmin=0.007, dataname='rxj1347_831', tail_name='rxj1347_PSW_nr_1_ext', \
				      template_names=['sze'], bias=[0.002, 0.002, 0.002], use_mask=True, max_nsrc=1000, make_post_plots=True, \
				      nsamp=2000, residual_samples=200, inject_sz_frac=1.0, float_fourier_comps=False, n_fc_terms=5, fc_amp_sig=None, inject_diffuse_comp=False, diffuse_comp_path=None, \
				      image_extnames=['SIGNAL'], add_noise=False, temp_sample_delay=30, initial_template_amplitude_dicts=None, \
				      color_mus=None, color_sigs=None, panel_list=None, nregion=5, burn_in_frac=0.7, err_f_divfac=1., template_moveweight=80., \
				      timestr_list_file=None, bkg_sig_fac=5.0, temp_prop_sig_fudge_facs=None, scalar_noise_sigma=None, \
				      movestar_moveweight=80, birth_death_moveweight=60, merge_split_moveweight=60, \
				      estimate_dust_first=False, nominal_nsrc=1000, nsamp_dustestimate=100, point_src_delay=30, fc_sample_delay=0., mask_file=None):

		''' Function for tests involving injecting SZ signals into mock data '''

		if initial_template_amplitude_dicts is None:
			initial_template_amplitude_dicts = dict({'sze': dict({'S':0.00, 'M':0.001, 'L':0.018})})



		if estimate_dust_first:
			
			panel_list = ['data0', 'model0', 'residual0', 'fourier_bkg0', 'residual_zoom0', 'dNdS0']

			# start with 250 micron image only

			ob = lion(band0=0, base_path=self.base_path, result_path=self.result_path, burn_in_frac=burn_in_frac, float_background=True, \
			  bkg_sample_delay=0, cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps, \
			  tail_name=tail_name, dataname=dataname, bias=bias, use_mask=use_mask, max_nsrc=max_nsrc, trueminf=fmin, nregion=nregion, \
			  make_post_plots=False, nsamp=nsamp_dustestimate, residual_samples=5, template_moveweight=template_moveweight, float_templates=False, \
			  inject_diffuse_comp=inject_diffuse_comp, diffuse_comp_path=diffuse_comp_path, image_extnames=image_extnames, add_noise=add_noise, \
			  panel_list=panel_list, err_f_divfac=err_f_divfac, bkg_sig_fac=bkg_sig_fac, \
			  scalar_noise_sigma=scalar_noise_sigma, movestar_moveweight=movestar_moveweight, nominal_nsrc=nominal_nsrc, birth_death_moveweight=birth_death_moveweight, merge_split_moveweight=merge_split_moveweight, \
			  float_fourier_comps=True, n_fourier_terms=n_fc_terms, fc_sample_delay=0., fourier_comp_moveweight=200., \
			  dfc_prob=1.0, nsrc_init=0, point_src_delay=point_src_delay, fc_amp_sig=fc_amp_sig, mask_file=mask_file)

			ob.main()
			_, filepath, _ = load_param_dict(ob.gdat.timestr, result_path=self.result_path)
			timestr = ob.gdat.timestr


			# use filepath from the last iteration to load estiamte of median background estimate
			chain = np.load(filepath+'/chain.npz')
			init_fourier_coeffs = np.median(chain['fourier_coeffs'][10:], axis=0)
			last_bkg_sample_250 = chain['bkg'][-1,0]

			print('last bkg sample is ', last_bkg_sample_250)

		else:
			init_fourier_coeffs = None

		dust_rel_SED = [self.dust_I_lams[self.band_dict[i]]/self.dust_I_lams[self.band_dict[0]] for i in range(3)]
		fc_rel_amps = [dust_rel_SED[i]*self.flux_density_conversion_dict[self.band_dict[i]]/self.flux_density_conversion_dict[self.band_dict[0]] for i in range(3)]

		panel_list = ['data0', 'data1', 'data2', 'residual0', 'residual1', 'residual2']


		ob = lion(band0=0, band1=1, band2=2, base_path=self.base_path, result_path=self.result_path, burn_in_frac=burn_in_frac, float_background=True, \
			  bkg_sample_delay=0, temp_sample_delay=temp_sample_delay, cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps, \
			  float_templates=True, template_names=template_names, init_template_amplitude_dicts=initial_template_amplitude_dicts, \
			  tail_name=tail_name, dataname=dataname, bias=bias, use_mask=use_mask, max_nsrc=max_nsrc, trueminf=fmin, nregion=nregion, \
			  make_post_plots=make_post_plots, nsamp=nsamp, residual_samples=residual_samples, inject_sz_frac=inject_sz_frac, template_moveweight=template_moveweight, \
			  inject_diffuse_comp=inject_diffuse_comp, diffuse_comp_path=diffuse_comp_path, image_extnames=image_extnames, add_noise=add_noise, \
			  color_mus=color_mus, color_sigs=color_sigs, init_fourier_coeffs=init_fourier_coeffs, panel_list=panel_list, err_f_divfac=err_f_divfac, timestr_list_file=timestr_list_file, bkg_sig_fac=bkg_sig_fac, temp_prop_sig_fudge_facs=temp_prop_sig_fudge_facs, \
			  scalar_noise_sigma=scalar_noise_sigma, movestar_moveweight=movestar_moveweight, nominal_nsrc=nominal_nsrc, birth_death_moveweight=birth_death_moveweight, merge_split_moveweight=merge_split_moveweight, \
			  float_fourier_comps=float_fourier_comps, fourier_comp_moveweight=0., dfc_prob=0.0, fc_rel_amps=fc_rel_amps, fc_sample_delay=fc_sample_delay, mask_file=mask_file)

		ob.main()


	def iter_fourier_comps(self, n_fc_terms=10, fc_amp_sig=0.0005, fmin_levels=[0.05, 0.02, 0.01, 0.007], final_fmin=0.007, \
							nsamps=[50, 100, 200, 500], final_nsamp=2000, \
							template_names=['sze'], nlast_fc=5, dataname='rxj1347_831', tail_name='rxj1347_PSW_nr_1_ext', \
							bias=[-0.004, -0.007, -0.008], max_nsrc=1000, visual=False, alph=1.0, show_input_maps=False, \
							inject_sz_frac=0.0, residual_samples=200, external_sz_file=False, timestr_list_file=None, \
							inject_diffuse_comp=False, nbands=3, mask_file=None, weighted_residual=False, diffuse_comp_path=None, float_templates=True, use_mask=True, image_extnames=['SIGNAL'], \
							n_frames_single=10, n_frames_joint=50, F_statistic_alph=False, nominal_nsrc = 700):


		''' 

		This function can be used to run PCAT with successive Fmin thresholds to enable stable estimation of structured background. From tests, I didn't see much of a difference
		using this approach versus just estimating background with no sources at all for some number of samples and then letting background parameters float. They usually converge to the same solution.

		Unique parameters
		-----------------

		fmin_levels : list, optional
			Specifies minimum flux density at each iteration of burn-in. Default is [0.05, 0.02, 0.01, 0.007] (in Jy).

		final_fmin : float, optional
			Final minimum flux density after iterative background estimation. Default is 0.007 [Jy].

		nsamps : list, optional
			Number of thinned samples for each step of the iteration. 
			Default is [50, 100, 200, 500]. 

		final_nsamp : float, optional
			Once iterative procedure completed, final_nsamp specifies number of thinned samples to run. Default is 2000.

		nlast_fc : int, optional
			Number of thinned samples from previous iteration to use to estimate current background level. Default is 5.
			(To be honest, this could probably just be 1, where the last state of the chain at a given Fmin is passed ot the next step.)


		Returns
		-------

		Nothing! This function is an inplace operation aside from normal PCAT outputs which are stored in the result_path.

		'''

		if residual_samples > final_nsamp:
			residual_samples = final_nsamp // 2
			print('residual samples changed to', residual_samples)

		if external_sz_file:
			template_filename=dict({'sze': self.sz_filename})
		else:
			template_filename = None

		timestr = None

		initial_template_amplitude_dicts = dict({'sze': dict({'S':0.00, 'M':0.002, 'L':0.025})})

		init_fc = np.zeros(shape=(n_fc_terms, n_fc_terms, 4))

		for i, fmin in enumerate(fmin_levels):
			if i==0:
				print('initial fourier coefficients set to zero')
				median_fc = init_fc
				point_src_delay = 30
			else:
				point_src_delay = 0
			
			panel_list = ['data0', 'model0', 'residual0', 'fourier_bkg0', 'residual_zoom0', 'dNdS0']

			# start with 250 micron image only
			ob = lion(band0=0, base_path=self.base_path, result_path=self.result_path, round_up_or_down='up', bolocam_mask=False, \
						float_background=True, burn_in_frac=0.75, bkg_sample_delay=0, float_templates=float_templates, template_moveweight=0.,\
		 				cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps, init_template_amplitude_dicts=initial_template_amplitude_dicts,\
		 				template_names=template_names, template_filename=template_filename, tail_name=tail_name, dataname=dataname, bias=None, load_state_timestr=timestr, max_nsrc=max_nsrc,\
		 				trueminf=fmin, nregion=5, make_post_plots=False, nsamp=nsamps[i], use_mask=use_mask,\
		 				residual_samples=5, init_fourier_coeffs=median_fc, float_fourier_comps=True, \
		 				n_fourier_terms=n_fc_terms, fc_sample_delay=0, fourier_comp_moveweight=200.,\
		 				alph=alph, dfc_prob=1.0, nsrc_init=0, mask_file=mask_file, point_src_delay=point_src_delay, \
		 				inject_sz_frac=0.0, raw_counts=True,nominal_nsrc = nominal_nsrc, F_statistic_alph=F_statistic_alph, fc_amp_sig=fc_amp_sig,n_frames=n_frames_single, panel_list=panel_list, weighted_residual=weighted_residual, image_extnames=image_extnames, inject_diffuse_comp=inject_diffuse_comp, diffuse_comp_path=diffuse_comp_path)

			ob.main()

			_, filepath, _ = load_param_dict(ob.gdat.timestr, result_path=self.result_path)
			timestr = ob.gdat.timestr

		# use filepath from the last iteration to load estiamte of median background estimate
		chain = np.load(filepath+'/chain.npz')
		median_fc = np.median(chain['fourier_coeffs'][-nlast_fc:], axis=0)
		last_bkg_sample_250 = chain['bkg'][-1,0]

		print('last bkg sample is ', last_bkg_sample_250)


		dust_rel_SED = [self.dust_I_lams[self.band_dict[i]]/self.dust_I_lams[self.band_dict[0]] for i in range(nbands)]
		fc_rel_amps = [dust_rel_SED[i]*self.flux_density_conversion_dict[self.band_dict[i]]/self.flux_density_conversion_dict[self.band_dict[0]] for i in range(nbands)]

		panel_list = ['data0', 'data1', 'data2', 'residual0', 'residual1', 'residual2']

		ob = lion(band0=0, band1=1, band2=2, base_path=self.base_path, result_path=self.result_path, round_up_or_down='up', \
					bolocam_mask=False, float_background=True, burn_in_frac=0.75, bkg_sample_delay=0,\
					cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps,\
					float_templates=float_templates, template_names=template_names, init_template_amplitude_dicts=initial_template_amplitude_dicts, \
					tail_name=tail_name, dataname=dataname, max_nsrc=max_nsrc, \
					init_fourier_coeffs=median_fc, template_filename=template_filename, trueminf=final_fmin, nregion=5, \
				    make_post_plots=False, nsamp=final_nsamp, use_mask=use_mask, residual_samples=residual_samples, \
				    float_fourier_comps=True, n_fourier_terms=n_fc_terms, fc_sample_delay=0, \
				    fourier_comp_moveweight=0., point_src_delay=0, temp_sample_delay=20, \
				    alph=alph, dfc_prob=0.0, fc_rel_amps=fc_rel_amps, inject_sz_frac=inject_sz_frac, timestr_list_file=timestr_list_file, \
				     inject_diffuse_comp=inject_diffuse_comp,F_statistic_alph=F_statistic_alph, nominal_nsrc = nominal_nsrc, n_frames=n_frames_joint, panel_list=panel_list, mask_file=mask_file, image_extnames=image_extnames, diffuse_comp_path=diffuse_comp_path, bias=[last_bkg_sample_250, 0.003, 0.003], load_state_timestr=timestr)

		ob.main()

