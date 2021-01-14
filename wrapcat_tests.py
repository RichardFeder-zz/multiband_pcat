from pcat_spire import *
from diffuse_gen import *
from spire_plotting_fns import *
import pandas as pd

''' this script is for testing the iterative background estimation suggested by Mike. the main idea is to 
gradually reduce the minimum flux threshold of the cataloger while fitting fourier coefficients to observations at 
250 micron. as Fmin gets reduced, the fourier coefficients shift as they do not need to model previously sub-Fmin sources.
After a few iterations, the best fit template is saved and used to model the background of all three bands. once the multiband fitting begins, 
the colors of the background model are fixed and so are its fourier coefficients.'''

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


	def empirical_parsimony_prior_estimate(self, band0=0, band1=None, band2=None, fmin=0.005, nsamp=2000, \
											 residual_samples=200, tail_name='rxj1347_PSW_sim0300', dataname='v3_sides_sims_corrected'):

		# run pcat in fixed dimensional mode for a range of source numbers
		# then collect average log-likelihood from last residual_samples samples


		ob = lion(band0=band0, band1=band1, band2=band2, nsrc_init=N, alph=0.0, birth_death_moveweight=0., merge_split_moveweight=0., \
			base_path=self.base_path, result_path=self.result_path, burn_in_frac=0.7, float_background=True, \
			  bkg_sample_delay=0, cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps, \
			  tail_name=tail_name, dataname=dataname, use_mask=use_mask, max_nsrc=N, trueminf=fmin, nregion=5, \
			  make_post_plots=make_post_plots, nsamp=nsamp, residual_samples=residual_samples)

		ob.main()

		_, filepath, _ = load_param_dict(ob.gdat.timestr, result_path=self.result_path)
		timestr = ob.gdat.timestr
		chain = np.load(filepath+'/chain.npz')

		nb = 0 
		for band in [band0, band1, band2]:
			if band is not None:
				nb += 1


		last_chi2_vals = chain['chi2']

		av_logLs = [np.mean(chain['chi2'][b, -residual_samples:]) for b in range(nb)]
		
		return av_logLs



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

	def validate_astrometry(self, band0=0, band1=1, band2=2, tail_name='rxj1347_PSW_nr_1_ext', dataname='rxj1347_831', use_mask=True, nregion=5, auto_resize=True,\
						ngrid=20, return_validation_figs=True):

		ob = lion(band0=band0, band1=band1, band2=band2, base_path=self.base_path, result_path=self.result_path, cblas=self.cblas, openblas=self.openblas, \
			  tail_name=tail_name, dataname=dataname, use_mask=use_mask, nregion=nregion)

		# ob.data.fast_astrom = wcs_astrometry(auto_resize, nregion=nregion)

		# ob.data.fast_astrom.load_wcs_header_and_dim(file_path, round_up_or_down=gdat.round_up_or_down)
		# separate for 

		for b in range(ob.gdat.nbands - 1):

			ob.data.fast_astrom.fit_astrom_arrays(b+1, 0, bounds0=ob.gdat.bounds[b+1], bounds1=ob.gdat.bounds[0])

		xr = np.arange(0, ob.gdat.imszs[0][0], ngrid)

		yr = np.arange(0, ob.gdat.imszs[0][1], ngrid)

		xv, yv = np.meshgrid(xr, yr)


		validation_figs = []

		for b in range(ob.gdat.nbands - 1):

			xnew, ynew = ob.data.fast_astrom.transform_q(xv, yv, b)

			x0, y0 = ob.data.fast_astrom.transform_q(np.array([0]), np.array([0]), b)
			print(x0, y0)

			# xnew -= np.min(xnew)
			# ynew -= np.min(ynew)

			# print(np.min(xnew), np.min(ynew))

			xv_rt, yv_rt = ob.data.fast_astrom.transform_q(xnew, ynew, ob.gdat.nbands-1+b)

			xnew_wcs, ynew_wcs = ob.data.fast_astrom.obs_to_obs(0, b+1, xv, yv)

			f = plt.figure(figsize=(12,5))
			plt.subplot(1,2,1)
			plt.title('band 0')
			plt.imshow(ob.data.data_array[0]-np.median(ob.data.data_array[0]), cmap='Greys', vmin=-0.005, vmax=0.02, origin='lower')
			plt.xlim(0, ob.data.data_array[0].shape[0]/2)
			plt.ylim(0, ob.data.data_array[0].shape[1]/2)

			plt.colorbar()
			plt.scatter(xv, yv, marker='x', color='r')
			plt.subplot(1,2,2)
			plt.title('band '+str(b+1))
			plt.imshow(ob.data.data_array[b+1]-np.median(ob.data.data_array[b+1]), cmap='Greys', vmin=-0.005, vmax=0.02, origin='lower')
			plt.colorbar()
			# plt.scatter(xnew, ynew, marker='x', color='r')
			plt.scatter(xnew_wcs, ynew_wcs, marker='x', color='r', label='WCS')
			plt.legend()
			plt.xlim(0, ob.data.data_array[b+1].shape[0]/2)
			plt.ylim(0, ob.data.data_array[b+1].shape[1]/2)

			plt.show()

			validation_figs.append(f)

		if return_validation_figs:
			return validation_figs



	def run_mocks_and_compare_numbercounts(self, visual=False, show_input_maps=False, fmin=0.007, dataname='sims_12_2_20', tail_name='rxj1347_PSW_nr_1_ext', \
		use_mask=False, bias=[0.002, 0.002, 0.002], max_nsrc=1000, make_post_plots=True, nsamp=2000, residual_samples=300, image_extnames=['SIG_PRE_LENSE', 'NOISE'], \
		add_noise=False, sim_idx=None, mockcat_path=None):

		mockcat_fpath = None
		if mockcat_path is not None:
			mockcat_fpath = gdat.base_path+mockcat_path

		ob = lion(band0=0, band1=1, band2=2, base_path=self.base_path, result_path=self.result_path, burn_in_frac=0.7, float_background=True, \
			  bkg_sample_delay=0, cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps, \
			  tail_name=tail_name, dataname=dataname, bias=bias, use_mask=use_mask, max_nsrc=max_nsrc, trueminf=fmin, nregion=5, \
			  make_post_plots=make_post_plots, nsamp=nsamp, residual_samples=residual_samples, \
			  image_extnames=image_extnames, add_noise=add_noise, mockcat_fpath=mockcat_fpath)

		ob.main()


	def run_sims_with_injected_sz(self, visual=False, show_input_maps=False, fmin=0.007, dataname='rxj1347_831', tail_name='rxj1347_PSW_nr_1_ext', \
				      template_names=['sze'], bias=[0.002, 0.002, 0.002], use_mask=True, max_nsrc=1000, make_post_plots=True, \
				      nsamp=2000, residual_samples=200, inject_sz_frac=1.0, inject_diffuse_comp=False, diffuse_comp_path=None, \
				      image_extnames=['SIGNAL'], add_noise=False, temp_sample_delay=50):



		initial_template_amplitude_dicts = dict({'sze': dict({'S':0.00, 'M':0.001, 'L':0.018})})

		ob = lion(band0=0, band1=1, band2=2, base_path=self.base_path, result_path=self.result_path, burn_in_frac=0.7, float_background=True, \
			  bkg_sample_delay=0, temp_sample_delay=temp_sample_delay, cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps, \
			  float_templates=True, template_names=template_names, init_template_amplitude_dicts=initial_template_amplitude_dicts, \
			  tail_name=tail_name, dataname=dataname, bias=bias, use_mask=use_mask, max_nsrc=max_nsrc, trueminf=fmin, nregion=5, \
			  make_post_plots=make_post_plots, nsamp=nsamp, residual_samples=residual_samples, inject_sz_frac=inject_sz_frac, template_moveweight=40., \
			  inject_diffuse_comp=inject_diffuse_comp, diffuse_comp_path=diffuse_comp_path, image_extnames=image_extnames, add_noise=add_noise)

		ob.main()

	def artificial_star_test(self, n_src_perbin=10, inject_fmin=0.01, inject_fmax=0.2, nbins=20, fluxbins=None, frac_flux_thresh=0.2, pos_thresh=0.5,\
		band0=0, band1=None, band2=None, fmin=0.01, nsamp=500, load_timestr=None, dataname='SMC_HERITAGE', tail_name='SMC_HERITAGE_mask2_PSW', \
		bias = None, max_nsrc=1000, visual=False, alph=1.0, show_input_maps=False, make_post_plots=True, \
		residual_samples=50, float_background=True, timestr_list_file=None, \
		nbands=None, mask_file=None, use_mask=True, image_extnames=['IMAGE'], float_fourier_comps=False, n_fc_terms=10, fc_sample_delay=0, \
		point_src_delay=0, nsrc_init=None, fc_prop_alpha=None, fourier_comp_moveweight=200., fc_amp_sig=0.001, n_frames=10, color_mus=None, color_sigs=None, im_fpath=None, err_fpath=None, \
		bkg_moore_penrose_inv=True, MP_order=5, ridge_fac=2., inject_catalog_path=None):

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
		 				n_fourier_terms=n_fc_terms, show_fc_temps=False, fc_sample_delay=fc_sample_delay, fourier_comp_moveweight=fourier_comp_moveweight,\
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
						catalog_inject[idxctr:idxctr+int(n_src_perbin[f]), 2] = np.array(np.random.uniform(fluxbins[f], fluxbins[f+1], n_src_perbin[f]), dtype=np.float32)
						idxctr += int(n_src_perbin[f])
					else:
						catalog_inject[f*n_src_perbin:(f+1)*n_src_perbin, 2] = np.array(np.random.uniform(fluxbins[f], fluxbins[f+1], n_src_perbin), dtype=np.float32)


			ob.gdat.catalog_inject = catalog_inject.copy()


		else:
			catalog_inject = np.load(self.result_path+load_timestr+'/inject_catalog.npz')['catalog_inject']
			ob.gdat.timestr = load_timestr
			print('ob.gdat.timestr is ', ob.gdat.timestr)

		flux_bin_idxs = [np.where((catalog_inject[:,2] > fluxbins[i])&(catalog_inject[:,2] < fluxbins[i+1]))[0] for i in range(len(fluxbins)-1)]

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

			inject_src_image, diff2 = image_model_eval(catalog_inject[:,0].astype(np.float32()), catalog_inject[:,1].astype(np.float32()), np.array(model.pixel_per_beam*ob.data.ncs[0]*catalog_inject[:,2]).astype(np.float32()), 0., model.imszs[0], \
													ob.data.ncs[0], np.array(ob.data.cfs[0]).astype(np.float32()), weights=np.array(ob.data.weights[0]).astype(np.float32()), \
													ref=resid, lib=libmmult.pcat_model_eval, regsize=model.regsizes[0], \
													margin=0, offsetx=0, offsety=0, template=None)

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
				ob.data.data_array[b] += inject_src_image

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

		completeness_ensemble = np.zeros((residual_samples, catalog_inject.shape[0]))

		fluxerror_ensemble = np.zeros((residual_samples, catalog_inject.shape[0]))

		for i in range(residual_samples):

			for j in range(catalog_inject.shape[0]):

				# make position cut 
				idx_pos = np.where(np.sqrt((xsrcs[-i] - catalog_inject[j,0])**2 +(ysrcs[-i] - catalog_inject[j,1])**2)  < pos_thresh)[0]
				fluxes_poscutpass = fsrcs[0][-i][idx_pos]

				# make flux cut
				mask_flux = np.where(np.abs(fluxes_poscutpass - catalog_inject[j,2])/catalog_inject[j,2] < frac_flux_thresh)[0]


				if len(mask_flux) >= 1:

					# print('we got one! true source is ', catalog_inject[j])
					# print('while PCAT source is ', xsrcs[-i][idx_pos][mask_flux], ysrcs[i][idx_pos][mask_flux], fluxes_poscutpass[mask_flux])

					completeness_ensemble[i,j] = 1.

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

					# fluxerror_ensemble[i,j] = brighter_flux/catalog_inject[j,2]
					# fluxerror_ensemble[i,j] = (brighter_flux-catalog_inject[j,2])/catalog_inject[j,2]
					fluxerror_ensemble[i,j] = (mindist_flux-catalog_inject[j,2])/catalog_inject[j,2]



		mean_frac_flux_error = np.zeros((catalog_inject.shape[0],))
		pct_16_fracflux = np.zeros(catalog_inject.shape[0],)
		pct_84_fracflux = np.zeros(catalog_inject.shape[0],)

		prevalences = []

		for j in range(catalog_inject.shape[0]):
			nonzero_fidx = np.where(fluxerror_ensemble[:,j] != 0)[0]
			prevalences.append(float(len(nonzero_fidx))/float(residual_samples))
			if len(nonzero_fidx) > 0:
				mean_frac_flux_error[j] = np.median(fluxerror_ensemble[nonzero_fidx, j])
				pct_16_fracflux[j] = np.percentile(fluxerror_ensemble[nonzero_fidx, j], 16)
				pct_84_fracflux[j] = np.percentile(fluxerror_ensemble[nonzero_fidx, j], 84)

		nonzero_ferridx = np.where(mean_frac_flux_error != 0)[0]

		prevalences = np.array(prevalences)

		mean_ferr_nonzero_ferridx = mean_frac_flux_error[nonzero_ferridx]
		pct_16_nonzero_ferridx = pct_16_fracflux[nonzero_ferridx]
		pct_84_nonzero_ferridx = pct_84_fracflux[nonzero_ferridx]

		yerrs = [mean_ferr_nonzero_ferridx - pct_16_nonzero_ferridx, pct_84_nonzero_ferridx-mean_ferr_nonzero_ferridx]

		mean_frac_flux_error_binned = np.zeros((len(fluxbins)-1))
		pct16_frac_flux_error_binned = np.zeros((len(fluxbins)-1))
		pct84_frac_flux_error_binned = np.zeros((len(fluxbins)-1))

		for f in range(len(fluxbins)-1):

			finbin = np.where((catalog_inject[nonzero_ferridx, 2] >= fluxbins[f])&(catalog_inject[nonzero_ferridx, 2] < fluxbins[f+1]))[0]
			
			mean_frac_flux_error_binned[f] = np.median(mean_ferr_nonzero_ferridx[finbin])

			pct16_frac_flux_error_binned[f] = np.percentile(mean_ferr_nonzero_ferridx[finbin], 16)

			pct84_frac_flux_error_binned[f] = np.percentile(mean_ferr_nonzero_ferridx[finbin], 84)


		g = plt.figure(figsize=(9, 6))
		# plt.title('Blank field test', fontsize=18)
		# plt.title('GOODS N', fontsize=18)
		# plt.title('HERITAGE Survey - SMC', fontsize=18)

		# plt.axhline(1.0, linestyle='dashed', color='b')

		plt.axhline(0.0, linestyle='solid', color='grey', alpha=0.4, linewidth=3, zorder=-10)
		# plt.errorbar(catalog_inject[nonzero_ferridx,2]*1e3, mean_frac_flux_error[nonzero_ferridx],\
		# 				 yerr=yerrs, color='k', c=prevalences[nonzero_ferridx], marker='.', fmt='.', capsize=2, alpha=0.2)
		# plt.errorbar(catalog_inject[nonzero_ferridx,2]*1e3, mean_frac_flux_error[nonzero_ferridx],\
		# 				 yerr=yerrs, color='k', marker='.', markersize=80*prevalences[nonzero_ferridx], fmt='.', capsize=2, alpha=0.2)
		# plt.errorbar(catalog_inject[nonzero_ferridx,2]*1e3, mean_frac_flux_error[nonzero_ferridx],\
		# 				 yerr=yerrs, color='k', marker='.',  fmt='none', capsize=2, alpha=0.4)
		# # plt.scatter(catalog_inject[nonzero_ferridx,2]*1e3, mean_frac_flux_error[nonzero_ferridx],\
		# 				 c=prevalences[nonzero_ferridx], marker='.', s=80, label='PCAT x Truth catalog (GOODS-N)')
		# plt.scatter(catalog_inject[nonzero_ferridx,2]*1e3, mean_frac_flux_error[nonzero_ferridx],\
						 # c=prevalences[nonzero_ferridx], alpha=0.7, marker='.', s=80, label='PCAT x Injected catalog (SMC)')
		# plt.scatter(catalog_inject[nonzero_ferridx,2]*1e3, mean_frac_flux_error[nonzero_ferridx],\
		# 				 s=160*prevalences[nonzero_ferridx], alpha=0.4, marker='.', color='k', label='PCAT x Injected catalog (SMC, 15th order FCs)')
		# # cbar = plt.colorbar()
		# cbar.set_label('Prevalence', fontsize=14)
		geom_mean = np.sqrt(fluxbins[1:]*fluxbins[:-1])
		xerrs = [[1e3*(geom_mean[f] - fluxbins[f]) for f in range(len(geom_mean))], [1e3*(fluxbins[f+1] - geom_mean[f]) for f in range(len(geom_mean))]]


		# plt.errorbar(geom_mean*1e3, mean_frac_flux_error_binned, xerr=xerrs, \
		# 				 yerr=[mean_frac_flux_error_binned-pct16_frac_flux_error_binned, pct84_frac_flux_error_binned-mean_frac_flux_error_binned], fmt='.', color='C3', \
		# 				 linewidth=3, capsize=5, alpha=1., capthick=2, label='PCAT (GOODS-N) \n single band fit, averaged', markersize=15)
		plt.errorbar(geom_mean*1e3, mean_frac_flux_error_binned, xerr=xerrs, \
						 yerr=[mean_frac_flux_error_binned-pct16_frac_flux_error_binned, pct84_frac_flux_error_binned-mean_frac_flux_error_binned], fmt='.', color='C3', \
						 linewidth=3, capsize=5, alpha=1., capthick=2, label='PCAT (SMC, 15th order FCs) \n single band fit, averaged', markersize=15)
		
		heritage_pcat = np.load('spire_results/20210109-134459/fracflux_errs_dpos='+str(np.round(pos_thresh, 1))+'_heritage.npz')
		geom_mean = heritage_pcat['geom_mean']
		mean_frac_flux_error_binned = heritage_pcat['mean_frac_flux_error_binned']
		pct16_frac_flux_error_binned = heritage_pcat['pct16_frac_flux_error_binned']
		pct84_frac_flux_error_binned = heritage_pcat['pct84_frac_flux_error_binned']
		fluxbins = heritage_pcat['fluxbins']
		xerrs = [[1e3*(geom_mean[f] - fluxbins[f]) for f in range(len(geom_mean))], [1e3*(fluxbins[f+1] - geom_mean[f]) for f in range(len(geom_mean))]]


		plt.errorbar(geom_mean*1e3, mean_frac_flux_error_binned, xerr=xerrs, \
						 yerr=[mean_frac_flux_error_binned-pct16_frac_flux_error_binned, pct84_frac_flux_error_binned-mean_frac_flux_error_binned], fmt='.', color='k', \
						 linewidth=3, capsize=5, alpha=1., capthick=2, label='PCAT (SMC, 10th order FCs) \n single band fit, averaged', markersize=15)


		# np.savez('spire_results/'+load_timestr+'/fracflux_errs_dpos='+str(np.round(pos_thresh, 1))+'_heritage.npz', geom_mean=geom_mean, mean_frac_flux_error_binned=mean_frac_flux_error_binned, \
		# 		pct16_frac_flux_error_binned=pct16_frac_flux_error_binned, pct84_frac_flux_error_binned=pct84_frac_flux_error_binned, fluxbins=fluxbins)

		import pandas as pd

		roseboom_medianx = np.array(pd.read_csv('~/Downloads/median_roseboom_250.csv', header=None)[0])
		roseboom_mediany = np.array(pd.read_csv('~/Downloads/median_roseboom_250.csv', header=None)[1])


		rb_pct16x = np.array(pd.read_csv('~/Downloads/roseboom_16_250.csv', header=None)[0])
		rb_pct16y = np.array(pd.read_csv('~/Downloads/roseboom_16_250.csv', header=None)[1])

		rb_pct84x = np.array(pd.read_csv('~/Downloads/roseboom_84_250.csv', header=None)[0])
		rb_pct84y = np.array(pd.read_csv('~/Downloads/roseboom_84_250.csv', header=None)[1])

		xid_medianx = np.array(pd.read_csv('~/Downloads/xid_median_250.csv', header=None)[0])
		xid_mediany = np.array(pd.read_csv('~/Downloads/xid_median_250.csv', header=None)[1])

		xid_pct16x = np.array(pd.read_csv('~/Downloads/xid_16_250.csv', header=None)[0])
		xid_pct16y = np.array(pd.read_csv('~/Downloads/xid_16_250.csv', header=None)[1])

		xid_pct84x = np.array(pd.read_csv('~/Downloads/xid_84_250.csv', header=None)[0])
		xid_pct84y = np.array(pd.read_csv('~/Downloads/xid_84_250.csv', header=None)[1])


		plt.plot(roseboom_medianx, roseboom_mediany/roseboom_medianx, label='XID (Deep) \n Roseboom et al. 2010', color='C2', marker='.', linewidth=3, markersize=15)
		plt.fill_between(rb_pct16x, rb_pct16y/roseboom_medianx, rb_pct84y/roseboom_medianx, color='C2', alpha=0.3)


		plt.plot(xid_medianx, xid_mediany, label='XID+ (COSMOS) \n Hurley et al. 2016', color='b', marker='.', linewidth=3, markersize=15)
		plt.fill_between(xid_pct16x, xid_pct16y, xid_pct84y, color='b', alpha=0.3)


		plt.legend(fontsize=13)
		plt.xlabel('$S_{250}^{True}$ [mJy]', fontsize=18)
		plt.ylabel('$(S_{250}^{Obs} - S_{250}^{True})/S_{250}^{True}$', fontsize=16)
		plt.xscale('log')
		# plt.yscale('log')
		# plt.ylim(0.1, 10.0)
		# plt.ylim(-1.5, 2)
		plt.ylim(-0.5, 0.5)
		# plt.xlim(1e-3, 10)
		# plt.xlim(3, 700)
		plt.xlim(10, 700)
		plt.tight_layout()
		plt.show()
		# g.savefig(filepath+'/fluxerr_vs_fluxdensity_dx='+str(pos_thresh)+'_dS-S='+str(frac_flux_thresh)+'_mindist.pdf', bbox_inches='tight')


		xstack = xsrcs[-20:,:].ravel()
		ystack = ysrcs[-20:,:].ravel()
		fstack = fsrcs[0][-20:,:].ravel()

		# average the completeness for each source over catalog samples and bin into fluxes
		avg_completeness = np.mean(completeness_ensemble, axis=0)
		std_completeness = np.std(completeness_ensemble, axis=0)
		completeness_vs_flux = [np.mean(avg_completeness[fbin_idxs]) for fbin_idxs in flux_bin_idxs]
		cvf_std = [np.mean(std_completeness[fbin_idxs]) for fbin_idxs in flux_bin_idxs]

		print(fluxbins)
		print(completeness_vs_flux)


		f = plt.figure(figsize=(10, 5.5))
		plt.subplot(1,2, 1)
		# plt.title('GOODS-N', fontsize=16)
		plt.title('HERITAGE Survey - SMC', fontsize=16)
		plt.imshow(91.*ob.data.data_array[0][:-5, :-5]-91.*np.nanmean(ob.data.data_array[0][:-5, :-5]), cmap='Greys', vmax=0.02*91., origin='lower')
		plt.xlabel('x [pix]', fontsize=14)
		plt.ylabel('y [pix]', fontsize=14)
		cbar = plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.12)
		cbar.set_label('MJy/sr', fontsize=14)
		plt.scatter(catalog_inject[:,0], catalog_inject[:, 1], marker='+', color='b', s=2e2*catalog_inject[:,2], label='Injected sources')
		plt.scatter(xstack, ystack, marker='x', color='r', s=2e2*fstack, alpha=0.05)
		plt.scatter([0], [0], marker='x', color='r', label='PCAT')
		plt.legend(loc=1)
		plt.xlim(0, ob.data.data_array[0].shape[0]-5)
		plt.ylim(0, ob.data.data_array[0].shape[0]-5)
		plt.subplot(1,2, 2)
		plt.title('$|\\delta \\vec{x}| < $'+str(np.round(pos_thresh, 2))+', $|\\delta S/S| < $'+str(np.round(frac_flux_thresh, 2)), fontsize=16)
		plt.errorbar(1e3*np.sqrt(fluxbins[:-1]*fluxbins[1:]), completeness_vs_flux, yerr=cvf_std/np.sqrt(completeness_ensemble.shape[0]), color='r', marker='x', capsize=5, label='PCAT')
		plt.plot([20, 50, 100, 200, 500, 1000], [0, 0.9, 1.0, 1.0, 1.0, 1.0], marker='+', color='k', markersize=10, label='Meixner et al. 2013 \n (low background)')
		import pandas as pd

		# psc_spire_cosmos_comp = pd.read_csv('~/Downloads/PSW_completeness.csv', header=None)
		# plt.plot(np.array(psc_spire_cosmos_comp[0]), np.array(psc_spire_cosmos_comp[1]), marker='.', markersize=10, label='SPIRE Point Source Catalog (2017)', color='k')
		plt.xscale('log')
		plt.xlabel('$S_{250}$ [mJy]', fontsize=14)
		plt.ylabel('Completeness', fontsize=14)
		# plt.xlim(4, 1e3)
		plt.xlim(15, 1e3)
		plt.legend()
		plt.ylim(-0.05, 1.05)
		plt.tight_layout()
		plt.show()

		f.savefig(filepath+'/completeness_vs_fluxdensity_dx='+str(pos_thresh)+'_dS-S='+str(frac_flux_thresh)+'.pdf', bbox_inches='tight')


		return fluxbins, completeness_vs_flux, f


	def real_dat_run(self, band0=0, band1=None, band2=None, fmin=0.007, nsamp=500, template_names=None, dataname='rxj1347_831', tail_name='rxj1347_PSW_nr_1_ext', \
		bias = [-0.004, -0.007, -0.008], max_nsrc=1000, visual=False, alph=1.0, show_input_maps=False, make_post_plots=True, \
		inject_sz_frac=0.0, residual_samples=50, float_background=True, timestr_list_file=None, \
		nbands=None, mask_file=None, weighted_residual=False, float_templates=True, use_mask=True, image_extnames=['SIGNAL'], \
		float_fourier_comps=False, n_fc_terms=10, fc_sample_delay=0, fourier_comp_moveweight=200., \
		template_moveweight=40., template_filename=None, \
		bkg_sample_delay=0, birth_death_sample_delay=0, movestar_sample_delay=0, merge_split_sample_delay=0, \
		load_state_timestr=None, nsrc_init=None, fc_prop_alpha=None, fc_amp_sig=0.0001, n_frames=10, color_mus=None, color_sigs=None, im_fpath=None, err_fpath=None, \
		bkg_moore_penrose_inv=True, MP_order=5., ridge_fac=None, point_src_delay=0, nregion=5):

		if nbands is None:
			nbands = 0
			if band0 is not None:
				nbands += 1
			if band1 is not None:
				nbands += 1
			if band2 is not None:
				nbands += 1

		ob = lion(band0=band0, band1=band1, band2=band2, base_path=self.base_path, result_path=self.result_path, round_up_or_down='down', \
					float_background=float_background, burn_in_frac=0.75, bkg_sample_delay=bkg_sample_delay, float_templates=float_templates, template_moveweight=template_moveweight, \
	 				cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps, \
	 				template_names=template_names, template_filename=template_filename, tail_name=tail_name, dataname=dataname, bias=bias, load_state_timestr=load_state_timestr, max_nsrc=max_nsrc,\
	 				trueminf=fmin, nregion=nregion, make_post_plots=make_post_plots, nsamp=nsamp, use_mask=use_mask,\
	 				residual_samples=residual_samples, float_fourier_comps=float_fourier_comps, \
	 				n_fourier_terms=n_fc_terms, show_fc_temps=False, fc_sample_delay=fc_sample_delay, fourier_comp_moveweight=fourier_comp_moveweight,\
	 				alph=alph, dfc_prob=1.0, nsrc_init=nsrc_init, mask_file=mask_file, birth_death_sample_delay=birth_death_sample_delay, movestar_sample_delay=movestar_sample_delay,\
	 				 merge_split_sample_delay=merge_split_sample_delay, color_mus=color_mus, color_sigs=color_sigs, n_frames=n_frames, raw_counts=False, weighted_residual=weighted_residual, image_extnames=image_extnames, fc_prop_alpha=fc_prop_alpha, \
	 				 im_fpath=im_fpath, err_fpath=err_fpath, point_src_delay=point_src_delay, fc_amp_sig=fc_amp_sig, MP_order=MP_order, bkg_moore_penrose_inv=bkg_moore_penrose_inv, ridge_fac=ridge_fac)

		ob.main()




	def iter_fourier_comps(self, n_fc_terms=10, fmin_levels=[0.05, 0.02, 0.01, 0.007], final_fmin=0.007, \
							nsamps=[50, 100, 200, 500], final_nsamp=2000, \
							template_names=['sze'], nlast_fc=20, dataname='rxj1347_831', tail_name='rxj1347_PSW_nr_1_ext', \
							bias=[-0.004, -0.007, -0.008], max_nsrc=1000, visual=False, alph=1.0, show_input_maps=False, \
							inject_sz_frac=0.0, residual_samples=200, external_sz_file=False, timestr_list_file=None, \
							inject_diffuse_comp=False, nbands=3, mask_file=None, weighted_residual=False, diffuse_comp_path=None, float_templates=True, use_mask=True, image_extnames=['SIGNAL']):



		if residual_samples > final_nsamp:
			residual_samples = final_nsamp // 2
			print('residual samples changed to', residual_samples)

		if external_sz_file:
			template_filename=dict({'sze': self.sz_filename})
		else:
			template_filename = None
		timestr = None

		initial_template_amplitude_dicts = dict({'sze': dict({'S':0.00, 'M':0.001, 'L':0.018})})

		init_fc = np.zeros(shape=(n_fc_terms, n_fc_terms, 4))

		for i, fmin in enumerate(fmin_levels):
			if i==0:
				print('initial fourier coefficients set to zero')
				median_fc = init_fc

			# start with 250 micron image only
			ob = lion(band0=0, base_path=self.base_path, result_path=self.result_path, round_up_or_down='down', bolocam_mask=False, \
						float_background=True, burn_in_frac=0.75, bkg_sample_delay=0, float_templates=float_templates, template_moveweight=0.,\
		 				cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps, init_template_amplitude_dicts=initial_template_amplitude_dicts,\
		 				template_names=template_names, template_filename=template_filename, tail_name=tail_name, dataname=dataname, bias=None, load_state_timestr=timestr, max_nsrc=max_nsrc,\
		 				trueminf=fmin, nregion=5, make_post_plots=False, nsamp=nsamps[i], use_mask=use_mask,\
		 				residual_samples=5, init_fourier_coeffs=median_fc, n_frames=3, float_fourier_comps=True, \
		 				n_fourier_terms=n_fc_terms, show_fc_temps=False, fc_sample_delay=0, fourier_comp_moveweight=200.,\
		 				alph=alph, dfc_prob=1.0, nsrc_init=0, mask_file=mask_file, birth_death_sample_delay=50, movestar_sample_delay=50, merge_split_sample_delay=50, \
		 				inject_sz_frac=1.0, raw_counts=True, weighted_residual=weighted_residual, image_extnames=image_extnames, inject_diffuse_comp=inject_diffuse_comp, diffuse_comp_path=diffuse_comp_path)

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

		ob = lion(band0=0, band1=1, band2=2, base_path=self.base_path, result_path=self.result_path, round_up_or_down='down', \
					bolocam_mask=False, float_background=True, burn_in_frac=0.75, bkg_sample_delay=0,\
					cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps,\
					float_templates=float_templates, template_names=template_names, init_template_amplitude_dicts=initial_template_amplitude_dicts, \
					tail_name=tail_name, dataname=dataname, max_nsrc=max_nsrc, \
					init_fourier_coeffs=median_fc, template_filename=template_filename, trueminf=final_fmin, nregion=5, \
				    make_post_plots=False, nsamp=final_nsamp, use_mask=use_mask, residual_samples=residual_samples, \
				    float_fourier_comps=True, show_fc_temps=False, n_fourier_terms=n_fc_terms, fc_sample_delay=0, \
				    fourier_comp_moveweight=0., movestar_sample_delay=0, merge_split_sample_delay=0, birth_death_sample_delay=0,\
				    alph=alph, dfc_prob=0.0, fc_rel_amps=fc_rel_amps, inject_sz_frac=1.0, timestr_list_file=timestr_list_file, \
				     inject_diffuse_comp=inject_diffuse_comp, mask_file=mask_file, image_extnames=image_extnames, diffuse_comp_path=diffuse_comp_path, bias=[last_bkg_sample_250, 0.003, 0.003], load_state_timestr=timestr)

		ob.main()

base_path='/Users/luminatech/Documents/multiband_pcat/'
result_path='/Users/luminatech/Documents/multiband_pcat/spire_results/'


# mask_file = base_path+'data/spire/gps_0/gps_0_PSW_mask.fits'
# mask_file = 'Data/spire/SMC_HERITAGE/SMC_HERITAGE_mask2_PSW.fits'
mask_file = base_path+'/data/spire/GOODSN/GOODSN_PSW_mask.fits'

# mask_file=None
# mask_file= None
# timestr_list_file='lensed_no_dust_gen3sims_rxj1347_11_10_20_timestrs.npz'
# started_idx_list_file = 'lensed_no_dust_gen3sims_rxj1347_11_10_20_simidxs.npz' 
# started_idx_list_file = 'simidxs.npz' 

sim_idx = 203
# pcat_test = pcat_test_suite(cluster_name='SMC_HERITAGE', base_path=base_path, result_path=result_path, cblas=True, openblas=False)
pcat_test = pcat_test_suite(cluster_name='GOODSN', base_path=base_path, result_path=result_path, cblas=True, openblas=False)

# figs = pcat_test.validate_astrometry(tail_name='rxj1347_PSW_sim0'+str(sim_idx), dataname='sims_12_2_20', ngrid=10, return_validation_figs=True)
# figs[0].savefig('test0.pdf')
# figs[1].savefig('test1.pdf')

# pcat_test.run_sims_with_injected_sz(dataname='gen_2_sims', add_noise=True, temp_sample_delay=10, image_extnames=['SIGNAL'], tail_name='rxj1347_PSW_sim0'+str(sim_idx), visual=True, show_input_maps=False)
# color_prior_sigs = dict({'S-M':1.5, 'M-L':1.5, 'L-S':1.5, 'M-S':1.5, 'S-L':1.5, 'L-M':1.5})

color_prior_sigs = dict({'S-M':0.5, 'M-L':0.5, 'L-S':0.5, 'M-S':0.5, 'S-L':0.5, 'L-M':0.5})

# pcat_test.iter_fourier_comps(dataname='GOODSN', tail_name='GOODSN_image_SMAP_PSW',nsamps=[50, 100, 200], float_templates=False, template_names=None, visual=False, show_input_maps=False, fmin_levels=[0.02,0.01, 0.005], final_fmin=0.003, alph=0.0, use_mask=True, image_extnames=['IMAGE'], max_nsrc=2500)
# pcat_test.real_dat_run(band0=0, band1=1, band2=2, nbands=3, dataname='GOODSN', tail_name='GOODSN_image_SMAP_PSW', float_fourier_comps=False,\
# 						 use_mask=True, bias=None, mask_file=mask_file, nsamp=3000, weighted_residual=False,\
# 						  float_templates=False, template_names=None, visual=False, show_input_maps=False, fmin=0.003, image_extnames=['IMAGE'],\
# 						   max_nsrc=3000, movestar_sample_delay=0, color_sigs=color_prior_sigs, alph=0.0, n_frames=30, birth_death_sample_delay=0, merge_split_sample_delay=0)


# pcat_test.iter_fourier_comps(dataname='gps_0', tail_name='gps_0_PSW', use_mask=True, bias=None, mask_file=mask_file, nsamps=[50], n_fc_terms=20, weighted_residual=False, float_templates=False, template_names=None, visual=True, show_input_maps=False, fmin_levels=[1.0], final_fmin=0.5, image_extnames=['IMAGE'], max_nsrc=200)
# color_prior_sigs = dict({'S-M':0.5, 'M-L':0.5, 'L-S':0.5, 'M-S':0.5, 'S-L':0.5, 'L-M':0.5})
# pcat_test.real_dat_run(nbands=1, dataname='gps_0', tail_name='gps_0_PSW', nsrc_init=0, float_fourier_comps=True,\
# 						 use_mask=True, bias=None, mask_file=mask_file, nsamp=2000, n_fc_terms=20, weighted_residual=False,\
# 						  float_templates=False, template_names=None, visual=False, show_input_maps=False, fmin=0.2, image_extnames=['IMAGE'],\
# 						   max_nsrc=300, movestar_sample_delay=50, color_sigs=color_prior_sigs, n_frames=3, birth_death_sample_delay=50, merge_split_sample_delay=50, fc_prop_alpha=-1.)
# pcat_test.artificial_star_test(n_src_perbin=20, nbins=15, nsamp=2000, residual_samples=200, visual=True,\
# 								  dataname='gps_0',tail_name='gps_0_PSW', use_mask=True, mask_file=mask_file, \
# 								  float_fourier_comps=True, n_fc_terms=10, n_frames=10, point_src_delay=10, nsrc_init=0, \
# 								  frac_flux_thresh=0.5, pos_thresh=1., inject_fmax=0.4, max_nsrc=500, show_input_maps=True)

# im_fpath = 'Data/spire/SMC_HERITAGE/cutouts/SMC_HERITAGE_cutsize200_199_PSW.fits'
im_fpath = 'Data/spire/LMC_HERITAGE/cutouts/LMC_HERITAGE_cutsize200_36_PSW.fits'

# im_fpath = 'Data/spire/SMC_HERITAGE/cutouts/SMC_HERITAGE_cutsize200_162_PSW.fits'

# load_timestr='20210105-042450'
# load_timestr = '20210105-142743'
# load_timestr = '20210105-162254'
# load_timestr = '20210106-012646'
# load_timestr = '20210106-230414'
# load_timestr = '20210109-134459'
# load_timestr = '20210109-050110'
# load_timestr = '20210111-133739'
load_timestr = None

fluxbins = np.logspace(np.log10(0.015), np.log10(1.0), 10)
fluxbins_bright = np.logspace(np.log10(0.1), np.log10(3.0), 8)
fluxbins_goodsn = np.logspace(np.log10(0.005), np.log10(0.5), 10)


print('flux bins are :', fluxbins)
print('flux bins bright : ', fluxbins_bright)

n_src_perbin = [100, 50, 50, 20, 20, 5, 3, 2, 2]
n_src_perbin_brightdust = [100, 50, 20, 20, 5, 5, 5]


n_src_perbin_goodsn = [100, 100, 50, 50, 10, 5, 5, 5, 5]

inject_catalog_path = 'spire_results/20210109-134459/inject_catalog.npz'
# inject_catalog_path = None


# n_src_perbin = 30

# pcat_test.artificial_star_test(n_src_perbin=20, inject_fmin=0.015, inject_fmax=1.0, nbins=15, nsamp=200, residual_samples=40, visual=True,\
# 								  dataname='SMC_HERITAGE/cutouts',tail_name='SMC_HERITAGE_cutsize200_168_PSW', im_fpath=im_fpath, use_mask=False, \
# 								  float_fourier_comps=True, n_fc_terms=10, n_frames=10, point_src_delay=10, nsrc_init=0, \
# 								  frac_flux_thresh=0.5, pos_thresh=1.0, max_nsrc=1000, load_timestr=load_timestr, show_input_maps=False, fmin=0.01)

# pcat_test.artificial_star_test(n_src_perbin=n_src_perbin, fluxbins=fluxbins, nbins=None, nsamp=2000, residual_samples=200, visual=False,\
# 								  dataname='SMC_HERITAGE/cutouts',tail_name='SMC_HERITAGE_cutsize200_168_PSW', im_fpath=im_fpath, use_mask=False, \
# 								  float_fourier_comps=True, n_fc_terms=10, n_frames=10, point_src_delay=10, nsrc_init=0, \
# 								  frac_flux_thresh=2.5, pos_thresh=0.0, max_nsrc=1500, load_timestr=load_timestr, show_input_maps=False, fmin=0.01)

# pcat_test.artificial_star_test(n_src_perbin=n_src_perbin_goodsn, fluxbins=fluxbins_goodsn, nbins=None, nsamp=2000, residual_samples=300, visual=True,\
# 								  dataname='GOODSN',tail_name='GOODSN_image_SMAP_PSW', use_mask=True, fc_amp_sig=0.0002, \
# 								  float_fourier_comps=False, n_fc_terms=5, n_frames=10, point_src_delay=0, nsrc_init=0, \
# 								  frac_flux_thresh=2.5, pos_thresh=1.0, max_nsrc=2500, mask_file=mask_file, load_timestr=load_timestr, show_input_maps=False, fmin=0.004)


# pcat_test.artificial_star_test(n_src_perbin=n_src_perbin, fluxbins=fluxbins, inject_catalog_path=inject_catalog_path, nbins=None, nsamp=3000, residual_samples=200, visual=False,\
# 								  dataname='SMC_HERITAGE/cutouts',tail_name='SMC_HERITAGE_cutsize200_199_PSW', im_fpath=im_fpath, use_mask=False, \
# 								  float_fourier_comps=True, n_fc_terms=15, n_frames=10, point_src_delay=10, nsrc_init=0, \
# 								  frac_flux_thresh=2.5, bkg_moore_penrose_inv=False, pos_thresh=1.0, max_nsrc=2000, MP_order=10, ridge_fac=1., fc_amp_sig=0.0005, load_timestr=load_timestr, show_input_maps=False, fmin=0.01)

# pcat_test.artificial_star_test(n_src_perbin=n_src_perbin_brightdust, fluxbins=fluxbins_bright, nbins=None, nsamp=200, residual_samples=40, visual=True,\
# 								  dataname='SMC_HERITAGE/cutouts',tail_name='SMC_HERITAGE_cutsize200_162_PSW', im_fpath=im_fpath, use_mask=False, \
# 								  float_fourier_comps=True, n_fc_terms=10, n_frames=10, point_src_delay=10, nsrc_init=0, \
# 								  frac_flux_thresh=0.5, pos_thresh=1., max_nsrc=1500, load_timestr=load_timestr, show_input_maps=False, fmin=0.05)

# pcat_test.real_dat_run(nbands=1, dataname='SMC_HERITAGE/cutouts', tail_name='SMC_HERITAGE_cutsize200_168_PSW', im_fpath=im_fpath, image_extnames=['IMAGE'], err_fpath=None, nsrc_init=0, float_fourier_comps=True,\
# 						 use_mask=False, bias=None, mask_file=mask_file, nsamp=200, n_fc_terms=10, weighted_residual=True,\
# 						  float_templates=False, template_names=None, visual=False, show_input_maps=False, fmin=0.015, \
# 						   max_nsrc=500, movestar_sample_delay=0, color_sigs=color_prior_sigs, n_frames=20, birth_death_sample_delay=0, merge_split_sample_delay=0,\
# 						    fc_prop_alpha=-1., fc_amp_sig=0.002, MP_order=6, bkg_moore_penrose_inv=True)

pcat_test.real_dat_run(nbands=1, dataname='LMC_HERITAGE/cutouts', tail_name='LMC_HERITAGE_cutsize200_36_PSW', im_fpath=im_fpath, image_extnames=['IMAGE'], err_fpath=None, nsrc_init=0, float_fourier_comps=True,\
						 use_mask=False, bias=None, mask_file=None, nsamp=3000, n_fc_terms=15, weighted_residual=True, residual_samples=300,\
						  float_templates=False, template_names=None, visual=True, show_input_maps=False, fmin=0.01, \
						   max_nsrc=1200, point_src_delay=0, color_sigs=color_prior_sigs, n_frames=20, \
						    fc_amp_sig=0.0005, bkg_moore_penrose_inv=True, MP_order=15, ridge_fac=1., nregion=10)




# result_plots(timestr='20210109-134459',cattype=None, burn_in_frac=0.7, boolplotsave=True, boolplotshow=False, plttype='png', gdat=None, \
# 			fourier_comp_plots=True)

# result_plots(timestr='20201221-011509',cattype=None, burn_in_frac=0.75, boolplotsave=True, boolplotshow=False, plttype='png', gdat=None)
# result_plots(timestr='20201221-010751',cattype=None, burn_in_frac=0.75, boolplotsave=True, boolplotshow=False, plttype='png', gdat=None)
# result_plots(timestr='20201221-010006',cattype=None, burn_in_frac=0.75, boolplotsave=True, boolplotshow=False, plttype='png', gdat=None)

# # def run_pcat_dust_and_sz_test(sim_idx=200, inject_dust=False, show_input_maps=False, inject_sz_frac=1.0):


# 	# ob = lion(band0=0, band1=1, band2=2, base_path=base_path, result_path=result_path, round_up_or_down='down', bolocam_mask=True, float_background=True, burn_in_frac=0.6, bkg_sig_fac=5.0, bkg_sample_delay=10, temp_sample_delay=20, \
# 	# 		 cblas=True, openblas=False, visual=False, show_input_maps=show_input_maps, float_templates=True, template_names=template_names, init_template_amplitude_dicts=initial_template_amplitude_dicts, tail_name='rxj1347_PSW_sim0'+str(sim_idx)+'_dust',\
# 	# 		  dataname='sim_w_dust', bias=None, max_nsrc=1500, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True,\
# 	# 		   make_post_plots=True, nsamp=50, delta_cp_bool=True, use_mask=True, residual_samples=100, template_filename=t_filenames, inject_dust=inject_dust, inject_sz_frac=inject_sz_frac)
# 	# ob.main()

