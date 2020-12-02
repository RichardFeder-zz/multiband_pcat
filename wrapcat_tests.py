from pcat_spire import *
from diffuse_gen import *
from spire_plotting_fns import *

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



	def run_sims_with_injected_sz(self, visual=False, show_input_maps=False, fmin=0.007, dataname='rxj1347_831', tail_name='rxj1347_PSW_nr_1_ext', \
				      template_names=['sze'], bias=[0.002, 0.002, 0.002], use_mask=True, max_nsrc=1000, make_post_plots=True, \
				      nsamp=2000, residual_samples=200, inject_sz_frac=1.0, inject_diffuse_comp=False, diffuse_comp_path=None, \
				      image_extnames=['SIGNAL']):

		# this assumes the SZ template is within the fits data struct 

		initial_template_amplitude_dicts = dict({'sze': dict({'S':0.00, 'M':0.001, 'L':0.018})})

		ob = lion(band0=0, band1=1, band2=2, base_path=self.base_path, result_path=self.result_path, burn_in_frac=0.7, float_background=True, \
			  bkg_sample_delay=0, temp_sample_delay=50, cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps, \
			  float_templates=True, template_names=template_names, init_template_amplitude_dicts=initial_template_amplitude_dicts, \
			  tail_name=tail_name, dataname=dataname, bias=bias, use_mask=use_mask, max_nsrc=max_nsrc, trueminf=fmin, nregion=5, \
			  make_post_plots=make_post_plots, nsamp=nsamp, residual_samples=residual_samples, inject_sz_frac=inject_sz_frac, template_moveweight=40., \
			  inject_diffuse_comp=inject_diffuse_comp, diffuse_comp_path=diffuse_comp_path, image_extnames=image_extnames)

		ob.main()

	def iter_fourier_comps(self, n_fc_terms=10, fmin_levels=[0.05, 0.02, 0.01, 0.007], final_fmin=0.007, \
							nsamps=[50, 100, 200, 500], final_nsamp=2000, \
							template_names=['sze'], nlast_fc=20, dataname='rxj1347_831', tail_name='rxj1347_PSW_nr_1_ext',\
							bias=[-0.004, -0.007, -0.008], max_nsrc=1000, visual=False, alph=1.0, show_input_maps=False, \
							inject_sz_frac=1.0, residual_samples=200, external_sz_file=False, timestr_list_file=None, \
							inject_diffuse_comp=False, diffuse_comp_path=None):

		nbands = len(bias)

		if residual_samples > final_nsamp:
			residual_samples = final_nsamp // 2
			print('residual samples changed to', residual_samples)

		if external_sz_file:
			template_filename=dict({'sze': self.sz_filename})
		else:
			template_filename = None
		timestr = None

		initial_template_amplitude_dicts = dict({'sze': dict({'S':0.00, 'M':0.001, 'L':0.018})})

		init_fc = np.zeros(shape=(n_fc_terms, n_fc_terms, 2))

		for i, fmin in enumerate(fmin_levels):
			if i==0:
				print('initial fourier coefficients set to zero')
				median_fc = init_fc

			# start with 250 micron image only
			ob = lion(band0=0, base_path=self.base_path, result_path=self.result_path, round_up_or_down='down', bolocam_mask=False, \
						float_background=True, burn_in_frac=0.75, bkg_sample_delay=0, float_templates=True, template_moveweight=0.,\
		 				cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps, init_template_amplitude_dicts=initial_template_amplitude_dicts,\
		 				template_names=template_names, template_filename=template_filename, tail_name=tail_name, dataname=dataname, bias=[bias[0]], load_state_timestr=timestr, max_nsrc=max_nsrc,\
		 				trueminf=fmin, nregion=5, make_post_plots=False, nsamp=nsamps[i], use_mask=True,\
		 				residual_samples=5, init_fourier_coeffs=median_fc, n_frames=3, float_fourier_comps=True, \
		 				n_fourier_terms=n_fc_terms, show_fc_temps=False, fc_sample_delay=0, fourier_comp_moveweight=200.,\
		 				alph=alph, dfc_prob=1.0, inject_sz_frac=1.0, inject_diffuse_comp=inject_diffuse_comp, diffuse_comp_path=diffuse_comp_path)

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
					temp_sample_delay=50, cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps,\
					float_templates=True, template_names=template_names, init_template_amplitude_dicts=initial_template_amplitude_dicts, \
					tail_name=tail_name, dataname=dataname, max_nsrc=max_nsrc, \
					init_fourier_coeffs=median_fc, template_filename=template_filename, trueminf=final_fmin, nregion=5, \
				    make_post_plots=False, nsamp=final_nsamp, use_mask=True, residual_samples=residual_samples, \
				    float_fourier_comps=True, show_fc_temps=False, n_fourier_terms=n_fc_terms, fc_sample_delay=0, \
				    fourier_comp_moveweight=0., movestar_sample_delay=0, merge_split_sample_delay=0, birth_death_sample_delay=0,\
				    alph=alph, dfc_prob=0.0, fc_rel_amps=fc_rel_amps, inject_sz_frac=1.0, timestr_list_file=timestr_list_file, \
				     inject_diffuse_comp=inject_diffuse_comp, diffuse_comp_path=diffuse_comp_path, bias=[last_bkg_sample_250, 0.003, 0.003], load_state_timestr=timestr)

		ob.main()

base_path='/Users/luminatech/Documents/multiband_pcat/'
result_path='/Users/luminatech/Documents/multiband_pcat/spire_results/'

# timestr_list_file='lensed_no_dust_gen3sims_rxj1347_11_10_20_timestrs.npz'
# started_idx_list_file = 'lensed_no_dust_gen3sims_rxj1347_11_10_20_simidxs.npz' 
# started_idx_list_file = 'simidxs.npz' 

sim_idx = 303
pcat_test = pcat_test_suite(cluster_name='rxj1347', base_path=base_path, result_path=result_path, cblas=True, openblas=False)
figs = pcat_test.validate_astrometry(tail_name='rxj1347_PSW_sim0'+str(sim_idx), dataname='sims_12_2_20', ngrid=10, return_validation_figs=True)
figs[0].savefig('test0.pdf')
figs[1].savefig('test1.pdf')

# pcat_test.run_sims_with_injected_sz(dataname='sims_12_2_20', image_extnames=['SIG_PRE_LENSE', 'NOISE'], tail_name='rxj1347_PSW_sim0'+str(sim_idx), visual=True, show_input_maps=True)



# # def run_pcat_dust_and_sz_test(sim_idx=200, inject_dust=False, show_input_maps=False, inject_sz_frac=1.0):


# 	# ob = lion(band0=0, band1=1, band2=2, base_path=base_path, result_path=result_path, round_up_or_down='down', bolocam_mask=True, float_background=True, burn_in_frac=0.6, bkg_sig_fac=5.0, bkg_sample_delay=10, temp_sample_delay=20, \
# 	# 		 cblas=True, openblas=False, visual=False, show_input_maps=show_input_maps, float_templates=True, template_names=template_names, init_template_amplitude_dicts=initial_template_amplitude_dicts, tail_name='rxj1347_PSW_sim0'+str(sim_idx)+'_dust',\
# 	# 		  dataname='sim_w_dust', bias=None, max_nsrc=1500, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True,\
# 	# 		   make_post_plots=True, nsamp=50, delta_cp_bool=True, use_mask=True, residual_samples=100, template_filename=t_filenames, inject_dust=inject_dust, inject_sz_frac=inject_sz_frac)
# 	# ob.main()

