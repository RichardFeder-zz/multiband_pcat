from pcat_spire import *
from diffuse_gen import *
from spire_plotting_fns import *
import pandas as pd
from wrapcat_tests import pcat_test_suite

base_path='/Users/luminatech/Documents/multiband_pcat/'
result_path='/Users/luminatech/Documents/multiband_pcat/spire_results/'
t_filenames = base_path+'Data/spire/rxj1347/rxj1347_PSW_nr.fits'
template_filenames = dict({'sze': t_filenames[0]})


color_prior_sigs = dict({'S-M':0.5, 'M-L':0.5, 'L-S':0.5, 'M-S':0.5, 'S-L':0.5, 'L-M':0.5})
# color_prior_sigs = dict({'S-M':5.5, 'M-L':5.5, 'L-S':5.5, 'M-S':5.5, 'S-L':5.5, 'L-M':5.5})

# pcat_test = pcat_test_suite(cluster_name='rxj1347_831', base_path=base_path, result_path=result_path, cblas=True, openblas=False)
# pcat_test.iter_fourier_comps(dataname='rxj1347_831', tail_name='rxj1347_PSW_nr_1_ext', nsamps=[100, 100, 200], float_templates=True, template_names=template_filenames, \
# 	visual=True, show_input_maps=False, fmin_levels=[0.02,0.01, 0.004], final_fmin=0.004, use_mask=True, image_extnames=['SIGNAL'], n_fc_terms=8, F_statistic_alph=True, nominal_nsrc = 700)

# result_plots(timestr='20210216-024518',cattype=None, burn_in_frac=0.75, boolplotsave=True, boolplotshow=False, plttype='png', gdat=None, \
# 				generate_condensed_cat=True, fourier_comp_plots=False, n_condensed_samp=100, prevalence_cut=0.5, mask_hwhm=5, condensed_catalog_plots=True)


pcat_test = pcat_test_suite(cluster_name='pre_lense_map_021621', base_path=base_path, result_path=result_path, cblas=True, openblas=False)
# figs = pcat_test.validate_astrometry(tail_name='rxj1347_PSW_sim0300', dataname='conley_030121', ngrid=15, return_validation_figs=True, image_extnames=[0], save=True)
# figs[0].savefig('Data/spire/pre_lense_map_021621/test0.pdf')
# figs[1].savefig('Data/spire/pre_lense_map_021621/test1.pdf')
# pcat_test.iter_fourier_comps(tail_name='rxj1347_PSW_sim0300_con', dataname='pre_lense_map_021621', nsamps=[30], float_templates=True, template_names=template_filenames, \
	# visual=True, show_input_maps=True, fmin_levels=[0.004], final_fmin=0.004, use_mask=True, image_extnames=['SIGNAL'], n_fc_terms=5, F_statistic_alph=True, nominal_nsrc = 700)
# panel_list = ['data0', 'data1', 'dNdS0', 'residual0', 'residual1', 'dNdS1']
panel_list = ['data0', 'data1', 'data2', 'residual0', 'dNdS0', 'dNdS1']

# panel_list = ['data0', 'data1', 'data2', 'residual0', 'residual1', 'residual2']
# panel_list = ['data0', 'data1', 'data_zoom1', 'residual0', 'residual1', 'residual_zoom1']


# ------------------ test ACT sources from HeLMS ------------------

# figs = pcat_test.validate_astrometry(tail_name='test_act_PSW_300arcsec_8', dataname='act_srcs_herschel_cutouts', use_mask=False, ngrid=10, return_validation_figs=True, image_extnames=['IMAGE'], save=True)
red_candidate_list_030421 = [2, 31, 43, 14, 18, 22, 25, 41, 45, 50, 52, 64, 38, 61, 80, 81, 82, 91, 97, 105, 110]
extended_object_list_030421 = [7, 11, 26, 28, 30, 40, 46, 49, 51, 66, 70, 88, 104, 107, 115, 117, 122, 126, 127]
red_candidate_list_hers_030921 = [102, 4, 7, 20, 25, 28, 38, 59, 62, 66, 75, 81, 82, 85, 90, 95, 96, 98, 100, 101]

# for a in range(111, 131):
# 	if a not in red_candidate_list_030421 and a not in extended_object_list_030421:
# 		print('we want to run this a = ', a)
# 		pcat_test.real_dat_run(nbands=3, band0=2, band1=1, band2=0, tail_name='act_helms_covered_030421_PSW_300arcsec_'+str(a), dataname='act_srcs_herschel_cutouts/300arcsec_images/', image_extnames=['IMAGE'], float_fourier_comps=False, \
# 			use_mask=False, bias=None, nsamp=500, weighted_residual=True, visual=False, show_input_maps=False, fmin=0.02, max_nsrc=50, color_sigs=color_prior_sigs, \
# 			n_frames=2, correct_misaligned_shift=False, residual_samples=100, generate_condensed_catalog=True, n_condensed_samp=100, prevalence_cut=0.5, panel_list=panel_list, alph=1.0, err_f_divfac=1, nominal_nsrc=50, nregion=2, raw_counts=True)
# for a in red_candidate_list_030421[10:]:
# for a in red_candidate_list_hers_030921:
# 	pcat_test.real_dat_run(nbands=3, band0=2, band1=1, band2=0, tail_name='act_hers_covered_030921_PSW_300arcsec_'+str(a), dataname='hers_act/300arcsec_images/', image_extnames=['IMAGE'], float_fourier_comps=False, \
# 		use_mask=False, bias=None, nsamp=500, weighted_residual=True, visual=True, show_input_maps=False, fmin=0.03, max_nsrc=50, color_sigs=color_prior_sigs, \
# 		n_frames=2, correct_misaligned_shift=False, residual_samples=100, generate_condensed_catalog=True, n_condensed_samp=100, panel_list=panel_list, alph=1.0, err_f_divfac=1, nominal_nsrc=50, nregion=2, raw_counts=True)

	# pcat_test.real_dat_run(nbands=3, band0=2, band1=1, band2=0, tail_name='full_act_PSW_600arcsec_'+str(a+1), dataname='act_srcs_herschel_cutouts_full/300arcsec/images', image_extnames=['IMAGE'], float_fourier_comps=False, \
	# 	use_mask=False, bias=None, nsamp=1000, weighted_residual=True, visual=True, show_input_maps=False, fmin=0.02, max_nsrc=60, color_sigs=color_prior_sigs, \
	# 	n_frames=2, correct_misaligned_shift=False, residual_samples=200, generate_condensed_catalog=False, panel_list=panel_list, alph=1.0, err_f_divfac=1, nominal_nsrc=50, nregion=4, raw_counts=True)
	# pcat_test.real_dat_run(nbands=3, band0=2, band1=1, band2=0, tail_name='full_act_PSW_300arcsec_'+str(a+1), dataname='act_srcs_herschel_cutouts_full', image_extnames=['IMAGE'], float_fourier_comps=False, \
	# 	use_mask=False, bias=None, nsamp=1000, weighted_residual=True, visual=True, show_input_maps=False, fmin=0.02, max_nsrc=500, color_sigs=color_prior_sigs, \
	# 	n_frames=2, correct_misaligned_shift=False, residual_samples=200, generate_condensed_catalog=False, panel_list=panel_list, alph=1.0, err_f_divfac=10, nominal_nsrc=50, nregion=5, raw_counts=True)
	# pcat_test.real_dat_run(nbands=3, band0=2, band1=1, band2=0, tail_name='act_helms_covered_030421_PSW_300arcsec_'+str(a), dataname='act_srcs_herschel_cutouts/300arcsec_images/', image_extnames=['IMAGE'], float_fourier_comps=False, \
	# 	use_mask=False, bias=None, nsamp=500, weighted_residual=True, visual=False, show_input_maps=False, fmin=0.02, max_nsrc=50, color_sigs=color_prior_sigs, \
	# 	n_frames=2, correct_misaligned_shift=False, residual_samples=100, generate_condensed_catalog=True, n_condensed_samp=100, panel_list=panel_list, alph=1.0, err_f_divfac=1, nominal_nsrc=50, nregion=2, raw_counts=True)

# timestring_list = ['20210304-071800', '20210304-065915', '20210304-064550', '20210304-063227', '20210304-062350', \
#                   '20210304-061106', '20210304-060337', '20210304-054724', '20210304-054142', '20210304-052931', \
#                   '20210304-052218', '20210304-050626', '20210304-050216', '20210304-044625', '20210304-044326', \
#                   '20210304-042405', '20210304-042314', '20210304-040619', '20210304-040520', '20210304-034957', \
#                   '20210304-034837']

# for timestr in timestring_list:
#     result_plots(timestr=timestr,cattype=None, generate_condensed_cat=True, n_condensed_samp=100, prevalence_cut=0.5, mask_hwhm=2, condensed_catalog_plots=True,\
#                  burn_in_frac=0.5, boolplotsave=True, boolplotshow=False, plttype='png', gdat=None, flux_color_color_plots=True, search_radius=0.75)


# 	pcat_test.real_dat_run(nbands=3, band0=2, band1=1, band2=0, tail_name='act_helms_covered_030421_PSW_300arcsec_'+str(a), dataname='act_srcs_herschel_cutouts/300arcsec_images/', image_extnames=['IMAGE'], float_fourier_comps=False, \
# 		use_mask=False, bias=None, nsamp=500, weighted_residual=True, visual=False, show_input_maps=False, fmin=0.02, max_nsrc=50, color_sigs=color_prior_sigs, \
# 		n_frames=2, correct_misaligned_shift=False, residual_samples=100, generate_condensed_catalog=True, n_condensed_samp=100, panel_list=panel_list, alph=1.0, err_f_divfac=1, nominal_nsrc=50, nregion=2, raw_counts=True)
# result_plots(timestr='20210304-120523',cattype=None, generate_condensed_cat=False, n_condensed_samp=100, prevalence_cut=0.5, mask_hwhm=2, condensed_catalog_plots=False,\
# 				 burn_in_frac=0.5, boolplotsave=True, boolplotshow=False, plttype='png', gdat=None, flux_color_color_plots=True, search_radius=0.75)

# ----------------- test blank HELMS field -----------------------

# for a in [3, 4, 5, 6, 7, 8, 9]:
# 	# pcat_test.real_dat_run(nbands=3, band0=2, band1=1, band2=0, tail_name='full_act_PSW_600arcsec_'+str(a+1), dataname='act_srcs_herschel_cutouts_full/300arcsec/images', image_extnames=['IMAGE'], float_fourier_comps=False, \
# 	# 	use_mask=False, bias=None, nsamp=1000, weighted_residual=True, visual=True, show_input_maps=False, fmin=0.02, max_nsrc=60, color_sigs=color_prior_sigs, \
# 	# 	n_frames=2, correct_misaligned_shift=False, residual_samples=200, generate_condensed_catalog=False, panel_list=panel_list, alph=1.0, err_f_divfac=1, nominal_nsrc=50, nregion=4, raw_counts=True)
# 	# pcat_test.real_dat_run(nbands=3, band0=2, band1=1, band2=0, tail_name='blank_helms_PSW_1200arcsec_'+str(a+1), dataname='blank_helms', image_extnames=['IMAGE'], float_fourier_comps=False, \
# 	# 	use_mask=False, bias=None, nsamp=1000, weighted_residual=True, visual=False, show_input_maps=False, fmin=0.02, max_nsrc=500, color_sigs=color_prior_sigs, \
# 	# 	n_frames=2, correct_misaligned_shift=False, residual_samples=200, generate_condensed_catalog=False, panel_list=panel_list, alph=1.0, err_f_divfac=10, nominal_nsrc=50, nregion=5, raw_counts=True)
# 	pcat_test.real_dat_run(nbands=3, band0=2, band1=1, band2=0, tail_name='blank_helms_030221_PSW_300arcsec_'+str(a+1), dataname='blank_helms', image_extnames=['IMAGE'], float_fourier_comps=False, \
# 		use_mask=False, bias=None, nsamp=500, weighted_residual=True, visual=False, show_input_maps=False, fmin=0.02, max_nsrc=60, color_sigs=color_prior_sigs, \
# 		n_frames=2, correct_misaligned_shift=False, residual_samples=200, generate_condensed_catalog=False, panel_list=panel_list, alph=1.0, err_f_divfac=1, nominal_nsrc=50, nregion=2, raw_counts=True)

# result_plots(timestr='blank_field_300/20210226-162901',cattype=None, generate_condensed_cat=False, n_condensed_samp=400, prevalence_cut=0.5, mask_hwhm=2, condensed_catalog_plots=True,\
# 				 burn_in_frac=0.5, boolplotsave=True, boolplotshow=False, plttype='png', gdat=None, flux_color_color_plots=True, search_radius=0.75)

# ---------------- GOODS-N search for red source false detections ------------------

# mask_file_goodsn_100_100 = 'Data/spire/GOODSN/GOODSN_PSW_mask_100_100_2_021921.fits'
# pcat_test.real_dat_run(nbands=3, band0=2, band1=1, band2=0,  dataname='GOODSN',tail_name='GOODSN_image_SMAP_PSW', image_extnames=['IMAGE'], float_fourier_comps=False, \
# 	use_mask=True, mask_file=mask_file_goodsn_100_100, bias=None, nsamp=1000, weighted_residual=True, visual=False, show_input_maps=False, fmin=0.004, max_nsrc=1000, color_sigs=color_prior_sigs, \
# 	n_frames=2, residual_samples=200, generate_condensed_catalog=False, panel_list=panel_list, alph=1.0, err_f_divfac=10., nominal_nsrc=1000, nregion=4, raw_counts=True)


# pcat_test.real_dat_run(nbands=3, band0=2, band1=1, band2=0,  dataname='GOODSN',tail_name='protocluster_goodsn_PSW_300arcsec', image_extnames=['IMAGE'], float_fourier_comps=False, \
# 	use_mask=False, bias=None, nsamp=1000, weighted_residual=True, visual=False, show_input_maps=False, fmin=0.004, max_nsrc=120, color_sigs=color_prior_sigs, \
# 	n_frames=2, residual_samples=200, generate_condensed_catalog=True, panel_list=panel_list, alph=1.0, err_f_divfac=5., nominal_nsrc=100, nregion=2, raw_counts=True, \
# 	bkg_sig_fac=[10., 5., 5.])

# pcat_test.real_dat_run(nbands=3, band0=2, band1=1, band2=0, dataname='Conley_sims_zitrin', tail_name='rxj1347_PSW_sim0351', image_extnames=['SIG_PRE_LENS', 'NOISE'], float_fourier_comps=False, \
# 	use_mask=False, bias=None, nsamp=1000, weighted_residual=True, visual=False, show_input_maps=True, fmin=0.006, max_nsrc=1100, color_sigs=color_prior_sigs, \
# 	n_frames=20, nregion=5, panel_list=panel_list, err_f_divfac=10.)


# -----------------------------------------------------------------
# pcat_test.real_dat_run(nbands=3, band0=0, band1=1, band2=2, tail_name='rxj1347_PSW_sim0300_con', dataname='pre_lense_map_021621', image_extnames=['SIG_PRE_LENS', 'NOISE'], float_fourier_comps=False, \
# 	use_mask=True, bias=None, nsamp=2500, weighted_residual=True, visual=False, show_input_maps=False, fmin=0.004, max_nsrc=1000, color_sigs=color_prior_sigs, \
# 	n_frames=20, correct_misaligned_shift=False, residual_samples=200, panel_list=panel_list, alph=0.0)

# mask_file = base_path+'/data/spire/GOODSN/GOODSN_PSW_mask_011721.fits'


# mask_file = base_path+'/data/spire/GOODSN/GOODSN_PSW_mask_100_100_021921.fits'
# pcat_test.real_dat_run(nbands=3, band0=0, band1=1, band2=2, dataname='GOODSN',tail_name='GOODSN_image_SMAP_PSW', image_extnames=['IMAGE'], float_fourier_comps=False, \
# 	use_mask=True, mask_file=mask_file, bias=None, nsamp=2500, weighted_residual=True, visual=False, show_input_maps=False, fmin=0.004, max_nsrc=1000, color_sigs=color_prior_sigs, \
# 	n_frames=20, correct_misaligned_shift=False, residual_samples=200, panel_list=panel_list)


# pcat_test.real_dat_run(nbands=2, band0=0, band1=1, tail_name='rxj1347_PSW_sim0300_con', dataname='pre_lense_map_021621', image_extnames=['SIG_PRE_LENS', 'NOISE'], float_fourier_comps=False, \
# 	use_mask=True, bias=None, nsamp=2000, weighted_residual=True, visual=False, show_input_maps=False, fmin=0.004, max_nsrc=1200, color_sigs=color_prior_sigs, \
# 	n_frames=20, correct_misaligned_shift=False, residual_samples=200, panel_list=panel_list)

# pcat_test.real_dat_run(nbands=1, band0=0, tail_name='rxj1347_PSW_sim0300_con', dataname='pre_lense_map_021621', image_extnames=['SIG_PRE_LENS', 'NOISE'], float_fourier_comps=False, \
# 	use_mask=True, bias=None, nsamp=2000, weighted_residual=True, visual=False, show_input_maps=False, fmin=0.004, max_nsrc=1000, color_sigs=color_prior_sigs, \
# 	n_frames=20, correct_misaligned_shift=False, residual_samples=200, panel_list=panel_list)


# mask_file = base_path+'data/spire/gps_0/gps_0_PSW_mask.fits'
# mask_file = 'Data/spire/SMC_HERITAGE/SMC_HERITAGE_mask2_PSW.fits'
# mask_file = base_path+'/data/spire/GOODSN/GOODSN_PSW_mask_011721.fits'
# mask_file = base_path+'/data/spire/GOODSN/GOODSN_PSW_mask_011721.fits'

# mask_file=None
# mask_file= None
# timestr_list_file='lensed_no_dust_gen3sims_rxj1347_11_10_20_timestrs.npz'
# started_idx_list_file = 'lensed_no_dust_gen3sims_rxj1347_11_10_20_simidxs.npz' 
# started_idx_list_file = 'simidxs.npz' 

# sim_idx = 350
# pcat_test = pcat_test_suite(cluster_name='SMC_HERITAGE', base_path=base_path, result_path=result_path, cblas=True, openblas=False)
# pcat_test = pcat_test_suite(cluster_name='victoria_test_021021', base_path=base_path, result_path=result_path, cblas=True, openblas=False)
# figs = pcat_test.validate_astrometry(tail_name='final_test_map_PSW', dataname='victoria_test_021021', ngrid=10, return_validation_figs=True, image_extnames=[0])

# figs = pcat_test.validate_astrometry(tail_name='rxj1347_PSW_sim0'+str(sim_idx), dataname='sims_12_2_20', ngrid=10, return_validation_figs=True)
# figs[0].savefig('test0.pdf')
# figs[1].savefig('test1.pdf')

# pcat_test.run_sims_with_injected_sz(dataname='gen_2_sims', add_noise=True, temp_sample_delay=10, image_extnames=['SIGNAL'], tail_name='rxj1347_PSW_sim0'+str(sim_idx), visual=True, show_input_maps=False)
# color_prior_sigs = dict({'S-M':1.5, 'M-L':1.5, 'L-S':1.5, 'M-S':1.5, 'S-L':1.5, 'L-M':1.5})


# pcat_test.iter_fourier_comps(dataname='GOODSN', tail_name='GOODSN_image_SMAP_PSW',nsamps=[50, 100, 200], float_templates=False, template_names=None, visual=False, show_input_maps=False, fmin_levels=[0.02,0.01, 0.005], final_fmin=0.003, alph=0.0, use_mask=True, image_extnames=['IMAGE'], max_nsrc=2500)
# pcat_test.real_dat_run(band0=0, band1=1, band2=2, nbands=3, dataname='GOODSN', tail_name='GOODSN_image_SMAP_PSW', float_fourier_comps=False,\
# 						 use_mask=True, bias=None, mask_file=mask_file, nsamp=3000, weighted_residual=False,\
# 						  float_templates=False, template_names=None, visual=True, show_input_maps=True, fmin=0.003, image_extnames=['IMAGE'],\
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
# im_fpath = 'Data/spire/LMC_HERITAGE/cutouts/LMC_HERITAGE_cutsize200_36_PSW.fits'
# im_fpath = 'Data/spire/SMC_HERITAGE/cutouts/SMC_HERITAGE_cutsize200_162_PSW.fits'

# load_timestr='20210105-042450'
# load_timestr = '20210105-142743'
# load_timestr = '20210105-162254'
# load_timestr = '20210106-012646'
# load_timestr = '20210106-230414'
# load_timestr = '20210109-134459'
# load_timestr = '20210109-050110'
# load_timestr = '20210111-133739'
# load_timestr = '20210114-054614'
# load_timestr = '20210117-193600'
# load_timestr = None

fluxbins = np.logspace(np.log10(0.015), np.log10(1.0), 10)
fluxbins_bright = np.logspace(np.log10(0.1), np.log10(3.0), 8)
fluxbins_goodsn = np.logspace(np.log10(0.005), np.log10(0.5), 10)
fluxbins_goodsn_deep = np.logspace(np.log10(0.002), np.log10(0.5), 12)

# print('flux bins are :', fluxbins)
# print('flux bins bright : ', fluxbins_bright)

n_src_perbin = [100, 50, 50, 20, 20, 5, 3, 2, 2]
n_src_perbin_brightdust = [100, 50, 20, 20, 5, 5, 5]
n_src_perbin_goodsn = [100, 100, 50, 50, 10, 5, 5, 5, 5]
n_src_perbin_goodsn_deep = [100, 100, 50, 50, 40, 30, 20, 2, 2, 2, 1]

# inject_catalog_path = 'spire_results/20210114-054614/inject_catalog.npz'
# inject_catalog_path = 'spire_results/'+load_timestr+'/inject_catalog.npz'

# inject_catalog_path = 'spire_results/20210117-193600/inject_catalog.npz'
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

# pcat_test.artificial_star_test(n_src_perbin=n_src_perbin_goodsn_deep, fluxbins=fluxbins_goodsn_deep, nbins=None, nsamp=3000, residual_samples=200, visual=True,\
# 								  dataname='GOODSN',tail_name='GOODSN_image_SMAP_PSW', use_mask=True, fc_amp_sig=0.0002, \
# 								  float_fourier_comps=False, n_fc_terms=5, n_frames=10, point_src_delay=0, nsrc_init=0, \
# 								  frac_flux_thresh=100., pos_thresh=1.0, max_nsrc=3000, mask_file=mask_file, inject_catalog_path=inject_catalog_path, load_timestr=load_timestr, show_input_maps=False, fmin=0.002)

# multiband GOODS-N
# pcat_test.artificial_star_test(band0=0, band1=1, band2=2, n_src_perbin=n_src_perbin_goodsn_deep, fluxbins=fluxbins_goodsn_deep, nbins=None, nsamp=3000, residual_samples=200, visual=True,\
# 								  dataname='GOODSN',tail_name='GOODSN_image_SMAP_PSW', use_mask=True, fc_amp_sig=0.0002, \
# 								  float_fourier_comps=False, n_fc_terms=5, n_frames=10, point_src_delay=0, nsrc_init=0, \
# 								  frac_flux_thresh=100., pos_thresh=1.0, max_nsrc=3000, mask_file=mask_file, load_timestr=load_timestr, show_input_maps=False, fmin=0.002)
# color_sigs = color_prior_sigs
# t = pcat_test.validate_astrometry(dataname='conley_sims_20200202', tail_name='rxj1347_PSW_sim0351', use_zero_point=False, correct_misaligned_shift=False, image_extnames=['SIG_PRE_LENS'], ngrid=20, return_validation_figs=True)


# --------------------- test on real data RXJ1347 -------------------------

pcat_test = pcat_test_suite(cluster_name='rxj1347', base_path=base_path, result_path=result_path, cblas=True, openblas=False)

# random initialization between 0.00 and 0.04 for PLW, 0.0 and 0.005 for PMW                                                                                                        
truealpha=2.5

initial_template_amplitude_dicts = dict({'sze': dict({'S':0.00026, 'M':0.001, 'L':0.018})})
initial_template_amplitude_dicts['sze']['L'] = np.random.uniform(0.01, 0.03)
initial_template_amplitude_dicts['sze']['M'] = np.random.uniform(-0.003, 0.003)

print('initial template amplitude dict is ', initial_template_amplitude_dicts)
# mask_file = base_path+'/Data/spire/bolocam_masks/bolocam_mask_PSW.fits'
mask_file = base_path+'/Data/spire/rxj1347_831/rxj1347_PSW_4arcmin_mask_040921_2.fits'

# pcat_test.validate_astrometry(dataname='rxj1347_831', tail_name='rxj1347_PSW_nr_1_ext', use_mask=True, mask_file=mask_file, image_extnames=['SIGNAL'], error_extname='ERROR', ngrid=10, return_validation_figs=True)
# pcat_test.validate_astrometry(dataname='rxj1347_040921', tail_name='rxj1347_PSW_360arcsec_test', use_mask=False, image_extnames=['IMAGE'], error_extname='ERROR', ngrid=10, return_validation_figs=True)

# pcat_test.real_dat_run(visual=True, band0=0, band1=1, band2=2, show_input_maps=False, fmin=0.005, dataname='rxj1347_831', tail_name='rxj1347_PSW_nr_1_ext', image_extnames=['SIGNAL'], \
#                        bias=[-0.0037, -0.0055, -0.0085], float_background=False,  use_mask=True, mask_file=mask_file, max_nsrc=300, make_post_plots=True, nsamp=4000, residual_samples=200, \
#                temp_sample_delay=20, color_sigs=color_prior_sigs, err_f_divfac=2., bkg_sig_fac=[10., 15., 20.], bkg_moveweight=40., temp_prop_sig_fudge_facs=[1., 4.0, 8.0], \
#                movestar_moveweight=60, birth_death_moveweight=15, float_templates=True, float_fourier_comps=True, n_fc_terms=6, merge_split_moveweight=15, estimate_dust_first=True, nsamp_dustestimate=20, point_src_delay=10, \
#                        initial_template_amplitude_dicts=initial_template_amplitude_dicts, template_names=['sze'], nregion=4, nominal_nsrc=200.)

# fixed background
# pcat_test.real_dat_run(visual=False, band0=0, band1=1, band2=2, show_input_maps=False, fmin=0.005, dataname='rxj1347_040921', tail_name='rxj1347_PSW_360arcsec_test', image_extnames=['IMAGE'], \
#                        bias=[-0.0037, -0.0055, -0.0085], float_background=False,  use_mask=False, mask_file=mask_file, max_nsrc=200, make_post_plots=True, nsamp=1000, residual_samples=200, \
#                temp_sample_delay=5, color_sigs=color_prior_sigs, err_f_divfac=10., bkg_sig_fac=[10., 15., 20.], bkg_moveweight=40., temp_prop_sig_fudge_facs=[1., 4.0, 8.0], \
#                movestar_moveweight=60, birth_death_moveweight=15, float_templates=True, float_fourier_comps=True, n_fc_terms=6, merge_split_moveweight=15, estimate_dust_first=True, nsamp_dustestimate=200, point_src_delay=10, \
#                        initial_template_amplitude_dicts=initial_template_amplitude_dicts, template_names=['sze'], nregion=4, nominal_nsrc=200.)

# pcat_test.real_dat_run(visual=False, band0=0, band1=1, band2=2, show_input_maps=False, fmin=0.005, dataname='rxj1347_040921', tail_name='rxj1347_PSW_240arcsec_test', image_extnames=['IMAGE'], \
#                        bias=[-0.0037, -0.0055, -0.0085], float_background=False,  use_mask=False, mask_file=mask_file, max_nsrc=100, make_post_plots=True, nsamp=3000, residual_samples=200, \
#                temp_sample_delay=0, color_sigs=color_prior_sigs, err_f_divfac=10., bkg_sig_fac=[10., 15., 20.], bkg_moveweight=20., fc_amp_sig=0.0005, temp_prop_sig_fudge_facs=[1., 15.0, 25.0], \
#                movestar_moveweight=60, birth_death_moveweight=15, float_templates=True, float_fourier_comps=True, n_fc_terms=3, merge_split_moveweight=15, estimate_dust_first=True, nsamp_dustestimate=50, point_src_delay=0, \
#                        initial_template_amplitude_dicts=initial_template_amplitude_dicts, template_names=['sze'], nregion=1, alph=0.0, nominal_nsrc=50., truealpha=3.)

# float background
pcat_test.real_dat_run(visual=True, band0=0, band1=1, band2=2, show_input_maps=False, fmin=0.005, dataname='rxj1347_831/rxj1347_10arcmin', tail_name='rxj1347_PSW_600arcsec_042021', image_extnames=['SIGNAL'], \
                       bias=[-0.0037, -0.0055, -0.0085], float_background=True,  use_mask=False, max_nsrc=500, make_post_plots=True, nsamp=3000, residual_samples=200, \
               temp_sample_delay=0, color_sigs=color_prior_sigs, err_f_divfac=15., bkg_sig_fac=[10., 15., 20.], bkg_moveweight=20., fc_amp_sig=0.0005, temp_prop_sig_fudge_facs=[1., 15.0, 25.0], \
               movestar_moveweight=60, birth_death_moveweight=15, float_templates=True, float_fourier_comps=True, n_fc_terms=6, merge_split_moveweight=15, estimate_dust_first=True, nsamp_dustestimate=50, point_src_delay=0, \
                       initial_template_amplitude_dicts=initial_template_amplitude_dicts, template_names=['sze'], nregion=5, alph=1.0, nominal_nsrc=400., truealpha=2.5)

# bestfitbkg = [-0.004, -0.006, -0.008]

# timestr_list_file = 'rxj1347_realdat_4arcmin_nfcterms=3_timestrs_042021_bestfitbkg.npz'

# REAL data 10 arcmin x 10 arcmin tests
#pcat_test.real_dat_run(visual=False, band0=0, band1=1, band2=2, show_input_maps=False, fmin=0.005, dataname='rxj1347_831/rxj1347_10arcmin', tail_name='rxj1347_PSW_600arcsec_042021', image_extnames=['SIGNAL'], bias=bkg_vals3, float_background=True, use_mask=False, max_nsrc=500, make_post_plots=True, nsamp=3000, \
#                       residual_samples=200, temp_sample_delay=10, color_sigs=color_prior_sigs, err_f_divfac=15., bkg_sig_fac=[10., 15., 20.], bkg_moveweight=40., temp_prop_sig_fudge_facs=[1., 20.0, 30.0], \
#                       movestar_moveweight=60, birth_death_moveweight=20, float_templates=True, float_fourier_comps=True, fc_amp_sig=0.0005, n_fc_terms=6, merge_split_moveweight=20, \
#                       estimate_dust_first=True, nsamp_dustestimate=300, point_src_delay=0, \
#                       initial_template_amplitude_dicts=initial_template_amplitude_dicts, template_moveweight=60., template_names=['sze'], nregion=5, nominal_nsrc=400, \
#timestr_list_file=timestr_list_file, truealpha=truealpha, alph=1.0)     

# REAL data 4 arcmin x 4 arcmin tests
# pcat_test.real_dat_run(visual=False, band0=0, band1=1, band2=2, show_input_maps=False, fmin=0.005, dataname='rxj1347_040921', tail_name='rxj1347_PSW_240arcsec_test', image_extnames=['IMAGE'], \
#                        bias=bestfitbkg, float_background=False, use_mask=False, mask_file=mask_file, max_nsrc=100, make_post_plots=True, nsamp=5000, residual_samples=200, \
#                temp_sample_delay=10, color_sigs=color_prior_sigs, err_f_divfac=8., bkg_sig_fac=[10., 15., 20.], bkg_moveweight=20., temp_prop_sig_fudge_facs=[1., 20.0, 30.0], \
#                movestar_moveweight=60, birth_death_moveweight=20, float_templates=True, float_fourier_comps=True, fc_amp_sig=0.0005, n_fc_terms=3, merge_split_moveweight=20, estimate_dust_first=True, nsamp_dustestimate=300, point_src_delay=0, \
#                        initial_template_amplitude_dicts=initial_template_amplitude_dicts, template_moveweight=60., template_names=['sze'], nregion=2, nominal_nsrc=50, timestr_list_file=timestr_list_file, truealpha=truealpha, alph=1.0)

# mock cluster realizations

#pcat_test.real_dat_run(visual=False, band0=0, band1=1, band2=2, show_input_maps=False, fmin=0.005, dataname='conley_030121/conley_030121_4arcmin', tail_name='rxj1347_PSW_240_simidx300', image_extnames=['SIG_PRE_LENS', 'NOISE', 'DUST'], \
#                       bias=[0.003, 0.0045, 0.007], float_background=False,  use_mask=False, mask_file=mask_file, max_nsrc=100, make_post_plots=True, nsamp=4000, residual_samples=200, \
#               temp_sample_delay=10, color_sigs=color_prior_sigs, err_f_divfac=10., inject_sz_frac=1.0, bkg_sig_fac=[10., 15., 20.], bkg_moveweight=40., temp_prop_sig_fudge_facs=[1., 15.0, 25.0], \
#               movestar_moveweight=60, birth_death_moveweight=20, float_templates=True, float_fourier_comps=True, fc_amp_sig=0.0005, n_fc_terms=3, merge_split_moveweight=15, estimate_dust_first=True, nsamp_dustestimate=500, point_src_delay=0, initial_template_amplitude_dicts=initial_template_amplitude_dicts, template_names=['sze'], nregion=1, nominal_nsrc=50, timestr_list_file=timestr_list_file, truealpha=truealpha, alph=0.0)




# exit()



# ---------------------- test on Conley_sims_zitrin, both without and with injected cirrus signal ------------------------------
# cirrus_gen_idx = 0
# mask_path = 'Data/spire/Conley_sims_zitrin/50pix_mask_swap.fits'

# panel_list = ['data0', 'model0', 'residual0', 'fourier_bkg0', 'injected_diffuse_comp0', 'dNdS0']
# mask_path = None
# dcomp_path = base_path+'/Data/spire/cirrus_gen/rxj1347_cirrus_sim_idx'+str(cirrus_gen_idx)+'_020821_1x_planck.npz'
# pcat_test.real_dat_run(nbands=1, band0=0, dataname='Conley_sims_zitrin', tail_name='rxj1347_PSW_sim0351', image_extnames=['SIG_PRE_LENS', 'NOISE'], nsrc_init=0, float_fourier_comps=True, \
# 	use_mask=False, bias=None, nsamp=1000, weighted_residual=True, visual=True, show_input_maps=False, mask_file=mask_path, fmin=0.004, max_nsrc=1500, color_sigs=color_prior_sigs, \
# 	n_frames=20, inject_diffuse_comp=True, diffuse_comp_path=dcomp_path, n_fc_terms=7, fc_amp_sig=0.0002, point_src_delay=5, nregion=2, panel_list=None)

# pcat_test.real_dat_run(nbands=1, band0=0, dataname='Conley_sims_zitrin', tail_name='rxj1347_PSW_sim0351', image_extnames=['SIG_PRE_LENS', 'NOISE'], nsrc_init=0, float_fourier_comps=False, \
# 	use_mask=False, bias=None, nsamp=2000, weighted_residual=True, visual=True, show_input_maps=True, fmin=0.004, max_nsrc=1300, color_sigs=color_prior_sigs, \
# 	n_frames=20)

# pcat_test.real_dat_run(nbands=3, band0=0, band1=1, band2=2, dataname='conley_030121', tail_name='rxj1347_PSW_sim0300', image_extnames=['SIG_POST_LENS', 'NOISE'], float_fourier_comps=False, \
# 	use_mask=True, bias=None, nsamp=2000, weighted_residual=True, visual=True, show_input_maps=False, fmin=0.004, max_nsrc=1000, color_sigs=color_prior_sigs, \
# 	n_frames=20, correct_misaligned_shift=True, residual_samples=200, panel_list=panel_list)

# -------------------- SZ inject test for Conley sims --------------------------------------------------

#pcat_test.run_sims_with_injected_sz(visual=False, show_input_maps=False, fmin=0.004, dataname='conley_030121', tail_name='rxj1347_PSW_sim0300', image_extnames=['NOISE'], \
#	bias=None, use_mask=True, max_nsrc=1000, make_post_plots=True, nsamp=2000, residual_samples=200, inject_sz_frac=1.0, inject_diffuse_comp=False, diffuse_comp_path=None, \
#	temp_sample_delay=20, color_sigs=color_prior_sigs, panel_list=panel_list, err_f_divfac=10.)



# pcat_test.real_dat_run(visual=True, band0=0, band1=1, band2=2, show_input_maps=False, fmin=0.005, dataname='conley_030121/conley_030121_10arcmin/', tail_name='rxj1347_PSW_600_simidx300', image_extnames=['SIG_PRE_LENS', 'NOISE', 'DUST'], \
#                        bias=[-0.0037, -0.0055, -0.0085], float_background=True, inject_sz_frac=1.0, use_mask=False, mask_file=mask_file, max_nsrc=500, make_post_plots=True, nsamp=3000, residual_samples=200, \
#                temp_sample_delay=10, color_sigs=color_prior_sigs, err_f_divfac=10., bkg_sig_fac=[10., 15., 20.], bkg_moveweight=20., fc_amp_sig=0.0005, temp_prop_sig_fudge_facs=[1., 15.0, 25.0], \
#                movestar_moveweight=60, birth_death_moveweight=15, float_templates=True, float_fourier_comps=True, n_fc_terms=6, merge_split_moveweight=15, estimate_dust_first=True, nsamp_dustestimate=100, point_src_delay=5, \
#                        initial_template_amplitude_dicts=initial_template_amplitude_dicts, template_names=['sze'], nregion=5, alph=0.0, nominal_nsrc=30., truealpha=3.)


# pcat_test.real_dat_run(visual=False, band0=0, band1=1, band2=2, show_input_maps=False, fmin=0.005, dataname='conley_030121/conley_030121_4arcmin/', tail_name='rxj1347_PSW_240_simidx300', image_extnames=['SIG_PRE_LENS', 'NOISE', 'DUST'], \
#                        bias=[-0.0037, -0.0055, -0.0085], float_background=False, inject_sz_frac=1.0, use_mask=False, mask_file=mask_file, max_nsrc=100, make_post_plots=True, nsamp=3000, residual_samples=200, \
#                temp_sample_delay=10, color_sigs=color_prior_sigs, err_f_divfac=10., bkg_sig_fac=[10., 15., 20.], bkg_moveweight=20., fc_amp_sig=0.0005, temp_prop_sig_fudge_facs=[1., 15.0, 25.0], \
#                movestar_moveweight=60, birth_death_moveweight=15, float_templates=True, float_fourier_comps=True, n_fc_terms=3, merge_split_moveweight=15, estimate_dust_first=True, nsamp_dustestimate=100, point_src_delay=0, \
#                        initial_template_amplitude_dicts=initial_template_amplitude_dicts, template_names=['sze'], nregion=1, alph=0.0, nominal_nsrc=30., truealpha=2.)

# pcat_test.real_dat_run(visual=False, band0=0, band1=1, band2=2, show_input_maps=False, fmin=0.005, dataname='conley_030121/conley_030121_4arcmin/', tail_name='rxj1347_PSW_240_simidx302', image_extnames=['SIG_PRE_LENS', 'NOISE', 'DUST'], \
#                        bias=[0.003, 0.004, 0.005], float_background=True, use_mask=False, mask_file=mask_file, max_nsrc=100, make_post_plots=True, nsamp=3000, residual_samples=200, \
#                temp_sample_delay=10, color_sigs=color_prior_sigs, inject_sz_frac=1.0, err_f_divfac=10., bkg_sig_fac=[10., 15., 20.], bkg_moveweight=20., fc_amp_sig=0.0005, temp_prop_sig_fudge_facs=[1., 15.0, 25.0], \
#                movestar_moveweight=60, birth_death_moveweight=15, float_templates=True, template_names=['sze'], initial_template_amplitude_dicts=initial_template_amplitude_dicts, float_fourier_comps=True, n_fc_terms=3, merge_split_moveweight=15, estimate_dust_first=True, nsamp_dustestimate=10, point_src_delay=0,  nregion=2, alph=0.0, nominal_nsrc=30.)


# mask_path = 'Data/spire/Conley_sims_zitrin/50pix_mask_swap.fits'

# panel_list = ['data0', 'model0', 'residual0', 'fourier_bkg0', 'injected_diffuse_comp0', 'dNdS0']
# mask_path = None
# dcomp_path = base_path+'/Data/spire/cirrus_gen/rxj1347_cirrus_sim_idx'+str(cirrus_gen_idx)+'_020821_1x_planck.npz'
# pcat_test.real_dat_run(nbands=1, band0=0, dataname='Conley_sims_zitrin', tail_name='rxj1347_PSW_sim0351', image_extnames=['SIG_PRE_LENS', 'NOISE'], nsrc_init=0, float_fourier_comps=True, \
# 	use_mask=False, bias=None, nsamp=1000, weighted_residual=True, visual=True, show_input_maps=False, mask_file=mask_path, fmin=0.004, max_nsrc=1500, color_sigs=color_prior_sigs, \
# 	n_frames=20, inject_diffuse_comp=True, diffuse_comp_path=dcomp_path, n_fc_terms=7, fc_amp_sig=0.0002, point_src_delay=5, nregion=2, panel_list=None)

# pcat_test.real_dat_run(nbands=1, band0=0, dataname='Conley_sims_zitrin', tail_name='rxj1347_PSW_sim0351', image_extnames=['SIG_PRE_LENS', 'NOISE'], nsrc_init=0, float_fourier_comps=False, \
# 	use_mask=False, bias=None, nsamp=2000, weighted_residual=True, visual=True, show_input_maps=True, fmin=0.004, max_nsrc=1300, color_sigs=color_prior_sigs, \
# 	n_frames=20)

# pcat_test.real_dat_run(nbands=3, band0=0, band1=1, band2=2, dataname='conley_030121', tail_name='rxj1347_PSW_sim0300', image_extnames=['SIG_POST_LENS', 'NOISE'], float_fourier_comps=False, \
# 	use_mask=True, bias=None, nsamp=2000, weighted_residual=True, visual=True, show_input_maps=False, fmin=0.004, max_nsrc=1000, color_sigs=color_prior_sigs, \
# 	n_frames=20, correct_misaligned_shift=True, residual_samples=200, panel_list=panel_list)


# -------------------- run on MACS for Ben ----------------------------

panel_list = ['data0', 'data1', 'fourier_bkg0','residual0', 'residual1', 'dNdS0']

# pcat_test.real_dat_run(visual=True, band0=0, show_input_maps=False, fmin=0.01, dataname='macs0717_spire_fields', tail_name='SPIRE_macs0717_PSW', image_extnames=['SIGNAL'], error_extname='SIGMA', \
# 	bias=None, use_mask=True, max_nsrc=300, generate_condensed_catalog=True, make_post_plots=True, nsamp=1000, n_condensed_samp=100, residual_samples=100, \
# 	color_sigs=color_prior_sigs, panel_list=panel_list, err_f_divfac=3., bkg_sig_fac=[5., 5., 10.],  \
# 	float_fourier_comps=True, n_fc_terms=6, nregion=4, point_src_delay=20, nsrc_init=0)
# pcat_test.validate_astrometry(dataname='spire_fields_v2', tail_name='SPIREmacs0717_PSW', use_mask=True, image_extnames=['SIGNAL'], error_extname='SIGMA', ngrid=10, return_validation_figs=True)
# result_plots('20210402-151022', cattype=None, burn_in_frac=0.7, boolplotsave=True, boolplotshow=False, plttype='png', gdat=None)

# panel_list = ['data0', 'data1', 'residual0', 'residual1', 'fourier_bkg0', 'fourier_bkg1']

# pcat_test.real_dat_run(visual=False, band0=0, band1=1, show_input_maps=False, fmin=0.015, dataname='spire_fields_v2', tail_name='SPIREmacs0717_PSW', image_extnames=['SIGNAL'], error_extname='SIGMA', \
# 	bias=None, use_mask=True, max_nsrc=300, generate_condensed_catalog=True, make_post_plots=True, nsamp=1000, n_condensed_samp=100, residual_samples=100, \
# 	color_sigs=color_prior_sigs, panel_list=panel_list, err_f_divfac=3., bkg_sig_fac=[5., 5., 10.],  \
# 	float_fourier_comps=True, n_fc_terms=6, nregion=4, estimate_dust_first=True, nsamp_dustestimate=200, point_src_delay=20, nsrc_init=0)


# -------------------- SZ inject test for Conley sims --------------------------------------------------
# dcomp_path = base_path+'/Data/spire/cirrus_gen/rxj1347_cirrus_sim_idx'+str(cirrus_gen_idx)+'_030521_1x_planck.npz'

# mask_file = 'Data/spire/conley_030121/rxj1347_PSW_8arcmin_mask_40_120.fits'

# pcat_test.run_sims_with_injected_sz(visual=True, show_input_maps=False, fmin=0.005, dataname='conley_030121', tail_name='rxj1347_PSW_sim0301', image_extnames=['SIG_POST_LENS', 'NOISE'], \
#  	bias=None, use_mask=True, max_nsrc=1000, make_post_plots=True, nsamp=1000, residual_samples=100, inject_sz_frac=1.0, inject_diffuse_comp=True, \
#  	temp_sample_delay=20, color_sigs=color_prior_sigs, mask_file=mask_file, panel_list=panel_list, err_f_divfac=15., bkg_sig_fac=[5., 5., 10.], temp_prop_sig_fudge_facs=[1., 1.5, 2.0], \
#  	movestar_moveweight=40, birth_death_moveweight=15, float_fourier_comps=True, merge_split_moveweight=15, diffuse_comp_path=dcomp_path, estimate_dust_first=True, nsamp_dustestimate=20, point_src_delay=10)

# initial_template_amplitude_dicts = dict({'sze': dict({'S':0.00, 'M':0.002, 'L':0.025})})

# panel_list = ['data0', 'data1', 'data2', 'residual0', 'residual1', 'residual2']
# result_plots(timestr='20210308-173224',cattype=None, burn_in_frac=0.5, boolplotsave=True, boolplotshow=False, plttype='png', gdat=None)
# sim_idx = 305
# for i in range(1):
# 	pcat_test.run_sims_with_injected_sz(visual=False, show_input_maps=False, fmin=0.005, dataname='conley_030121', tail_name='rxj1347_PSW_sim0'+str(sim_idx+i), image_extnames=['NOISE'], \
# 	 	bias=None, use_mask=True, max_nsrc=10, make_post_plots=True, nsamp=1000, residual_samples=100, inject_sz_frac=1.0, inject_diffuse_comp=False, \
# 	 	temp_sample_delay=0, panel_list=panel_list, movestar_moveweight=10, birth_death_moveweight=5, merge_split_moveweight=5, estimate_dust_first=False, point_src_delay=0)

# mask_file = base_path+'/data/spire/conley_030121/rxj1347_PSW_8arcmin_mask.fits'
# sim_idx = 350
# pcat_test.run_sims_with_injected_sz(visual=True, show_input_maps=True, fmin=0.005, dataname='conley_030121', tail_name='rxj1347_PSW_sim0'+str(sim_idx), image_extnames=['SIGNAL', 'NOISE'], \
#  	bias=None, use_mask=True, max_nsrc=1000, make_post_plots=True, nsamp=1000, residual_samples=100, inject_sz_frac=1.0, inject_diffuse_comp=False, \
#  	temp_sample_delay=20, panel_list=panel_list, mask_file=mask_file, err_f_divfac=10., temp_prop_sig_fudge_facs=[1., 3, 5.0], estimate_dust_first=False, nregion=5, initial_template_amplitude_dicts=initial_template_amplitude_dicts)

# mask_file = base_path+'/Data/spire/bolocam_masks/bolocam_mask_PSW.fits'

# pcat_test.real_dat_run(visual=True, band0=0, band1=1, band2=2, show_input_maps=True, fmin=0.005, dataname='rxj1347_831', tail_name='rxj1347_PSW_nr_1_ext', image_extnames=['SIGNAL'], \
# 	bias=None, use_mask=True, max_nsrc=1000, make_post_plots=True, nsamp=1000, residual_samples=100, \
# 	temp_sample_delay=20, color_sigs=color_prior_sigs, panel_list=panel_list, err_f_divfac=15., bkg_sig_fac=[5., 5., 10.], temp_prop_sig_fudge_facs=[1., 1.5, 2.0], \
# 	movestar_moveweight=40, birth_death_moveweight=15, float_templates=True, mask_file=mask_file, float_fourier_comps=True, n_fc_terms=6, merge_split_moveweight=15, estimate_dust_first=True, nsamp_dustestimate=200, point_src_delay=20, \
# 	initial_template_amplitude_dicts=initial_template_amplitude_dicts, template_names=['sze'])



# ---------------------------- runs on SMC/LMC --------------------------

# pcat_test.artificial_star_test(n_src_perbin=n_src_perbin, fluxbins=fluxbins, inject_catalog_path=inject_catalog_path, nbins=None, nsamp=3000, residual_samples=200, visual=True,\
# 								  dataname='SMC_HERITAGE/cutouts',tail_name='SMC_HERITAGE_cutsize200_199_PSW', im_fpath=im_fpath, use_mask=False, \
# 								  float_fourier_comps=True, n_fc_terms=15, n_frames=10, point_src_delay=10, nsrc_init=0, \
# 								  frac_flux_thresh=2.5, bkg_moore_penrose_inv=False, pos_thresh=1.0, max_nsrc=2000, MP_order=10, ridge_fac=1., fc_amp_sig=0.0005, load_timestr=load_timestr, show_input_maps=False, fmin=0.01)

# pcat_test.artificial_star_test(n_src_perbin=n_src_perbin_brightdust, fluxbins=fluxbins_bright, nbins=None, nsamp=200, residual_samples=40, visual=True,\
# 								  dataname='SMC_HERITAGE/cutouts',tail_name='SMC_HERITAGE_cutsize200_162_PSW', im_fpath=im_fpath, use_mask=False, \
# 								  float_fourier_comps=True, n_fc_terms=10, n_frames=10, point_src_delay=10, nsrc_init=0, \
# 								  frac_flux_thresh=0.5, pos_thresh=1., max_nsrc=1500, load_timestr=load_timestr, show_input_maps=False, fmin=0.05)

# pcat_test.real_dat_run(nbands=1, dataname='SMC_HERITAGE/cutouts', tail_name='SMC_HERITAGE_cutsize200_199_PSW', image_extnames=['IMAGE'], err_fpath=None, nsrc_init=0, float_fourier_comps=True,\
# 						 use_mask=False, bias=None, nsamp=1500, n_fc_terms=15, weighted_residual=True, residual_samples=200,\
# 						  float_templates=False, template_names=None, visual=True, show_input_maps=True, fmin=0.012, \
# 						   max_nsrc=1500, point_src_delay=50, color_sigs=color_prior_sigs, n_frames=50,\
# 						    fc_amp_sig=0.0005)
# pcat_test.real_dat_run(nbands=1, dataname='SMC_HERITAGE/cutouts', tail_name='SMC_HERITAGE_cutsize200_168_PSW', im_fpath=im_fpath, image_extnames=['IMAGE'], err_fpath=None, nsrc_init=0, float_fourier_comps=True,\
# 						 use_mask=False, bias=None, mask_file=mask_file, nsamp=200, n_fc_terms=10, weighted_residual=True,\
# 						  float_templates=False, template_names=None, visual=False, show_input_maps=False, fmin=0.015, \condensed_catalog_overlaid_data_allbands
# 						   max_nsrc=500, movestar_sample_delay=0, color_sigs=color_prior_sigs, n_frames=20, birth_death_sample_delay=0, merge_split_sample_delay=0,\
# 						    fc_prop_alpha=-1., fc_amp_sig=0.002, MP_order=6, bkg_moore_penrose_inv=True)


# pcat_test.real_dat_run(nbands=1, dataname='LMC_HERITAGE/cutouts', tail_name='LMC_HERITAGE_cutsize200_36_PSW', im_fpath=im_fpath, image_extnames=['IMAGE'], err_fpath=None, nsrc_init=0, float_fourier_comps=True,\
# 						 use_mask=False, bias=None, mask_file=None, nsamp=3000, n_fc_terms=15, weighted_residual=True, residual_samples=300,\
# 						  float_templates=False, template_names=None, visual=True, show_input_maps=False, fmin=0.01, \
# 						   max_nsrc=1200, point_src_delay=0, color_sigs=color_prior_sigs, n_frames=20, \
# 						    fc_amp_sig=0.0005, bkg_moore_penrose_inv=True, MP_order=15, ridge_fac=1., nregion=10)

# psf_fwhms_arcsec = [18, 24.9, 36.0]
# pixel_size_arcsec = [6, 10., 14.]
# psf_pix_fwhms = [psf_fwhms_arcsec[i]/pixel_size_arcsec[i] for i in range(len(pixel_size_arcsec))]
# print("psf pix fwhms are ", psf_pix_fwhms)
# pcat_test.real_dat_run(nbands=3, band0=0, band1=1, band2=2, dataname='LMC_HERITAGE/cutouts', tail_name='test_lmc_PSW_100', image_extnames=['IMAGE'], err_fpath=None, nsrc_init=0, float_fourier_comps=True,\
# 						 use_mask=False, bias=None, mask_file=None, psf_fwhms=psf_pix_fwhms, nsamp=3000, n_fc_terms=10, weighted_residual=True, residual_samples=300,\
# 						  float_templates=False, template_names=None, visual=True, show_input_maps=True, fmin=0.01, \
# 						   max_nsrc=1200, ridge_fac=1., bkg_moore_penrose_inv=False, point_src_delay=20, color_sigs=color_prior_sigs, n_frames=20, \
# 						    fc_amp_sig=0.0005, nregion=5)

# pcat_test.validate_astrometry(dataname='LMC_HERITAGE/cutouts', tail_name='test_lmc_PSW_100', ngrid=10, return_validation_figs=True)
# figs = pcat_test.validate_astrometry(tail_name='rxj1347_PSW_sim0'+str(sim_idx), dataname='sims_12_2_20', ngrid=10, return_validation_figs=True)

# --------- condensed catalog results ----------------
# cond_cat_fpath = 'spire_results/20210111-133739/condensed_catalog_nsamp=100_prevcut=0.8_searchradius=0.75_maskhwhm=2.txt'
# cond_cat_fpath = 'spire_results/20210114-054614/condensed_catalog_nsamp=100_prevcut=0.8_searchradius=0.75_maskhwhm=2.txt'
# cond_cat_fpath = 'spire_results/20210117-193600/condensed_catalog_nsamp=100_prevcut=0.5_searchradius=0.75_maskhwhm=5.txt'
# cond_cat_fpath = None
# result_plots(timestr='20210322-000000',cattype=None, burn_in_frac=0.7, boolplotsave=True, boolplotshow=False, plttype='png', gdat=None, \
# 			fourier_comp_plots=False, condensed_catalog_plots=False, condensed_catalog_fpath = cond_cat_fpath, generate_condensed_cat=False, \
# 			n_condensed_samp=200, prevalence_cut=0.5, mask_hwhm=5, nregion=2, search_radius=0.75, matching_dist=0.75, residual_plots=False, flux_color_color_plots=True)

# -----------------------------------------
# result_plots(timestr='20210224-122047',cattype=None, generate_condensed_cat=True, n_condensed_samp=400, prevalence_cut=0.5, mask_hwhm=1, condensed_catalog_plots=True,\
# 				 burn_in_frac=0.75, boolplotsave=True, boolplotshow=False, plttype='png', gdat=None, flux_color_color_plots=True, search_radius=0.75)
# result_plots(timestr='20210322-000000',cattype=None, burn_in_frac=0.75, boolplotsave=True, boolplotshow=False, plttype='png', gdat=None)
# result_plots(timestr='20210109-134459',cattype=None, residual_plots=False, dc_background_plots=False, burn_in_frac=0.9, boolplotsave=True, boolplotshow=False, plttype='png', gdat=None)

# # def run_pcat_dust_and_sz_test(sim_idx=200, inject_dust=False, show_input_maps=False, inject_sz_frac=1.0):


# 	# ob = lion(band0=0, band1=1, band2=2, base_path=base_path, result_path=result_path, round_up_or_down='down', bolocam_mask=True, float_background=True, burn_in_frac=0.6, bkg_sig_fac=5.0, bkg_sample_delay=10, temp_sample_delay=20, \
# 	# 		 cblas=True, openblas=False, visual=False, show_input_maps=show_input_maps, float_templates=True, template_names=template_names, init_template_amplitude_dicts=initial_template_amplitude_dicts, tail_name='rxj1347_PSW_sim0'+str(sim_idx)+'_dust',\
# 	# 		  dataname='sim_w_dust', bias=None, max_nsrc=1500, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True,\
# 	# 		   make_post_plots=True, nsamp=50, delta_cp_bool=True, use_mask=True, residual_samples=100, template_filename=t_filenames, inject_dust=inject_dust, inject_sz_frac=inject_sz_frac)
# 	# ob.main()



