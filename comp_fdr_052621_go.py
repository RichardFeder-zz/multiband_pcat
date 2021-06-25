import numpy as np
import pickle
from spire_data_utils import load_param_dict

def load_truth_cat(timestr):
    if simidx is None:
        gdat, filepath, result_path = load_param_dict(timestr, result_path=result_path)
        simidx = int(gdat.tail_name[-3:])


def load_cat_flux_dists(timestr_list, result_path='spire_results/', nsamp=300, nbands=3):
    print('timestr_list is ', timestr_list)
    for t, timestr in enumerate(timestr_list):
        print(timestr)
        pcat_chain = np.load(result_path+timestr+'/chain.npz')

        #gdat, filepath, result_path = load_param_dict(timestr, result_path=result_path)                                                                                   
        #simidx = int(gdat.tail_name[-3:])                                                                                                                                 

        pcat_fluxes = np.array([pcat_chain['f'][b][-nsamp:,:] for b in range(nbands)])
        if t==0:
            nsrc_max = len(pcat_fluxes[0][0])
            print('nsrc max is ', nsrc_max)
            if nbands==1:
                all_fluxes_array = np.zeros((nsamp*len(timestr_list), nsrc_max))
            else:
                all_fluxes_array = np.zeros((nbands, nsamp*len(timestr_list), nsrc_max))

        if nbands==1:
            print('we are here')
            all_fluxes_array[t*nsamp:(t+1)*nsamp, :] = pcat_fluxes[0]
        else:
            print('now were here')
            all_fluxes_array[:,t*nsamp:(t+1)*nsamp,:] = pcat_fluxes[0]


    return all_fluxes_array	

def load_cat_result_data(timestr, simidx=None, basepath='Data/spire/', cat_basepath='Data/spire/conley_030121/p2_pre_lens_cat/', result_path='spire_results/', \
                                         nsamp=300, mask_hwhm=2.0, pos_thresh=1.0, frac_flux_thresh=10.0, \
                                         fmin=0.002, fmax=0.1, nfluxbins=15, cutout_key='SIG_PRE_LENS', nbands=3):
    # get chain from timestring                                                                                                                                            
    pcat_chain = np.load(result_path+timestr+'/chain.npz')

    # grab conley sim idx if not explicitly provided                                                                                                                       
    if simidx is None:
        gdat, filepath, result_path = load_param_dict(timestr, result_path=result_path)
        simidx = int(gdat.tail_name[-3:])


    # get chain from timestring                                                                                                                                            
    bandstrs = ['PSW', 'PMW', 'PLW']

    truth_fs = []
    for b in range(nbands):
        cat_tail = 'pre_lens_cat_rxj1347_'+str(simidx-300)+'_'+bandstrs[b]
        catpath = cat_basepath+cat_tail+'.npz'

        load_cat = np.load(catpath, allow_pickle=True)

        if b==0:
            truth_x = load_cat['x']
            truth_y = load_cat['y']

        truth_fs.append(load_cat['flux'])

    truth_fs = np.array(truth_fs)
    truth_x -= np.min(truth_x)
    truth_y -= np.min(truth_y)

    ref_img = fits.open('Data/spire/conley_030121/rxj1347_sims/rxj1347_PSW_sim0'+str(simidx)+'.fits')
    cut_obs = fits.open('Data/spire/conley_030121/conley_041921_10arcmin/rxj1347_PSW_600_simidx'+str(simidx)+'.fits')

    pcat_x_raw = pcat_chain['x'][-nsamp:,:]
    pcat_y_raw = pcat_chain['y'][-nsamp:,:]
    pcat_fluxes_raw = [pcat_chain['f'][b][-nsamp:,:] for b in range(nbands)]

    pcat_xs, pcat_ys = [[] for x in range(2)]
    pcat_fluxes = [[] for b in range(nbands)]

    for n in range(nsamp):

        fluxmask = (pcat_fluxes_raw[0][n,:] > 0.)
        pcat_xs.append(pcat_x_raw[n,fluxmask])
        pcat_ys.append(pcat_y_raw[n,fluxmask])
        for b in range(nbands):
            pcat_fluxes[b].append(pcat_fluxes_raw[b][n,fluxmask])

    wcs_ref = WCS(ref_img['SIGNAL'].header)
    wcs_con = WCS(cut_obs[cutout_key].header)

    zero_coord_con = wcs_con.wcs_pix2world(0., 0., 1)
    zero_coord_con_to_mock = wcs_ref.wcs_world2pix(zero_coord_con[0], zero_coord_con[1], 1)

    dx_cut = zero_coord_con_to_mock[0]
    dy_cut = zero_coord_con_to_mock[1]

    truth_x -= dx_cut
    truth_y -= dy_cut

    imdim = np.array([cut_obs[cutout_key].shape[0], cut_obs[cutout_key].shape[1]])

    print('imdim is ', imdim)
    truth_cat_mask = (truth_fs[0] > fmin)*(truth_x > mask_hwhm)*(truth_y > mask_hwhm)*(truth_x < imdim[0]-mask_hwhm)*(truth_y < imdim[1]-mask_hwhm)

    truth_cat = [truth_x[truth_cat_mask], truth_y[truth_cat_mask]]

    for b in range(nbands):
        truth_cat.append(truth_fs[b,truth_cat_mask])

    truth_cat = np.array(truth_cat).transpose()

    return truth_cat, pcat_xs, pcat_ys, pcat_fluxes, imdim, simidx
    


def get_completeness_and_fdr(timestr, simidx=None, basepath='Data/spire/', cat_basepath='Data/spire/conley_030121/p2_pre_lens_cat/', result_path='spire_results/', \
                                         nsamp=300, mask_hwhm=2.0, pos_thresh=1.0, frac_flux_thresh=10.0, \
                                         fmin=0.002, fmax=0.1, nfluxbins=15, cutout_key='SIG_PRE_LENS', nbands=3, b=0):


    truth_cat, pcat_xs, pcat_ys, pcat_fluxes, imdim, simidx = load_cat_result_data(timestr, simidx=simidx, basepath=basepath, cat_basepath=cat_basepath, result_path=result_path, \
                                                                   nsamp=nsamp, mask_hwhm=mask_hwhm, fmin=fmin, fmax=fmax, cutout_key=cutout_key, nbands=nbands)

    fluxbins = np.logspace(np.log10(fmin), np.log10(fmax), 15)

    completeness_ensemble, completeness_vs_flux = completeness_basic_pcat(truth_cat, pcat_xs, pcat_ys, pcat_fluxes[b], fluxbins, frac_flux_thresh=10., pos_thresh=1.0, \
                           imdim=imdim, mask_hwhm=2.0, residual_samples=nsamp)

    fdr_vs_flux = false_discovery_rate_basic_pcat(truth_cat, pcat_xs, pcat_ys, pcat_fluxes[b], fluxbins, frac_flux_thresh=10., pos_thresh=1.0, \
                           imdim=imdim, mask_hwhm=2.0, residual_samples=nsamp)

    return fluxbins, completeness_vs_flux, fdr_vs_flux, simidx


def flux_density_errors(truth_catalog, pcat_xs, pcat_ys, pcat_fluxes, nbands=3, residual_samples=300, \
                       pos_thresh = 1.0):

    n_src_truth = truth_catalog.shape[0]

    recovered_flux_ensemble = np.zeros((residual_samples, nbands, n_src_truth))
    fluxerror_ensemble = np.zeros((residual_samples, nbands, n_src_truth))

    for b in range(nbands):

        for i in range(residual_samples):

            for j in range(n_src_truth):

                idx_pos = np.where(np.sqrt((pcat_xs[-i] - truth_catalog[j,0])**2 +(pcat_ys[-i] - truth_catalog[j,1])**2)  < pos_thresh)[0]

                fluxes_poscutpass = pcat_fluxes[b][-i][idx_pos]

                if frac_flux_thresh is None:
                    flux_candidates = fluxes_poscutpass
        else:
                    mask_flux = np.where(np.abs(fluxes_poscutpass - truth_catalog[j,2+b])/truth_catalog[j,2+b] < frac_flux_thresh)[0]
                    flux_candidates = fluxes_poscutpass[mask_flux]


                if len(flux_candidates) >= 1:

                    dists = np.sqrt((pcat_xs[-i][mask_flux] - truth_catalog[j,0])**2 + (pcat_ys[-i][mask_flux] - truth_catalog[j,1])**2)

                    mindist_idx = np.argmin(dists)
                    mindist_flux = flux_candidates[mindist_idx]

                    recovered_flux_ensemble[i,b,j] = mindist_flux

                    fluxerror_ensemble[i,b,j] = (mindist_flux-truth_catalog[j,2+b])


    mean_flux_error = np.zeros((n_src_truth,nbands))
    pct_16_flux = np.zeros((n_src_truth,nbands))
    pct_84_flux = np.zeros((n_src_truth,nbands))

    mean_recover_flux = np.zeros((n_src_truth,nbands))
    pct_16_recover_flux = np.zeros((n_src_truth,nbands))
    pct_84_recover_flux = np.zeros((n_src_truth,nbands))


    for b in range(nbands):
        for j in range(n_src_truth):
            nonzero_fidx = np.where(fluxerror_ensemble[:,b,j] != 0)[0]
            if len(nonzero_fidx) > 0:
                mean_flux_error[j,b] = np.median(fluxerror_ensemble[nonzero_fidx,b, j])
                pct_16_flux[j,b] = np.percentile(fluxerror_ensemble[nonzero_fidx,b, j], 16)
                pct_84_flux[j,b] = np.percentile(fluxerror_ensemble[nonzero_fidx,b, j], 84)

                mean_recover_flux[j,b] = np.median(recovered_flux_ensemble[nonzero_fidx, b, j])
                pct_16_recover_flux[j,b] = np.percentile(recovered_flux_ensemble[nonzero_fidx, b, j], 16)
                pct_84_recover_flux[j,b] = np.percentile(recovered_flux_ensemble[nonzero_fidx, b, j], 84)

        nonzero_ferridx = np.where(mean_flux_error[:,b] != 0)[0]
        ferr_recover = 1e3*np.array([mean_recover_flux[nonzero_ferridx, b] - pct_16_recover_flux[nonzero_ferridx, b], pct_84_recover_flux[nonzero_ferridx, b]-mean_recover_flux[nonzero_ferridx, b]])

    return ferr_recover, nonzero_ferridx, recovered_flux_ensemble, fluxerror_ensemble, mean_flux_error, pct_16_flux, pct_84_flux, mean_recover_flux, pct_16_recover_flux, pct_84_recover_flux

def group_cat_fluxerrs_into_bins(truth_cat, mean_flux_error, pct_16_flux, pct_84_flux, nonzero_ferridx, fluxbins=None, fmin=0.002, fmax=0.1, n_flux_bins=15, nbands=3):

    if fluxbins is None:
        fluxbins = np.logspace(np.log10(fmin), np.log10(fmax), n_flux_bins)

    mean_ferr_nonzero_ferridx = mean_flux_error[nonzero_ferridx,:]
    pct_16_nonzero_ferridx = pct_16_flux[nonzero_ferridx,:]
    pct_84_nonzero_ferridx = pct_84_flux[nonzero_ferridx,:]

    mean_flux_error_binned = np.zeros((len(fluxbins)-1, nbands))
    pct16_flux_error_binned = np.zeros((len(fluxbins)-1, nbands))
    pct84_flux_error_binned = np.zeros((len(fluxbins)-1, nbands))

    flux_bin_idxs = [[np.where((truth_cat[:,2+b] > fluxbins[i])&(truth_cat[:,2+b] < fluxbins[i+1]))[0] for i in range(len(fluxbins)-1)] for b in range(nbands)]

    for b in range(nbands):
        for f in range(len(fluxbins)-1):

            finbin = np.where((truth_cat[nonzero_ferridx,2+b] >= fluxbins[f])&(truth_cat[nonzero_ferridx,2+b] < fluxbins[f+1]))[0]

            if len(finbin)>0:
                mean_flux_error_binned[f,b] = np.median(mean_ferr_nonzero_ferridx[finbin,b])

                pct16_flux_error_binned[f,b] = np.percentile(mean_ferr_nonzero_ferridx[finbin,b], 16)

                pct84_flux_error_binned[f,b] = np.percentile(mean_ferr_nonzero_ferridx[finbin,b], 84)

    all_nsrc_perfbin = [[len(fbin_idxs) for fbin_idxs in flux_bin_idxs[b]] for b in range(nbands)]

    return all_nsrc_perfbin, mean_flux_error_binned, pct16_flux_error_binned, pct84_flux_error_binned, fluxbins




# --------------- from SPORC ---------------

result_path='spire_results/'
#timestr_list = np.load('conley_unlensed_10arcmin_1band_1mJy_beam_inst_052521_timestrs.npz')['timestr_list']                                                               
#timestr_list = np.load('conley_unlensed_10arcmin_2band_1mJy_beam_inst_052921_timestrs.npz')['timestr_list']                                                               

#cat_basepath='Data/spire/conley_030121/pre_lens_cat_rxj1347/'                                                                                                             

#dir_name = 'conley_unlensed_10arcmin_1band_1mJy_beam_inst_052921'                                                                                                         
#dir_name = 'conley_unlensed_10arcmin_1band_1mJy_beam_inst_053021_2x_Planck_nfc=6'                                                                                         
dir_name = 'conley_unlensed_10arcmin_1band_1mJy_beam_060121_vary_nfc_4x_Planck_simidx_301'

if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
mask_hwhm = 2.0
timestr_list = np.load(dir_name.replace('21_', '21_timestrs_')+'.npz')['timestr_list']

fmin = 0.002
all_fluxes_array = load_cat_flux_dists(timestr_list, nsamp=300, nbands=1)
truth_cats = []
for timestr in timestr_list:
    truth_cat, pcat_xs, pcat_ys, pcat_fluxes, imdim, _ = load_cat_result_data(timestr, nbands=1, nsamp=300, result_path=result_path, mask_hwhm=mask_hwhm, fmin=fmin)
    truth_cats.append(truth_cat)

print('all_fluxes array has shape', all_fluxes_array.shape)
np.savez(dir_name+'/'+dir_name+'_all_fluxes_array.npz', timestr_list=timestr_list, all_fluxes_array=all_fluxes_array, truth_cats=truth_cats)
exit()


fmin = 0.002
fmax = 0.1

mask_hwhm=2.0
pos_thresh=1.0
frac_flux_thresh=10.
nsamp = 500
nfluxbins=15


list_of_comps = []
list_of_fdrs = []
list_of_simidxs = []
list_of_nsrc_perfbin, list_of_mean_flux_error_binned, list_of_pct16_flux_error_binned, list_of_pct84_flux_error_binned = [[] for x in range(4)]
list_of_ferr_recover, list_of_nonzero_ferridx, list_of_mean_fe, list_of_pct16_fe, list_of_pct84_fe = [[] for x in range(5)]



for timestr in timestr_list:
    print('timestr = ', timestr)

    fluxbins, comp_vs_flux, fdr_vs_flux, simidx = get_completeness_and_fdr(timestr, nsamp=nsamp, result_path=result_path, fmin=fmin, fmax=fmax, mask_hwhm=mask_hwhm, \
                                                                           pos_thresh=pos_thresh, frac_flux_thresh=frac_flux_thresh, nfluxbins=nfluxbins, nbands=1)

    list_of_comps.append(comp_vs_flux)
    list_of_fdrs.append(fdr_vs_flux)
    list_of_simidxs.append(simidx)
    print('simidx is ', simidx)
    truth_cat, pcat_xs, pcat_ys, pcat_fluxes, imdim, _ = load_cat_result_data(timestr, nbands=1, nsamp=300, result_path=result_path, mask_hwhm=mask_hwhm, fmin=fmin)

    print('truth_cat has shape ', truth_cat.shape)

    ferr_recover, nonzero_ferridx, recovered_flux_ensemble,\
            fluxerror_ensemble, mean_fe, pct16_fe, pct84_fe, \
                        mean_rf, pct16_rf, pct84_rf = flux_density_errors(truth_cat, pcat_xs, pcat_ys, pcat_fluxes, nbands=1)

    print('mean_fe has shape', mean_fe.shape)

    n_flux_bins=10

    list_of_ferr_recover.append(ferr_recover)
    list_of_nonzero_ferridx.append(nonzero_ferridx)
    list_of_mean_fe.append(mean_fe)
    list_of_pct16_fe.append(pct16_fe)
    list_of_pct84_fe.append(pct84_fe)



    #all_nsrc_perfbin, mean_flux_error_binned,\                                                                                                                            
    #        pct16_flux_error_binned, pct84_flux_error_binned, fluxbins = group_cat_fluxerrs_into_bins(truth_cat, mean_fe, pct16_fe, pct84_fe, nonzero_ferridx, fmin=fmin,\
    
    #fmax=fmax, n_flux_bins=n_flux_bins)                                                                                                                                       

    #list_of_nsrc_perfbin.append(all_nsrc_perfbin)                                                                                                                         
    #list_of_mean_flux_error_binned.append(mean_flux_error_binned)                                                                                                         
    #list_of_pct16_flux_error_binned.append(pct16_flux_error_binned)                                                                                                       
    #list_of_pct84_flux_error_binned.append(pct84_flux_error_binned)                                                                                                       
    np.savez(dir_name+'/'+timestr+'.npz', ferr_recover=ferr_recover, nonzero_ferridx=nonzero_ferridx, mean_fe=mean_fe, pct16_fe=pct16_fe, pct84_fe=pct84_fe, simidx=simidx)

    #np.savez('conley_unlensed_10arcmin_2band_1mJy_beam_inst_052821_fluxerrs/'+timestr+'.npz', ferr_recover=ferr_recover, nonzero_ferridx=nonzero_ferridx, mean_fe=mean_fe, pct16_fe=pct16_fe, pct84_fe=pct84_fe, simidx=simidx)                                                                                                                     

#np.savez('conley_unlensed_10arcmin_1band_6mJy_beam_inst_052821_fluxerrs.npz', list_of_ferr_recover=list_of_ferr_recover, list_of_nonzero_ferridx=list_of_nonzero_ferridx,list_of_mean_fe=list_of_mean_fe, list_of_pct16_fe=list_of_pct16_fe, list_of_pct84_fe=list_of_pct84_fe)                                                                    

np.savez(dir_name+'/'+dir_name+'_comps_fdrs.npz', timestr_list=timestr_list, fluxbins=fluxbins, list_of_comps=list_of_comps, list_of_fdrs=list_of_fdrs, list_of_simidxs=li\
st_of_simidxs)

#np.savez('conley_unlensed_10arcmin_2band_1mJy_beam_inst_052521_comps_fdrs.npz', timestr_list=timestr_list, fluxbins=fluxbins, list_of_comps=list_of_comps, list_of_fdrs=list_of_fdrs, list_of_simidxs=list_of_simidxs)   


print(fluxbins)
print(comp_vs_flux)
print(fdr_vs_flux)

exit()



# ------------------------ local machine ---------------------------


timestr_list = np.load('conley_unlensed_10arcmin_3band_1mJy_beam_inst_052521_timestrs.npz')['timestr_list']



fmin = 0.002
fmax = 0.1

mask_hwhm=2.0
pos_thresh=1.0
frac_flux_thresh=10.
nsamp = 500
nfluxbins=15


list_of_comps = []
list_of_fdrs = []

list_of_nsrc_perfbin, list_of_mean_flux_error_binned, list_of_pct16_flux_error_binned, list_of_pct84_flux_error_binned = [[] for x in range(4)]

for timestr in timestr_list:

	fluxbins, comp_vs_flux, fdr_vs_flux = get_completeness_and_fdr(timestr, nsamp=nsamp, result_path=result_path, fmin=fmin, fmax=fmax, mask_hwhm=mask_hwhm, \
		pos_thresh=pos_thresh, frac_flux_thresh=frac_flux_thresh, nfluxbins=nfluxbins)

	list_of_comps.append(comp_vs_flux)
	list_of_fdrs.append(fdr_vs_flux)


	truth_cat, pcat_xs, pcat_ys, pcat_fluxes = load_cat_result_data(timestr, nsamp=300, simidx=8, result_path=result_path)

	ferr_recover, nonzero_ferridx, recovered_flux_ensemble,\
	        fluxerror_ensemble, mean_fe, pct16_fe, pct84_fe, \
	            mean_rf, pct16_rf, pct84_rf = flux_density_errors(truth_cat, pcat_xs, pcat_ys, pcat_fluxes, nbands=3)

	n_flux_bins=10

	all_nsrc_perfbin, mean_flux_error_binned,\
	        pct16_flux_error_binned, pct84_flux_error_binned, fluxbins = group_cat_fluxerrs_into_bins(truth_cat, mean_fe, pct16_fe, pct84_fe, nonzero_ferridx, fmin=fmin, fmax=fmax, n_flux_bins=n_flux_bins)



	list_of_nsrc_perfbin.append(all_nsrc_perfbin)
	list_of_mean_flux_error_binned.append(mean_flux_error_binned)
	list_of_pct16_flux_error_binned.append(pct16_flux_error_binned)
	list_of_pct84_flux_error_binned.append(pct84_flux_error_binned)

# np.savez('conley_unlensed_10arcmin_3band_1mJy_beam_inst_052521_comps_fdrs.npz', timestr_list=timestr_list, fluxbins=fluxbins, list_of_comps=list_of_comps, list_of_fdrs=list_of_fdrs)

# np.savez('conley_unlensed_10arcmin_3band_1mJy_beam_inst_052521_comps_fdrs.npz', timestr_list=timestr_list, fluxbins=fluxbins, list_of_comps=list_of_comps, list_of_fdrs=list_of_fdrs)

print(fluxbins)
print(comp_vs_flux)
print(fdr_vs_flux)