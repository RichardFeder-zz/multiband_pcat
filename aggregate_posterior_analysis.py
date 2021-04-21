import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from spire_data_utils import *
import pickle
import corner
import os

# from pcat_spire import *


# def load_param_dict(timestr, result_path='/Users/richardfeder/Documents/multiband_pcat/spire_results/'):

def return_step_func_hist(xvals, hist_bins, hist_vals):
    all_step_vals = np.zeros_like(xvals)
    for i, x in enumerate(xvals):
        for j in range(len(hist_bins)-1):
            if hist_bins[j] <= x < hist_bins[j+1]:
                all_step_vals[i] = hist_vals[j]
                
    return all_step_vals

def handselect_residual(timestr_list_file=None, fmin_subtract=0.01, timestr_list=None, inject_sz_frac=None, tail_name='9_24_20', datatype='real', pdf_or_png ='.png', save=False, dust=False, ref_dust_amp=1.):

    if timestr_list_file is not None:
        timestr_list = np.load(timestr_list_file)['timestr_list']
    elif timestr_list is None:
        print('no list of runs specified, ending now')
        return

    band_dict = dict({0:'250 micron', 1:'350 micron', 2:'500 micron'})

    temp_mock_amps = [0.0111, 0.1249, 0.6912]
    flux_density_conversion_facs = [86.29e-4, 16.65e-3, 34.52e-3]

    temp_mock_amps_dict = dict({'S':0.0111, 'M': 0.1249, 'L': 0.6912})
    figs = []
    median_select_resid_list = []
    for i in np.arange(1):
        all_resids = []

        if inject_sz_frac is not None:
            inject_sz_amp = inject_sz_frac*temp_mock_amps[i]
            ref_vals.append(inject_sz_amp)
            print('inject sz amp is ', inject_sz_amp)

        f = plt.figure(figsize=(10,10))
        plt.title('median residual, $f_{min}$='+str(np.round(fmin_subtract, 4))+', '+band_dict[i], fontsize=18)
        for j, timestr in enumerate(timestr_list):
            if timestr=='':
                continue
            gdat, filepath, result_path = load_param_dict(timestr, result_path='spire_results/')
            
            datapath = gdat.base_path+'/Data/spire/'+gdat.dataname+'/'
            
#            print('filepath:', filepath)
#            print('datapath:', datapath)

            dat = pcat_data(gdat.auto_resize, nregion=gdat.nregion)
            dat.load_in_data(gdat)


            chain = np.load(filepath+'/chain.npz')

            xsrcs = chain['x']
            ysrcs = chain['y']
            fsrcs = chain['f']
            chi2modl = chain['chi2sample']

#            print('Minimum chi2 is ', np.min(chi2modl, axis=0))
#            print(np.array(xsrcs).shape, np.array(fsrcs.shape))

            bkgs = chain['bkg']

            template_amplitudes = chain['template_amplitudes']

            print(np.array(xsrcs).shape, np.array(fsrcs.shape))
            for s in range(100):
            #for s in range(len(xsrcs)):
                
                fmask = (fsrcs[0,:,-s] > fmin_subtract)
                print('fmask has shape', fmask.shape)
                x_fmask = xsrcs[fmask,-s]
                y_fmask = ysrcs[fmask,-s]
                fs_fmask = fsrcs[0,fmask,-s]
                print('x_fmask has shape ', x_fmask.shape)
                print('f_fmask has shape ', fs_fmask.shape)
                print(fs_fmask)
                dtemp = []
                for t, temp in enumerate(dat.template_array[i]):
                    if temp is not None and template_amplitudes[s,t,i]:
                        dtemp.append(template_amplitudes[s,t,i]*temp)

                if len(dtemp) > 0:
                    dtemp = np.sum(np.array(dtemp), axis=0).astype(np.float32)

                libmmult = ctypes.cdll['./blas.so']
                lib = libmmult.clib_eval_modl
                print('imszs:', gdat.imszs[i])
                pixel_per_beam = 2*np.pi*((3.)/2.355)**2
                dmodel, diff2 = image_model_eval(x_fmask, y_fmask, pixel_per_beam*dat.ncs[i]*fs_fmask, bkgs[i,-s], gdat.imszs[i].astype(np.int32), dat.ncs[i],\
                                                 np.array(dat.cfs[i]).astype(np.float32()), weights=dat.weights[i],\
                                                 lib=lib, template=dtemp)
            

                
                r = dat.data_array[i]-dmodel[i]
                print('residual has shape', r.shape)
                print(r)
                all_resids.append(r)

            median_select_resid = np.median(np.array(all_resids), axis=0)
            median_select_resid_list.append(median_select_resid)

            plt.imshow(median_select_resid, cmap='Greys', vmin=np.percentile(median_select_resid, 5), vmax=np.percentile(median_select_resid, 95))
        plt.xlabel('x [pixel]',fontsize=16)
        plt.ylabel('y [pixel]', fontsize=16)
            

        if save:
            plt.savefig('agg_posts/median_residual_fminsub='+str(fmin_subtract)+'_'+tail_name+'.'+pdf_or_png)
        figs.append(f)
        plt.close()

        return median_select_resid_list, figs

#ms, figs = handselect_residual(fmin_subtract=0.01, timestr_list_file='rxj1347_mock_test_9_24_20_10sims.npz')

class pcat_agg():
    
    flux_density_conversion_facs = [86.29e-4, 16.65e-3, 34.52e-3]
    band_dict = dict({0:'250 micron', 1:'350 micron', 2:'500 micron'})

    def __init__(self, base_path='/home/mbzsps/multiband_pcat/', result_path='/home/mbzsps/multiband_pcat/spire_results/'):
        self.base_path = base_path
        self.result_path = result_path
        self.chain = None

    def load_chain(self, timestr, inplace=True):
        chain = np.load(self.result_path+timestr+'/chain.npz')
        if inplace:
            self.chain = chain
        else:
            return chain

    def load_timestr_list(self, timestr_list_file, inplace=False):
        timestr_list = np.load(timestr_list_file)['timestr_list']
        if inplace:
            self.timestr_list = timestr_list
        else:
            return timestr_list

    def grab_acceptance_fracs(self, chain=None):
        if chain is None:
            chain = self.chain
        acceptance_fracs = chain['accept']
        print('acceptance_fracs has shape', acceptance_fracs.shape)
        return acceptance_fracs

    def grab_chi2_stats(self, chain=None):
        if chain is None:
            chain = self.chain
        chi2_stats = chain['chi2']
        print('chi2 stats has shape', chi2_stats.shape)
        return chi2_stats
    
    def grab_bkg_vals(self, chain=None):
        if chain is None:
            chain = self.chain
            
        bkg_vals = chain['bkg']
        #print(bkg_vals)
        print('bkg vals has shape', bkg_vals.shape)
        return bkg_vals

    def compile_stats(self, mode='accept', timestr_list_file=None, inplace=False):
        
        all_stats = []

        if timestr_list_file is not None:
            self.load_timestr_list(timestr_list_file, inplace=True)

        for t, timestr in enumerate(self.timestr_list):
            print(timestr)
            self.load_chain(timestr, inplace=True)
            if mode=='accept':
                stats = self.grab_acceptance_fracs()
            elif mode=='chi2':
                stats = self.grab_chi2_stats()
            elif mode=='bkg':
                stats = self.grab_bkg_vals()
                #if t==0:
            #    stats_shape = stats.shape
            #    all_stats_shape = stats_shape.copy()
            #    all_stats_shape[0] *= len(timestr_list_file)
            #    print('all stats shape is now ', all_stats_shape)
            #    all_stats = np.zeros((all_stats_shape))
            
            all_stats.append(stats)

            #if len(all_stats_shape)==2:
            #    all_stats[i*stats_shape[0]:(i+1)*stats_shape[0],:] = stats
            #elif len(all_stats_shape)==3:
            #    all_stats[i*stats_shape[0]:(i+1)*stats_shape[0],:,:] = stats

        all_stats = np.array(all_stats)

        print('all stats has shape', all_stats.shape)
        if inplace:
            self.all_stats = all_stats
        else:
            return all_stats

# for getting bkg means
#timestr_sim_file = 'rxj1347_conley_10arcmin_041921_timestrs_fitbkg.npz'
#pcat_agg_obj = pcat_agg()
#burn_in = 1400
#timestr_list = pcat_agg_obj.load_timestr_list(timestr_sim_file, inplace=False)

#all_bkg_vals = pcat_agg_obj.compile_stats(mode='bkg', timestr_list_file=timestr_sim_file)

#print('all_bkg_vals has shape', all_bkg_vals.shape)
#all_bkg_vals = all_bkg_vals[:,burn_in:, :]

#mean_bkg_vals = np.mean(all_bkg_vals, axis=1)

#print('mean bkg vals is ', mean_bkg_vals)
#print(mean_bkg_vals.shape)

#sim_idx_list = []
#mean_list = []

#for timestr in timestr_list:
#    gdat, filepath, result_path = load_param_dict(timestr, result_path='spire_results/')
#    sim_idx = int(gdat.tail_name[-3:])
#    print('sim_idx is ', sim_idx)
    
#    sim_idx_list.append(sim_idx)

#print(sim_idx_list)
#print(mean_bkg_vals)

#np.savez('rxj1347_conley_10arcmin_041921_bkg_best_fits.npz', sim_idx_list=sim_idx_list, mean_bkg_vals=mean_bkg_vals)


#exit()



def gather_posteriors(timestr_list=None, timestr_list_file=None, inject_sz_frac=None, tail_name='6_4_20', band_idx0=0, datatype='real', pdf_or_png='.png', save=False, dust=False, ref_dust_amp=1., integrate_sz_prof=False, burn_in_frac=None):

    if timestr_list_file is not None:
        timestr_list = np.load(timestr_list_file)['timestr_list']
    elif timestr_list is None:
        print('no list of runs specified, ending now')
        return

#     all_temp_posteriors = []
    figs = []

    band_dict = dict({0:'250 micron', 1:'350 micron', 2:'500 micron'})

    # temp_mock_amps = [None, 0.3, 0.5] # MJy/sr
    #temp_mock_amps = [0.0111, 0.1249, 0.6912]
    temp_mock_amps = [0.03, 0.20, 0.80] # updated values from X-ray measurement of RXJ1347
    flux_density_conversion_facs = [86.29e-4, 16.65e-3, 34.52e-3]

    #temp_mock_amps_dict = dict({'S':0.0111, 'M': 0.1249, 'L': 0.6912})
    temp_mock_amps_dict = dict({'S':0.03, 'M':0.20, 'L':0.80}) # udpated values from X-ray measurement
    medians = []
    pcts_5 = []
    pcts_16, pcts_84, pcts_95 = [[] for x in range(3)]
    ref_vals = []
    dont_include_idxs = []
    #dont_include_idxs = [331, 332, 333, 334]
    #lensed_cat_idxs = np.load('lensed_cat_criteria_45arcsec_20mJy.npz')
    #dont_include_idxs = lensed_cat_idxs['unsatisfied_idxs']
    #print(dont_include_idxs)
    f = plt.figure(figsize=(15, 5), dpi=200)
    plt.suptitle(tail_name, fontsize=20, y=1.04)

    indiv_sigmas_list = []
    indiv_medians_list = []
    indiv_84pcts_list = []
    indiv_16pcts_list = []
    list_of_posts = []
    list_of_chains = []
    for i in np.arange(band_idx0, 3):
        mocksim_names = []
        all_temp_posteriors = []
        all_temp_ravel = []
        indiv_sigmas, indiv_medians, indiv_84pcts, indiv_16pcts = [], [], [], []
        all_temp_chains = []
        all_bkg_chains = []
        sim_idx_list = []
        all_nsrc_chains = []
        bkg_means = []
        plt.subplot(1,3, i+1)

        if inject_sz_frac is not None:
            inject_sz_amp = inject_sz_frac*temp_mock_amps[i]
            ref_vals.append(inject_sz_amp)
            print('inject sz amp is ', inject_sz_amp)

        sim_idxs = []
        for j, timestr in enumerate(timestr_list):
  #          print('timestr issssss', timestr)
            gdat, filepath, result_path = load_param_dict(timestr, result_path='spire_results/')
            if j==0:
                print('gdat file name is ', gdat.tail_name, ' and injected sz frac is ', gdat.inject_sz_frac)
            # print('inject sz frac is ', inject_sz_frac, ' while in gdat it is ', gdat.inject_sz_frac)
            #if gdat.tail_name in mocksim_names or '0230' in gdat.tail_name:
            #    print('already have this one, going to next')
            #    continue
            #dont_include_idxs = []
            print(gdat.bands)
            boolvar = 0
            for dont_include_idx in dont_include_idxs:
                if str(dont_include_idx) == gdat.tail_name[-3:]:
                    print('PASS!:', dont_include_idx, gdat.tail_name)
                    boolvar = 1
                    #print(dont_include_idx)
                    #print('pass!')
                    continue

            if gdat.tail_name[-3:] in sim_idx_list and datatype != 'real':
                print('we already have this sim, skipping it', gdat.tail_name[-3:])
                print(sim_idx_list)
                boolvar = 1

            if boolvar==1:
                continue

            print(gdat.tail_name)
            print(gdat.image_extnames)
            
            mocksim_names.append(gdat.tail_name)
            sim_idx_list.append(gdat.tail_name[-3:])    
            chain = np.load(filepath+'/chain.npz')
            #print(chain['template_amplitudes'].shape())
            chi2modl = chain['chi2']
            bkgs = chain['bkg']
            
            bkg_means.append(np.mean(bkgs, axis=0)[i])
            #print('mean_bkg is ', np.mean(bkgs, axis[i])
 #           print('Minimum chi2 is ', np.min(chi2modl, axis=0))
            #print(i)
            #print(i, gdat.bands[i])
            #band=band_dict[gdat.bands[i]]
            band = band_dict[i]
            sim_idxs.append(gdat.tail_name[-3:])
            if j==0:
            #   plt.title(band+', $\\langle \\delta F\\rangle = $'+str(np.median(all_temp)))
                if datatype=='real':
                    label='Indiv. chains'
                else:
                    label='Indiv. mock realizations'

            else:
                label = None

            if burn_in_frac is None:
                burn_in_frac = gdat.burn_in_frac
            burn_in = int(gdat.nsamp*burn_in_frac)
            # burn_in = int(gdat.nsamp*0.75)

            if i==0:
                nsrc_chains = chain['n']
                print(nsrc_chains)
                all_nsrc_chains.append(nsrc_chains)
            
            bkg_chains = chain['bkg']
            
            all_bkg_chains.append(bkg_chains[:,i])

            if dust:
                template_amplitudes = chain['template_amplitudes'][burn_in:, 1, i]
            else:
                template_amplitudes = chain['template_amplitudes'][burn_in:, 0, i]/flux_density_conversion_facs[i]
                full_chain_amps = chain['template_amplitudes'][:,0,i]/flux_density_conversion_facs[i]
                
                if integrate_sz_prof:
                    
                    print('INTEGRATE SZ PROF')
                    t = 0 # use sz template                                                                                                                                                            
                    dat = pcat_data(gdat.auto_resize, nregion=gdat.nregion)
                    dat.load_in_data(gdat)
                    pixel_sizes = dict({'S':6, 'M':8, 'L':12}) # arcseconds                                                                                                                        
                    #print('max of template is ', np.max(dat.template_array[i][t]))
                    npix = dat.template_array[i][t].shape[0]*dat.template_array[i][t].shape[1]
                    geom_fac = (np.pi*pixel_sizes[gdat.band_dict[gdat.bands[i]]]/(180.*3600.))**2
                    #print('geometric factor is ', geom_fac)

                    print('integrating sz profiles..')

                    template_amplitudes = np.array([np.sum(amp*dat.template_array[i][t]) for amp in template_amplitudes])     
                    template_amplitudes *= geom_fac
                    template_amplitudes *= 1e6 # MJy to Jy                                                                                                                                        
                    #print('final template flux densities are ', template_amplitudes)

                    
   #         print('indiv median:', np.median(template_amplitudes))
            indiv_medians.append(np.median(template_amplitudes))
            indiv_sigmas.append(np.std(template_amplitudes))
            indiv_84pcts.append(np.percentile(template_amplitudes, 84))
            indiv_16pcts.append(np.percentile(template_amplitudes, 16))
            all_temp_posteriors.append(template_amplitudes)
            all_temp_ravel.extend(template_amplitudes)
            all_temp_chains.append(full_chain_amps)
        print('length of mocksim_names is ', len(mocksim_names))
        print(mocksim_names)
        print('average background for band ', i, 'is ', np.mean(np.array(bkg_means)), np.std(np.array(bkg_means)))
        print('average sig for band', i, 'is ', np.mean(np.array(indiv_sigmas)))
        all_n, bins, _  = plt.hist(all_temp_ravel, label='Aggregate Posterior', histtype='step', bins=20, color='k')


        for k, t in enumerate(all_temp_posteriors):
            n, _, _ = plt.hist(t, bins=bins, color='black', histtype='stepfilled', linewidth=1.5, alpha=0.15)
            idx = np.argmax(n)
            plt.text(bins[idx], 1.1*n[idx], sim_idxs[k], fontsize=12)
            print('mean/median:', np.mean(t), np.median(t))

        all_temp = np.array(all_temp_ravel)

        print(all_temp.shape, np.median(all_temp), np.std(all_temp), np.percentile(all_temp, 84)-np.median(all_temp), np.median(all_temp)-np.percentile(all_temp, 16))
        
        medians.append(np.median(all_temp_ravel))
        pcts_5.append(np.percentile(all_temp_ravel, 5))
        pcts_16.append(np.percentile(all_temp_ravel, 16))
        pcts_84.append(np.percentile(all_temp_ravel, 84))
        pcts_95.append(np.percentile(all_temp_ravel, 95))

        indiv_medians_list.append(indiv_medians)
        indiv_sigmas_list.append(indiv_sigmas)
        indiv_84pcts_list.append(indiv_84pcts)
        indiv_16pcts_list.append(indiv_16pcts)
        median_str_noinj = str(np.round(np.median(all_temp_ravel), 4))
        str_plus_noinj = str(np.round(np.percentile(all_temp_ravel, 84)-np.median(all_temp_ravel), 4))
        str_minus_noinj = str(np.round(-np.percentile(all_temp_ravel, 16)+np.median(all_temp_ravel), 4))
        if inject_sz_frac is not None:

            
            if dust:
                median_str = str(np.round(np.median(all_temp)-1.,4))

                str_plus = str(np.round(np.percentile(all_temp, 84) - 1., 4))
                str_minus = str(np.round(1. - np.percentile(all_temp, 16), 4))
                unit = ''
            else:
                median_str = str(np.round(np.median(all_temp)-float(inject_sz_amp),4))
                str_plus = str(np.round(np.percentile(all_temp, 84) - np.median(all_temp),4))
                str_minus = str(np.round(np.median(all_temp)-np.percentile(all_temp, 16),4))
                unit = ' MJy/sr'
                
                if integrate_sz_prof:
                    unit = ' Jy'

            if dust:
                plt.title(band+', $\\langle \\delta A_{dust}\\rangle = $'+median_str+'$^{+'+str_plus+'}_{-'+str_minus+'}$'+unit)
            else:
                plt.title(band+', $\\langle \\delta I\\rangle = $'+median_str+'$^{+'+str_plus+'}_{-'+str_minus+'}$'+unit)
        else:
            unit = ' MJy/sr'
            if integrate_sz_prof:
                unit = ' Jy'
                plt.title(band+', $\\langle \\int \\delta I d\\Omega \\rangle = $'+median_str_noinj+'$^{+'+str_plus_noinj+'}_{-'+str_minus_noinj+'}$'+unit)
            else:
                plt.title(band+', $\\langle \\delta I \\rangle = $'+median_str_noinj+'$^{+'+str_plus_noinj+'}_{-'+str_minus_noinj+'}$'+unit)
        #medians.append(np.median(all_temp_ravel))
        #pcts_5.append(np.percentile(all_temp_ravel, 5))
        #pcts_16.append(np.percentile(all_temp_ravel, 16))
        #pcts_84.append(np.percentile(all_temp_ravel, 84))
        #pcts_95.append(np.percentile(all_temp_ravel, 95))
        
        if inject_sz_frac is not None:
            if dust:
                plt.axvline(1., label='Fiducial dust amp.', linestyle='solid', color='k', linewidth=4)
            else:  
                plt.axvline(inject_sz_amp, label='Injected SZ Amplitude', linestyle='solid', color='k', linewidth=4)

        
        plt.axvline(medians[i], label='Median', linestyle='dashed', color='r')

        bin_cents = 0.5*(bins[1:]+bins[:-1])
    
        pm_1sig_fine = np.linspace(pcts_16[i], pcts_84[i], 300)
        all_n_fine_1sig = return_step_func_hist(pm_1sig_fine, bins, all_n)
        pm_2sig_fine = np.linspace(pcts_5[i], pcts_95[i], 300)
        all_n_fine_2sig = return_step_func_hist(pm_2sig_fine, bins, all_n)

        plt.fill_between(pm_1sig_fine, 0, all_n_fine_1sig, interpolate=True, color='royalblue')    
        plt.fill_between(pm_2sig_fine, 0, all_n_fine_2sig, interpolate=True, color='royalblue', alpha=0.4)    

        #plt.xlim(-0.05, 0.2)

        #plt.legend()
        if dust:
            plt.xlabel('Template amplitude')
        else:
            if integrate_sz_prof:
                plt.xlabel('Integrated flux density [Jy]')
            else:
                plt.xlabel('Template amplitude [MJy/sr]')

        plt.ylabel('Number of samples')
        if i==0:
            list_of_chains.append(all_nsrc_chains)
        list_of_posts.append(all_temp_posteriors)
        list_of_chains.append(all_temp_chains)
        list_of_chains.append(all_bkg_chains)
    if save:
        if dust:
            
            plt.savefig('agg_posts/agg_post_dust_'+tail_name+pdf_or_png, bbox_inches='tight')
        else:
            if integrate_sz_prof:
                plt.savefig('agg_posts/agg_post_sz_integrated_'+tail_name+pdf_or_png, bbox_inches='tight')
            else:
                plt.savefig('agg_posts/agg_post_sz_'+tail_name+pdf_or_png, bbox_inches='tight')



    plt.show()


    return f, medians, pcts_16, pcts_84, pcts_5, pcts_95, ref_vals, indiv_medians_list, indiv_sigmas_list, indiv_84pcts_list, indiv_16pcts_list, list_of_posts, list_of_chains





# timestr_list_no_sz = ['20200511-091740', '20200511-090238', '20200511-085314', '20200511-040620', '20200511-040207', '20200511-035415', '20200510-230221', \
# 							'20200510-230200', '20200510-230147', '20200512-170853', '20200512-170912', '20200512-170920', '20200512-231130', \
# 							'20200512-231147', '20200512-231201', '20200513-041608', '20200513-042104', '20200513-043145', '20200513-112022', \
# 							'20200513-112041', '20200513-112030', '20200513-205034', '20200513-205025', '20200513-205018', '20200514-023951',\
# 							'20200514-024001', '20200514-024009', '20200514-145434', '20200514-145442', \
# 							'20200514-205559', '20200514-205546', '20200514-205608', '20200515-022433', '20200515-023105', '20200515-023151', \
# 							'20200515-075700', '20200515-080337', '20200515-080822']

timestr_list_no_sz = ['20200531-224825', '20200531-224833', '20200531-224841', '20200601-025648', '20200601-030306', '20200601-031630', \
						'20200601-071608', '20200601-071708', '20200601-073958', '20200601-232446', '20200601-232509', '20200601-232518', '20200602-033906', \
						'20200602-035510', '20200602-035940', '20200602-080030', '20200602-081620', '20200602-084239']

timestr_list_sz_0p5 = ['20200603-183737', '20200603-182513', '20200603-182023', '20200603-131227', '20200603-131154', '20200603-131052', '20200604-000848', '20200604-001450', '20200603-235206']

# timestr_list_threeband_sz = ['20200607-010226', '20200607-010206', '20200607-010142', '20200605-172052']

timestr_list_threeband_sz2 = ['20200617-120842', '20200617-120909', '20200617-120933']


timestr_list_twoband_sz = ['20200607-130436', '20200607-130411', '20200607-130345', '20200605-172157']

timestrs_nullsz_730 = ['20200728-004231', '20200728-004306', '20200728-004356', '20200728-043328', \
                      '20200728-043350', '20200728-043420', '20200728-082301', '20200728-082327', \
                      '20200728-082341', '20200804-152424']

timestrs_sz_1p0_730 = ['20200729-022132', '20200729-021825', '20200729-021824', '20200728-223152', \
                      '20200728-223137', '20200728-223005', '20200728-184222', '20200728-184131', \
                      '20200728-184042', '20200804-113402']

timestrs_sz_0p5_730 = ['20200729-220548', '20200729-220511', '20200729-220442', '20200730-015523', \
						'20200730-015740', '20200730-015846', '20200730-055527', '20200730-055646', \
						'20200730-055803', '20200804-173357']

timestrs_sz_1p5_730 = ['20200730-134309', '20200730-134211', '20200730-134143', '20200731-014033',\
						 '20200731-013931', '20200731-013605', '20200730-194245', '20200730-194134', \
						 '20200730-193651', '20200804-184129']


timestrs_nulldust = ['20200731-110251', '20200731-110027', '20200731-155433', '20200731-181601', '20200731-182915', '20200801-000748', \
						'20200801-003000', '20200801-061410', '20200801-063612']

timestrs_injectdust = ['20200731-182740', '20200801-002542', '20200801-062844']

timestrs_dust_and_sz = ['20200801-125355', '20200801-130110', '20200801-130130', '20200801-203252',\
						 '20200801-203301', '20200801-203308', '20200802-044741', '20200802-045054', \
						 '20200802-045118', '20200804-111222']

timestrs_dust_and_nullsz = ['20200802-115653', '20200802-115712', '20200802-115726', '20200802-155425', \
							'20200802-155743', '20200802-155850', '20200802-194705', '20200802-234708', \
							 '20200803-080815', '20200803-035127']

timestrs_dust_and_sz_0p5 = ['20200803-010027', '20200803-010051', '20200803-052550', '20200803-052911',\
							 '20200803-094623', '20200803-094323', '20200803-182209', '20200803-143551',\
							  '20200804-020753', '20200803-221317']

timestrs_dust_and_sz_1p5 = ['20200803-183056', '20200803-183003', '20200803-144027', '20200803-144006',\
							 '20200804-021936', '20200804-021820', '20200803-222228', '20200803-222122',\
							  '20200804-111941', '20200804-151002']


timestrs_dust_sz_1p0_delta_cp = ['20200823-034301', '20200823-005729', '20200823-005721', '20200822-235450', '20200822-210855', '20200822-210827', \
									'20200822-201010', '20200822-173252', '20200822-173142', '20200822-164134']
# fs, _, _, _, _, _, _ = gather_posteriors(timestrs_nulldust, save=True, tail_name='731')




# save_figs = True

# fs, medians_nszd, pcts_16_nszd, pcts_84_nszd, pcts_5_nszd, pcts_95_nszd, ref_vals_nszd = gather_posteriors(timestrs_dust_sz_1p0_delta_cp, save=save_figs, tail_name='inject_dust_sz_delta_cp_dust', inject_sz_frac=1.0)


# fs, medians_nszd, pcts_16_nszd, pcts_84_nszd, pcts_5_nszd, pcts_95_nszd, ref_vals_nszd = gather_posteriors(timestrs_dust_and_nullsz, save=save_figs, tail_name='inject_dust_nullsz', inject_sz_frac=0.0)
# fs, medians_szd, pcts_16_szd, pcts_84_szd, pcts_5_szd, pcts_95_szd, ref_vals_szd = gather_posteriors(timestrs_dust_and_sz, save=save_figs, tail_name='inject_dust_sz1p0', inject_sz_frac=1.0)
# fs, medians_szd0p5, pcts_16_szd0p5, pcts_84_szd0p5, pcts_5_szd0p5, pcts_95_szd0p5, ref_vals_szd0p5 = gather_posteriors(timestrs_dust_and_sz_0p5, save=save_figs, tail_name='inject_dust_sz0p5', inject_sz_frac=0.5)
# fs, medians_szd1p5, pcts_16_szd1p5, pcts_84_szd1p5, pcts_5_szd1p5, pcts_95_szd1p5, ref_vals_szd1p5 = gather_posteriors(timestrs_dust_and_sz_1p5, save=save_figs, tail_name='inject_dust_sz1p5', inject_sz_frac=1.5)


# fs, medians_1p0, pcts_16_1p0, pcts_84_1p0, pcts_5_1p0, pcts_95_1p0, ref_vals_1p0  = gather_posteriors(timestrs_sz_1p0_730, inject_sz_frac=1.0, tail_name='inject_sz1p0', save=save_figs)
# fs, medians_0p0, pcts_16_0p0, pcts_84_0p0, pcts_5_0p0, pcts_95_0p0, ref_vals_0p0  = gather_posteriors(timestrs_nullsz_730, inject_sz_frac=0.0, tail_name='nullsz', save=save_figs)
# fs, medians_0p5, pcts_16_0p5, pcts_84_0p5, pcts_5_0p5, pcts_95_0p5, ref_vals_0p5  = gather_posteriors(timestrs_sz_0p5_730, inject_sz_frac=0.5, tail_name='inject_sz0p5', save=save_figs)
# fs, medians_1p5, pcts_16_1p5, pcts_84_1p5, pcts_5_1p5, pcts_95_1p5, ref_vals_1p5  = gather_posteriors(timestrs_sz_1p5_730, inject_sz_frac=1.5, tail_name='inject_sz1p5', save=save_figs)

# band_dict = dict({0:'250 micron', 1:'350 micron', 2:'500 micron'})



# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey='row',
#                          figsize=(9, 6))
# # plt.suptitle('whooop')
# plt.suptitle('Fiducial $\\Delta I$: S:0.0111, M: 0.1249, L: 0.691 [MJy/sr]', fontsize=16, y=1.03)
# ax1.set_title('CIB + SZ', fontsize=16)
# for i in range(3):
# 	m0 = medians_0p0[i]
# 	m1 = medians_1p0[i]
# 	m0p5 = medians_0p5[i]
# 	m1p5 = medians_1p5[i]


# 	ref0 = ref_vals_0p0[i]
# 	ref1 = ref_vals_1p0[i]
# 	ref0p5 = ref_vals_0p5[i]
# 	ref1p5 = ref_vals_1p5[i]


# 	ax1.errorbar([0.0+0.02*(i-1),0.5+0.02*(i-1), 1.0+0.02*(i-1), 1.5+0.02*(i-1)], [m0 - ref0, m0p5-ref0p5, m1 - ref1, m1p5-ref1p5], yerr=[[m0 - pcts_16_0p0[i], m0p5-pcts_16_0p5[i], m1 - pcts_16_1p0[i], m1p5 - pcts_16_1p5[i]], [pcts_84_0p0[i]- m0, pcts_84_0p5[i]-m0p5, pcts_84_1p0[i] - m1, pcts_84_1p5[i] - m1p5]], capsize=5, marker='.', label=band_dict[i])


# 	# plt.errorbar([0.0+0.02*(i-1), 1.0+0.02*(i-1)], [m0 - ref0, m1 - ref1], yerr=[[m0 - pcts_16_0p0[i], m1 - pcts_16_1p0[i]], [pcts_84_0p0[i]- m0, pcts_84_1p0[i] - m1]], capsize=5, marker='.', label=band_dict[i])
# 	# plt.errorbar([ref_vals_0p0[i], ref_vals_1p0[i]], [medians_0p0[i], medians_1p0[i]], yerr=[[pcts_16_0p0[i], pcts_16_1p0[i]], [pcts_84_0p0[i], pcts_84_1p0[i]]], label=band_dict[i])
# ax1.legend(loc=0)
# ax1.set_xlabel('Injected SZ Fraction', fontsize=16)
# ax1.set_ylabel('Recovered - Injected SZ [MJy/sr]', fontsize=16)
# ax1.set_ylim(-0.4, 0.05)
# ax1.set_xlim(-0.1, 1.6)

# ax2.set_title('CIB + SZ + Dust', fontsize=16)

# ax2.set_xlim(-0.1, 1.6)

# for i in range(3):
# 	mnszd = medians_nszd[i]
# 	mszd = medians_szd[i]
# 	mszd0p5 = medians_szd0p5[i]
# 	mszd1p5 = medians_szd1p5[i]
# 	refnszd = ref_vals_nszd[i]
# 	refszd = ref_vals_szd[i]
# 	refszd0p5 = ref_vals_szd0p5[i]
# 	refszd1p5 = ref_vals_szd1p5[i]


# 	ax2.errorbar([0.0+0.02*(i-1), 0.5+0.02*(i-1), 1.0+0.02*(i-1), 1.5+0.02*(i-1)], [mnszd - refnszd, mszd0p5-refszd0p5, mszd - refszd, mszd1p5 - refszd1p5], yerr=[[mnszd - pcts_16_nszd[i], mszd0p5 - pcts_16_szd0p5[i],  mszd - pcts_16_szd[i], mszd1p5 - pcts_16_szd1p5[i] ], [pcts_84_nszd[i]- mnszd, pcts_84_szd0p5[i]-mszd0p5,  pcts_84_szd[i] - mszd, pcts_84_szd1p5[i] - mszd1p5]], capsize=5, marker='.', label=band_dict[i])

# ax2.set_xlabel('Injected SZ Fraction', fontsize=16)
# # ax1.set_ylim(-0.4, 0.1)
# ax2.legend(loc=0)


# plt.tight_layout()
# plt.savefig('input_recover_sz_804_v3.png', bbox_inches='tight', dpi=150)
# plt.close()
# # fs = gather_posteriors(timestr_list_sz_0p5, inject_sz_frac=0.5)

# # fs = gather_posteriors(timestr_list_twoband_sz, tail_name='two_band_sz')


#timestr_realdat_file = 'rxj1347_realdat_4arcmin_nfcterms=3_timestrs_041621_bkg_vals3.npz'
timestr_realdat_file = 'rxj1347_realdat_10arcmin_nfcterms=6_timestrs_042021_fitbkg.npz'
#timestr_bkg_file = 'rxj1347_conley_10arcmin_041921_timestrs_fitbkg.npz'

#timestr_bkg_file = 'rxj1347_conley_4arcmin_041921_timestrs_bestfitbkgs.npz'
#timestr_bkg_file = 'rxj1347_conley_4arcmin_unlensed_042021_timestrs_bestfitbkgs.npz'
#timestr_sim_file = 'lensed_1xdust_nfcterms=6_simidx300_30chains_conley_rxj1347_timestrs_3_23_21_narrow_cprior_Fmin=5.npz'
#timestr_sim_file = 'group_timestrs/unlensed_nodust_conley_rxj1347_timestrs_withnoise_3_2_21_narrow_cprior_Fmin=5.npz'
#timestr_sim_file = 'unlensed_1xdust_nfcterms=3_conley_rxj1347_4arcmin_simidx300_withnoise_timestrs_041021.npz'
#timestr_sim_file = 'lensed_1xdust_nfcterms=6_conley_rxj1347_timestrs_withnoise_3_31_21_smallmask_narrow_cprior_Fmin=5.npz'
#truealpha = 2.5
#timestr_dust_file = 'lensed_1xdust_nfcterms=6_truealpha='+str(truealpha)+'_conley_rxj1347_timestrs_3_17_21_narrow_cprior_Fmin=5.npz'
#timestr_dust_file = 'lensed_16xdust_nfcterms='+str(n_fc_terms)+'_conley_rxj1347_timestrs_withnoise_3_22_21_narrow_cprior_Fmin=5.npz'

#fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _, all_temp_posteriors, full_chains = gather_posteriors(timestr_list_file=timestr_sim_file, save=save_figs, tail_name='rxj1347_sims_nfcterms='+str(n_fc_terms)+'_unlensed_4arcmin_041021_smallmask_narrow_cprior_Fmin=5', integrate_sz_prof=False, dust=False, inject_sz_frac=1.0, datatype='real', burn_in_frac=0.5)

#out_tail = '4arcmin_041021_fixbkg_errdivfac=15'
#out_tail = '4arcmin_041021_fixbkg'
#out_tail = '4arcmin_041221_fixbkg_nreg2_longo'

#out_tail = '4arcmin_041921_bestfitbkg_unlensed'
# out_tail = '10arcmin_042021_fitbkg_realdat'
# dirpath = out_tail
# if not os.path.isdir(dirpath):
    # os.makedirs(dirpath)

# pcagg = pcat_agg()
# all_accept_stats = pcagg.compile_stats(mode='accept', timestr_list_file=timestr_realdat_file)
# np.savez(dirpath+'/all_acceptance_stats_'+out_tail+'.npz', all_accept_stats = all_accept_stats, timestr_list_file=timestr_realdat_file)
# all_chi2_stats = pcagg.compile_stats(mode='chi2', timestr_list_file=timestr_realdat_file)
# np.savez(dirpath+'/all_chi2_stats_'+out_tail+'.npz', all_chi2_stats = all_chi2_stats, timestr_list_file=timestr_realdat_file)

# np.savez(dirpath+'/all_stats_'+out_tail+'.npz', all_chi2_stats = all_chi2_stats, all_accept_stats = all_accept_stats, timetsr_list_file=timestr_realdat_file)

# fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _, all_temp_posteriors, full_chains = gather_posteriors(timestr_list_file=timestr_realdat_file, save=save_figs, tail_name=out_tail, integrate_sz_prof=False, dust=False, inject_sz_frac=None, datatype='real', burn_in_frac=0.7)

# ------------ save chains for convergence diagnostic analysis ----------------
#np.savez('list_of_sz_chains_realdat_031621_take1_with_nsrc.npz', all_temp_posterior_sz=all_temp_posteriors, fmin=0.005, n_fc_terms=6, cprior_widths=0.5, nsrc_chains=all_temp_posteriors[-1])
#np.savez('list_of_sz_chains_smallmask_full_realdat_033021_with_nsrc.npz', all_temp_posterior_sz=[full_chains[0], full_chains[1]], sz_band_order=['PMW', 'PLW'], fmin=0.005, n_fc_terms=6, cprior_widths=0.5, nsrc_chains=full_chains[2])
# np.savez(dirpath+'/list_of_chains_full_'+out_tail+'.npz', chains = full_chains, fmin=0.005, n_fc_terms=6, cprior_widths=0.5)
#np.savez('list_of_chains_full_unlensed_smallmask_041021_simidx300.npz', chains=full_chains, fmin=0.005, n_fc_terms=6, cprior_widths=0.5)

#fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _, all_temp_posteriors, full_chains = gather_posteriors(timestr_list_file=timestr_dust_file, save=save_figs, tail_name='lensed_16xdust_nfcterms='+str(n_fc_terms)+'_conley_rxj1347_timestrs_withnoise_3_22_21_narrow_cprior_Fmin=5', integrate_sz_prof=False, dust=False, inject_sz_frac=1.0, datatype='mock')


#fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _, all_temp_posteriors = gather_posteriors(timestr_list_file=timestr_dust_file, save=save_figs, tail_name=timestr_dust_file[:-4], integrate_sz_prof=False, dust=False, inject_sz_frac=1.0, datatype='mock') 

#np.savez('list_of_sz_chains_realdat_nfcterms=6_031621_take2.npz', all_temp_posterior_sz=all_temp_posteriors, fmin=0.005, n_fc_terms=6, cprior_widths=0.5)

#fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _ = gather_posteriors(timestr_list_file="lensed_nodust_conley_rxj1347_timestrs_withnoise_3_2_21_narrow_cprior_Fmin=2p5.npz", save=save_figs, tail_name='lensed_nodust_conley_rxj1347_timestrs_withnoise_3_2_21_narrow_cprior_Fmin=2p5', integrate_sz_prof=False, dust=False, inject_sz_frac=1.0)

#fs, medians_fctest, pcts_16_fctest, pcts_84_fctest, pcts_5_fctest, pcts_95_fctest, ref_vals_fctest, _, _,_, _ = gather_posteriors(timestr_list_file="lensed_nodust_conley_rxj1347_timestrs_withnoise_3_2_21_narrow_cprior_Fmin=10.npz", save=save_figs, tail_name='lensed_nodust_conley_rxj1347_timestrs_withnoise_3_2_21_narrow_cprior_Fmin=10', integrate_sz_prof=False, dust=False, inject_sz_frac=1.0)







def aggregate_posterior_corner_plot(timestr_list, temp_amplitudes=True, bkgs=True, nsrcs=True, \
							bkg_bands=[0, 1, 2], temp_bands=[0, 1, 2], n_burn_in=1500, plot_contours=True, tail_name='', pdf_or_png='.pdf', nbands=3, save=True):

	flux_density_conversion_facs = dict({0:86.29e-4, 1:16.65e-3, 2:34.52e-3})
	bkg_labels = dict({0:'$B_{250}$', 1:'$B_{350}$', 2:'$B_{500}$'})
	temp_labels = dict({0:'$A_{250}^{SZ}$', 1:'$A_{350}^{SZ}$', 2:'$A_{500}^{SZ}$'})


	nsrc = []
	bkgs = [[] for x in range(nbands)]
	temp_amps = [[] for x in range(nbands)]
	samples = []

	for i, timestr in enumerate(timestr_list):
		chain = np.load('spire_results/'+timestr+'/chain.npz')

		if nsrcs:
			nsrc.extend(chain['n'][n_burn_in:])
		
		for b in bkg_bands:
			if bkgs:
				bkgs[b].extend(chain['bkg'][n_burn_in:,b].ravel()/flux_density_conversion_facs[b])
		
		for b in temp_bands:
			if temp_amplitudes:
				temp_amps[b].extend(chain['template_amplitudes'][n_burn_in:,b].ravel()/flux_density_conversion_facs[b])


	corner_labels = []

	agg_name = ''

	if bkgs:
		agg_name += 'bkg_'
		for b in bkg_bands:
			samples.append(np.array(bkgs[b]))
			corner_labels.append(bkg_labels[b])
	if temp_amplitudes:
		agg_name += 'temp_amp_'
		for b in temp_bands:
			samples.append(np.array(temp_amps[b]))
			corner_labels.append(temp_labels[b])

	if nsrcs:
		agg_name += 'nsrc_'
		samples.append(np.array(nsrc))
		corner_labels.append("N_{src}")


	samples = np.array(samples).transpose()

	print('samples has shape', samples.shape)


	figure = corner.corner(samples, labels=corner_labels, quantiles=[0.05, 0.16, 0.5, 0.84, 0.95],plot_contours=plot_contours,
						   show_titles=True,title_fmt='.3f', title_kwargs={"fontsize": 14}, label_kwargs={"fontsize":16})



	if save:
		figure.savefig('corner_plot_'+agg_name+tail_name+pdf_or_png, bbox_inches='tight')

	return figure

def compute_gelman_rubin_diagnostic(list_of_chains):
    
    m = len(list_of_chains)
    n = len(list_of_chains[0])
    
    print('n=',n,' m=',m)
        
    B = (n/(m-1))*np.sum((np.mean(list_of_chains, axis=1)-np.mean(list_of_chains))**2)
    
    W = 0.
    for j in range(m):
        
        sumsq = np.sum((list_of_chains[j]-np.mean(list_of_chains[j]))**2)
        
        W += (1./m)*(1./(n-1.))*sumsq
    
    var_th = ((n-1.)/n)*W + (B/n)
    
    
    Rhat = np.sqrt(var_th/W)
    
    print("rhat = ", Rhat)
    
    return Rhat
    
def compute_chain_rhats(all_chains, labels=['']):
    
    rhats = []
    for chains in all_chains:
        rhat = compute_gelman_rubin_diagnostic(chains)
        
        rhats.append(rhat)
        
    
    f = plt.figure()
    plt.title('Gelman Rubin statistic $\\hat{R}$', fontsize=16)
    plt.bar(labels, rhats, width=0.5, alpha=0.4)
    plt.axhline(1.2, linestyle='dashed', label='$\\hat{R}$=1.2')
    plt.axhline(1.1, linestyle='dashed', label='$\\hat{R}$=1.1')

    plt.legend()
    plt.xticks(fontsize=16)
    plt.ylabel('$\\hat{R}$', fontsize=16)
    plt.show()
    
    return f, rhats



def gather_posteriors_different_amps(timestr_list, label_dict):

	all_temp_posteriors = []
	figs = []


	band_dict = dict({0:'250 micron', 1:'350 micron', 2:'500 micron'})


	temp_mock_amps = [None, 0.3, 0.5] # MJy/sr
	flux_density_conversion_facs = [86.29e-4, 16.65e-3, 34.52e-3]

	ratios = [0.0, 0.5, 1.0, 1.5]

	temp_mins = [-0.003, -0.012]
	temp_maxs = [0.011, 0.027]


	for i in range(2):

		f = plt.figure(figsize=(12, 4))


		for j, timestr in enumerate(timestr_list):
			gdat, filepath, result_path = load_param_dict(timestr, result_path='spire_results/')

			chain = np.load(filepath+'/chain.npz')

			band=band_dict[gdat.bands[i+1]]

			if j==0:
				linelabel = 'Recovered SZ'
				plt.title('Injected and Recovered SZ amplitudes (RX J1347.5-1145, '+band+')', fontsize=18)
			else:
				linelabel = None
			# if j==0:
			# 	plt.title(band)
			# 	label='Indiv. mock realizations'
			# else:
			# 	label = None

			colors = ['C0', 'C1', 'C2', 'C3']

			burn_in = int(gdat.nsamp*gdat.burn_in_frac)

			template_amplitudes = chain['template_amplitudes'][burn_in:, i+1, 0]

			bins = np.linspace(temp_mins[i], temp_maxs[i], 50)

			plt.hist(template_amplitudes, histtype='stepfilled',alpha=0.7,edgecolor='k', bins=bins,label=linelabel,  color=colors[j], linewidth=2)

			plt.axvline(ratios[j]*temp_mock_amps[i+1]*flux_density_conversion_facs[i+1], color=colors[j], label=label_dict[i+1][j], linestyle='dashed', linewidth=3)

			# all_temp_posteriors.extend(template_amplitudes)


		# plt.hist(all_temp_posteriors, label='Aggregate Posterior', histtype='step', bins=30)

		# plt.axvline(0., label='Injected SZ Amplitude', linestyle='dashed', color='g')

		plt.legend()
		plt.xlabel('Template amplitude [mJy/beam]', fontsize=14)
		plt.ylabel('Number of posterior samples', fontsize=14)
		plt.savefig('recover_injected_sz_template_amps_band_'+str(i+1)+'.pdf', bbox_inches='tight')

		# plt.savefig('aggregate_posterior_sz_template_no_injected_signal_band_'+str(i+1)+'.pdf', bbox_inches='tight')
		plt.show()

		figs.append(f)

	return figs


# figs = aggregate_posterior_corner_plot(timestr_list_threeband_sz2, tail_name='testing', temp_bands=[0, 1, 2], nsrcs=False)

# 
# amplitudes = dict({1:[]})
# labels = dict({1:['No SZ signal', '0.15 MJy/sr', '0.3 MJy/sr', '0.45 MJy/sr'], 2:['No signal', '0.25 MJy/sr', '0.5 MJy/sr', '0.75 MJy/sr']})


# timestrs = ['20200510-230147', '20200512-101717', '20200512-101738', '20200512-103101']

# fs = gather_posteriors_different_amps(timestrs, labels)





