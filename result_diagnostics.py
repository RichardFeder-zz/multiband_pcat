import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
# from astropy.visualization import (MinMaxInterval, SqrtStretch, ImageNormalize)
from astropy.io import fits
from helpers import *


sizefac = 10.*136
color_bins = np.linspace(-3, 3, 40)
dpi_val = 300
burn_in_frac = 0.3

mag_bins = np.linspace(15, 23, 15)
hubble_dpos = np.array([[0.1528, 0.797],[0.1045,0.6760]])


def results(nchain, fchain, truef, color, nsamp, timestats, tq_times,plt_times, chi2, bkgsample, accept_stats, result_directory, nbands, bands, multiband, nmgy_per_count, datatype):
    # plt.rc('text', usetex=True) #use for latex quality characters and such

    if datatype == 'mock':
        label = 'Mock Truth'
    else:
        label = datatype
    burn_in = int(nsamp*burn_in_frac)

    # ------------------- SOURCE NUMBER ---------------------------

    #plt.figure(1)
    plt.title('Posterior Source Number Histogram')
    plt.hist(nchain[burn_in:], histtype='step', label='Posterior', color='b')
    plt.axvline(np.median(nchain[burn_in:]), label='Median=' + str(np.median(nchain[burn_in:])), color='b', linestyle='dashed')
    plt.xlabel('nstar')
    plt.legend()
    plt.savefig(result_directory +'/posterior_histogram_nstar.pdf', bbox_inches='tight')

    # -------------------- CHI2 ------------------------------------
    
    sample_number = list(xrange(nsamp-burn_in))
    full_sample = xrange(nsamp)
   # plt.figure(2)
    plt.title('Chi-Squared Distribution over Samples')
    for b in xrange(nbands):
        plt.plot(sample_number, chi2[burn_in:,b], label=bands[b])
        plt.axhline(np.min(chi2[burn_in:,b]), linestyle='dashed', alpha=0.5, label=str(np.min(chi2[burn_in:,b]))+' (' + str(bands[b]) + ')')
    plt.xlabel('Sample')
    plt.ylabel('Chi2')
    plt.yscale('log')
    plt.legend()
    plt.savefig(result_directory + '/chi2_sample.pdf', bbox_inches='tight')

    # ---------------------------- COMPUTATIONAL RESOURCES --------------------------------

    time_array = np.zeros(4, dtype=np.float32)
    labels = ['Proposal', 'Likelihood', 'Implement', 'asTrans']
    for samp in xrange(nsamp):
        time_array += np.array([timestats[samp][2][0], timestats[samp][3][0], timestats[samp][4][0], tq_times[samp]])
   # plt.figure(3)
    plt.title('Computational Resources')
    plt.pie(time_array, labels=labels, autopct='%1.1f%%', shadow=True)
    plt.savefig(result_directory+ '/time_resource_statistics.pdf', bbox_inches='tight')

   # plt.figure(4)
    plt.title('Time Histogram for asTrans Transformations')
    plt.hist(tq_times[burn_in:], bins=20, histtype='step')
    plt.xlabel('Time (s)')
    plt.axvline(np.median(tq_times), linestyle='dashed', color='r', label='Median: ' + str(np.median(tq_times)))
    plt.legend()
    plt.savefig(result_directory+ '/asTrans_time_resources.pdf', bbox_inches='tight')

   # plt.figure(5)
    plt.title('Time Histogram for Plotting')
    plt.hist(plt_times, bins=20, histtype='step')
    plt.xlabel('Time (s)')
    plt.axvline(np.median(plt_times), linestyle='dashed', color='r', label='Median: ' + str(np.median(plt_times)))
    plt.legend()
    plt.savefig(result_directory + '/plot_time_resources.pdf', bbox_inches='tight')

    # ------------------------------ ACCEPTANCE FRACTION -----------------------------------------

    proposals = ['All', 'Move', 'Birth/Death', 'Merge/Split', 'Background Shift']
   # plt.figure(6)
    plt.title('Proposal Acceptance Fractions')
    for x in xrange(len(accept_stats[0])):
        if not np.isnan(accept_stats[0,x]):
            plt.plot(full_sample, accept_stats[:,x], label=proposals[x])
    plt.legend()
    plt.xlabel('Sample Number')
    plt.ylabel('Acceptance Fraction')
    plt.savefig(result_directory+'/acceptance_fraction.pdf', bbox_inches='tight')


    for b in xrange(nbands):
# ------------------ BACKGROUND SAMPLES --------------------------------
        if np.median(bkgsample[:,b]) != bkgsample[0,b]: #don't plot this if we're not sampling the background
    #        plt.figure()
            plt.title('Background Distribution over Samples')
            plt.hist(bkgsample[burn_in:,b], label=bands[b], histtype='step')
            plt.axvline(np.median(bkgsample[burn_in:,b]), label='Median in ' + str(bands[b])+'-band: ' + str(np.median(bkgsample[burn_in:,b])), linestyle='dashed', color='b')
            # plt.axvline(actual_background[b], label='True Bkg ' + str(bands[b])+'-band: ' + str(actual_background[b]), color='g', linestyle='dashed')
            plt.xlabel('Background (ADU)')
            plt.ylabel('$n_{samp}$')
            plt.legend()
            plt.savefig(result_directory + '/bkg_sample_' + str(bands[b]) + '.pdf')

# ------------------------------ MAGNITUDE DISTRIBUTION ---------------------------------------

     #   plt.figure(7)
        plt.title('Posterior Magnitude Distribution - ' + str(bands[b]))
        true_mags = adu_to_magnitude(truef[:,b], nmgy_per_count[b])
        (n, bins, patches) = plt.hist(true_mags, histtype='step', label=label, color='g')
        post_hist = []
        for samp in xrange(burn_in, nsamp):
            hist = np.histogram([adu_to_magnitude(x, nmgy_per_count[b]) for x in fchain[b][samp] if x>0], bins=bins)
            post_hist.append(hist[0]+0.01)
        medians = np.median(np.array(post_hist), axis=0)
        stds = np.std(np.array(post_hist), axis=0)
        bincentres = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
        plt.errorbar(bincentres, medians, yerr=stds, fmt='o', label='Posterior')
        plt.legend()
        plt.yscale('log', nonposy='clip')
        plt.ylim(0.05, 1000)
        plt.xlabel('Magnitude - ' + str(bands[b]))
        plt.savefig(result_directory+ '/posterior_flux_histogram_' + str(bands[b]) + '.pdf', bbox_inches='tight')


# ----------------------------------------- COLOR PLOTS --------------------------------------------
    color_post_bins = np.linspace(-1.5, 1.5, 30)
    for b in xrange(nbands-1):
        plt.figure(7)
        plt.title('Posterior Color Distribution (Normalized to 1)')
        nmpc = [nmgy_per_count[0], nmgy_per_count[b+1]]
        plt.hist(adus_to_color(truef[:,0], truef[:,b+1], nmpc), histtype='step',bottom=0.1, bins=color_post_bins, label=label, color='C1', linewidth=4, normed=1, alpha=0.5)
        post_hist = []
        bright_hist = []
        for samp in xrange(burn_in, nsamp):
            mask = fchain[0][samp] > 0
            n_bright = min(300, int(len(fchain[0][samp][mask])/3))
            brightest_idx = np.argpartition(fchain[0][samp][mask], n_bright)[-n_bright:]
            bright_h = np.histogram(adus_to_color(fchain[0][samp][mask][brightest_idx], fchain[b+1][samp][mask][brightest_idx], nmpc), bins=color_post_bins)
            bright_hist.append(bright_h[0]+0.01)
            hist = np.histogram([x for x in color[b][samp]], bins=color_post_bins)
            post_hist.append(hist[0]+0.01)
        medians = np.median(np.array(post_hist), axis=0)
        medians /= (np.sum(medians)*(color_post_bins[1]-color_post_bins[0]))  
        medians_bright = np.median(np.array(bright_hist), axis=0)
        medians_bright /= (np.sum(medians_bright)*(color_post_bins[1]-color_post_bins[0]))  
        bincentres = [(color_post_bins[i]+color_post_bins[i+1])/2. for i in range(len(color_post_bins)-1)]
        plt.step(bincentres, medians, where='mid', color='b', label='Posterior', alpha=0.5)
        plt.step(bincentres, medians_bright, where='mid', color='k', label='Brightest Third', alpha=0.5)
        plt.legend()
        plt.xlabel(str(bands[0]) + ' - ' + str(bands[b+1]))
        plt.legend()
        plt.savefig(result_directory+'/posterior_histogram_'+str(bands[0])+'_'+str(bands[b+1])+'_color.pdf', bbox_inches='tight')
        plt.close()

# ---------------------------------------- COLOR COLOR DISTRIBUTION ----------------------------

    #posterior color-color marginalized plot
    # if nbands==3:
    #     plt.figure(figsize=(5,11))
    #     plt.title('Posterior Color-Color Histogram')
    #     color_post_bins = list(color_post_bins)
    #     post_2dhist = []
    #     bright_2dhist = []
    #     for samp in xrange(burn_in, nsamp):
    #         samp_colors = []
    #         n_bright = int(len(fchain[0][samp][mask])/3)
    #         brightest_idx = np.argpartition(fchain[0][samp][mask], n_bright)[-n_bright:]
    #         for b in xrange(nbands-1):
    #             nmpc = [nmgy_per_count[b], nmgy_per_count[b+1]]
    #             samp_colors.append(adus_to_color(fchain[b][samp][mask][brightest_idx], fchain[b+1][samp][mask][brightest_idx], nmpc))
    #         bright_2dh = np.histogram2d(samp_colors[1], samp_colors[0], bins=(color_post_bins, color_post_bins))
    #         bright_2dhist.append(bright_2dh[0])
    #         dhist = np.histogram2d([x for x in color[1][samp]], [x for x in color[0][samp]], bins=(color_post_bins, color_post_bins))
    #         post_2dhist.append(dhist[0])
    #     d_medians = np.median(np.array(post_2dhist), axis=0)
    #     d_bright_medians = np.median(np.array(bright_2dhist), axis=0)


    #     plt.subplot(2,1,1)
    #     plt.title('All Sources', fontsize=10)
    #     norm = ImageNormalize(d_medians, interval=MinMaxInterval(),stretch=SqrtStretch())
    #     plt.imshow(d_medians, interpolation='none',norm=norm, origin='low', extent=[color_post_bins[0], color_post_bins[-1], color_post_bins[0], color_post_bins[-1]])
    #     plt.colorbar()
    #     plt.ylabel(bands[0]+' - '+bands[1])
    #     # plt.subplot(3,1,2)
    #     # plt.title('Brightest Third', fontsize=10)
    #     # norm = ImageNormalize(d_bright_medians, interval=MinMaxInterval(),stretch=SqrtStretch())
    #     # plt.imshow(d_bright_medians, interpolation='none', norm=norm, origin='low', extent=[color_post_bins[0], color_post_bins[-1], color_post_bins[0], color_post_bins[-1]])
    #     # plt.colorbar()
    #     # plt.ylabel(bands[0]+' - '+bands[1])
    #     r_i_color = adus_to_color(truef[:,0], truef[:,1], [nmgy_per_count[0], nmgy_per_count[1]])
    #     g_r_color = adu_to_magnitude(truef[:,2], nmgy_per_count[2]) - adu_to_magnitude(truef[:,0], nmgy_per_count[0])
    #     plt.subplot(2,1,2)
    #     plt.title(label, fontsize=10)
    #     true_colorcolor = np.histogram2d(g_r_color, r_i_color, bins=(color_post_bins, color_post_bins))
    #     norm = ImageNormalize(true_colorcolor[0], interval=MinMaxInterval(),stretch=SqrtStretch())
    #     plt.imshow(true_colorcolor[0], interpolation='none', origin='low', norm=norm, extent=[color_post_bins[0], color_post_bins[-1], color_post_bins[0], color_post_bins[-1]])
    #     plt.xlabel(bands[1]+' - '+bands[2])
    #     plt.ylabel(bands[0]+' - '+bands[1])
    #     plt.colorbar()
    #     plt.tight_layout()
    #     plt.savefig(result_directory+'/posterior_color_color_histogram.png', bbox_inches='tight') 


def multiband_sample_frame(data_array, x, y, f, ref_x, ref_y, ref_f, truecolor, h_coords, hf, resids, weights, bands, nmgy_per_count, nstar, frame_dir, c, pixel_transfer_mats, mean_dpos, visual=0, savefig=0, include_hubble=0, datatype='mock'):
    # plt.rc('text', usetex=True) #use for latex quality characters and such
    
    if datatype=='mock':
        labldata = 'Mock Truth'
    else:
        labldata = datatype


    sizefac = 10.*136
    
    # if datatype != 'mock':
    if include_hubble:
        posmask = np.logical_and(h_coords[0]<99.5, h_coords[1]<99.5)
        hf = hf[posmask]
        hmask = hf < 22

    first_frame_band_no = 1
    first_resid_no = 0
    second_resid_no = 0
    first_mag_no = 0
    second_mag_no = 1
    color_no_1 = 0
    color_no_2 = 1

    if len(bands)==1:

        # hf = hf[posmask]
        # hmask = hf < 22

        plt.gcf().clear()
        plt.figure()
        plt.imshow(data_array[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data_array[0]), vmax=np.percentile(data_array[0], 95))    
        mask = ref_f[:,0] > 25
        # if datatype != 'mock':
        if include_hubble:
            plt.scatter(h_coords[0][posmask][hmask], h_coords[1][posmask][hmask], marker='+', s=3*2*mag_to_cts(hf[hmask], nmgy_per_count[0]) / sizefac, color='lime') #hubble
        plt.scatter(x, y, marker='x', s=(10000/15**2)*f[0]/(2*sizefac), color='r')
        plt.scatter(ref_x[mask], ref_y[mask]-1, marker='+', s=(10000/15**2)*ref_f[mask,0]/(2*sizefac), color='g') #daophot
        mask = np.logical_not(mask)
        plt.scatter(ref_x[mask], ref_y[mask]-1, marker='+', s=(10000/15**2)*ref_f[mask,0]/(2*sizefac), color='g')
        plt.xlim(50, 70)
        plt.ylim(70, 90)
        if savefig:
            plt.savefig(frame_dir + '/single_frame_' + str(c) + '.png', bbox_inches='tight')

        
        plt.gcf().clear()
        plt.figure(figsize=(15,5))
        plt.clf()
        plt.subplot(1,3,1)
        plt.imshow(data_array[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data_array[0]), vmax=np.percentile(data_array[0], 95))
        plt.colorbar()
        sizefac = 10.*136
        if include_hubble:
            plt.scatter(h_coords[0][posmask][hmask], h_coords[1][posmask][hmask], marker='+', s=2*mag_to_cts(hf[hmask], nmgy_per_count[0]) / sizefac, color='lime') #hubble
        
        # mask = ref_f[:,0] > 250 # will have to change this for other data sets
        # plt.scatter(ref_x[mask], ref_y[mask], marker='+', s=ref_f[mask] / sizefac, color='lime')
        # mask = np.logical_not(mask)
        # plt.scatter(ref_x[mask], ref_y[mask], marker='+', s=ref_f[mask] / sizefac, color='g')

        plt.scatter(x, y, marker='x', s=(10000/len(data_array[0])**2)*f[0]/(2*sizefac), color='r')
        plt.xlim(-0.5, len(data_array[0])-0.5)
        plt.ylim(-0.5, len(data_array[0])-0.5)

        plt.subplot(1,3,2)
        plt.imshow(resids[0]*np.sqrt(weights[0]), origin='lower', interpolation='none', cmap='bwr', vmin=-5, vmax=5)
        plt.xlim(-0.5, len(data_array[0])-0.5)
        plt.ylim(-0.5, len(data_array[0])-0.5)
        plt.colorbar()
        plt.subplot(1,3,3)
        if datatype == 'mock':
            (n, bins, patches) = plt.hist(adu_to_magnitude(f[0], nmgy_per_count[0]), bins=mag_bins, alpha=0.5, color='r', label='Chain - ' + bands[0], histtype='step')
            plt.hist(adu_to_magnitude(ref_f[:,0], nmgy_per_count[0]), bins=bins,alpha=0.5, label=labldata, color='g', histtype='step')
        else:
            (n, bins, patches) = plt.hist(adu_to_magnitude(f[0], nmgy_per_count[0]), bins=mag_bins, alpha=0.5, color='r', label='Chain - ' + bands[0], histtype='step')
        plt.legend()
        plt.xlabel(str(bands[0]))
        plt.yscale('log')

        plt.ylim((0.5, int(nstar/2)))
        plt.draw()
        if savefig:
            plt.savefig(frame_dir + '/frame_' + str(c) + '.pdf', bbox_inches='tight')
        plt.pause(1e-5)

        fits.writeto(frame_dir + '/residual_'+str(c)+'.fits', resids[0])

        return 


    # fits.writeto(frame_dir + '/residualr_'+str(c)+'.fits', resids[0])
    # fits.writeto(frame_dir + '/residuali_'+str(c)+'.fits', resids[1])
    # fits.writeto(frame_dir + '/residualg_'+str(c)+'.fits', resids[2])


# ---------------------- DECIDE WHICH BAND TO USE/ WHICH COORDINATES TO PLOT --------------------------------

    if first_frame_band_no != 0:
        # if datatype != 'mock':
        if include_hubble:
            hx1, hy1 = transform_q(h_coords[0][posmask], h_coords[1][posmask], pixel_transfer_mats[first_frame_band_no-1])
            hx1 -= mean_dpos[first_frame_band_no-1, 0]
            hy1 -= mean_dpos[first_frame_band_no-1, 1]
        x1, y1 = transform_q(x, y, pixel_transfer_mats[first_frame_band_no-1])
        x1 -= mean_dpos[first_frame_band_no-1, 0]
        y1 -= mean_dpos[first_frame_band_no-1, 1]
    else:
        # if datatype != 'mock':
        if include_hubble:
            hx1 = h_coords[0][posmask]
            hy1 = h_coords[1][posmask]
        x1 = x
        y1 = y

# ------------------------------------------ MULTIBAND SAMPLE FRAME ---------------------------------------------
# ----------------------------- 1 ----------------------------

    

    if len(bands)==2:
        plt.gcf().clear()
        plt.figure(figsize=(8,10))
        # plt.subplot(2,2,1)

        # plt.imshow(data_array[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data_array[0]), vmax=np.percentile(data_array[0], 95))    
        # plt.colorbar()
        # mask = ref_f[:,0] > 25
        # # hf = hf[posmask]
        # plt.scatter(h_coords[0][posmask], h_coords[1][posmask], marker='+', s=2*mag_to_cts(hf, nmgy_per_count[0]) / sizefac, color='lime') #hubble
        # # if labldata != 'mock':
        #     # hf = hf[posmask]
        #     # plt.scatter(hx1[hmask], hy1[hmask], marker='+', s=2*mag_to_cts(hf[hmask], nmgy_per_count[0]) / sizefac, color='lime') #hubble
        # plt.scatter(x1, y1, marker='x', s=(10000/len(data_array[0])**2)*f[0]/(2*sizefac), color='r')
        # plt.xlim(-0.5, len(data_array[0])-0.5)
        # plt.ylim(-0.5, len(data_array[0][0])-0.5)

        plt.subplot(2, 2, 1)
        # plt.subplot(3,2,2)

        plt.title('Residual in ' + str(bands[0]) + ' band')
        plt.imshow(resids[0]*np.sqrt(weights[0]), origin='lower', interpolation='none', cmap='Greys', vmax=5, vmin=-5)
        # plt.colorbar()

        # plt.subplot(2,2,3)

        # plt.imshow(data_array[1], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data_array[1]), vmax=np.percentile(data_array[1], 95))    
        # plt.colorbar()
        # mask = ref_f[:,0] > 25
        # if labldata != 'mock':
        #     # hf = hf[posmask]
        #     plt.scatter(hx1[hmask], hy1[hmask], marker='+', s=2*mag_to_cts(hf[hmask], nmgy_per_count[0]) / sizefac, color='lime') #hubble
        # plt.scatter(x1, y1, marker='x', s=(10000/len(data_array[0])**2)*f[0]/(2*sizefac), color='r')
        # plt.xlim(-0.5, len(data_array[0])-0.5)
        # plt.ylim(-0.5, len(data_array[0][0])-0.5)

        plt.subplot(2, 2, 2)
        # plt.subplot(3,2,2)

        plt.title('Residual in ' + str(bands[1]) + ' band')
        plt.imshow(resids[1]*np.sqrt(weights[1]), origin='lower', interpolation='none', cmap='Greys', vmax=5, vmin=-5)
        # plt.colorbar()
     

        plt.subplot(2,2,3)
        # plt.subplot(3, 2, 5)

        plt.title('Posterior Magnitude Histogram (' + str(bands[first_mag_no]) + ')')
        (n, bins, patches) = plt.hist(adu_to_magnitude(f[first_mag_no], nmgy_per_count[first_mag_no]), bins=mag_bins, bottom=0.1, alpha=0.5, color='r', label='Chain - ' + bands[first_mag_no], histtype='step')
        plt.hist(adu_to_magnitude(ref_f[:,first_mag_no], nmgy_per_count[first_mag_no]), bins=bins,alpha=0.5, bottom=0.1, label=labldata, color='g', histtype='step')
        # plt.hist(adu_to_magnitude(f[0,brightest_idx], nmgy_per_count[0]), bins=mag_bins, bottom=0.1, alpha=0.5, color='k', label='Brightest Third', histtype='step')
        plt.legend(loc=2)
        plt.xlabel(bands[first_mag_no])
        plt.ylim((0.5, nstar))
        plt.yscale('log')

        plt.subplot(2, 2, 4)
        # plt.subplot(3, 2, 6)

        plt.title('Posterior Magnitude Histogram (' + str(bands[second_mag_no]) + ')')
        (n, bins, patches) = plt.hist(adu_to_magnitude(f[second_mag_no], nmgy_per_count[second_mag_no]), bins=mag_bins, bottom=0.1, alpha=0.5, color='r', label='Chain - ' + bands[second_mag_no], histtype='step')
        plt.hist(adu_to_magnitude(ref_f[:,second_mag_no], nmgy_per_count[second_mag_no]), bins=bins,alpha=0.5, bottom=0.1, label=labldata, color='g', histtype='step')
        # plt.hist(adu_to_magnitude(f[1,brightest_idx], nmgy_per_count[1]), bins=mag_bins, bottom=0.1, alpha=0.5, color='k', label='Brightest Third', histtype='step')
        plt.legend(loc=2)
        plt.xlabel(bands[second_mag_no])
        plt.ylim((0.5, nstar))
        plt.yscale('log')

        plt.tight_layout()
        plt.savefig(frame_dir+'/2x2_r_i_frame_' + str(c) + '.pdf')
        plt.close()





    if len(bands)==3:

        # else:
        plt.gcf().clear()
        plt.figure(1, dpi=300)
        plt.subplot(2,3,1)
        # plt.subplot(3,2,1)

        plt.imshow(data_array[first_frame_band_no], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data_array[first_frame_band_no]), vmax=np.percentile(data_array[first_frame_band_no], 95))    
        mask = ref_f[:,0] > 25
        # if labldata != 'mock':
        #     hf = hf[posmask]
        #     plt.scatter(hx1[hmask], hy1[hmask], marker='+', s=2*mag_to_cts(hf[hmask], nmgy_per_count[0]) / sizefac, color='lime') #hubble
        plt.scatter(x1, y1, marker='x', s=(10000/len(data_array[0])**2)*f[0]/(2*sizefac), color='r')
        plt.xlim(-0.5, len(data_array[0])-0.5)
        plt.ylim(-0.5, len(data_array[0][0])-0.5)

        # plt.scatter(ref_x[mask], ref_y[mask], marker='+', s=2*ref_f[mask,0] / sizefac, color='lime') #daophot
        # mask = np.logical_not(mask)
        # plt.scatter(ref_x[mask], ref_y[mask], marker='+', s=2*ref_f[mask,0] / sizefac, color='g')

    # ----------------------------- 2 ----------------------------

        plt.subplot(2, 3, 2)
        # plt.subplot(3,2,2)

        plt.title('Residual in ' + str(bands[first_resid_no]) + ' band')
        plt.imshow(resids[first_resid_no]*np.sqrt(weights[first_resid_no]), origin='lower', interpolation='none', cmap='Greys', vmax=5, vmin=-5)
        plt.colorbar()

    # ----------------------------- 3 ----------------------------

        plt.subplot(2,3,3)
        # plt.subplot(3,2,3)

        plt.title('Residual in ' + str(bands[second_resid_no]) + ' band (Frame ' + str(c)+')')
        plt.imshow(resids[second_resid_no]*np.sqrt(weights[second_resid_no]), origin='lower', interpolation='none', cmap='Greys', vmax=5, vmin=-5)
        plt.colorbar()

    # ----------------------------- 4 ----------------------------

        bolo_flux = np.sum(np.array(f), axis=0) 
        n_bright = min(len(ref_x), int(len(bolo_flux)/3))
        brightest_idx = np.argpartition(bolo_flux, n_bright)[-n_bright:]

        #Color histogram
        plt.subplot(2, 3, 4)
        # plt.subplot(3, 2, 4)

        plt.title('Posterior Color Histogram')
        plt.hist(adus_to_color(f[color_no_1], f[color_no_2], nmgy_per_count), label='Chain', alpha=0.5, bottom=0.1, bins=color_bins, histtype='step', color='r')
        plt.hist(truecolor[0], label=labldata, bins=color_bins, color='g', bottom=0.1, histtype='step')
        plt.hist(adus_to_color(f[color_no_1, brightest_idx], f[color_no_2, brightest_idx], nmgy_per_count), bottom=0.1, alpha=0.5, bins=color_bins, label='Brightest 300', histtype='step', color='k')
        plt.ylim(0.5, 300)
        plt.legend(loc=2)
        plt.yscale('log')
        plt.xlabel(bands[color_no_1] + ' - ' + bands[color_no_2])

    # ----------------------------- 5 ----------------------------

        plt.subplot(2,3,5)
        # plt.subplot(3, 2, 5)

        plt.title('Posterior Magnitude Histogram (' + str(bands[first_mag_no]) + ')')
        (n, bins, patches) = plt.hist(adu_to_magnitude(f[first_mag_no], nmgy_per_count[first_mag_no]), bins=mag_bins, bottom=0.1, alpha=0.5, color='r', label='Chain - ' + bands[first_mag_no], histtype='step')
        plt.hist(adu_to_magnitude(ref_f[:,first_mag_no], nmgy_per_count[first_mag_no]), bins=bins,alpha=0.5, bottom=0.1, label=labldata, color='g', histtype='step')
        # plt.hist(adu_to_magnitude(f[0,brightest_idx], nmgy_per_count[0]), bins=mag_bins, bottom=0.1, alpha=0.5, color='k', label='Brightest Third', histtype='step')
        plt.legend(loc=2)
        plt.xlabel(bands[first_mag_no])
        plt.ylim((0.5, nstar))
        plt.yscale('log')


    # ----------------------------- 6 ----------------------------

        plt.subplot(2, 3, 6)
        # plt.subplot(3, 2, 6)

        plt.title('Posterior Magnitude Histogram (' + str(bands[second_mag_no]) + ')')
        (n, bins, patches) = plt.hist(adu_to_magnitude(f[second_mag_no], nmgy_per_count[second_mag_no]), bins=mag_bins, bottom=0.1, alpha=0.5, color='r', label='Chain - ' + bands[second_mag_no], histtype='step')
        plt.hist(adu_to_magnitude(ref_f[:,second_mag_no], nmgy_per_count[second_mag_no]), bins=bins,alpha=0.5, bottom=0.1, label=labldata, color='g', histtype='step')
        # plt.hist(adu_to_magnitude(f[1,brightest_idx], nmgy_per_count[1]), bins=mag_bins, bottom=0.1, alpha=0.5, color='k', label='Brightest Third', histtype='step')
        plt.legend(loc=2)
        plt.xlabel(bands[second_mag_no])
        plt.ylim((0.5, nstar))
        plt.yscale('log')


        if visual:
            plt.draw()
        if savefig:
            plt.savefig(frame_dir + '/frame_' + str(c) + '.png', bbox_inches='tight')
            plt.gcf().clear()

    # ---------------------------- COLOR HISTOGRAM  n = 2 -------------------------------

            if len(bands) == 2:
                r_i_color = adus_to_color(f[0], f[1], nmgy_per_count)
                plt.figure()
                plt.title('Posterior Color Histogram (Frame ' + str(c)+')')
                plt.hist(r_i_color, label='Chain', alpha=0.5, bins=color_bins, bottom=0.1, histtype='step', color='r')
                plt.hist(truecolor[0], label=labldata, bins=color_bins, color='g', bottom=0.1, histtype='step')
                plt.hist(r_i_color[brightest_idx], alpha=0.5, bins=color_bins, bottom=0.1, label='Brightest 300', histtype='step', color='k')
                plt.legend(loc=2)
                plt.yscale('log')
                plt.xlabel(bands[0] + ' - ' + bands[1])
                plt.savefig(frame_dir + '/color_histograms_sample_' + str(c) + '.pdf', bbox_inches='tight')


    # ------------------------------ COLOR COLOR HISTOGRAM ----------------------------------------

            elif len(bands)==3:

                r_i_color = adu_to_magnitude(f[0], nmgy_per_count[0]) - adu_to_magnitude(f[1], nmgy_per_count[1])
                z_r_color = adu_to_magnitude(f[2], nmgy_per_count[2]) - adu_to_magnitude(f[0], nmgy_per_count[0])
                # plt.figure(2)
                # plt.title('Posterior Color-Color Histogram (Frame ' + str(c)+')')
                # plt.scatter(g_r_color[brightest_idx], r_i_color[brightest_idx], label='Brightest 300', s=2, color='k', alpha=0.5)
                # plt.xlabel('g-r', fontsize=14)
                # plt.ylabel('r-i', fontsize=14)
                # plt.scatter(g_r_color, r_i_color, label='Chain', s=1, alpha=0.2)
                # plt.scatter(-truecolor[1], truecolor[0], label=labldata, s=2, alpha=0.5)
                # plt.xlim(-2,2)
                # plt.ylim(-2,2)
                # plt.legend(loc=2)
                # plt.savefig(frame_dir + '/'+bands[0]+'_'+bands[1]+'_'+bands[2]+'_sample_'+str(c)+'.pdf', bbox_inches='tight')
                # plt.gcf().clear()

    # --------------------------------------- COLOR HISTOGRAMS ------------------------------------

                #color histograms
                plt.figure(3, figsize=(10,5))
                plt.subplot(1,2,1)
                plt.title('Posterior Color Histogram (Frame ' + str(c)+')')
                plt.hist(r_i_color, label='Chain', alpha=0.5, bins=color_bins, bottom=0.1, histtype='step', color='r')
                plt.hist(truecolor[0], label=labldata, bins=color_bins, color='g', bottom=0.1, histtype='step')
                plt.hist(r_i_color[brightest_idx], alpha=0.5, bins=color_bins, bottom=0.1, label='Brightest 300', histtype='step', color='k')
                plt.legend(loc=2)
                plt.ylim(0.5, 300)
                plt.yscale('log')
                plt.xlabel(bands[0] + ' - ' + bands[1])

                plt.subplot(1,2,2)
                plt.title('Posterior Color Histogram (Frame ' + str(c)+')')
                plt.hist(-z_r_color, label='Chain', alpha=0.5, bins=color_bins, bottom=0.1, histtype='step', color='r')
                plt.hist(truecolor[1], label=labldata, bins=color_bins, bottom=0.1, color='g', histtype='step')
                plt.hist(-z_r_color[brightest_idx], alpha=0.5, bins=color_bins, bottom=0.1, label='Brightest 300', histtype='step', color='k')
                plt.legend(loc=2)
                plt.yscale('log')
                plt.ylim(0.5, 300)
                plt.xlabel(bands[0] + ' - ' + bands[2])
                plt.savefig(frame_dir + '/color_histograms_sample_' + str(c) + '.pdf', bbox_inches='tight')


    # -------------------------------- ZOOM IN FRAME --------------------------------------------
            # hf = hf[posmask]
            # hmask = hf < 22


            plt.gcf().clear()
            plt.figure(4)
            plt.imshow(data_array[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data_array[0]), vmax=np.percentile(data_array[0], 95))    
            mask = ref_f[:,0] > 25
            if labldata != 'mock':
                plt.scatter(h_coords[0][posmask][hmask], h_coords[1][posmask][hmask], marker='+', s=3*2*mag_to_cts(hf[hmask], nmgy_per_count[0]) / sizefac, color='lime') #hubble
            plt.scatter(x, y, marker='x', s=(10000/15**2)*f[0]/(2*sizefac), color='r')
            plt.scatter(ref_x[mask], ref_y[mask]-1, marker='+', s=(10000/15**2)*ref_f[mask,0]/(2*sizefac), color='g') #daophot
            mask = np.logical_not(mask)
            plt.scatter(ref_x[mask], ref_y[mask]-1, marker='+', s=(10000/15**2)*ref_f[mask,0]/(2*sizefac), color='g')
            plt.xlim(50, 70)
            plt.ylim(70, 90)
            if savefig:
                plt.savefig(frame_dir + '/single_frame_r' + str(c) + '.png', bbox_inches='tight')

            x1, y1 = transform_q(x, y, pixel_transfer_mats[0])
            x1 -= mean_dpos[0, 0]
            y1 -= mean_dpos[0, 1]

            plt.gcf().clear()
            plt.figure(5)
            plt.imshow(data_array[1], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data_array[1]), vmax=np.percentile(data_array[1], 95))    
            mask = ref_f[:,0] > 25
            if labldata != 'mock':
                plt.scatter(h_coords[2][posmask][hmask], h_coords[3][posmask][hmask], marker='+', s=3*2*mag_to_cts(hf[hmask], nmgy_per_count[0]) / sizefac, color='lime') #hubble
            plt.scatter(x1, y1, marker='x', s=(10000/15**2)*f[0]/(2*sizefac), color='r')
            # plt.scatter(ref_x[mask], ref_y[mask]-1, marker='+', s=(10000/15**2)*ref_f[mask,0]/(2*sizefac), color='g') #daophot
            # mask = np.logical_not(mask)
            # plt.scatter(ref_x[mask], ref_y[mask]-1, marker='+', s=(10000/15**2)*ref_f[mask,0]/(2*sizefac), color='g')
            plt.xlim(50, 70)
            plt.ylim(70, 90)
            if savefig:
                plt.savefig(frame_dir + '/single_frame_i' + str(c) + '.png', bbox_inches='tight')

            x2, y2 = transform_q(x, y, pixel_transfer_mats[1])
            x2 -= mean_dpos[1, 0]
            y2 -= mean_dpos[1, 1]


            plt.gcf().clear()
            plt.figure(6)
            plt.imshow(data_array[2], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data_array[2]), vmax=np.percentile(data_array[2], 95))    
            mask = ref_f[:,0] > 25
            if labldata != 'mock':
                plt.scatter(h_coords[4][posmask][hmask], h_coords[5][posmask][hmask], marker='+', s=3*2*mag_to_cts(hf[hmask], nmgy_per_count[0]) / sizefac, color='lime') #hubble
            plt.scatter(x2, y2, marker='x', s=(10000/15**2)*f[0]/(2*sizefac), color='r')
            # plt.scatter(ref_x[mask], ref_y[mask]-1, marker='+', s=(10000/15**2)*ref_f[mask,0]/(2*sizefac), color='g') #daophot
            # mask = np.logical_not(mask)
            # plt.scatter(ref_x[mask], ref_y[mask]-1, marker='+', s=(10000/15**2)*ref_f[mask,0]/(2*sizefac), color='g')
            plt.xlim(50, 70)
            plt.ylim(70, 90)
            if savefig:
                plt.savefig(frame_dir + '/single_frame_g' + str(c) + '.png', bbox_inches='tight')


    plt.pause(1e-5)


