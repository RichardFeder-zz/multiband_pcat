import matplotlib.pyplot as plt
import numpy as np

sizefac = 10.*136
color_bins = np.linspace(-4, 4, 50)
dpi_val = 300

# color_bins = np.linspace(-2, 2, 30)
mag_bins = np.linspace(15, 23, 15)

def adus_to_color(flux0, flux1, nm_2_cts):
    colors = adu_to_magnitude(flux0, nm_2_cts[0]) - adu_to_magnitude(flux1, nm_2_cts[1])
    # colors = -2.5*np.log10((np.array(flux0)*nm_2_cts[0])/(np.array(flux1)*nm_2_cts[1]))
    return colors
def adu_to_magnitude(flux, nm_2_cts):
    mags = 22.5-2.5*np.log10((np.array(flux)*nm_2_cts))
    return mags

def mag_to_cts(mags, nm_2_cts):
    flux = 10**((22.5-mags)/2.5)/nm_2_cts
    return flux

def get_pint_dp(p):
    pint = np.floor(p+0.5)
    dp = p - pint
    return pint.astype(int), dp

def transform_q(x,y, mats):
    if len(x) != len(y):
        print('Unequal number of x and y coordinates')
        return
    xtrans, ytrans, dxpdx, dypdx, dxpdy, dypdy = mats
    xints, dxs = get_pint_dp(x)
    yints, dys = get_pint_dp(y)
    xnew = xtrans[yints,xints] + dxs*dxpdx[yints,xints] + dys*dxpdy[yints,xints]
    ynew = ytrans[yints,xints] + dxs*dypdx[yints,xints] + dys*dypdy[yints,xints] 
    return np.array(xnew).astype(np.float32), np.array(ynew).astype(np.float32)


def results(nchain, fchain, truef, color, nsamp, timestats, tq_times,plt_times, chi2, bkgsample, directory_path, timestr, nbands, bands, multiband, nmgy_per_count):
    plt.figure()
    plt.title('Posterior Source Number Histogram')
    plt.hist(nchain[int(len(nchain)/5):], histtype='step', label='Posterior', color='b')
    plt.axvline(np.median(nchain), label='Median=' + str(np.median(nchain)), color='b', linestyle='dashed')
    plt.xlabel('nstar')
    plt.legend()
    plt.savefig(directory_path + '/' + timestr + '/posterior_histogram_nstar.png', dpi=dpi_val)
    
    sample_number = list(xrange(nsamp))
    plt.figure()
    plt.title('Chi-Squared Distribution over Samples')
    for b in xrange(nbands):
        plt.plot(sample_number, chi2[:,b], label=bands[b])
    plt.xlabel('Sample')
    plt.ylabel('Chi2')
    plt.yscale('log')
    plt.legend()
    plt.savefig(directory_path + '/' + timestr + '/chi2_sample.png', dpi=dpi_val)

    # time stat analysis
    time_array = np.zeros(3, dtype=np.float32)
    labels = ['Proposal', 'Likelihood', 'Implement']
    for samp in xrange(nsamp):
        time_array += np.array([timestats[samp][2][0], timestats[samp][3][0], timestats[samp][4][0]])
    plt.figure()
    plt.title('Computational Resources')
    plt.pie(time_array, labels=labels, autopct='%1.1f%%', shadow=True)
    plt.savefig(directory_path + '/' + timestr + '/time_resource_statistics.png', dpi=dpi_val)

    plt.figure()
    plt.title('Time Histogram for asTrans Transformations')
    plt.hist(tq_times, bins=20, histtype='step')
    plt.xlabel('Time (s)')
    plt.axvline(np.median(tq_times), linestyle='dashed', color='r', label='Median: ' + str(np.median(tq_times)))
    plt.legend()
    plt.savefig(directory_path + '/' + timestr + '/asTrans_time_resources.png', dpi=dpi_val)

    plt.figure()
    plt.title('Time Histogram for Plotting')
    plt.hist(plt_times, bins=20, histtype='step')
    plt.xlabel('Time (s)')
    plt.axvline(np.median(plt_times), linestyle='dashed', color='r', label='Median: ' + str(np.median(plt_times)))
    plt.legend()
    plt.savefig(directory_path + '/' + timestr + '/plot_time_resources.png', dpi=dpi_val)


    for b in xrange(nbands):
        if np.median(bkgsample[:,b]) != bkgsample[0,b]: #don't plot this if we're not sampling the background
            plt.figure()
            plt.title('Background Distribution over Samples')
            plt.hist(bkgsample[:,b], label=bands[b], histtype='step')
            plt.axvline(np.median(bkgsample[:,b]), label='Median in ' + str(bands[b])+'-band: ' + str(np.median(bkgsample[:,b])), linestyle='dashed', color='b')
            # plt.axvline(actual_background[b], label='True Bkg ' + str(bands[b])+'-band: ' + str(actual_background[b]), color='g', linestyle='dashed')
            plt.xlabel('Background (ADU)')
            plt.ylabel('$n_{samp}$')
            plt.legend()
            plt.savefig(directory_path + '/' + timestr + '/bkg_sample_' + str(bands[b]) + '.png', dpi=dpi_val)

        plt.figure()
        if multiband:
            plt.title('Posterior Magnitude Distribution - ' + str(bands[b]))
        else:
            plt.title('Posterior Magnitude Distribution')
        true_mags = adu_to_magnitude(truef[:,b], nmgy_per_count[b])
        (n, bins, patches) = plt.hist(true_mags, histtype='step', label='DAOPHOT', color='g')
        post_hist = []
        for samp in xrange(nsamp):
            if multiband:
                hist = np.histogram([adu_to_magnitude(x, nmgy_per_count[b]) for x in fchain[b][samp] if x>0], bins=bins)
            else:
                hist = np.histogram([adu_to_magnitude(x, nmgy_per_count[b]) for x in fchain[samp] if x>0], bins=bins)
            post_hist.append(hist[0])
        medians = np.median(np.array(post_hist), axis=0)
        stds = np.std(np.array(post_hist), axis=0)
        bincentres = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
        plt.errorbar(bincentres, medians, yerr=stds, fmt='o', label='Posterior')
        plt.legend()
        plt.yscale('log', nonposy='clip')
        plt.ylim(0.1, 1000)
        if multiband:
            plt.xlabel('Magnitude - ' + str(bands[b]))
            plt.savefig(directory_path + '/' + timestr + '/posterior_flux_histogram_' + str(bands[b]) + '.png', dpi=dpi_val)
        else:
            plt.xlabel('Magnitude')
            plt.savefig(directory_path + '/' + timestr + '/posterior_flux_histogram.png', dpi=dpi_val)

    if multiband:
        # color plots
        plt.figure()
        plt.title('Posterior Color Distribution (Normalized to 1)')
        color_post_bins = np.linspace(-1.5, 1.5, 30)
        plt.hist(adus_to_color(truef[:,0], truef[:,1], nmgy_per_count), histtype='step', bins=color_post_bins, label='DAOPHOT', color='g', normed=1, alpha=0.5)
        post_hist = []
        bright_hist = []
        for samp in xrange(nsamp):
            brightest_idx = np.argpartition(fchain[0][samp], -100)[-100:]
            bright_h = np.histogram(adus_to_color(fchain[0][samp][brightest_idx], fchain[1][samp][brightest_idx], nmgy_per_count), bins=color_post_bins)
            bright_hist.append(bright_h[0])
            hist = np.histogram([x for x in color[samp]], bins=color_post_bins)
            post_hist.append(hist[0])
        medians = np.median(np.array(post_hist), axis=0)
        medians /= (np.sum(medians)*(color_post_bins[1]-color_post_bins[0]))  
        medians_bright = np.median(np.array(bright_hist), axis=0)
        medians_bright /= (np.sum(medians_bright)*(color_post_bins[1]-color_post_bins[0]))  
        bincentres = [(color_post_bins[i]+color_post_bins[i+1])/2. for i in range(len(color_post_bins)-1)]
        plt.step(bincentres, medians, where='mid', color='b', label='Posterior', alpha=0.5)
        plt.step(bincentres, medians_bright, where='mid', color='k', label='Brightest 100', alpha=0.5)
        plt.legend()
        plt.xlabel(str(bands[0]) + ' - ' + str(bands[1]), fontsize=14)
        plt.legend()
        plt.savefig(directory_path + '/' + timestr + '/posterior_histogram_r_i_color.png', dpi=dpi_val)


def multiband_sample_frame(data_array, x, y, f, ref_x, ref_y, ref_f, truecolor, resids, weights, bands, nmgy_per_count, nstar, frame_dir, c, pixel_transfer_mats, visual=0, savefig=0):
    # plt.rc('text', usetex=True) #use for latex quality characters and such
    #sdss specific thing
    if nmgy_per_count[0] == nmgy_per_count[1]:
        offset = 0
    else:
        offset = 2

    plt.gcf().clear()
    plt.figure(1)
    plt.subplot(2,3,1)
    plt.imshow(data_array[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data_array[0]), vmax=np.percentile(data_array[0], 95))
    plt.scatter(x, y, marker='x', s=(10000/len(data_array[0])**2)*f[0]/(2*sizefac), color='r')
    mask = ref_f[:,0] > 25
    plt.scatter(ref_x[mask], ref_y[mask]-offset, marker='+', s=2*ref_f[mask,0] / sizefac, color='lime')
    mask = np.logical_not(mask)
    plt.scatter(ref_x[mask], ref_y[mask]-offset, marker='+', s=2*ref_f[mask,0] / sizefac, color='g')
    plt.xlim(-0.5, len(data_array[0])-0.5)
    plt.ylim(-0.5, len(data_array[0][0])-0.5)
    

    x1, y1 = transform_q(x, y, pixel_transfer_mats[0])
    refx1, refy1 = transform_q(ref_x, ref_y, pixel_transfer_mats[0])

    
    plt.subplot(2, 3, 2)
    plt.title('Residual in ' + str(bands[0]) + ' band')
    plt.imshow(resids[0]*np.sqrt(weights[0]), origin='lower', interpolation='none', cmap='bwr', vmax=5, vmin=-5)
    plt.colorbar()

    plt.subplot(2,3,3)
    plt.imshow(data_array[1], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data_array[1]), vmax=np.percentile(data_array[1], 95))
    mask = ref_f[:,1] > 25
    plt.scatter(refx1[mask], refy1[mask]-offset, marker='+', s=2*ref_f[mask,1] / sizefac, color='lime')
    mask = np.logical_not(mask)
    plt.scatter(refx1[mask], refy1[mask]-offset, marker='+', s=2*ref_f[mask,1] / sizefac, color='g')
    plt.scatter(x1, y1, marker='x', s=(10000/len(data_array[0])**2)*f[1]/(2*sizefac), color='r')
    plt.xlim(-0.5, len(data_array[1])-0.5)
    plt.ylim(-0.5, len(data_array[1][0])-0.5)


    # plt.subplot(2,3,3)
    # plt.title('Residual in ' + str(bands[1]) + ' band')
    # plt.imshow(resids[1]*np.sqrt(weights[1]), origin='lower', interpolation='none', cmap='bwr', vmax=5, vmin=-5)
    # plt.colorbar()

    bolo_flux = np.sum(np.array(f), axis=0) 

    n_bright = 100
    # n_bright = np.minimum(int(nstar/10), len(bolo_flux))
    brightest_idx = np.argpartition(bolo_flux, n_bright)[-n_bright:]

    #Color histogram
    plt.subplot(2, 3, 4)
    plt.title('Posterior Color Histogram')
    plt.hist(adus_to_color(f[0], f[1], nmgy_per_count), label='Chain', alpha=0.5, bins=color_bins, histtype='step', color='r')
    plt.hist(adus_to_color(f[0, brightest_idx], f[1, brightest_idx], nmgy_per_count), alpha=0.5, bins=color_bins, label='Brightest ' + str(n_bright), histtype='step', color='k')
    plt.hist(truecolor, label='DAOPHOT', bins=color_bins, color='g', histtype='step')
    plt.legend(loc=2)
    plt.yscale('log')
    plt.xlabel(bands[0] + ' - ' + bands[1])


    plt.subplot(2,3,5)
    plt.title('Posterior Flux Histogram (' + str(bands[0]) + ')')
    (n, bins, patches) = plt.hist(adu_to_magnitude(f[0], nmgy_per_count[0]), bins=mag_bins, alpha=0.5, color='r', label='Chain - ' + bands[0], histtype='step')
    plt.hist(adu_to_magnitude(ref_f[:,0], nmgy_per_count[0]), bins=bins,alpha=0.5, label='DAOPHOT', color='g', histtype='step')
    plt.hist(adu_to_magnitude(f[0,brightest_idx], nmgy_per_count[0]), bins=mag_bins, alpha=0.5, color='k', label='Brightest ' + str(n_bright), histtype='step')
    plt.legend(loc=2)
    plt.xlabel(bands[0])
    plt.ylim((0.5, nstar))
    plt.yscale('log')


    plt.subplot(2, 3, 6)
    plt.title('Posterior Flux Histogram (' + str(bands[1]) + ')')
    (n, bins, patches) = plt.hist(adu_to_magnitude(f[1], nmgy_per_count[1]), bins=mag_bins, alpha=0.5, color='r', label='Chain - ' + bands[1], histtype='step')
    plt.hist(adu_to_magnitude(ref_f[:,1], nmgy_per_count[1]), bins=bins,alpha=0.5, label='DAOPHOT', color='g', histtype='step')
    plt.hist(adu_to_magnitude(f[1,brightest_idx], nmgy_per_count[1]), bins=mag_bins, alpha=0.5, color='k', label='Brightest ' + str(n_bright), histtype='step')
    plt.legend(loc=2)
    plt.xlabel(bands[1])
    plt.ylim((0.5, nstar))
    plt.yscale('log')

    if visual:
        plt.draw()
    if savefig:
        plt.savefig(frame_dir + '/frame_' + str(c) + '.png', dpi=dpi_val)
    plt.pause(1e-5)


def zoom_in_frame(data_array, x, y, f, hx, hy, hf, bounds, frame_dir, c, nmgy_per_count, pixel_transfer_mats):
    bound_dim = bounds[1]-bounds[0]
    orig_size = len(data_array[0])
    factor = orig_size / bound_dim
    hfr = mag_to_cts(hf[:,0], nmgy_per_count[0])
    fig = plt.figure()
    plt.imshow(data_array[0], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data_array[0]), vmax=np.percentile(data_array[0], 95))
    plt.colorbar()
    plt.scatter(x, y, marker='x', s=factor*f[:,0]/(2*sizefac), color='r')
    mask = hfr >25
    hfr = mag_to_cts(hf[:,0], nmgy_per_count[0])
    plt.scatter(hx[mask], hy[mask], marker='+', s=factor*hfr[mask]/(2*sizefac), color='lime')
    plt.xlim(bounds[0],bounds[1])
    plt.ylim(bounds[2],bounds[3])
    plt.savefig(frame_dir + '/zoom_in_' + str(c) + '.png', dpi=dpi_val)
    plt.close(fig)


    # plt.scatter(x, y, marker='x', s=f[0]/(2*sizefac), color='r')
    # mask = ref_f[:,0] > 25
    # plt.scatter(ref_x[mask], ref_y[mask]-2, marker='+', s=2*ref_f[mask,0] / sizefac, color='lime')
    # mask = np.logical_not(mask)
    # plt.scatter(ref_x[mask], ref_y[mask]-2, marker='+', s=2*ref_f[mask,0] / sizefac, color='g')
    # plt.xlim(-0.5, len(data_array[0])-0.5)
    # plt.ylim(-0.5, len(data_array[0][0])-0.5)


# def single_band_sample_frame(data_array, x, y, f, ref_x, ref_y, ref_f, resid, weight, nstar, visual=0, savefig=0, bounds=None)
#     plt.gcf().clear()
#     plt.figure(1)
#     plt.clf()
#     plt.subplot(1,3,1)
#     plt.imshow(data_array, origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data_array), vmax=np.percentile(data_array, 95))
#     sizefac = 10.*136

#             mask = truef[:,0] > 250 # will have to change this for other data sets
#             plt.scatter(truex[mask], truey[mask], marker='+', s=truef[mask] / sizefac, color='lime')
#             mask = np.logical_not(mask)
#             plt.scatter(truex[mask], truey[mask], marker='+', s=truef[mask] / sizefac, color='g')
#     plt.scatter(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], marker='x', s=self.stars[self._F, 0:self.n]/sizefac, color='r')
#     plt.xlim(-0.5, imsz[0]-0.5)
#     plt.ylim(-0.5, imsz[1]-0.5)
#     # plt.xlim(-0.5*imsz[0], 1.5*imsz[0])
#     # plt.ylim(-0.5*imsz[1], imsz[1]*1.5)
#     plt.subplot(1,3,2)
#     plt.imshow(resid*np.sqrt(weight), origin='lower', interpolation='none', cmap='bwr', vmin=-30, vmax=30)
#     plt.xlim(-0.5, imsz[0]-0.5)
#     plt.ylim(-0.5, imsz[1]-0.5)
#     plt.colorbar()
#     if j == 0:
#         plt.tight_layout()
#     plt.subplot(1,3,3)
#     if datatype == 'mock':
#         plt.hist(np.log10(truef), range=(np.log10(self.trueminf), np.log10(np.max(truef))), log=True, alpha=0.5, label=labldata, histtype='step')
#         plt.hist(np.log10(self.stars[self._F, 0:self.n]), range=(np.log10(self.trueminf), np.log10(np.max(truef))), log=True, alpha=0.5, label='Chain', histtype='step')
#     else:
#         plt.hist(np.log10(self.stars[self._F, 0:self.n]), range=(np.log10(self.trueminf), np.ceil(np.log10(np.max(self.f[0:self.n])))), log=True, alpha=0.5, label='Chain', histtype='step')
#     plt.legend()
#     plt.xlabel('log10 flux')
#     plt.ylim((0.5, self.nstar))
#     plt.draw()
#     if savefig:
#         plt.savefig(frame_dir + '/frame_' + str(c) + '.png')
#     plt.pause(1e-5)



