import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import astropy.io.fits
import astropy.wcs
import scipy.spatial
import sys
from astropy.io import fits

dataname = sys.argv[1]
run_name = sys.argv[2]

print 'run_name', run_name
datatype = 'notmock'

max_num_sources = 2500
bounds = [310.0, 410.0, 630.0, 730.0]

def err_f(f):
        back = 179
        gain = 4.62
        return 1./np.sqrt(gain*np.sum(psf0*psf0/(back+psf0*f)))
def err_mag(mag):
        f = 10**((22.5 - mag)/2.5) / 0.00546689
        return 1.08573620476 * np.sqrt((err_f(f) / f)**2 + 0.01**2)
def adutomag(adu):
        return 22.5 - 2.5 * np.log10(0.00546689 * adu)

hwhm=2.5

nsamp = 50

#----------------- HUBBLE ------------------

if datatype == 'mock':
    mock_catalog = np.loadtxt('Data/'+dataname+'/'+dataname+'-tru.txt')
    mock_x = mock_catalog[:,0]
    mock_y = mock_catalog[:,1]
    mock_f = mock_catalog[:,2:]
    mock_coords = np.zeros((mock_x.shape[0], 2))
    mock_coords[:, 0] = mock_x
    mock_coords[:, 1] = mock_y
    mock_kd = scipy.spatial.KDTree(mock_coords)

    mock_rmag = adutomag(mock_f[:,0])


    print np.sum(mock_rmag < 22), 'Mock Truth Brighter than 22'

else:
    fitshubble = np.loadtxt('hubble_catalog_2583-2-0136_astrans.txt')
    HTx_fits = fitshubble[:,0] - 310
    HTy_fits = fitshubble[:,1] - 630
    fitsmask = np.logical_and(fitshubble[:,2]>0, fitshubble[:,3]>0)
    fitsmask = np.logical_and(np.logical_and(np.logical_and(HTx_fits > 0+hwhm, HTx_fits < 99-hwhm), np.logical_and(HTy_fits > 0+hwhm, HTy_fits < 99-hwhm)), fitsmask)
    HTx_fits = HTx_fits[fitsmask]
    HTy_fits = HTy_fits[fitsmask]
    HTcat = np.loadtxt('Data/NGC7089R.RDVIQ.cal.adj.zpt', skiprows=1)
    HTra = HTcat[:,21]
    HTdc = HTcat[:,22]
    HT606= HTcat[:,3]
    HT606err = HTcat[:,4]
    HT814= HTcat[:,7]
    hdulist = astropy.io.fits.open('Data/'+dataname+'/frame-r-002583-2-0136.fits')
    w = astropy.wcs.WCS(hdulist['PRIMARY'].header)
    pix_coordinates = w.wcs_world2pix(HTra, HTdc, 0)
    # HTx = pix_coordinates[0] - bounds[0]
    # HTy = pix_coordinates[1] - bounds[2]
    # mask = np.logical_and(HT606 > 0, HT814 > 0)
    # mask = np.logical_and(np.logical_and(np.logical_and(HTx > 0+hwhm, HTx < 99-hwhm), np.logical_and(HTy > 0+hwhm, HTy < 99-hwhm)), mask)
    # HTx = HTx[mask]
    # HTy = HTy[mask]
    HT606 = fitshubble[:,2]
    HT814 = fitshubble[:,3]
    HT606 = HT606[fitsmask]
    HT814 = HT814[fitsmask]
    HTc = np.zeros((HTx_fits.shape[0], 2))
    HTc[:, 0] = HTx_fits
    HTc[:, 1] = HTy_fits
    HTkd = scipy.spatial.KDTree(HTc)

    print np.sum(HT606 < 22), 'HST brighter than 22'


# if datatype == 'mock': 
lion = np.load('pcat-lion-results/'+run_name+'/chain.npz')
PC3n = lion['n'][-nsamp:].astype(np.int)
PC3x = lion['x'][-nsamp:,:]
PC3y = lion['y'][-nsamp:,:]
print len(PC3y)
PC3f = lion['f']
# print PC3f
PC3f = PC3f[0,-nsamp:]

print len(PC3f)

# print PC3f
PC3r = adutomag(PC3f)

print PC3r
mask = (PC3f > 0) * (PC3x > 0+hwhm) * (PC3x < 99-hwhm) * (PC3y > 0+hwhm) * (PC3y < 99-hwhm)
PCc3_all = np.zeros((np.sum(mask), 2))
PCc3_all[:, 0] = PC3x[mask].flatten()
PCc3_all[:, 1] = PC3y[mask].flatten()
PCr3_all = PC3r[mask].flatten()
PC3kd = scipy.spatial.KDTree(PCc3_all)


#------------------- PORTILLO ET AL 2017 -----------------------

if datatype != 'mock':
    PCcat = np.loadtxt('run-alpha-20-new/posterior_sample.txt')
    maxn = 3000
    PCn = PCcat[-nsamp:,10003].astype(np.int)
    PCx = PCcat[-nsamp:,10004:10004+maxn]
    PCy = PCcat[-nsamp:,10004+maxn:10004+2*maxn]
    PCf = PCcat[-nsamp:,10004+2*maxn:10004+3*maxn]
    print PCn.shape, PCx.shape, PCy.shape, PCf.shape
    PCr = adutomag(PCf)
    mask = (PCf > 0) * (PCx > 0+hwhm) * (PCx < 99-hwhm) * (PCy > 0+hwhm) * (PCy < 99-hwhm)
    PCc_all = np.zeros((np.sum(mask), 2))
    PCc_all[:, 0] = PCx[mask].flatten()
    PCc_all[:, 1] = PCy[mask].flatten()
    PCr_all = PCr[mask].flatten()
    PCkd = scipy.spatial.KDTree(PCc_all)

    print np.mean(PCn), 'mean PCAT sources'


sizefac = 1360.




def associate(a, mags_a, b, mags_b, dr, dmag, confs_b = None, sigfs_b = None):
	allmatches = a.query_ball_tree(b, dr)
	goodmatch = np.zeros(mags_a.size, np.bool)
	if confs_b is not None:
		confmatch = np.zeros(mags_a.size)
	if sigfs_b is not None:
		sigfmatch = np.zeros(mags_a.size) + float('inf')

	for i in xrange(len(allmatches)):
		matches = allmatches[i]
		if len(matches):
			mag_a = mags_a[i]
			goodmatch[i] = False
			for j in matches:
				mag_b = mags_b[j]
				if np.abs(mag_a - mag_b) < dmag:
					goodmatch[i] = True
					if (confs_b is not None) and (confs_b[j] > confmatch[i]):
						confmatch[i] = confs_b[j]
					if (sigfs_b is not None) and (sigfs_b[j] < sigfmatch[i]):
						sigfmatch[i] = sigfs_b[j]

	if confs_b is not None:
		if sigfs_b is not None:
			return goodmatch, confmatch, sigfmatch
		else:
			return goodmatch, confmatch
	else:
		if sigfs_b is not None:
			return goodmatch, sigfmatch
		else:
			return goodmatch

dr = 0.75
dmag = 0.75
nbins = 16
minr, maxr = 15.5, 23.5
binw = (maxr - minr) / float(nbins)
precPC = np.zeros(nbins)
precPC3= np.zeros(nbins)

if datatype == 'mock':
    goodmatchPC3 = associate(PC3kd, PCr3_all, mock_kd, mock_rmag, dr, dmag)
else:
    goodmatchPC = associate(PCkd, PCr_all, HTkd, HT606, dr, dmag)
    goodmatchPC3 = associate(PC3kd, PCr3_all, HTkd, HT606, dr, dmag)

for i in xrange(nbins):
    rlo = minr + i * binw
    rhi = rlo + binw

    if datatype != 'mock':
        inbin = np.logical_and(PCr_all >= rlo, PCr_all < rhi)
        precPC[i] = np.sum(np.logical_and(inbin, goodmatchPC)) / float(np.sum(inbin))

    inbin = np.logical_and(PCr3_all >= rlo, PCr3_all < rhi)
    precPC3[i] = np.sum(np.logical_and(inbin, goodmatchPC3)) / float(np.sum(inbin))
print 'precPC3', precPC3

plt.figure()


svals = np.loadtxt('pcat-lion-results/'+run_name+'/completeness_'+str(run_name)+'.txt')

print('Saving pcat-lion-results/'+run_name+'/fdr-lion'+str(run_name)+'_'+str(dr)+'_'+str(dmag)+'.pdf')
if datatype != 'mock':
    label = 'Portillo et al. (2017)'
    plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precPC, c='r', label=label, marker='x', markersize=10, mew=2)
else:
    label = 'Mock'

# plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precPC, c='r', label=label, marker='x', markersize=10, mew=2)
plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precPC3, c='b', label='Multiple Band Catalog Ensemble', marker='+', markersize=10, mew=2)
# plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precPC3, c='b', label='this work', marker='+', markersize=10, mew=2)

# if datatype != 'mock':
#     plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precPC, c='r', label='Portillo et al. (2017)', marker='x', markersize=10, mew=2)
# else:
#     plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precPC2, c='m', label='this work', marker='x', markersize=10, mew=2)
    # plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precPC2, c='m', label='this work', marker='x', markersize=10, mew=2)
plt.xlabel('SDSS r magnitude')
plt.ylabel('false discovery rate')
plt.ylim((-0.05, 0.9))
plt.xlim((15,24))
plt.legend(prop={'size':12}, loc = 'best')
plt.savefig('pcat-lion-results/'+run_name+'/fdr-lion'+str(run_name)+'_'+str(dr)+'_'+str(dmag)+'.pdf')
plt.show()

fdr3 = 1-precPC3
np.savetxt('pcat-lion-results/'+run_name+'/fdr_'+str(run_name)+'.txt', fdr3)


plt.gcf().clear()





nbins = 18
minr, maxr = 15.5, 24.5
binw = (maxr - minr) / float(nbins)


if datatype != 'mock':

    #if doPCAT:
    completePC = np.zeros((PCx.shape[0], HTx_fits.size))
    for i in xrange(PCx.shape[0]):
        print i
        n = PCn[i]
        CCc_one = np.zeros((n,2))
        CCc_one[:, 0] = PCx[i, 0:n]
        CCc_one[:, 1] = PCy[i, 0:n]
        CCr_one = PCr[i, 0:n]
        completePC[i, :] = associate(HTkd, HT606, scipy.spatial.KDTree(CCc_one), CCr_one, dr, dmag)
    completePC = np.sum(completePC, axis=0) / float(PCx.shape[0])

    completePC3 = np.zeros((PC3x.shape[0], HTx_fits.size))
    for i in xrange(PC3x.shape[0]):
        print 'B', i
        n = PC3n[i]
        CCc_one = np.zeros((n,2))
        CCc_one[:, 0] = PC3x[i,0:n]
        CCc_one[:, 1] = PC3y[i,0:n]
        CCr_one = PC3r[i, 0:n]
        completePC3[i,:] = associate(HTkd, HT606, scipy.spatial.KDTree(CCc_one), CCr_one, dr, dmag)
    completePC3 = np.sum(completePC3, axis=0) / float(PC3x.shape[0])


else:

    completePC3 = np.zeros((PC3x.shape[0], mock_x.size))
    for i in xrange(PC3x.shape[0]):
        print 'B', i
        n = PC3n[i]
        CCc_one = np.zeros((n,2))
        CCc_one[:, 0] = PC3x[i,0:n]
        CCc_one[:, 1] = PC3y[i,0:n]
        CCr_one = PC3r[i, 0:n]
        completePC3[i,:] = associate(mock_kd, mock_rmag, scipy.spatial.KDTree(CCc_one), CCr_one, dr, dmag)
    completePC3 = np.sum(completePC3, axis=0) / float(PC3x.shape[0])



    # completePC3 = np.zeros((PCx.shape[0], mock_x.size))
    # for i in xrange(PC2x.shape[0]):
    #     print 'B', i
    #     n = PC2n[i]
    #     CCc_one = np.zeros((n,2))
    #     CCc_one[:, 0] = PC2x[i,0:n]
    #     CCc_one[:, 1] = PC2y[i,0:n]
    #     CCr_one = PC2r[i, 0:n]
    #     completePC2[i,:] = associate(mock_kd,mock_f[:,0], scipy.spatial.KDTree(CCc_one), CCr_one, dr, dmag)
    # completePC2 = np.sum(completePC2, axis=0) / float(PC2x.shape[0])



reclPC = np.zeros(nbins)
reclPC2= np.zeros(nbins)
reclPC3= np.zeros(nbins)



for i in xrange(nbins):
    rlo = minr + i * binw
    rhi = rlo + binw
    if datatype != 'mock':
        inbin = np.logical_and(HT606 >= rlo, HT606 < rhi)
        reclPC[i] = np.sum(completePC[inbin]) / float(np.sum(inbin))
        reclPC3[i] = np.sum(completePC3[inbin]) / float(np.sum(inbin))
    else:
        inbin = np.logical_and(mock_rmag >= rlo, mock_rmag< rhi)
        reclPC3[i] = np.sum(completePC3[inbin]) / float(np.sum(inbin))

# svals = np.loadtxt('pcat-lion-results/'+run_name+'/completeness_'+str(run_name)+'.txt')
# binss = np.arange(19)/2.0 + 15.0
# bin_centerss = np.zeros(18)
# for i in range(0, len(binss) - 1):
#     bin_centerss[i] = (binss[i] + binss[i+1])/2.0


# print svals
plt.figure()
plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclPC3, c='b', label='Three Band Catalog Ensemble', marker='+', markersize=10, mew=2)
# plt.plot(minr + (np.arange(nbins)+0.5)*binw, svals, c='b', label='Two Band Catalog Ensemble', marker='+', markersize=10, mew=2)


if datatype != 'mock':
    plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclPC, c='r', label='Portillo et al. (2017)', marker='x', markersize=10, mew=2)
    # plt.plot(bin_centerss, svals, label='DAOPHOT', marker='+', c='g', markersize=10, mew=2)
# else:
#     plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclPC2, c='m', label='this work', marker='x', markersize=10, mew=2)
plt.xlabel('HST F606W magnitude', fontsize='large')
plt.ylabel('completeness', fontsize='large')
plt.ylim((-0.1,1.1))
plt.legend(loc='best', fontsize='large')
plt.savefig('pcat-lion-results/'+run_name+'/completeness-lion_3_band'+str(run_name)+'_'+str(dr)+'_'+str(dmag)+'.pdf')
plt.show()

np.savetxt('pcat-lion-results/'+run_name+'/completeness_3_band'+str(run_name)+'.txt', reclPC3)
