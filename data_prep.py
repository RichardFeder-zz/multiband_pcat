from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import random
import astropy
import math
import astropy.io.fits
import astropy.wcs
from matplotlib.colors import LogNorm 
from scipy.optimize import curve_fit
import pandas as pd
from copy import deepcopy
from __future__ import division
import scipy.signal
import pyfits as pyf
from scipy.spatial import cKDTree

xmin = 310
xmax = 409
ymin = 630+1361
ymax = 729+1361
imdim = 100

coords = [xmin, xmax, ymin, ymax]


base_path = '/Users/richardfeder/Documents/multiband_pcat/pcat-lion-master/'
run = '002583'
camcol = '2'
field = '0136'
rerun = '301'
bands = ['r', 'i', 'g']


run_cam_field = run + '-'+camcol+'-'+field
dataname = 'idR-'+run_cam_field
data_path = base_path+'/Data/'+dataname

# Run, rerun, CamCol, DAOPHOTID, RA, DEC, xu, yu, u (mag), uErr, chi, sharp, flag, xg, yg, g, gErr, chi, sharp, flag, 
# xr, yr, r, rerr, chi, sharp, flag, xi, yi, i, ierr, chi, sharp, flag, xz, yz, z, zerr, chi, sharp, flag

def magnitudes_to_counts(frame_file, mags):
    fits_frame = fits.open(frame_file)
    frame_header = fits_frame[0].header
    nanomaggy_per_count = frame_header['NMGY']
    print('nanomaggy_per_count: ' + str(nanomaggy_per_count))
    nmgy = 10**((22.5-np.array(mags))/2.5)
    source_counts = [ x/nanomaggy_per_count for x in nmgy ]
    return source_counts

def extract_sdss_catalog(catalog_file, bands, outfile, bounds):

	band_idx = dict('r'=22, 'i'=29, 'g'=15, 'z'=36)

	with open(catalog_file, 'r') as p:
        lines = p.read().splitlines()
    sources = []
    for line in lines:
        sources.append(line.split())

    subregion_sources = [x for x in sources if float(x[20])>bounds[0] and float(x[20])<bounds[1] and float(x[21])>bounds[2] and float(x[21])<bounds[3]]
    catalog_sources = [[float(x[20])-bounds[0]-1, float(x[21])-bounds[2]-1] for x in subregion_sources] 

    for band in bands:
    	mags = [float(x[band_idx['r']]) for x in subregion_sources]
    	frame_name = base_path+'/Data/'+dataname+'/frames/frame-'+band+'-'+run_cam_field+'.fits'
    	counts = magnitudes_to_counts(frame_name, mags)
    	for s in xrange(len(catalog_sources)):
    		catalog_sources[s].append(counts[s])

    print len(catalog_sources), 'sources included in catalog from subregion.'

    with open(outfile, 'w') as file:
    	file.writelines(' '.join(str(j) for j in source) + '\n' for source in catalog_sources)

        
def get_hubble_catalog(filename, outfile,save='no'):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
        sources = []
        for line in lines:
            sources.append(line.split()) 
        sources = sources[3:]
        
        HTcat = np.loadtxt('pcat-lion-master/Data/NGC7089R.RDVIQ.cal.adj.zpt', skiprows=1)
        HTra = HTcat[:,21]
        HTdc = HTcat[:,22]
        HTrmag = HTcat[:,9]
        HTimag = HTcat[:,10]
    
        hdulist = astropy.io.fits.open('pcat-lion-master/Data/idR-002583-2-0136/frames/frame-r-002583-2-0136.fits')
        w = astropy.wcs.WCS(hdulist['PRIMARY'].header)
        
        pix_coordinates = w.wcs_world2pix(HTra, HTdc, 0)
        HTx = pix_coordinates[0]-310
        HTy = pix_coordinates[1]-630
 
        border = 2.5   
        border_mask = np.logical_and(np.logical_and(HTx > border, HTx < 99.0 - border), np.logical_and(HTy > border, HTy < 99.0 - border))
        back_to_radec = [[HTra[x], HTdc[x], HTrmag[x], HTimag[x]] for x in xrange(len(HTcat)) if HTx[x]>0 and HTx[x]<100 and HTy[x]>0 and HTy[x]<100]
        
        hubble_pix_coordinates = fits.open('pcat-lion-master/Data/idR-002583-2-0136/hubble_pixel_coords-2583-2-0136.fits')
        print len(hubble_pix_coordinates[0].data), len(np.array(back_to_radec)[:,0])
        #         np.savetxt('hubble_ra_decs_2.txt', np.array(back_to_radec), fmt=['%f','%f'])

        hubble_catalog = np.array([hubble_pix_coordinates[0].data, hubble_pix_coordinates[1].data, np.array(back_to_radec)[:,2],np.array(back_to_radec)[:,3]])
        if save:
            np.savetxt('hubble_catalog_2583-2-0136_astrans.txt', hubble_catalog.transpose(), fmt=['%f', '%f', '%f', '%f'])
        return hubble_catalog


def save_cts(infile, outfile, vmin=0, vmax=0, save=0, colorbar=0):
    counts = pyf.getdata(infile)
    np.savetxt(outfile, counts)
        
def save_psf_resampled(data_path, band):
    filename = data_path + '/psfs/sdss-'+run_cam_field+'-psf-'+band+'.fits'
    sdss_psf = pyf.getdata(filename)
    psf = np.zeros((50,50))
    psf[0:25,0:25] = sdss_psf
    psf = scipy.misc.imresize(psf, (250, 250), interp='lanczos', mode='F')
    psfnew = np.array(psf[0:125, 0:125])
    psfnew[0:123,0:123] = psf[2:125,2:125]  # shift due to lanczos kernel
    outfile = data_path+'/psfs/'+dataname+'-psf'+band+'.txt'
	np.savetxt(outfile, psfnew, header='25\t5')
    
def get_nanomaggy_per_count(frame_path):
    fits_frame = fits.open(frame_path)
    frame_header = fits_frame[0].header
    nanomaggy_per_count = frame_header['NMGY']
    return nanomaggy_per_count

def make_pix_file(data_path, band):

	nmgy_to_cts = get_nanomaggy_per_count(data_path+'/frames/frame-'+band+'-'+run_cam_field+'.fits')

	bias, gain = get_bias_gain(data_path)

	f = open(data_path+'/pixs/'+dataname+'-pix'+band+'.txt', 'w')
	f.write('%1d\t%1d\t1\n0.\t%0.3f\n%0.8f' % (imdim, imdim, bias, gain, nmgy_to_cts))
	f.close()

def get_bias_gain(data_path):
    ecalib_lines = []
    with open(data_path+'/opECalib-'+rerun+'-'+run, 'r') as p:
        for line in p:
            split = line.split()
            if len(split)>0:
                if split[0]=='ECALIB':
                    ecalib_lines.append(split)

    camrows = dict(r='1', i='2', u='3', g='4', z='5')

    desired_row = [row for row in ecalib_lines if row[2]==camrows['r'] and row[3]=='2']

    assert len(desired_row)==1
    gain, bias = desired_row[0][6], desired_row[0][7]

    return bias, gain




# sdss catalog extraction

extract_sdss_catalog(data_path+'/m2_2583.phot', bands, data_path+'/'+dataname+'-tru.txt', coords)

for band in bands:
	save_cts(data_path+'/cts/idR-'+run+'-'+band+camcol+'-'+field+'_subregion_cts.fits', data_path+'/cts/'+dataname+'-cts'+band+'.txt')
	save_psf_resampled(data_path, band)
	make_pix_file(data_path, band)













