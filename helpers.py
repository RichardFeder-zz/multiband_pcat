import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from image_eval import psf_poly_fit

def adus_to_color(flux0, flux1, nm_2_cts):
    colors = adu_to_magnitude(flux0, nm_2_cts[0]) - adu_to_magnitude(flux1, nm_2_cts[1])
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
    assert len(x)==len(y)
    xtrans, ytrans, dxpdx, dypdx, dxpdy, dypdy = mats
    xints, dxs = get_pint_dp(x)
    yints, dys = get_pint_dp(y)
    try:
        xnew = xtrans[yints,xints] + dxs*dxpdx[yints,xints] + dys*dxpdy[yints,xints]
        ynew = ytrans[yints,xints] + dxs*dypdx[yints,xints] + dys*dypdy[yints,xints] 
        return np.array(xnew).astype(np.float32), np.array(ynew).astype(np.float32)
    except:
        print xints, dxs, yints, dys
        raise ValueError('problem accessing elements')

def best_fit_transform(mat): 
    #generate random points to get best fit to be used for x and y vals
    randx = np.random.uniform(1, 99, 1000000)
    randy = np.random.uniform(1, 99, 1000000)
    newx, newy = transform_q(randx, randy, mat)
    diffx = newx - randx
    diffy = newy - randy
    fitx = np.poly1d(np.polyfit(randx, diffx, 1))
    fity = np.poly1d(np.polyfit(randy, diffy, 1))
    return fitx, fity

def linear_transform_astrans(x, y, linex, liney):
    xp = x+linex(x)
    yp = y+liney(y)
    return xp, yp

def gaussian(x, mu, sig):
    return -np.power(x - mu, 2.) / (2 * np.power(sig, 2.))

def generate_default_astrans(imsz):
    astransx, astransy, mat3, mat4, mat5, mat6 = [[] for x in xrange(6)]
    for x in xrange(imsz[0]):
        astransx.append(np.linspace(0, imsz[0]-1, imsz[0]))
        astransy.append(np.full((imsz[1]), x).transpose())
    mat3 = np.full((imsz[0], imsz[1]), 1)
    mat4 = np.zeros((imsz[0], imsz[1]))
    mat5 = np.zeros((imsz[0], imsz[1]))
    mat6 = np.full((imsz[0], imsz[1]), 1)
    pixel_transfer_mats = np.zeros((6, imsz[0],imsz[1]))
    pixel_transfer_mats = np.array([astransx, astransy, mat3, mat4, mat5, mat6])
    return pixel_transfer_mats

def read_astrans_mats(data_path):
    hdu = fits.open(data_path)
    mats = np.array([hdu[0].data, hdu[1].data,hdu[2].data, hdu[3].data, hdu[4].data, hdu[5].data])
    return mats

def find_mean_offset(filename, dim=100):
    mats = read_astrans_mats(filename)
    x, y = [np.random.uniform(10, dim-10, 10000) for p in xrange(2)]
    x1, y1 = transform_q(x, y, mats)
    dx, dy = np.mean(x1-x), np.mean(y1-y)
    return dx, dy

def get_psf_and_vals(path):
    f = open(path)
    psf = np.loadtxt(path, skiprows=1).astype(np.float32)
    nc, nbin = [np.int32(i) for i in f.readline().split()]
    f.close()
    cf = psf_poly_fit(psf, nbin=nbin)
    return psf, nc, cf

def get_nanomaggy_per_count(frame_path):
    fits_frame = fits.open(frame_path)
    frame_header = fits_frame[0].header
    nanomaggy_per_count = frame_header['NMGY']
    return nanomaggy_per_count

def get_hubble(hubble_cat_path, xoff=310, yoff=630):
    hubble_pos = fits.open(hubble_cat_path)

    hxr = hubble_pos[0].data-xoff
    hyr = hubble_pos[1].data-yoff
    hxi = hubble_pos[2].data-xoff
    hyi = hubble_pos[3].data-yoff
    hxg = hubble_pos[4].data-xoff
    hyg = hubble_pos[5].data-yoff

    hubble_coords = [hxr, hyr, hxi, hyi, hxg, hyg]

    hubble_cat = np.loadtxt(hubble_cat_path)
    hx = hubble_cat[:,0]
    hy = hubble_cat[:,1]
    hf = hubble_cat[:,2:]

    posmask = np.logical_and(hx+0.5<imdim, hy+0.5<imdim)
    hf = hf[posmask]
    hmask = hf < 22

    return hubble_coords, hf, hmask




