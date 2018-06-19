import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import (MinMaxInterval, SqrtStretch, ImageNormalize)
from astropy.io import fits

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
    if len(x) != len(y):
        print('Unequal number of x and y coordinates')
        return
    xtrans, ytrans, dxpdx, dypdx, dxpdy, dypdy = mats
    xints, dxs = get_pint_dp(x)
    yints, dys = get_pint_dp(y)
    xnew = xtrans[yints,xints] + dxs*dxpdx[yints,xints] + dys*dxpdy[yints,xints]
    ynew = ytrans[yints,xints] + dxs*dypdx[yints,xints] + dys*dypdy[yints,xints] 
    return np.array(xnew).astype(np.float32), np.array(ynew).astype(np.float32)

def gaussian(x, mu, sig):
    return -np.power(x - mu, 2.) / (2 * np.power(sig, 2.))