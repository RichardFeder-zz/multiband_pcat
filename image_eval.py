import numpy as np

def psf_poly_fit(psf0, nbin):
        assert psf0.shape[0] == psf0.shape[1] # assert PSF is square
        npix = psf0.shape[0]

        # pad by one row and one column
        psf = np.zeros((npix+1, npix+1), dtype=np.float32)
        psf[0:npix, 0:npix] = psf0

        # make design matrix for each nbin x nbin region
        # print(type(npix), type(nbin))
        nc = int(npix/nbin) # dimension of original psf
        nx = nbin+1
        y, x = np.mgrid[0:nx, 0:nx] / np.float32(nbin)
        x = x.flatten()
        y = y.flatten()
        A = np.column_stack([np.full(nx*nx, 1, dtype=np.float32), x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y]).astype(np.float32)
        # output array of coefficients
        
        cf = np.zeros((A.shape[1], nc, nc), dtype=np.float32)

        # loop over original psf pixels and get fit coefficients
        for iy in range(nc):
            for ix in range(nc):
                # solve p = A cf for cf
                p = psf[iy*nbin:(iy+1)*nbin+1, ix*nbin:(ix+1)*nbin+1].flatten()
                AtAinv = np.linalg.inv(np.dot(A.T, A))
                ans = np.dot(AtAinv, np.dot(A.T, p))
                cf[:,iy,ix] = ans

        return cf.reshape(cf.shape[0], cf.shape[1]*cf.shape[2])

def image_model_eval(x, y, f, back, imsz, nc, cf, regsize=None, margin=0, offsetx=0, offsety=0, weights=None, ref=None, lib=None, template=None):
    assert x.dtype == np.float32
    assert y.dtype == np.float32
    # assert f.dtype == np.float32
    # not sure what to do with cf
    #assert cf.dtype == np.float32
    if ref is not None:
        assert ref.dtype == np.float32

    if weights is None:
        weights = np.full(imsz, 1., dtype=np.float32)

    if regsize is None:
        regsize = max(imsz[0], imsz[1])

    # FIXME sometimes phonions are outside image... what is best way to handle?
    goodsrc = (x > 0) * (x < imsz[0] - 1) * (y > 0) * (y < imsz[1] - 1)
    x = x.compress(goodsrc)
    y = y.compress(goodsrc)
    f = f.compress(goodsrc)

    nstar = x.size
    rad = nc/2 # 12 for nc = 25

    nregy = int(imsz[1]/regsize + 1) # assumes imsz % regsize = 0?
    nregx = int(imsz[0]/regsize + 1)

    ix = np.ceil(x).astype(np.int32)
    dx = ix - x
    iy = np.ceil(y).astype(np.int32)
    dy = iy - y

    dd = np.column_stack((np.full(nstar, 1., dtype=np.float32), dx, dy, dx*dx, dx*dy, dy*dy, dx*dx*dx, dx*dx*dy, dx*dy*dy, dy*dy*dy)).astype(np.float32) * f[:, None]
    if lib is None:
        image = np.full((imsz[1]+2*rad+1,imsz[0]+2*rad+1), back, dtype=np.float32)
        recon2 = np.dot(dd, cf).reshape((nstar,nc,nc))
        recon = np.zeros((nstar,nc,nc), dtype=np.float32)
        recon[:,:,:] = recon2[:,:,:]
        for i in range(nstar):
            image[iy[i]:iy[i]+rad+rad+1,ix[i]:ix[i]+rad+rad+1] += recon[i,:,:]

        image = image[rad:imsz[1]+rad,rad:imsz[0]+rad]

        if ref is not None:
            diff = ref - image
            diff2 = np.zeros((nregy, nregx), dtype=np.float64)
            for i in range(nregy):
                y0 = max(i*regsize - offsety - margin, 0)
                y1 = min((i+1)*regsize - offsety + margin, imsz[1])
                for j in range(nregx):
                    x0 = max(j*regsize - offsetx - margin, 0)
                    x1 = min((j+1)*regsize - offsetx + margin, imsz[0])
                    subdiff = diff[y0:y1,x0:x1]
                    diff2[i,j] = np.sum(subdiff*subdiff*weights[y0:y1,x0:x1])
    else:
        # image = np.full((imsz[1], imsz[0]), back, dtype=np.float32)
        image = np.full((imsz[0], imsz[1]), back, dtype=np.float32)

        recon = np.zeros((nstar,nc*nc), dtype=np.float32)
        reftemp = ref
        if ref is None:
            reftemp = np.zeros((imsz[0], imsz[1]), dtype=np.float32)
            # reftemp = np.zeros((imsz[1], imsz[0]), dtype=np.float32)
        diff2 = np.zeros((nregy, nregx), dtype=np.float64)

        if template is not None: # template
            image += np.array(template)
        
        lib(imsz[0], imsz[1], nstar, nc, cf.shape[0], dd, cf, recon, ix, iy, image, reftemp, weights, diff2, regsize, margin, offsetx, offsety)


    if ref is not None:
        return image, diff2
    else:
        return image
