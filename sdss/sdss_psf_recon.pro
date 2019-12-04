;+
; NAME:
;   sdss_psf_recon
;
; PURPOSE:
;   Reconstruct an SDSS PSF using a psField-file structure.
;
; CALLING SEQUENCE:
;   psfimage = sdss_psf_recon(psfield, xpos, ypos, [ normalize=, trimdim= ] )
;
; INPUTS:
;   psfield    - psField file structure describing the PSF.  This is
;                from HDU's 1 through 5 in a psField file, corresponding
;                to the five filters u,g,r,i,z.
;   xpos       - Column position (0-indexed, not 0.5-indexed as PHOTO outputs)
;   ypos       - Row position (0-indexed, not 0.5-indexed as PHOTO outputs)
;
; OPTIONAL INPUTS:
;   normalize  - If set, then normalize the image to this value.
;   trimdim    - Trimmed dimensions; for example, set to [25,25] to trim
;                the output PSF image to those dimensions.  These dimensions
;                must be odd-valued.
;
; OUTPUTS:
;   psfimage   - PSF image, typically dimensioned [51,51].  The center of
;                the PSF is always the central pixel; this function will
;                not apply any sub-pixel shifts.
;
; OPTIONAL OUTPUTS:
;
; COMMENTS:
;   The SDSS photo PSF is described as a set of eigen-templates, where the
;   mix of these eigen-templates is a simple polynomial function with (x,y)
;   position on the CCD.  Typically, there are 4 such 51x51 pixel templates.
;   The polynomial functions are typically quadratic in each dimension,
;   with no cross-terms.
;
;   The formula is the following, where i is the index of row polynomial order,
;   j is the index of column polynomial order, and k is the template index:
;      acoeff_k = SUM_i{ SUM_j{ (0.001*ROWC)^i * (0.001*COLC)^j * C_k_ij } }
;      psfimage = SUM_k{ acoeff_k * RROWS_k }
;   The polynomial terms need not be of the same order for each template.
;
;   Lupton's stand-alone C code to do the same thing adds a "software bias"
;   value of 1000 counts.  It also normalizes the PSF such that the central
;   pixel value (not the peak value) is 30000 (or 29000, after subtracting
;   the software bias).
;
;   This C code can be found in the "photo" product in the file
;     photo/readAtlasImages/variablePsf.c
;   as the function phPsfKLReconstruct().
; 
; EXAMPLES:
;   Reconstruct the PSF for run 259, camcol 1, field 100 in the r-band
;   (HDU #3) at position (x,y) = (600.,500.):
;     IDL> psfield = mrdfits('psField-000259-1-0100.fit', 3)
;     IDL> psfimage = sdss_psf_recon(psfield, 600., 500.)
;
; BUGS:
;
; PROCEDURES CALLED:
;
; REVISION HISTORY:
;   18-Sep-2002  Written by D. Schlegel, Princeton
;-  
;------------------------------------------------------------------------------
function sdss_psf_recon, psfield, xpos, ypos, psfimage, normalize=normalize, $
 trimdim=trimdim

   rc_scale = 1.e-3 ; Hard-wired scale factor for ypos/xpos coefficients

   ;----------
   ; Assume that the dimensions of each eigen-template are the same.

   rncol = psfield[0].rncol
   rnrow = psfield[0].rnrow
   npix = rncol * rnrow

   ;----------
   ; These are the polynomial coefficients as a function of x,y.
   ; Only compute these coefficients for the maximum polynomial order in use.
   ; In general, this order can be different for each eigen-template.

   nr_max = max(psfield.nrow_b)
   nc_max = max(psfield.ncol_b)
   coeffs = ((ypos+0.5) * rc_scale)^lindgen(nr_max) $
    # ((xpos+0.5) * rc_scale)^lindgen(nc_max)

   ;----------
   ; Reconstruct the image by summing each eigen-template.

   neigen = n_elements(psfield)
   psfimage = 0
   for i=0, neigen-1 do $
    psfimage = psfimage $
     + total( psfield[i].c[0:psfield[i].nrow_b-1,0:psfield[i].ncol_b-1] $
     * coeffs[0:psfield[i].nrow_b-1,0:psfield[i].ncol_b-1] ) $
     * psfield[i].rrows

   ;----------
   ; We have reconstructed the PSF as a vector using all the
   ; pixels in PSFIELD.RROWS.  The arithmetic is fastest in IDL doing
   ; this reconstruction with all these pixels even if some are not used
   ; (otherwise we are forcing a de-reference, which copies the eigen-template).
   ; So, at the end we trim this vector to only those pixels used, and
   ; reform it into a 2-dimensional image.

   psfimage = reform(psfimage[0:npix-1], rncol, rnrow)

   if (keyword_set(trimdim)) then $
    psfimage = psfimage[ (rncol-trimdim[0])/2:(rncol+trimdim[0]-1)/2, $
     (rnrow-trimdim[1])/2:(rnrow+trimdim[1]-1)/2 ]

   if (keyword_set(normalize)) then $
    psfimage = psfimage * float(normalize / total(psfimage))

   return, psfimage
end
;------------------------------------------------------------------------------
