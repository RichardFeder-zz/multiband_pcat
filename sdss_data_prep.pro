pro mpcat_asTrans, band1, band2, run, camcol, field, rerun, corner_x, corner_y, dim, data_repository

  run_cam_field = string(run,'-',camcol,'-',field, format='(I6.6,A,I1,A,I4.4)')
  
; ------- Set $PHOTO_REDUX
  setenv, 'PHOTO_REDUX='+data_repository

  run =fix(uint(run))
  camcol = fix(uint(camcol))
; -------- Check path stuff is set up
                                    
  print, run
  print, camcol

  print, sdss_path('asTrans', run, camcol, rerun=rerun)
; -------- and that we can generate a correct file name
  
  print, sdss_name('asTrans', run, camcol, field, rerun=rerun)

; -------- Read astrometry data 

  as1 = sdss_astrom(run, camcol, field, rerun=rerun, filter=band1)
  as2 = sdss_astrom(run, camcol, field, rerun=rerun, filter=band2)
  
; -------- Now, we want to do some transformations. Let's make
;          a grid of x and y
  nx = dim
  ny = dim

  xbox = lindgen(nx, ny) mod nx
  ybox = lindgen(nx, ny)  /  nx
; derivatives for band 1 to band 2
                                                                                                                                                                                                                
  get_derivative, as1, as2, xbox, ybox, 0.5d, 0.0d, dxpdx12, dypdx12, corner_x, corner_y
  get_derivative, as1, as2, xbox, ybox, 0.0d, 0.5d, dxpdy12, dypdy12, corner_x, corner_y

  ;print, dypdy12


  astrans_band2band, as1, as2, xbox+corner_x, ybox+corner_y, xp12, yp12
  xp12 = xp12-corner_x
  yp12 = yp12-corner_y
  ;print, 'xp12', xp12
  ;print, 'yp12', yp12

  asGrid_path = data_repository+'/asGrid/asGrid'
  ;fname = string(asGrid_path, run, '-', camcol, '-', field, '-', nx, 'x', ny, '-', band1, '-', band2, '-', corner_x, '-', corner_y, '_cterms_0p15_1p0.fits', format='(A,I6.6,A,I1,A,I4.4,A,I4.4,A,I4.4,A,A,A,A,A,I4.4,A,I4.4,A)')
 
  fname = string(asGrid_path, run, '-', camcol, '-', field, '-', nx, 'x', ny, '-', band1, '-', band2, '-', corner_x, '-', corner_y, '_cterms_0p01_0p01.fits', format='(A,I6.6,A,I1,A,I4.4,A,I4.4,A,I4.4,A,A,A,A,A,I4.4,A,I4.4,A)')

  print, 'Writing ', fname
  mwrfits, xp12, fname, /create
  mwrfits, yp12, fname
  mwrfits, dxpdx12, fname
  mwrfits, dypdx12, fname
  mwrfits, dxpdy12, fname
  mwrfits, dypdy12, fname

  return
end

pro psField, run_cam_field, data_repository
  
  psfield_path = data_repository+'/psfs/psField-'+run_cam_field+'.fit'
  print, psfield_path
  bands = ["g", "r", "i", "u", "z"]
  color_dict = DICTIONARY("g", 2, "r", 3, "i", 4, "u", 1, "z", 5)
  psf_base_path = data_repository+'/psfs/sdss-'+run_cam_field+'-psf'

  FOREACH band, bands DO BEGIN
     psfield = mrdfits(psfield_path, color_dict[band])
     psfimage_recenter = sdss_psf_recon(psfield, 630+50, 310+50, trimdim=[25,25])
     psfimage = sdss_psf_recon(psfield, 600., 500., trimdim=[25,25])
     psf_path = psf_base_path+'-'+band+'.fits'
     psf_path_recenter = psf_base_path+'-'+band+'-680-360.fits'
     mwrfits, psfimage, psf_path, /create
     mwrfits, psfimage_recenter, psf_path_recenter, /create
     ;mwrfits, psfimage, data_repository+'/psfs/'
     ;mwrfits, psfimage, '~/Data/idR-'+run_cam_field+'/psfs/sdss-'+run_cam_field+'-psf-'+band+'-680-360.fits', /create
  ENDFOREACH

end

pro get_subregion_idr, infile, outfile, offx, offy, dim, dx, dy
  idr_in = mrdfits(infile)
  mwrfits, idr_in[40+offx+dx :40+offx+dim-1+dx,offy+dy :offy+dim-1+dy], outfile, /create
end

pro astrans_band2band, as0, as1, x, y, x1, y1

  astrans_xy2eq, as0, x, y, ra=ra, dec=dec, cterm=0.01

  ;astrans_eq2xy, as0, ra, dec, xpix=xback, ypix=yback

  astrans_eq2xy, as1, ra, dec, xpix=x1, ypix=y1, cterm=0.01
  ;astrans_xy2eq, as1, x, y, ra=ra, dec=dec, cterm=0.15
  ;astrans_eq2xy, as1, ra, dec, xpix=x1, ypix=y1, cterm=1.0 ;when transforming to g band, cterm should be <g-r>

  return
end

pro get_derivative, as0, as1, x, y, epsx, epsy, dxp, dyp, offx, offy

; -------- transform this (x,y) grid to (ra,dec) via as0 astrometry
  astrans_band2band, as0, as1, offx+x+epsx, offy+y+epsy, x1, y1

; -------- Now transform BACK to (x,y) for as1
  astrans_band2band, as0, as1, offx+x-epsx, offy+y-epsy, x0, y0

  dxp = x1-x0
  dyp = y1-y0
  
  ;print, 'dxp', dxp
  ;print, 'dyp', dyp
  return
end


;bands = ["r", "i", "g"]
bands = ["g"]
;run = '002583'
;camcol = '2'
;field = '0136'
;rerun = '301'

run = '008151'
camcol = '4'
field = '0063'
rerun = '301'

run_cam_field = run+'-'+camcol+'-'+field
data_repository = '/n/fink1/rfeder/mpcat/multiband_pcat/Data/idR-'+run_cam_field
;x0 = 310.
;y0 = 630.
x0 = 100.
y0 = 100.
;dim = 100
dim = 500
dx = -2
dy = 12
;psField, run_cam_field, data_repository

;astrans_dpos_ints = [-1, 3, -2, 12]

FOREACH band, bands DO BEGIN
   band_col = '-'+band+camcol+'-'
   run_cam_field_band = repstr(run_cam_field, '-'+camcol+'-', band_col)
   cts_path = data_repository+'/cts/idR-'+run_cam_field_band+'_subregion_500_cts_0203.fits'
   in_path = data_repository+'/idrs/idR-'+run_cam_field_band+'.fit'
   get_subregion_idr, in_path, cts_path, x0, y0, dim, dx, dy
   ; for copying to home directory for jupyter notebook validation
   get_subregion_idr, in_path, '~/Data/idR-'+run_cam_field+'/cts/idR-'+run_cam_field_band+'_subregion_500_cts_0203.fits', x0, y0, dim, dx, dy
ENDFOREACH

;astrans_bands = ["i", "g", "z"]
;astrans_bands = ["i", "g"]
;FOREACH band, astrans_bands DO BEGIN
;   print, 'r - ', band
;   mpcat_asTrans, 'r', band, run, camcol, field, rerun, x0, y0, dim, data_repository  
;ENDFOREACH

end
