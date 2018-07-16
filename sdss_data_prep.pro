pro mpcat_asTrans, band1, band2, run, camcol, field, rerun, corner_x, corner_y, dim, data_repository

  run_cam_field = string(run,'-',camcol,'-',field, format='(I6.6,A,I1,A,I4.4)')
  
  setenv, 'PHOTO_REDUX='+data_repository

  run =fix(uint(run))
  camcol = fix(uint(camcol))
  
; -------- Set $PHOTO_REDUX
  ;setenv, 'PHOTO_REDUX=/n/fink1/rfeder/mpcat/multiband_pcat/Data/idR-002583-2-0134'
; -------- Check path stuff is set up                                                                                                                                                                           
                                      
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

  astrans_band2band, as1, as2, xbox, ybox, xp12, yp12
  xp12 = xp12
  yp12 = yp12

  asGrid_path = data_repository+'/asGrid'
  fname = string(asGrid_path, run, '-', camcol, '-', field, '-', nx, 'x', ny, '-', band1, '-', band2, '-', corner_x, '-', corner_y, '_cterms0_0.fits', format='(A,I6.6,A,I1,A,I4.4,A,I4.4,A,I4.4,A,A,A,A,A,I4.4,A,I4.4,A)')

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
     psfimage = sdss_psf_recon(psfield, 600., 500., trimdim=[25,25])
     psf_path = psf_base_path+'-'+band+'.fits'
     mwrfits, psfimage, psf_path
  ENDFOREACH

end

pro get_subregion_idr, infile, outfile, offx, offy
  idr_in = mrdfits(infile)
  mwrfits, idr_in[40+offx :40+offx+499,offy :offy+499], outfile
end

pro astrans_band2band, as0, as1, x, y, x1, y1

  astrans_xy2eq, as0, x, y, ra=ra, dec=dec, cterm=0.0
  astrans_eq2xy, as1, ra, dec, xpix=x1, ypix=y1, cterm=0.0

  return
end

pro get_derivative, as0, as1, x, y, epsx, epsy, dxp, dyp, offx, offy

; -------- transform this (x,y) grid to (ra,dec) via as0 astrometry
  astrans_band2band, as0, as1, offx+x+epsx, offy+y+epsy, x1, y1

; -------- Now transform BACK to (x,y) for as1
  astrans_band2band, as0, as1, offx+x-epsx, offy+y-epsy, x0, y0

  dxp = x1-x0
  dyp = y1-y0

  return
end


bands = ["r", "i", "g", "z"]
run = '002583'
camcol = '2'
field = '0134'
rerun = '301'
run_cam_field = run+'-'+camcol+'-'+field
data_repository = '/n/fink1/rfeder/mpcat/multiband_pcat/Data/idR-'+run_cam_field
x0 = 310.
y0 = 630.
dim = 20


psField, run_cam_field, data_repository

FOREACH band, bands DO BEGIN
   band_col = '-'+band+camcol+'-'
   run_cam_field_band = repstr(run_cam_field, '-'+camcol+'-', band_col)
   cts_path = data_repository+'/cts/idR-'+run_cam_field_band+'_subregion_cts.fits'
   in_path = data_repository+'/idR-'+run_cam_field_band+'.fit'
   get_subregion_idr, in_path, cts_path, x0, y0
ENDFOREACH

astrans_bands = ["i", "g", "z"]

FOREACH band, astrans_bands DO BEGIN
   print, 'r - ', band
   mpcat_asTrans, 'r', band, run, camcol, field, rerun, x0, y0, dim, data_repository  
ENDFOREACH

end
