import numpy as np
import os
import time




#add_xue_bins.bash, stowedflux_smallfov.bash, change_evt2_names.bash, do_all_with_xue.bash, download_fov.bash, merge_obs_loop/noloop.bash

frac_exps = np.loadtxt(frac_exp.txt)

def __init__(marx=False):
    os.system('ciao')
    if marx:
        os.system('source marx-5.3.2/setup_marx.sh')

def change_fname(fname, obsid, e=None, ccd=None):
    fname = fname.replace('*obsid*', str(obsid))
    if e is not None:
        fname = fname.replace('*e*', str(e))
    if ccd is not None:
        fname = fname.replace('*ccd*', str(ccd))
    return fname

def mv_files(infile, outfile):
    cmd = 'mv '+infile+' '+outfile
    print cmd
    os.system(cmd)

def chmod(fname):
    cmd = 'chmod +w '+fname

    print cmd
    # os.system(cmd)
    return

def download_chandra(obsid, datatype):
    cmd = 'download_chandra_obsid '+str(obsid)+' '+datatype
    print cmd
    os.system(cmd)
    cmd2 = 'gunzip '+str(obsid)+'/primary/*'+datatype+'*'
    print cmd2
    os.system(cmd2)

def dmimgcalc(infile, out, infile2='none', weight=None, clobber='no'):
    cmd = 'dmimgcalc infile='+infile+' infile2='+infile2+' out='+out+' op="imgout=img1+img2+img3+img4" clobber=yes'
    if weight is not None:
        cmd += ' weight='+weight
    print cmd
    os.system(cmd)

def dmreadpar(parpath, stowed_file):
    cmd = 'dmreadpar '+parpath+' "'+stowed_file+'[events]" clobber+'
    print cmd
    # os.system(cmd) 
    return 


def get_sky_limits(imgfile):
    cmd = 'get_sky_limits '+imgfile
    print cmd
    # os.system(cmd)
    return

def dmf():
    dmf = os.system('pget get_sky_limits dmfilter') # ? 
    print 'dmf: ', dmf
    return dmf

def dmhedit(fname):
    cmd = 'dmhedit '+fname+' file= op=add key=DETNAM value=ACIS-0123'
    print cmd
    os.system(cmd)

def dmcopy(source, destination, bins=None, clobber='no', emin=None, emax=None):
    cmd = 'dmcopy '+source+' '+destination+' clobber='+clobber
    if bins is not None:
        cmd = cmd.replace(source, '"'+source+'[bin '+bins+']"')
    if emin is not None:
        cmd = cmd.replace(source, '"'+source+'[energy='+str(emin)+':'+str(emax)+']"')
    print cmd
    # os.system(cmd)
    return

def repropject_events(infile, outfile, aspect, match, random='0', clobber='no'):
    cmd = 'reproject_events infile='+infile+'"[cols -time]" outfile='+outfile+' aspect='+aspect+' match='+match+' random='+random+' clobber='+clobber
    print cmd
    # os.system(cmd)



def make_inst_map(outfile, spectrum_fname, obsfile, ccd, monoenergy='1.0'):
    cmd = 'mkinstmap outfile='+outfile+' spectrumfile='+spectrum_fname+' monoenergy='+monoenergy+' obsfile="'+obsfile+'" detsubsys="ACIS-'+str(ccd)+'" mirror="HRMA;AREA=1" clobber=no maskfile=NONE pixelgrid="1:1024:#1024,1:1024:#1024" grating=NONE pbkfile=NONE'
    print cmd
    os.system(cmd)
    return


class xray_cdfs():

    source_directory = '/n/fink1/rfeder/xray_pcat/obsids/full'
    obsid_dir = '/Users/richardfeder/Documents/xray_pcat/obsids'
    
    primary_path = self.source_directory+'/full/*obsid*/primary/' #replace obsid with actual obsid in functions
    fov_path = primary_path+'*obsid*fov_0123.fits' # same here
    img_path = primary_path+'instfiles/*obsid*_*energy*_i0123.img' # replace obsid and energy 
    asol_path = primary_path+'*obsid*_asol.fits'
    asphist_path = primary_path+'instfiles/*obsid*_i*ccd*.asphist'
    evt0123_path = primary_path+'*obsid*evt2_0123.fits'
    instfiles = primary_path+'/instfiles/'


    Ms_periods = ['2', '4', '7']
    exp_time = 6.03e5

    energies = [0.5, 0.91028, 1.65723, 3.01709, 5.4928, 10.0]
    center_of_e = [np.round(np.sqrt(energies[e]*energies[e+1]), 4) for e in xrange(len(energies)-1)]

    xue_energies = [0.5, 2, 8]
    center_of_e_xue = [1, 4]

    centers_x = [3944.5, 4174.5, 4182.5]
    centers_y = [4308, 4105.5, 4097.5]
    obs_idx = dict({'2':0, '4':1, '7':2})

    nx, ny  = 600, 600
    xmin=3796.5
    ymin=3796.5

    bounds = [xmin, xmin+nx, ymin, ymin+ny]

    bad_obsids ['2406']
    obsids_deflare=['1431_1', '16176', '16184', '17542']


    num_ccd = 4

    obsid_list, obsid_string_list = [], []



    def get_obsid_lists(self):
        for pd in self.Ms_periods:
            fpath = self.obsid_dir+'/'+str(pd)+'Ms_obsids.txt'
            text_file = open(fpath, 'r')
            t2 = open(fpath, 'r')
            obsid_string = t2.read().replace('\n', '').replace('-', '_')
            self.obsid_string_list.append(obsid_string)
            obsids = text_file.read().split(',')
            obsids = [ob.replace('\n', '').replace('-', '_') for ob in obsids]
            self.obsid_list.append(obsids)
        return 
            
    def mv_evt_files(self, bin_no, from_primary=0):
        for obsid in self.obsid_list[2]:
            bin_path = self.source_directory+'/binned/'+str(obsid)+'_bin/'+str(bin_no)+'/'
            evt_fname = str(obsid)+'_0_10_bin'+str(bin_no)+'_evt2.fits'
        
            if from_primary:
                mv_files(change_fname(self.primary_path, obsid)+evt_fname, bin_path)
            else:
                mv_files(bin_path+evt_fname, change_fname(self.primary_path, obsid))
        return


    def merge_obs_command(self, obs_set_idx, bin_no, xue=0):
        if xue:
            es = self.xue_energies
            coe = self.center_of_e_xue
            strbin = 'xue'+str(bin_no)
        else:
            es = self.energies
            coe = self.center_of_e
            strbin = str(bin_no)
            
        mname = 'merged_'+self.Ms_periods[obs_set_idx]+'Ms'
        cmd = 'merge_obs '+self.obsid_string_list[obs_set_idx]+' '+src+'/'+mname+'/rest_fov/'+strbin+'/ bin=1 '
        cmd += 'bands='+str(es[bin_no])+':'+str(es[bin_no+1])+':'+str(coe[bin_no])
        cmd += ' xygrid='+str(self.bounds[0])+':'+str(self.bounds[1])+':#'+str(self.nx)+','+str(self.bounds[2])+':'+\
                str(self.bounds[3])+':#'+str(self.ny)+' clobber=yes'

        print cmd
    #     os.system(cmd)
        return

    
    def merge_for1_bin(self, bin_no, xue=0):
        print 'Merging bin', bin_no, '..'
        print 'xue=', xue
        self.mv_evt_files(bin_no, from_primary=0)
        for o in xrange(len(self.Ms_periods)):
            self.merge_obs_command(bin_no, o, xue=xue)
        self.mv_evt_files(bin_no, from_primary=1)
        return

    def merge_all(self):
        t0 = time.clock()

        print 'Initializing CIAO..'
        os.system('ciao')
        
        print 'Merging log-spaced bins..'
        for b in xrange(len(self.center_of_e)):
            merge_for1_bin(b)
        
        print 'Merging bins from Xue et al. 2017'
        for b in xrange(len(center_of_e_xue)):
            merge_for1_bin(b, xue=1)
            
        dt = time.clock()-t0
        print 'Total time (s):', dt
        return


    def make_aspect_histograms(self):
        print 'Making aspect histograms...'
        for c in xrange(self.num_ccd):
            print 'Working on CCD #'+str(c)+'..'
            for obsid in self.obsid_list[2]:
                obs = str(obsid)

                asol_path = change_fname(self.asol_path, obsid)
                if obsid in self.bad_obsids:
                    print 'hooya'
                    asol_path = '"'+asol_path.replace(obs+'_asol', 'pcad*')+'"'

                asphist_path = change_fname(self.asphist_path, obsid, ccd=c)
                evt_file_path = change_fname(self.primary_path, obsid)+obs+'evt2_0123.fits'

                cmd = 'asphist '+asol_path+' '+asphist_path+' evtfile="'+evt_file_path+'[ccd_id='+str(c)+']" clobber=no'
                print cmd
                # os.system(cmd)
        return


    def copy_image_4_xygrid(self, obsid, e):
        obs = str(obsid)
        evt_path = change_fname(evt_path, obsid)+obs+'evt2_0123.fits'
        fov_path = change_fname(self.fov_path, obsid)
        cmd = 'dmcopy "'+evt_path+'[sky=region('+fov_path+')][bin sky=1]" '+change_fname(img_path, obsid, e=e)+' clobber=no'
        print cmd
        # os.system(cmd)
        return


    def make_param_file(self, obsid):
        obs = str(obsid)
        headerpar = self.instfiles+obs+'_evt_header.par'
        pntpar = change_fname(self.instfiles, obsid)+'/'+obs+'_pnt.par'
        cmd = 'dmmakepar '+self.evt0123_path+' '+headerpar +' clobber=no'
        cmd = change_fname(cmd, obsid)
        print cmd
        # os.system(cmd)

        cmd2 = 'grep _pnt '+headerpar+' > '+pntpar
        print cmd2
        # os.system(cmd2)

        cmd3 = 'grep _avg '+headerpar+' >> '+pntpar
        print cmd3
        # os.system(cmd3)

        cmd4 = 'cat '+pntpar
        print cmd4
        # os.system(cmd4)

    def make_exp_map(self, obsid, obs_set_idx):
        cmd = 'mkexpmap asphistfile='+asphistfile+' outfile='+outfile+' instmapfile='+instmapfile+'xygrid=xygrid=3644.5:4244.5:#'+str(self.nx)+',4008:4608:#'+str(self.ny)+' useavgaspect=yes clobber=yes'



#marx log bins, marx 10 sim

class MARX():

    marx_sim_dir = '/n/fink1/rfeder/xray_pcat/marx_psf_sims/'

    ras = ['53.1167', '53.1267', '53.1367', '53.1467', '53.1567', '53.1667']
    exposure_time = 100000
    det_type = 'ACIS-I'

    marx ExposureTime=100000 OutputDir=marx_psf_sims/off_axis_1ks_exp/10/ DetectorType=ACIS-I SourceRA=${ras[$counter]} SourceDEC=-27.8061 RA_Nom=53.1167 Dec_Nom=-27.8061 Roll_Nom=0 MinEnergy=10 MaxEnergy=10


    SourceDEC=-27.8061 
    RA_Nom=53.1167 
    Dec_Nom=-27.8061 
    Roll_Nom=0 
    MinEnergy=10 
    MaxEnergy=10

    ra_min = [4081.5, 4016.25, 3951.5, 3885.75, 3820.5, 3755.25]
    ra_max = [4111.5, 4046.25, 3981.5, 3915.75, 3850.5, 3785.25]


    def marx_init():
        os.system('source marx-5.3.2/setup_marx.sh')

    def do_simulation(self, outdir, src_ra, src_dec, nom_ra, nom_dec, min_e, max_e, roll_nom=0, spect_file=None, src_flux=None):
        cmd = 'marx ExposureTime='+str(self.exposure_time)+' OutputDir='+outdir+' DetectorType='+self.det_type
        cmd += ' SourceRA='+str(src_ra)+' SourceDEC='+str(src_dec)+' RA_Nom='+str(nom_ra)+' Dec_Nom='+str(nom_dec)+' Roll_Nom='+str(roll_nom)
        cmd += ' MinEnergy='+str(min_e)+' MaxEnergy='+str(max_e)
        if spect_file is not None:
            cmd += ' SpectrumType=FILE SpectrumFile='+spect_file
        if src_flux is not None:
            cmd += ' SourceFlux='+str(src_flux)
        print cmd
        os.system(cmd)
        return


    def marx2fits(inpath, outfile, pixadj=None):
        cmd = 'marx2fits '+inpath+' '+outfile
        if pixadj is not None:
            cmd = cmd.replace('marx2fits', 'marx2fits --pixadj='+pixadj)
        print cmd
        os.system(cmd)
        return


class stowed():

    #define constants
    exp_time = 6.03e5
    nbins = 200
    xmin = 3009
    xmax = 5099
    ymin = 2109
    ymax = 4200

    e_low = 0.5 #keV
    e_high = 10 #keV
    eV_to_keV = 0.001

    e_bins = [0.5, 0.91, 1.66, 3.02, 5.49, 10]
    log_bins = np.logspace(np.log10(self.e_low/self.eV_to_keV), np.log10(self.e_high/self.eV_to_keV), 6) #log spaced energy bins in eV, n+1 values
    norm_log = np.diff(log_bins) #width of each energy bin in eV
    bin_com = np.sqrt(log_bins[1:]*log_bins[:-1]) #center of mass for each log bin in eV

    bins = [[] for x in xrange(len(e_bins)-1)]
    final_spectrum = np.zeros(len(e_bins)-1)

    total_pixel_area = (2048*0.492)**2 #in arcsecond^2
    arcsec_to_sterad = 4.25e10 
    total_steradians = total_pixel_area/arcsec_to_sterad

    energy = []
    eas = []

    def load_eff_area(fname):
        dat = np.loadtxt(fname) #    dat = np.loadtxt('effective_area_cycle3.txt')
        self.energy = np.array(dat[:,0], dtype=np.float32)
        self.eas = np.array(dat[:,1], dtype=np.float32)

    def bin_evt_file(self, ea_fname, evt_fname):
        load_eff_area(ea_fname)

        split_idx = np.digitize(self.energy, self.e_bins)
        for x in xrange(len(split_idx)):
            bins[split_idx].append(eas[x])
        ea_weighted = [np.average(bini) for bini in bins]

        hstow = fits.open(fname)
        datastow = hstow[1].data
        hstow.close()
        x_coords = datastow['x']
        y_coords = datastow['y']
        e_values = datastow['energy']
        cropped_evts = [[x_coords[i], y_coords[i], e_values[i]] for i in xrange(len(x_coords)) if x_coords[i]>self.xmin and x_coords[i]<self.xmax\
                         and y_coords[i]>self.ymin and y_coords[i]<self.ymax]
        cropped_evts = np.array(cropped_evts)
        
        # energy binning
        div = eV_to_keV*(np.min(cropped_evts[:,2])+np.max(cropped_evts[:,2]))/self.nbins
        energyhist = np.array(self.eV_to_keV*np.histogram(cropped_evts[:,2], self.nbins)[1], dtype='float')
        counts = energyhist[0]

        log_hist = np.histogram(cropped_evts[:,2], bins=self.log_bins)

        self.final_spectrum = log_hist[0]/self.exp_time/norm_log/self.eV_to_keV/self.total_steradians/ea_weighted


    def plt_effective_area(self, saveas=None):
        plt.figure()
        plt.plot(self.energy, self.ea)
        plt.xscale('log')
        plt.xlim(self.e_low,self.e_high)
        plt.xlabel('Energy (keV)')
        plt.ylabel('Effective Area (cm^2)')
        plt.title('Effective Area of ACIS-I Detector')
        if saveas is not None:
            plt.savefig(saveas)
        plt.show()


    def plt_stowed_image(self, cropped_evts, nbins, saveas=None):   
        img_zero, yedges, xedges = np.histogram2d(cropped_evts[:,0], cropped_evts[:,1], self.nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.figure(figsize=(10,10))
        plt.imshow(img_zero, extent=extent, interpolation='nearest', cmap='gist_yarg', origin='lower')
        plt.xlabel('Chip Y Coordinates (pixel)')
        plt.ylabel('Chip X Coordinates (pixel)')
        if saveas is not None:
            plt.savefig(saveas)
        plt.show()

    def plt_stowed_spectrum(self, stowed_energy, counts, div, log_hist, saveas=None):
        plt.figure(figsize=(8,6))
        plt.plot(stowed_energy[0:200], counts/self.exp_time/div, color='black', linewidth=1)
        plt.plot(stowed_energy[0:200], counts/self.exp_time/div, 'ro')
        plt.plot(self.bin_com*self.eV_to_keV, log_hist[0]/self.norm_log/self.exp_time/self.eV_to_keV, 'bo') #in keV
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(self.e_low,self.e_high)
        plt.ylim(5e-2, 1)
        plt.title('Log-Binned Spectrum of ACIS-I Stowed Background: ' + str(self.e_low) + "-" + str(self.e_high) + ' keV', fontsize=15)
        plt.xlabel('Energy (keV)', fontsize=14)
        plt.ylabel('S (Counts/s/keV)', fontsize=14)
        if saveas is not None:
            # plt.savefig(saveas)
            plt.savefig(saveas+'_log_hist_stowed_' + str(self.e_low)+ '_' + str(self.e_high) + '_keV.png')
        plt.show()

    def plt_final_stowed_spectrum(self, saveas=None):
        plt.figure(figsize=(10,8))
        plt.plot(self.bin_com*self.eV_to_keV, self.final_spectrum, color='black', linewidth=1)
        plt.plot(self.bin_com*self.eV_to_keV, self.final_spectrum, 'ro')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(self.e_low,self.e_high)
        # plt.ylim(5e-2,5e-1)
        plt.xlabel('Energy (keV)', fontsize=14)
        plt.ylabel('S (Counts/s/keV/sr/cm^2)', fontsize=14)
        plt.title('X-ray Spectrum of Chandra Stowed Background (DE): ' + str(self.e_low) + ' - ' + str(self.e_high) + 'keV', fontsize=15)
        if saveas is not None:
            plt.savefig('binned_spectrum_'+saveas+'.png')
        plt.show()



# plt_stowed_image(cropped_evts, nbins)

# cropped_evts = bin_evt_file('./acis_DE_01236_stowed_evt_110210.fits')






