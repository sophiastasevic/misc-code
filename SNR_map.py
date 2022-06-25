"""
Calculates signal to noise ratio of reduced image using the no-disk image reduced
with negative parallactic angle + creates SNR map
Inputs:
    -Reduction (+normalisation) method of image(s) to process
    -Reduced image + no-disk image
    -Star PSF
"""

import numpy as np
import os
from astropy.io import fits
import vip_hci as vip
from datetime import datetime
from photutils import CircularAnnulus

EPOCH='2015-04-12'
TARGET='HD110058'
INSTRUMENT='ird'
SPHERE_FILTER='H23'
FILTER='H23'

SAVE_DIR='/mnt/c/Users/stasevis/Documents/RDI/ADI/output'
SPHERE_DIR='/mnt/c/Users/stasevis/Documents/sphere_data'

use_fake = False
fake_lum = 'faint'

path_end = {False: 'convert_recenter', True: 'fake_disk_injection'}
cube_path_end = {False: '', True: '_{0:s}_disk'.format(fake_lum)}
save_dir_end = {False: '', True: '/fake_disk'}

DATA_PATH = '_'.join(x for x in [TARGET, SPHERE_FILTER, EPOCH, INSTRUMENT, path_end[use_fake]])
psf_path = os.path.join(TARGET, DATA_PATH, 'SCIENCE_PSF_MASTER_CUBE-median_unsat')

if INSTRUMENT=='ird':
    PXSCALE=0.01225 ##arcsec/px
else: ##ifs
    PXSCALE=0.00746 ##arcsec/px
mask=round(0.092/PXSCALE)##int radius of mask in px
#mask = 40

method=['RDI-VIP-PCA','RDI-PCA']
norm=['temp-mean_best_frames','spat-mean_best_frames']
"""
-------------------------------------------- FUNCTIONS --------------------------------------------
"""
def DataInitialisation(method, save_path):
    if FILTER != SPHERE_FILTER:
        save_path+= '_stack'
    try:
        data_path = save_path.replace('SNR_map/', '')
        hdr = fits.getheader(os.path.join(SPHERE_DIR,data_path) +'.fits')
        reduced_cube, nodisk_cube = LoadCube(data_path)
    except FileNotFoundError:
        print('{0:s} does not exist, continuing to next method.'.format(data_path +'.fits'))
        return

    try:
        fwhm = hdr['FWHM']
    except:
        fwhm = CalcFWHM(os.path.join(SPHERE_DIR,psf_path)+'.fits')[0]

    if type(fwhm) == str:
        digits = [i for i,x in enumerate(fwhm) if x.isdigit()]
        fwhm = fwhm[min(digits):max(digits)+1].split(' ')
        fwhm = [float(x) for x in fwhm if len(x)>0]

    bands,ncomp=(1,1) ##for loops to still work even if only single band or ncomp image
    if len(reduced_cube.shape)==2:
        x_px,y_px=reduced_cube.shape ##2D reduced image
        fwhm=np.nanmean(fwhm)
    elif len(reduced_cube.shape)==3:
        if method=='CADI':
            bands,x_px,y_px=reduced_cube.shape
        else:
            ncomp,x_px,y_px=reduced_cube.shape ##channel combined data
            fwhm=np.nanmean(fwhm)
    else:
        bands,ncomp,x_px,y_px=reduced_cube.shape

    try:
        if bands>1 and ncomp>1:
            snr_map = np.zeros((bands,ncomp,x_px,y_px))
            for i in range(bands):
                for j in range(ncomp):
                    snr_map[i][j]=SNRMap(reduced_cube[i][j],nodisk_cube[i][j],fwhm[i])
        elif bands>1:
            snr_map = np.zeros((bands,x_px,y_px))
            for i in range(bands):
                snr_map[i]=SNRMap(reduced_cube[i],nodisk_cube[i],fwhm[i])
        elif ncomp>1:
            snr_map = np.zeros((ncomp,x_px,y_px))
            for j in range(ncomp):
                snr_map[j]=SNRMap(reduced_cube[j],nodisk_cube[j],fwhm)
        else:
            snr_map=SNRMap(reduced_cube,nodisk_cube,fwhm)

        hdr['FWHM'] = (str(fwhm).replace('\n',''), 'mean FWHM of star PSF')
        hdu_new = fits.PrimaryHDU(data=snr_map)
        hdu_new.writeto(save_path + '_SNR.fits', overwrite=True)

    except KeyboardInterrupt:
        print("Program interrupted. Saving current reduced data.")
        hdu_new = fits.PrimaryHDU(data=snr_map)
        hdu_new.writeto(save_path + '_SNR_autosave.fits', overwrite=True)


def LoadCube(path):

    reduced_cube=fits.getdata(path + '.fits')
    nodisk_cube=fits.getdata(path + '_neg_parang.fits')

    return reduced_cube, nodisk_cube


def CalcFWHM(psf_path):
    try:
        psf=fits.getdata(psf_path)
        if len(psf.shape)>3:
            if INSTRUMENT=='ifs' and EPOCH=='2015-04-03':
                psf=psf[:,1] ##2015-04-03 1st psf measurement not good
            else:
                psf=np.nanmean(psf,axis=1)

        bands=psf.shape[0]
        fwhm=np.zeros(bands)
        for i in range(bands):
            DF_fit=vip.var.fit_2dgaussian(psf[i], crop=True, cropsize=22, debug=False)
            fwhm[i]=np.mean([DF_fit['fwhm_x'],DF_fit['fwhm_y']])
    except FileNotFoundError:
        fwhm=4 ##typical value
        psf=None

    return fwhm, psf

##calculates the standard deviation and mean of the flux within the annulus
def Noise(image,annul):

    ##calculating mean + stdev NOT using fractional pixel values
    annul_mask=annul.to_mask(method='center')
    annul_data=annul_mask.multiply(image)

    annul_xy=np.where(np.array(annul_mask)>0)
    data=annul_data[annul_xy]

    noise=data.std(ddof=1)
    bg_flux_mean=data.sum()/annul.area

    return noise, bg_flux_mean

##calculates signal to noise ratio by subtracting mean background flux (of no-disk image)
##from reduced image then dividing by noise of no-disk image
def SNRMap(reduced_image,nodisk_image,fwhm):

    x,y = reduced_image.shape
    cxy=(x//2,y//2)

    r_min,r_max=(mask+1, min(cxy)) ##buffer of 1 px to avoid including non-physical parts of the image
    rad=np.arange(r_min,r_max,round(fwhm))
    if rad[-1]<min(cxy)-1:
        rad=np.append(rad,min(cxy)-1)
    r_inner, r_outer=(rad[:-1],rad[1:])

    snr_map=np.zeros_like(reduced_image)

    for i in range(len(rad)-1):
        annul=CircularAnnulus((cxy),r_inner[i],r_outer[i])
        annul_mask=annul.to_mask(method='exact') ##exact mode gives partial pixel values
        annul_mask=annul_mask.to_image((x,y))
        noise, bg_flux = Noise(nodisk_image,annul)
        snr=(reduced_image-bg_flux)/noise
        snr_map+=annul_mask*snr ##mask with fractional pixel values so pixel
                                ##overlap between annuli are accounted for

    return snr_map
#%%

if __name__=='__main__':

    start_time=datetime.now()
    print("Start time:",start_time.isoformat(' ',timespec='seconds'))

    method_tmp=[]
    for m in method:
        if 'PCA' in m:
            for i in range(len(norm)):
                method_tmp.append(m+'_'+norm[i])

    output_dir = os.path.join(SAVE_DIR, TARGET, EPOCH + save_dir_end[use_fake], 'SNR_map')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for mt in method_tmp:
        print("Running process for",mt)
        file_name = INSTRUMENT + '_' + FILTER + cube_path_end[use_fake] + '_' + mt
        if INSTRUMENT == '': file_name = file_name[1:]
        DataInitialisation(mt, os.path.join(output_dir, file_name))

    print("Total run time:", str(datetime.now()-start_time))
