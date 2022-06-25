#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Jan 18 00:00:07 2022

@author: stasevis

Stacks wavelength channels of reduced data cubes

Inputs:
    - Reduced cube
    - Channel wavelength list
    - Star PSF
    - (if weightings = 'best' or 'SNR') SNR map for reduced cube

Outputs:
    - Combined broadband image(s)

If combining broadband images from more than one epoch, epoch and instrument
must both be lists of the same size. Currently only works if combined broadband
images for each individual epoch already exist (uses weighted SNR)

'''

from astropy.io import fits
import numpy as np
import os
import cv2
import vip_hci as vip
import photutils
from photutils import CircularAnnulus, CircularAperture
from var_mod import modify_params

TARGET = 'HD110058'
FILTER = 'H23' #['YJ','YJH', 'H23']
epoch = '2015-04-12' #['2015-04-12', '2015-04-03', '2015-04-12']
instrument = 'ird' #['ifs','ifs', 'ird']
stack_band = None

SAVE_DIR = '/mnt/c/Users/stasevis/Documents/RDI/ADI/output'
SPHERE_DIR = '/mnt/c/Users/stasevis/Documents/sphere_data'

use_fake = False
fake_lum = 'faint'

path_end = {False: '_convert_recenter', True: '_fake_disk_injection'}
cube_path_end = {False: '_', True: '_{0:s}_disk_'.format(fake_lum)}
save_dir_end = {False: '', True: '/fake_disk'}

pxscale_dict = {'ird': 12.25, 'ifs': 7.46}
##combining channels in a single data cube
if type(epoch) != list:
    path_tmp = TARGET + '/' + TARGET + '_' + FILTER + '_' + epoch + '_' + instrument
    psf_path = os.path.join(path_tmp + path_end[use_fake], 'SCIENCE_PSF_MASTER_CUBE-median_unsat')
    lambda_path = os.path.join(path_tmp + path_end[False], 'SCIENCE_LAMBDA_INFO-lam')

    pxscale = pxscale_dict[instrument]

##combining more than one data cube from different epochs
else:
    psf_path=[]
    lambda_path=[]
    pxscale=[]
    for i in range(len(epoch)):
        path_tmp = TARGET + '/' + TARGET + '_' + FILTER[i] + '_' + epoch[i] + '_' + instrument[i]
        psf_path.append(os.path.join(path_tmp + path_end[use_fake], 'SCIENCE_PSF_MASTER_CUBE-median_unsat'))
        lambda_path.append(os.path.join(path_tmp + path_end[False], 'SCIENCE_LAMBDA_INFO-lam'))

        pxscale.append(pxscale_dict[instrument[i]])

ird_to_ifs_pxscale = pxscale_dict['ird']/pxscale_dict['ifs']
ifs_to_ird_pxscale = 1/ird_to_ifs_pxscale

px_resize = lambda px, scale: px*pxscale_dict['ifs']/scale

size = 256
r_aper = 18 ##radius of aperture for measuring star PSF flux in px
r_annul = (22,30) ##inner and outer radius of annulus for measuring star psf background flux
params = {'r_min': 30, 'r_max': 70, 'height': 26, 'pa': -67.5, 'sides': 2}

method = ['RDI-VIP-PCA','RDI-PCA'] ##reduction method of input cube
norm = ['temp-mean_best_frames','spat-mean_best_frames']  ##normalisation method of input cube
weighting = ['best'] ##method of channel selection and weighting to use: 'flux', 'SNR' or 'best'

ds9 = vip.Ds9Window()

#%%
def FileIn(dir_path, file_path=None):
    if file_path is not None:
        path=os.path.join(dir_path,file_path)
    else:
        path = dir_path
    data=fits.getdata(path + '.fits')

    return data


## if combining channels of single data cube, gets cube header, otherwise creates new header
def AddHeader(method, params, fwhm, pxscale, band=FILTER, path=None, pcs=None):
    if path != None:
        hdr = fits.getheader(path+'.fits')
    else:
        hdr = fits.Header()
        hdr['Instr'] = str(instrument)
        hdr['Filter'] = band
        hdr['Object'] = TARGET
        hdr['Epoch'] = str(epoch)
        if '_' in method:
            method = method.replace('_',' ')
        hdr['Reductn'] = method
        hdr['PXSCALE'] = pxscale
        if pcs is None:
            pcs = str(np.arange(1,11))
        hdr['PCS'] = pcs

    hdr['FWHM'] = (fwhm, 'mean FWHM of star PSF')
    hdr['RMIN_SNR'] = (params['r_min'],'inner of SNR measurement area')
    hdr['RMAX_SNR'] = (params['r_max'],'outer radius of SNR measurement area')
    hdr['H_SNR'] = (params['height'],'height of slice used for SNR measurement when disk is horizontal')
    hdr['PA'] = (params['pa'],'disk PA used to align to horizontal')

    return hdr


## read in psf + calculate fwhm of each psf channel
def CalcFWHM(psf_path):
    psf = FileIn(psf_path)
    if len(psf.shape) > 3:
        if 'ifs' in psf_path and '2015-04-03' in psf_path:
            psf = psf[:,1]
        else:
            psf = np.nanmean(psf,axis = 1)

    bands=psf.shape[0]
    fwhm=np.zeros(bands)
    for i in range(bands):
        DF_fit=vip.var.fit_2dgaussian(psf[i], crop=True, cropsize=22, debug=False)
        fwhm[i]=np.mean([DF_fit['fwhm_x'],DF_fit['fwhm_y']])

    return psf, fwhm


def MeasureFlux(psf, aper, annul_bg):
    apers = (aper,annul_bg)
    flux = photutils.aperture_photometry(psf,apers,method = 'exact')
    bkg_mean = flux['aperture_sum_1']/annul_bg.area
    flux_mean = flux['aperture_sum_0']/aper.area-bkg_mean

    return flux_mean


## calculates weighting of each channel based on the flux of the star psf
def WeightedFlux(channels, psf, pxscale, return_flux=False):
    cx,cy = (psf.shape[1]//2,psf.shape[2]//2)
    fluxes = np.zeros(channels)
    aper = CircularAperture((cx,cy), int(px_resize(r_aper,pxscale)))
    annul_bg = CircularAnnulus((cx,cy), int(px_resize(r_annul[0],pxscale)),
                                        int(px_resize(r_annul[1],pxscale)))

    for i in range(channels):
        fluxes[i] = MeasureFlux(psf[i],aper,annul_bg)

    w_flux = fluxes/fluxes.sum()

    if return_flux == True:
        return fluxes
    else:
        return w_flux


## measures max SNR of an annulus
def MeasureAnnulSNR(image, fwhm, r_min, r_max):
    cx,cy=(image.shape[0]//2,image.shape[1]//2)
    rad = np.arange(r_min, r_max, fwhm)
    r_inner,r_outer = (rad[:-1], rad[1:])
    max_snr = 0

    for n in range(len(r_inner)):
        annul = CircularAnnulus((cx-0.5,cy-0.5), r_inner[n], r_outer[n])
        annul_mask = annul.to_mask(method='center')
        annul_data = annul_mask.multiply(image)

        max_snr+= np.max(annul_data)
    mean_max_snr = max_snr/len(r_inner)

    return mean_max_snr


## measures max SNR of bins in specified region
def MeasureSNR(cube, fwhm, var, loop, example_cube = None):
    cx,cy = (cube.shape[-2]//2, cube.shape[-1]//2)
    if len(cube.shape)>2: pcs = cube.shape[0]
    else: pcs = 1

    ## allowing for parameters of measurement region to be modified after seeing image
    fin_params = False
    while fin_params == False:

        r_min, r_max, height = int(var['r_min']), int(var['r_max']), int(var['height'])
        r_size = r_max - r_min
        pa = float(var['pa'])
        if height%2: height+=1; var['height'] = int(height)

        rot_cube = vip.preproc.cube_derotate(cube, np.ones(pcs)*pa, imlib='opencv')

        if loop > 0:
            fin_params = True
        else:
            if example_cube is None:
                ds9.display(rot_cube)
            else:
                example_rot_cube = vip.preproc.cube_derotate(example_cube, np.ones(example_cube.shape[0])*pa, imlib='opencv')
                ds9.display(example_rot_cube)
            ds9.set("regions command {{box {0:d} {1:d} {2:d} {3:d}}}".format(cx-r_min-r_size//2 +1, cy+1, r_size, height))
            ds9.set("regions command {{box {0:d} {1:d} {2:d} {3:d}}}".format(cx+r_min+r_size//2 +1, cy+1, r_size, height))

            var, fin_params = modify_params(var)

    separation = np.arange(r_min, r_max, fwhm)
    nb_prof = len(separation)
    prof = np.zeros((pcs, var['sides'], height, nb_prof))
    disk_snr=np.zeros((pcs, var['sides'], nb_prof))

    if fwhm % 2: buffer = 1
    else: buffer = 0

    vmin = cy - height//2
    vmax = cy + height//2
    for sepi, sep in enumerate(separation):
        hmin = {0: cx - sep - fwhm//2, 1: cx + sep - fwhm//2}
        hmax = {0: cx - sep + fwhm//2 + buffer, 1: cx + sep + fwhm//2 + buffer}

        for sidei in range(var['sides']):
            if pcs == 1:
                prof[:,sidei,:,sepi] = np.mean(rot_cube[vmin:vmax,hmin[sidei]:hmax[sidei]], axis=1)
            else:
                prof[:,sidei,:,sepi] = np.mean(rot_cube[:,vmin:vmax,hmin[sidei]:hmax[sidei]], axis=2)

            for p in range(pcs):
                disk_snr[p,sidei,sepi]=max(prof[p,sidei,:,sepi])

    mean_snr = disk_snr.sum(axis=(1,2))/nb_prof

    return mean_snr, disk_snr


## calculates weighting of each channel based on its disk SNR in the reduced image
def WeightedSNR(channels, snr_map, params, fwhm, loop):
    if len(snr_map.shape)>3:
        pcs = snr_map.shape[1]
    else:
        pcs = 1

    mean_snr = np.zeros(channels)
    for i in range(channels):
        mean_snr[i] = np.sum(MeasureSNR(snr_map[i], round(fwhm[i]), params, loop+i, snr_map[:,0])[0])/pcs

    w_snr = mean_snr/mean_snr.sum()

    return w_snr


## if weighting = best: removes channels bellow a certain SNR and flux threshold
## based on the max weigthed SNR and flux + recalculates weightings for remaining channels
def RemoveChannel(cube, neg_cube, channels, w_snr, w_flux):
    keep_frames = BadFrames(cube,channels)

    if w_flux is not None:
        keep_flux = np.where(w_flux>=(max(w_flux[keep_frames])/2))[0]  ##adjust threshold after flux scaling for epochs
        keep_frames = np.array(list(set(keep_frames)&set(keep_flux)))

    if w_snr is not None:
        keep_snr = np.where(w_snr>=(max(w_snr[keep_frames])/1.5))[0] ##max so changing keep_frames doesn't affect next threshold
        keep_frames = np.array(list(set(keep_frames)&set(keep_snr)))

    new_channels = len(keep_frames)

    if new_channels == channels:
        return cube, neg_cube, w_snr, w_flux, channels

    else:
        new_cube = cube[keep_frames]
        new_negcube = neg_cube[keep_frames]
        new_snr,new_flux = (None,None)

        if w_snr is not None:
            new_snr = w_snr[keep_frames]*w_snr.sum()/w_snr[keep_frames].sum()

        if w_flux is not None:
            new_flux = w_flux[keep_frames]*w_flux.sum()/w_flux[keep_frames].sum()

    return new_cube, new_negcube, new_snr, new_flux, new_channels


## finds max flux of each channel in reduced cube + returns index array of channels to keep
## based on the previous and next frame to account for cubes with data from different epochs
def BadFrames(cube, channels):
    frame_max = np.zeros(channels)
    for i in range(channels):
        frame_max[i] = np.nanmax(cube[i])

    a = np.where(frame_max<(5*np.roll(frame_max,1)))[0]
    b = np.where(frame_max<(5*np.roll(frame_max,-1)))[0]
    good_frames = np.array(list(set(a)|set(b)))

    return good_frames


## removes bad channels, multiplies each channel by its weighting + sums them
def StackFrames(cube, neg_cube, channels, save_path, hdr, w_snr=None, w_flux=None):
    cube, neg_cube, w_snr, w_flux, channels = RemoveChannel(cube, neg_cube, channels, w_snr, w_flux)

    if w_snr is None:
        weights = w_flux
    else:
        weights = w_snr

    combined_cube = np.zeros(cube.shape[1:])
    combined_noise = np.zeros_like(combined_cube)

    for i in range(channels):
        combined_cube+= cube[i]*weights[i]
        combined_noise+= neg_cube[i]*weights[i]

    hdu_new = fits.PrimaryHDU(data=combined_cube,header=hdr)
    hdu_new.writeto(save_path + '_stack.fits', overwrite=True)

    hdr['Background'] = 'cube reduced using -ve parang'
    hdu_neg = fits.PrimaryHDU(data=combined_noise,header=hdr)
    hdu_neg.writeto(save_path + '_stack_neg_parang.fits', overwrite=True)

    print('Stacked image saved as',save_path.split('/')[-1]+'_stack.fits')

    return combined_cube


def Combine(cube, neg_cube, channels, save_path, hdr, weighting, w_snr, w_flux):
    if 'SNR' in weighting:
        combined_cube = StackFrames(cube, neg_cube, channels, save_path+'_snr_weighted', hdr, w_snr=w_snr)
    if 'flux' in weighting:
        combined_cube = StackFrames(cube, neg_cube, channels, save_path+'_flux_weighted', hdr, w_flux=w_flux)
    if 'best' in weighting:
        combined_cube = StackFrames(cube, neg_cube, channels, save_path, hdr, w_snr=w_snr, w_flux=w_flux)

    return combined_cube


## scale ird images when combining with ifs images in order to match mas/px scale of instruments
def ScaleImage(cube, pxscale):
    ##TODO: could run spine fitting for each interpolation option to test
    if pxscale == pxscale_dict['ird']: scale = ird_to_ifs_pxscale; interp = cv2.INTER_AREA
    elif pxscale == pxscale_dict['ifs']: scale = ifs_to_ird_pxscale; interp = cv2.INTER_LANCZOS4

    #cube_flux = np.sum(cube)
    channels, pcs, x, y = cube.shape
    scaled_x, scaled_y = round(scale*x), round(scale*y)
    scaled_cube = np.zeros((channels,pcs,scaled_x,scaled_y))
    buffer = abs(np.min(cube))
    flux_scale = scale**2

    for i in range(channels):
        for j in range(pcs):
            scaled_cube[i,j] = cv2.resize(cube[i,j]+buffer, (0,0), fx=scale, fy=scale,
                                          interpolation=interp) - buffer ##TODO: scale before subtracting
    scaled_cube/= flux_scale
    #scaled_cube_flux = np.sum(scaled_cube)
    #print(cube_flux, scaled_cube_flux)

    if scaled_x > x:
        if (x+scaled_x)%2: scaled_x+=1
        if (y+scaled_y)%2: scaled_y+=1

        xmin, xmax, ymin, ymax = (scaled_x-x)//2, (scaled_x+x)//2, (scaled_y-y)//2, (scaled_y+y)//2
        resize_cube = scaled_cube[:,:,xmin:xmax,ymin:ymax]

    elif scaled_x < x:
        if (size+scaled_x)%2: scaled_x-=1; scaled_cube = scaled_cube[...,1:,:]
        if (size+scaled_y)%2: scaled_y-=1; scaled_cube = scaled_cube[...,1:]

        resize_cube = np.zeros((channels, pcs, size, size))
        xpad, ypad = (size-scaled_x)//2, (size-scaled_y)//2
        for i in range(channels):
            for j in range(pcs):
                resize_cube[i,j] = np.pad(scaled_cube[i,j], ((xpad,),(ypad,)))

    return resize_cube


def GetBands(filters, stack_band=None):
    if stack_band is None:
        elements = {} ##only concidering bands in more than one data cube
        for char in [x for x in ''.join(filters) if not x.isdigit()]:
            if elements.get(char,None) != None:
                elements[char]+=1
            else:
                elements[char] = 1
        broad_bands = [k for k,v in elements.items() if v>1]

        if 'K12' in filters:
            broad_bands += 'K'

        if instrument.count(instrument[0]) == len(instrument):
            broad_bands = [k for k,v in elements.items() if v>=1]
        return broad_bands

    else:
        return stack_band

#%%

## combine broad bnd frames from different epochs
def EpochCombine(methods, weighting, wl_index, params):
##TODO: check for consistant PCs + save PC list in header
#%%
    nfiles = len(epoch)
    broad_bands = GetBands(FILTER,stack_band)

    scale_to_inst = 'ifs' ##instrument pxscale that data is scaled to
    flag_inst = ['ifs' if scale_to_inst == 'ird' else 'ird'][0]

    avg_flux = np.zeros(nfiles)
    avg_band_flux = np.full((nfiles,3),np.nan)

    wavelengths = []
    psf = []
    fwhm = []

    for i in range(nfiles):
        wavelengths.append(FileIn(SPHERE_DIR,lambda_path[i]))
        psf_tmp, fwhm_tmp = CalcFWHM(os.path.join(SPHERE_DIR,psf_path[i]))

        if instrument.count(instrument[i]) != len(instrument) and instrument[i] == flag_inst:
            channels, psf_x, psf_y = psf_tmp.shape
            psf_reshape = np.zeros((channels, 1, psf_x, psf_y))
            psf_reshape[:,0,:,:] = psf_tmp
            psf_tmp = ScaleImage(psf_reshape, pxscale[i])[:,0]

        psf.append(psf_tmp)
        fwhm.append(px_resize(fwhm_tmp,pxscale[i]))

        flux = WeightedFlux(psf_tmp.shape[0], psf_tmp, pxscale_dict[scale_to_inst], return_flux=True)
        avg_flux[i] = flux.mean()
        for n,band in enumerate(['H','YJ','K']):
            try:
                avg_band_flux[i][n] = flux[wl_index[band](wavelengths[i])].mean()
            except IndexError:
                next
#%%
    wavelengths = np.concatenate(wavelengths)
    channels = len(wavelengths)
    fwhm = np.concatenate(fwhm)

    elig_data = [i for i,x in enumerate(FILTER) if x=='YJH'] ##data sets with YJH channels
    if len(elig_data) > 0:
        max_band = elig_data[np.argmax(np.nansum(avg_band_flux,axis=1)[elig_data])]

        scale_epochs = np.nanmean(avg_band_flux[max_band]/avg_band_flux, axis=1)
        scale_epochs[elig_data] = avg_flux[max_band]/avg_flux[elig_data]
    else:
        scale_epochs = max(avg_flux)/avg_flux
#%%
    for i in range(nfiles):
        psf[i]*= scale_epochs[i]

    psf = np.concatenate(psf)
    w_flux = WeightedFlux(channels, psf, pxscale_dict[scale_to_inst])
#%%
    for loop,mthd in enumerate(methods):
#%%
        cube = []
        neg_cube = []
        snr_map = []

        for i in range(nfiles):
            data_path = os.path.join(TARGET, epoch[i] + save_dir_end[use_fake], instrument[i] + '_' +
                                     FILTER[i] + cube_path_end[use_fake] + mthd)

            cube_tmp = FileIn(SAVE_DIR,data_path) #*scale_epochs[i]
            neg_cube_tmp = FileIn(SAVE_DIR,data_path+'_neg_parang') #*scale_epochs[i]

            #pcs = fits.getheader(os.path.join(SAVE_DIR,data_path+'.fits'))['PCS']
            print('{0:s} data cube loaded'.format(data_path.split('/')[-1]))

            if instrument.count(instrument[i]) != len(instrument) and instrument[i] == flag_inst:
                cube_tmp = ScaleImage(cube_tmp, pxscale[i])
                neg_cube_tmp = ScaleImage(neg_cube_tmp, pxscale[i])

                print('> frames rescaled from {0:.2f} px/mas to {1:.2f} px/mas'
                      .format(pxscale_dict[instrument[i]], pxscale_dict[scale_to_inst]))

            cube.append(cube_tmp)
            neg_cube.append(neg_cube_tmp)
#%%
        cube = np.concatenate(cube)
        neg_cube = np.concatenate(neg_cube)

        save_path = os.path.join(SAVE_DIR, TARGET, 'broad_band' + save_dir_end[use_fake],
                                 'X' + cube_path_end[use_fake] + mthd)
        if instrument.count(instrument[0]) == len(instrument):
            save_path = save_path.replace('X', 'X_{0:s}'.format(instrument[0]))
#%%
        try:
            for i in range(nfiles):
                snr_path = os.path.join(TARGET, epoch[i] + save_dir_end[use_fake],'SNR_map',
                                        instrument[i] + '_' + FILTER[i] + cube_path_end[use_fake] +
                                        mthd + '_SNR')

                snr_map_tmp = FileIn(SAVE_DIR,snr_path)

                if instrument.count(instrument[i]) != len(instrument) and instrument[i] == 'ird':
                    snr_map_tmp = ScaleImage(snr_map_tmp, pxscale[i])

                snr_map.append(snr_map_tmp)

            snr_map = np.concatenate(snr_map)
            w_snr = WeightedSNR(channels, snr_map, params, fwhm, loop)

        except FileNotFoundError:
            print('No SNR map available, using flux weighting to stack.')
            weighting = ['flux']
            w_snr = np.ones_like(w_flux)

        for band in broad_bands:
            save_path_tmp = save_path.replace('X',band)
            band_frames = wl_index[band](wavelengths)

            w_snr_tmp = w_snr[band_frames]*w_snr.sum()/w_snr[band_frames].sum()
            w_flux_tmp = w_flux[band_frames]*w_flux.sum()/w_flux[band_frames].sum()

            hdr = AddHeader(mthd, params, fwhm[band_frames].mean(), pxscale_dict[scale_to_inst], band=band)
            combined_cube = Combine(cube[band_frames], neg_cube[band_frames], len(band_frames),
                                    save_path_tmp, hdr, weighting, w_snr_tmp, w_flux_tmp)
        if len(broad_bands)>1:
            save_path_tmp = save_path.replace('X',''.join(broad_bands))
            hdr = AddHeader(mthd, params, fwhm.mean(), pxscale_dict[scale_to_inst], band=''.join(broad_bands))
            combined_cube = Combine(cube, neg_cube, channels, save_path_tmp, hdr, weighting, w_snr, w_flux)

        print('\n')

    return combined_cube


## combine frames of different channels within a data cube
def ChannelCombine(methods, weighting, wl_index, params):
    psf, fwhm = CalcFWHM(os.path.join(SPHERE_DIR,psf_path))

    channels = psf.shape[0]
    w_flux = WeightedFlux(channels, psf, pxscale)
    broad_bands = [x for x in FILTER if not x.isdigit()]

    wavelengths = FileIn(SPHERE_DIR,lambda_path)

    for loop,method in enumerate(methods):
        data_path = os.path.join(TARGET, epoch + save_dir_end[use_fake], instrument +
                                 '_' + FILTER + cube_path_end[use_fake] + method)

        cube = FileIn(SAVE_DIR,data_path)
        neg_cube = FileIn(SAVE_DIR,data_path +'_neg_parang')

        save_path = os.path.join(SAVE_DIR, data_path)

        try:
            snr_path = os.path.join(TARGET, epoch + save_dir_end[use_fake], 'SNR_map', instrument +
                                    '_' + FILTER + cube_path_end[use_fake] + method + '_SNR')
            snr_map = FileIn(SAVE_DIR, snr_path)
            w_snr = WeightedSNR(channels, snr_map, params, fwhm, loop)

        except FileNotFoundError:
            print('No SNR map available, using flux weighting to stack.')
            weighting = ['flux']
            w_snr = np.ones_like(w_flux)

        for band in broad_bands:
            save_path_tmp = save_path.replace(FILTER,band)
            band_frames = wl_index[band](wavelengths)
            w_snr_tmp = w_snr[band_frames]*w_snr.sum()/w_snr[band_frames].sum()
            w_flux_tmp = w_flux[band_frames]*w_flux.sum()/w_flux[band_frames].sum()

            hdr = AddHeader(method, params, fwhm[band_frames].mean(), pxscale, path=os.path.join(SAVE_DIR,data_path))
            combined_cube = Combine(cube[band_frames], neg_cube[band_frames], len(band_frames),
                                    save_path_tmp, hdr, weighting, w_snr_tmp, w_flux_tmp)
        if len(broad_bands)>1:
            hdr = AddHeader(method, params, fwhm.mean(), pxscale, path=os.path.join(SAVE_DIR,data_path))
            combined_cube = Combine(cube, neg_cube, channels, save_path, hdr, weighting, w_snr, w_flux)

    return combined_cube

#%%
if __name__=='__main__':
#%%
    wl_index = {'Y': lambda wl: np.where(wl<1.11)[0],
                'J': lambda wl: np.array(list(set(np.where(wl>1.14)[0])&
                                              set(np.where(wl<1.35)[0]))),
                'H': lambda wl: np.array(list(set(np.where(wl>1.45)[0])&
                                              set(np.where(wl<1.78)[0]))),
                'K': lambda wl: np.where(wl>1.99)[0],
                'YJ': lambda wl: np.array(list(set(np.where(wl<=1.45)[0])|
                                               set(np.where(wl>=1.78)[0])))}
    methods = []
    for m in method:
        if 'PCA' in m:
            for n in norm:
                methods.append(m+'_'+n)
#%%
    if type(epoch) != list:
        combined_cube = ChannelCombine(methods, weighting, wl_index, params)
    else:
        combined_cube = EpochCombine(methods, weighting, wl_index, params)
