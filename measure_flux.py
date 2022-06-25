#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 23:56:03 2022

@author: stasevis
measure disk flux for reflectance spectra
"""
import numpy as np
import os
import math
import fnmatch
import vip_hci as vip
from astropy.io import fits
from ADI_reduction import FormatMethods
from var_mod import modify_params
import photutils
from photutils import CircularAnnulus, CircularAperture
import matplotlib.pyplot as plt
import matplotlib.cm as cm


EPOCH = '2015-04-12'
TARGET = 'HD110058'
IFS_BAND = {'2015-04-03': 'YJH', '2015-04-12': 'YJ'}

SPHERE_DIR = '/mnt/c/Users/stasevis/Documents/sphere_data'
SAVE_DIR = '/mnt/c/Users/stasevis/Documents/RDI/ADI/output'
USE_FAKE = False
fake_lum = 'faint'

PATH_END = {False: '_convert_recenter', True: '_{0:s}_disk_'.format(fake_lum)}
DIR_END = {False: '', True: 'fake_disk'}

DATA_DIR = os.path.join(SAVE_DIR, TARGET, EPOCH, DIR_END[USE_FAKE])

PATH_TMP = TARGET + '/' + TARGET + '_' + IFS_BAND[EPOCH] + '_' + EPOCH + '_ifs'
lambda_path = os.path.join(PATH_TMP + PATH_END[False], 'SCIENCE_LAMBDA_INFO-lam.fits')
psf_path = os.path.join(PATH_TMP + PATH_END[False], 'SCIENCE_PSF_MASTER_CUBE-median_unsat.fits')

pxscale_dict = {'ird': 12.25, 'ifs': 7.46} ##mas/px ##12.2340 and 12.2210 for K12 H23 and
px_resize = lambda px,scale: int(px*pxscale_dict['ifs']/scale)

method = ['PCA']
norm = ['spat-mean_best_frames']

var = {'r_min': 36, 'r_max': 56, 'height': 6} #, 'pa': -66.5, 'sides': 2}
pa, sides = -66.5, 2

ds9 = vip.Ds9Window()

def in_data(data_dir,filename):
    path=os.path.join(data_dir,filename)
    data=fits.getdata(path)

    return data


## calculates weighting of each channel based on the flux of the star psf
def PSFFlux(psf):
    r_aper = 22 ##radius of aperture for measuring star PSF flux in px
    r_annul = (30,32)

    if len(psf.shape)>3:
        if EPOCH=='2015-04-03':
            psf = psf[:,1] ##1st psf measurement for these data not good
        else:
            psf = np.nanmean(psf,axis=1)

    channels, x, y = psf.shape
    cx,cy = x//2, y//2
    fluxes = np.zeros(channels)
    aper = CircularAperture((cx,cy), r_aper)
    annul_bg = CircularAnnulus((cx,cy), r_annul[0], r_annul[1])
    for i in range(channels):
        apers = (aper,annul_bg)
        flux = photutils.aperture_photometry(psf[i],apers,method = 'exact')
        bkg_mean = flux['aperture_sum_1']/annul_bg.area
        fluxes[i] = flux['aperture_sum_0']/aper.area-bkg_mean

    return fluxes



def MeanFlux(image, bg_image, r_min, r_max, r_size, height):
    channels,pcs,x,y = image.shape
    cx,cy = x//2, y//2

    ymin, ymax = cy-height//2, cy+height//2
    xmin, xmax = (cx-r_max, cx+r_min), (cx-r_min, cx+r_max)

    npa = round((2*math.pi*r_min)/height)
    err_pa = np.arange(0,360,360/npa)

    flux = np.zeros((channels, pcs, sides))
    flux_bg = np.zeros((channels, pcs, sides, npa))
    for pci in range(pcs):
        rot_image = vip.preproc.cube_derotate(image[:,pci], np.ones(channels)*pa, imlib='opencv')
        rot_bg_image = vip.preproc.cube_derotate(bg_image[:,pci], np.ones(channels)*pa, imlib='opencv')

        for wli in range(channels):
            for sidei in range(sides):
                profile = rot_image[wli,ymin:ymax,xmin[sidei]:xmax[sidei]] - rot_bg_image[wli,ymin:ymax,xmin[sidei]:xmax[sidei]]
                flux[wli,pci,sidei] = np.sum(profile)/(height*r_size)

            for pai in range(npa):
                rot_bg = vip.preproc.frame_rotate(bg_image[wli,pci], err_pa[pai], imlib='opencv')
                bg_flux = np.sum(rot_bg[ymin:ymax,xmin[0]:xmax[0]])/(height*r_size)
                flux_bg[wli,pci,:,pai] = bg_flux, bg_flux

    return flux, flux_bg


def MeasureFlux(image, bg_image, instrument, var, loop):
    channels,pcs,x,y = image.shape
    cx,cy = x//2, y//2

    pxscale = pxscale_dict[instrument]
    fin_params = False
    while fin_params == False:
        height = var['height']
        if height%2: height+=1; var['height'] = int(height)
        height = px_resize(height,pxscale)
        r_min = px_resize(var['r_min'],pxscale)
        r_max = px_resize(var['r_max'],pxscale)
        r_size = r_max - r_min

        example_rot_cube = vip.preproc.cube_derotate(image[:,0], np.ones(channels)*pa, imlib='opencv')
        ds9.display(example_rot_cube)
        ds9.set("regions command {{box {0:d} {1:d} {2:d} {3:d}}}".format(cx-r_min-r_size//2 +1, cy+1, r_size, height))
        ds9.set("regions command {{box {0:d} {1:d} {2:d} {3:d}}}".format(cx+r_min+r_size//2 +1, cy+1, r_size, height))


        if loop == 0: var, fin_params = modify_params(var)
        else: fin_params = True

    flux, flux_bg = MeanFlux(image, bg_image, r_min, r_max, r_size, height)
    if instrument == 'ird': flux/= (12.25/7.46); flux_bg/= (12.25/7.46)

    return flux, flux_bg, var
#%%
wavelengths = in_data(SPHERE_DIR,lambda_path)
psf = in_data(SPHERE_DIR,psf_path)
psf_flux = PSFFlux(psf)
methods = FormatMethods(method, norm)

files = []
for m in methods:
    files+=[x for x in fnmatch.filter(os.listdir(DATA_DIR),
            '*_{0:s}.fits'.format(m)) if 'outer' not in x and 'bigger' not in x]
ifs_files = [[i,x] for i,x in enumerate(files) if 'ifs' in x]
ird_files = np.array([i for i,x in enumerate(files) if 'ird' in x], dtype=int)

nfiles = len(files)
instruments = [x.split('_')[0] for x in files]
bands = [x.split('_')[1] for x in files]

fluxes = []
fluxes_bg = []
for filei, f in enumerate(files):
    image = in_data(DATA_DIR,f)
    bg_image = in_data(DATA_DIR,f.replace('.','_neg_parang.'))
    measured_flux, measured_error, var = MeasureFlux(image, bg_image, instruments[filei], var, filei)
    fluxes.append(measured_flux)
    fluxes_bg.append(measured_error)
#%%
def Plot(flux, flux_bg, wavelengths, path):
    colors = iter(cm.rainbow(np.linspace(0, 1, 2)))
    markers = ['o','^']
    lines = ['-','--']
    fig, ax = plt.subplots(1,1,figsize=(20,14))

    ax.set_title('{0:s} {1:s} IFS {2:s} reflectance spectrum'.format(TARGET, EPOCH, IFS_BAND[EPOCH]), y=1.05, fontsize=32)
    ax.set_ylabel('Normalised Flux', fontsize=32)
    ax.set_xlabel('Wavelength [\u03bcm]', fontsize=32)

    pc=5; pci=pc-1
    wl0 = np.where(wavelengths == min(abs(wavelengths-1))+1)[0]

    channels, pcs, sides, npa = flux_bg.shape
    norm_flux = np.zeros_like(flux)
    norm_flux_bg = np.zeros_like(flux_bg)
    for i in range(channels):
        norm_flux[i,:,:] = flux[i,pci,:]/psf_flux[i]
        norm_flux_bg[i,:,:] = flux_bg[i,pci,:]/psf_flux[i]

    normalisation = norm_flux[wl0,pci,0] #SE flux at 1 micron
    norm_flux/=normalisation
    norm_flux_bg/=normalisation

    for sidei, side in enumerate(['SE','NW']):
        c=next(colors)
        m=markers[sidei]
        l=lines[sidei]

        ax.errorbar(wavelengths, norm_flux[:,pci,sidei], np.std(norm_flux_bg[:,pci,sidei,:]), color=c, marker=m, ls=l, lw=3, capsize=6, elinewidth=2.5, label=side)

    #ax.set_xticks(np.arange(1,11))
    ax.tick_params(labelsize=28)
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0 - box.height * 0.03, box.width, box.height * 0.97])
    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(flip(handles, 4), flip(labels, 4), frameon=True, loc='lower center', bbox_to_anchor=(0.5,0.97), borderaxespad=0, ncol=4, fontsize=14)
    ax.legend(frameon=True, loc='best', fontsize=28)


    ax.grid(True)

    plt.savefig(path.replace('.fits','_pc{0:d}_ref_spec.png'.format(pc)))

for i,file in ifs_files:
    Plot(fluxes[i], fluxes_bg[i], wavelengths, file)