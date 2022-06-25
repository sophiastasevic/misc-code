#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:47:13 2022

@author: stasevis
"""

import numpy as np
import os
import fnmatch
from astropy.io import fits
#from ADI.ADI_reduction import FormatMethods
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from channel_combine import MeasureSNR
import vip_hci as vip
from ADI_reduction import FormatMethods

ds9 = vip.Ds9Window()

def in_data(data_dir,filename):
    path=os.path.join(data_dir,filename)
    data=fits.getdata(path)

    return data

epoch='2015-04-12'
#bands=['Y','J','H','YJH']
#method = ['VIP-PCA','PCA']
#norm = ['temp-mean','spat-mean']
bands=['H']
method = ['PCA','RDI-PCA']
norm = ['temp-mean']

DATA_DIR='/mnt/c/Users/stasevis/Documents/RDI/ADI/output/HD110058/{0:s}/SNR_map/'.format(epoch)
#DATA_DIR_K='/mnt/c/Users/stasevis/Documents/RDI/ADI/output/HD110058/2015-04-03/SNR_map'
SPHERE_DIR='/mnt/c/Users/stasevis/Documents/sphere_data/HD110058'

method_tmp = FormatMethods(method,norm)

r_aper= 16 ##radius of aperture for measuring star PSF flux in px
r_annul=(25,31) ##inner and outer radius of annulus for measuring star psf background flux
params = {'r_min': 30, 'r_max': 60, 'height': 24, 'pa': -68.5, 'sides': 2}

pcs=np.arange(1,11)
#pcs_rdi=(1,5,10,15,20,25,35,50,75,100)

if 'CADI' in method:
    n_maps=len(bands)*((len(method)-1)*len(norm)+1)
else:
    n_maps=len(bands)*len(method)*len(norm)
#%%
files={}
snr={}
for i,band in enumerate(bands):
    files[band]=[]
    snr[band]=[] #np.zeros((len(method_tmp),len(pcs)))

    for mthd in method_tmp:
        file_prefix = '_'.join(x for x in [band, mthd, 'stack*'])
        files[band] += [x for x in fnmatch.filter(os.listdir(DATA_DIR),file_prefix)]

    hdr = fits.getheader(os.path.join(DATA_DIR.replace('SNR_map/',''),files[band][0].replace('_SNR','')))
    fwhm=round(hdr['FWHM'])
    for m in range(len(method_tmp)):
        snr[band].append(MeasureSNR(in_data(DATA_DIR,files[band][m]), fwhm, params, i+m)[1])

#%%
def SinglePlotSNR():
    colors = iter(cm.rainbow(np.linspace(0, 1, len(norm)+1)))

    fig, ax = plt.subplots(1,1,figsize=(10,7))
    lab=['40px PCA mask','8px PCA mask']
    for i,m in enumerate(method_tmp):
        c=next(colors)
        m=m.replace('_',' ')

        ax.plot(pcs,snr['H'][i][0],color=c,label=lab[i])

    ax.set_title('HD110058 2015-04-12 H temp-mean VIP-PCA disk SNR',fontsize=18,loc='left')
    #ax.set_ylim(2.,6.)
    ax.set_ylabel('SNR', fontsize=18)
    ax.set_xlabel('PCs', fontsize=18)
    ax.legend(frameon=False,loc='upper right', fontsize=18)
    ax.grid(True)

    plt.savefig('2015-04-12_H_ird_VIP-PCA_temp-mean_disk_SNR_mask_comp.png')


def PlotSNR():
    colors = iter(cm.rainbow(np.linspace(0, 1, len(method_tmp))))

    fig = plt.figure(1, figsize=(10,14))
    gs = gridspec.GridSpec(4,1, height_ratios=[1,1,1,1], width_ratios=[1])
    gs.update(left=0.1, right=0.95, bottom=0.1, top=0.94, wspace=0.2, hspace=0.2)
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[2,0])
    ax4 = plt.subplot(gs[3,0])

    fig.suptitle("HD110058 epoch combined broad band max disk SNR", fontsize=16,
                 fontweight='bold', x=0.1, horizontalalignment='left')

    for i,m in enumerate(method_tmp):
        c=next(colors)
        if m == 'PCA_temp-mean': c='lime'
        m=m.replace('_',' ')

        ax1.plot(pcs,snr['Y'][i],color=c,label=m)
        ax1.set_title('broad band Y',fontsize=16,loc='left')
        ax1.set_ylim(5.,14.)
        ax1.set_ylabel('SNR', fontsize=16)
        #ax1.legend(frameon=False,loc='upper right', fontsize=14)
        #ax1.set_xlabel('PCs', fontsize=18)

        ax2.plot(pcs,snr['J'][i],color=c,label=m)
        ax2.set_title('broad band J',fontsize=16,loc='left')
        ax2.set_ylim(5.,14.)
        ax2.set_ylabel('SNR', fontsize=16)
        #ax2.legend(frameon=False,loc='lower right', fontsize=18)
        #ax2.set_xlabel('PCs', fontsize=18)

        ax3.plot(pcs,snr['H'][i],color=c,label=m)
        ax3.set_title('broad band H',fontsize=16,loc='left')
        ax3.set_ylim(5.,14.)
        ax3.set_ylabel('SNR', fontsize=16)
        #ax3.legend(frameon=False,loc='lower right', fontsize=14)
        #ax3.set_xlabel('PCs', fontsize=16)

        ax4.plot(pcs,snr['YJH'][i],color=c,label=m)
        ax4.set_title('combined YJH',fontsize=16,loc='left')
        ax4.set_ylim(5.,14.)
        ax4.set_ylabel('SNR', fontsize=16)
        ax4.legend(frameon=False,loc='lower right', fontsize=14)
        ax4.set_xlabel('PCs', fontsize=16)

    for ax in [ax1,ax2,ax3,ax4]:
        ax.grid(True)
        ax.tick_params(labelsize=12)

    plt.savefig(DATA_DIR+'snr_compare.png')

PlotSNR()
#SinglePlotSNR()

#%%

"""

r_aper= 16
r_annul=(25,31)
r_min,r_max=(17,65)
fwhm=4

method=['ssPCA_no_unnorm','ssPCA_sep_unnorm']
norm=['spat-mean','spat-standard','temp-mean','temp-standard']

data_dir='/mnt/c/Users/stasevis/Documents/RDI/ADI/output/HD110058/2015-04-12/norm_test/SNR'

H_files = fnmatch.filter(os.listdir(data_dir),'*H_*')
x = fnmatch.filter(H_files,'*none*')
H_files=[f for f in H_files if f not in x]
no_files = fnmatch.filter(H_files,'*no_*')
sep_files = fnmatch.filter(H_files,'*sep*')

snr = {'no':np.zeros((len(norm),10)),
        'sep':np.zeros((len(norm),10))}

for i in range(len(norm)):
    snr['sep'][i]=get_SNR(in_data(data_dir, sep_files[i]))
    snr['no'][i]=get_SNR(in_data(data_dir, no_files[i]))

pcs=np.arange(1,11)
snr_dif=snr['no']-snr['sep']


colors = iter(cm.rainbow(np.linspace(0, 1, len(norm)+1)))
fig = plt.subplots(figsize=(16,16))
ax1 = plt.subplot2grid((3,2),(0,0),colspan=1,rowspan=1)
ax2 = plt.subplot2grid((3,2),(0,1),colspan=1,rowspan=1)
ax3 = plt.subplot2grid((3,2),(1,0),colspan=2,rowspan=1)
ax4 = plt.subplot2grid((3,2),(2,0),colspan=2,rowspan=1)

for i,n in enumerate(norm):
    c=next(colors)
    ax1.plot(pcs,snr['no'][i],color=c,label=n)
    ax1.set_title('not unnormalised',fontsize=14,loc='left')
    ax1.set_ylim(3.8,6.3)
    ax1.set_xlim(1,10)
    ax1.set_ylabel('SNR', fontsize=14)
    ax1.set_xlabel('PCs', fontsize=14)
    ax1.legend(frameon=False,loc='best', fontsize=14)

    ax2.plot(pcs,snr['sep'][i],color=c,label=n)
    ax2.set_title('unnormalise before subtracting PCs from data',fontsize=14,loc='left')
    ax2.set_ylim(3.7,6.2)
    ax2.set_xlim(1,10)
    ax2.set_ylabel('SNR', fontsize=14)
    ax2.set_xlabel('PCs', fontsize=14)
    #ax2.legend(frameon=False,loc='best', fontsize=14)

    if 'standard' in n:
        ax3.plot(pcs,snr_dif[i],color=c,label=n)
    else:
        ax3.plot(pcs,snr_dif[i],color=c)
    ax3.set_title('not unnormalised - unnormalise after',fontsize=14,loc='center')
    ax3.set_ylim(-.3,1)
    ax3.set_xlim(1,10)
    ax3.set_ylabel('$\Delta$ SNR', fontsize=14)
    ax3.set_xlabel('PCs', fontsize=14)
    ax3.legend(frameon=False,loc='upper left', fontsize=14)

    if 'mean' in n:
        ax4.plot(pcs,snr_dif[i],color=c,label=n)
        ax4.set_title('temp not unnormalised - unnormalise after', fontsize=14,loc='center')
        ax4.set_ylim(-7e-7,7e-7)
        ax4.set_xlim(1,10)
        ax4.set_ylabel('$\Delta$ SNR', fontsize=14)
        ax4.set_xlabel('PCs', fontsize=14)
        ax4.legend(frameon=False,loc='upper left', fontsize=14)

for ax in [ax1,ax2,ax3,ax4]:
    ax.grid(True)
plt.savefig(os.path.join(data_dir,'no_unnormalisation_snr_dif.png'))
"""