#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:00:22 2017

@author: jmilli
"""

import os
#from pathlib import Path
from astropy.io import fits
import numpy as np
import vip_hci as vip
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import profile_fit as pfit
import position_angle as fpa
from scipy.ndimage import gaussian_filter
# from matplotlib import colors as mcolors
# from sklearn import linear_model
# from scipy import polyfit,polyval
#from numpy.polynomial.polynomial import polyfit

ds9 = vip.Ds9Window()
#%%
TARGET_EPOCH="2015-04-12"
TARGET_NAME="HD110058"
INSTRUMENT="ird"
FILTER="H"

root_path = '/mnt/c/Users/stasevis/Documents'
data_path = os.path.join(root_path,'RDI/ADI/output',TARGET_NAME,TARGET_EPOCH)
if INSTRUMENT =="" and FILTER =="":
    file_prefix = ''
elif INSTRUMENT =="":
    file_prefix = FILTER
elif FILTER =="":
    file_prefix = INSTRUMENT + '_'
else:
    file_prefix = INSTRUMENT + '_' + FILTER

size = 180
pxscale_dict = {'ird': 12.25, 'ifs': 7.46, '': 7.46} ##mas/px
pxscale = pxscale_dict[INSTRUMENT]

px_resize = lambda px: int(px*pxscale_dict['ifs']/pxscale)

dstar = 1/0.0076932

iterate_pa = False
fit_sides_indv = False

fake = ['']
bands = [] #'Y','J','H','YJH']
methods = ['RDI-VIP-PCA','RDI-PCA']#,'high_pass_filter_VIP-PCA']
pcs = [35,50,75,100]
norms = ['temp-mean','spat-mean']
##TODO: try with 55.3 (isophot)
pa = [-67]
nb_m, nb_n, nb_pc, nb_b = len(methods), len(norms), len(pcs), len(bands)

pci = lambda index: index - nb_pc*(index//nb_pc)
mthdi = lambda index: index//(nb_n*nb_pc) - nb_m*(index//(nb_m*nb_n*nb_pc))

norm_tmp, mthd_tmp, pc_tmp, fake_tmp, band_tmp = ('','','','','')
if nb_b == 1: band_tmp = bands[0]
if nb_n == 1: norm_tmp = '_' + norms[0]
if nb_m == 1: mthd_tmp = '_' + methods[0]
if nb_pc == 1: pc_tmp = '_pc{0:d}'.format(pcs[0])
if not all(x == '' for x in fake):
    fake_type = '_'.join('{0:s}'.format(x) for x in sorted(dict.fromkeys(fake)) if x != '')
    fake_tmp='_{0:s}_fake'.format(fake_type)

#details that apply to all algo
save_name = file_prefix + '{b}{f}{m}{n}{p}'.format(b=band_tmp, f=fake_tmp, m=mthd_tmp, n=norm_tmp, p=pc_tmp)

"""
-------------------------------------------- FUNCTIONS --------------------------------------------
"""
def FormatDescription(methods, pcs, norms, pa, fake, band):
    method_name=[]
    for i in range(len(methods)):
        if 'PCA' in methods[i]:
            method_name.append(methods[i].replace('-',' '))
        else:
            method_name.append(methods[i])

    p_tmp,n_tmp,m_tmp,f_tmp=('','','','')
    if len(pcs)==1:
        p_tmp='PC={p} '.format(p=pcs[0])
    if len(norms)==1:
        n_tmp='{n} '.format(n=norms[0])
    if len(methods)==1 and 'convolved' not in methods[0]:
        m_tmp='{m} '.format(m=method_name[0])
    if len(fake)==1 and fake[0]!='':
        f_tmp='Fake {0:s} disk '.format(fake[0])

    algo_master='{f}{n}{m}{p}'.format(f=f_tmp,n=n_tmp,m=m_tmp,p=p_tmp)
    if len(algo_master)>0 and algo_master[-1]==' ': #remove trailing spaces
        algo_master=algo_master[:-1].replace('_',' ')
    #algo_master+=' PA={pa}$^\circ$'.format(pa=90-pa[0])

    #if len(pa)>1:
    #   pcs=list(np.ones(len(pa),dtype=int)*pcs)
    algo = Descriptions(methods,pcs,norms,pa,method_name,fake, band)

    return algo[0], algo[1], algo_master



def Descriptions(methods, pcs, norms, pa, method_name, fake, band=None):
    algo_description=[]
    if len(pa)>1:
        if len(pcs) + len(norms) + len(methods) > 3:
            raise ValueError('Reduction methods, normalisation methods, and no. PCs \
                             must be single values when using multiple input PAs.')
        for i in range(len(pa)):
            name='PA = {pa}$^\circ$'.format(pa=90-pa[i])
            algo_description.append(name)
    else:
        for i in range(len(methods)):
            if 'convolved' in methods[i]:
                algo_description.append(methods[i] + ' disk')
                continue
            for j in range(len(norms)):
                for k in range(len(pcs)):
                    if FILTER == "": name = band.replace('_',' ') + ' '
                    else: name=''

                    if len(fake)!=1 and fake[i]!='':
                        name+='{0:s} fake disk '.format(fake[i])
                    if len(norms)!=1 and 'PCA' in methods[i]:
                        name+=norms[j].replace('_',' ')+' '
                    if len(methods)!=1:
                        name+=method_name[i].replace('_',' ')+' '
                    if len(pcs)!=1 and 'PCA' in methods[i]:
                        name+='PC {p}'.format(p=pcs[k])
                    if len(name)>0 and name[-1]==' ': #remove trailing spaces
                        name=name[:-1]
                    algo_description.append(name)

    return algo_description, len(algo_description)



def FileNames(algo_description, algo_master, nb_algo, norms, pcs, methods, mthdi, file_prefix):
    files=[]
    for i in range(nb_algo):
        if 'unconvolved' in algo_description[i] or 'Unconvolved' in algo_master:
            path ='/mnt/c/Users/stasevis/Documents/sphere_data/HD110058/HD110058_H23_2015-04-12_fake_disk_injection/fake_disk_bright_unconvolved.fits'

        elif 'convolved' in algo_description[i] or 'Convolved' in algo_master:
            path ='/mnt/c/Users/stasevis/Documents/sphere_data/HD110058/HD110058_H23_2015-04-12_fake_disk_injection/fake_disk_bright_convolved.fits'

        else:
            method_tmp = methods[mthdi(i)]
            if method_tmp =='CADI':
                norm_tmp=''
            else:
                norm_tmp ='_' + norms[i//len(pcs)-len(norms)*(i//(len(norms)*len(pcs)))]

            if 'bright' in algo_description[i] or 'bright' in algo_master:
                path = os.path.join(data_path,'fake_disk','{f}_bright_disk_{m}{n}.fits'.format(f=file_prefix,m=method_tmp,n=norm_tmp))
            elif 'faint' in algo_description[i] or 'faint' in algo_master:
                path = os.path.join(data_path,'fake_disk','{f}_faint_disk_{m}{n}.fits'.format(f=file_prefix,m=method_tmp,n=norm_tmp))
            else:
                if 'filter' in method_tmp:
                    method_tmp = method_tmp.split('_')[-1]
                path = os.path.join(data_path,'{f}_{m}{n}.fits'.format(f=file_prefix,m=method_tmp,n=norm_tmp))
        if INSTRUMENT == '' or FILTER not in set(['YJH','K12','YJ','H23']):
            path = path.replace('.fits','_stack.fits')
        if os.path.isfile(path.replace(path.split('_')[-1], 'best_frames_'+path.split('_')[-1])):
            path = path.replace(path.split('_')[-1], 'best_frames_'+path.split('_')[-1])
        files.append(path)

    return files



def InData(nb_algo, files, pcs, pci, size):
    images = np.ndarray((nb_algo,size,size))
    noise_images = np.ndarray((nb_algo,size,size))

    for i in range(nb_algo):
        image_tmp = fits.getdata(files[i])
        try:
            noise_tmp=fits.getdata(files[i].replace('.','_neg_parang.'))
        except FileNotFoundError:
            noise_tmp=np.zeros_like(image_tmp)

        size_tmp = image_tmp.shape[-1]
        if size_tmp < size:
            size = size_tmp
        elif (size+size_tmp)%2:
            size_tmp+=1
        size_min,size_max=((size_tmp-size)//2, (size_tmp+size)//2)

        if image_tmp.shape[0] == 2 or image_tmp.shape[0] == 39 or len(image_tmp.shape)==4:
            image_tmp = image_tmp[0]
            noise_tmp = noise_tmp[0]
        if len(image_tmp.shape) == 3: #has different PCs
            try:
                hdr=fits.getheader(files[i])
                pc_list = np.array([int(x) for x in hdr['PCS'][1:-1].split(', ') if x.isdigit()])
                #pc_list = np.array((1,5,10,15,20,25,35,50,75,100))
                pc_tmp = int(np.where(pcs[pci(i)]==pc_list)[0])

            except KeyError: #no pc list in header, assume value of pc= index -1
                pc_tmp = pcs[pci(i)]-1

            images[i,:,:] = image_tmp[pc_tmp,size_min:size_max,size_min:size_max]
            noise_images[i,:,:] = noise_tmp[pc_tmp,size_min:size_max,size_min:size_max]
        else:
            images[i,:,:] = image_tmp[size_min:size_max,size_min:size_max]
            noise_images[i,:,:] = noise_tmp[size_min:size_max,size_min:size_max]

    return images,noise_images


##apply high pass gaussian filter to images
def HighPass(images, noise, files, nb_algo):
    sigma = lambda fwhm: fwhm/2.355

    sigmas = np.zeros(nb_algo)
    filtered_images = np.zeros_like(images)
    filtered_noise = np.zeros_like(noise)

    for i in range(nb_algo):
        fwhm = fits.getheader(files[i])['FWHM']
        filtered_images[i] = gaussian_filter(images[i],sigma(fwhm))
        filtered_noise[i] = gaussian_filter(noise[i],sigma(fwhm))
        sigmas[i] = sigma(fwhm)

    return filtered_images, filtered_noise, sigmas



def PlotSingleFit(horizontal_separation, gauss_fit, files, algo_description, algo_master, pcs):
    for filei,f in enumerate(files):
        plt.close(1)
        fig = plt.figure(1, figsize=(18,21))
        gs = gridspec.GridSpec(3,1, height_ratios=[1,1,1], width_ratios=[1])
        gs.update(left=0.1, right=0.95, bottom=0.1, top=0.96, wspace=0.2, hspace=0.1)
        ax1 = plt.subplot(gs[0,0]) # Area for the first plot
        ax2 = plt.subplot(gs[1,0]) # Area for the second plot
        ax3 = plt.subplot(gs[2,0]) # Area for the colorbar

        fig.suptitle(algo_description[filei] + ' ' + algo_master, fontsize=20, fontweight='bold')

        # amp
        ax1.plot(horizontal_separation,gauss_fit['A'][filei,1,:],color='tomato')
        ax1.plot(-horizontal_separation[::-1],gauss_fit['A'][filei,0,::-1],color='tomato')
        ax1.set_ylabel('amplitude in ADU', fontsize=18)

        #x0
        ax2.plot(horizontal_separation,gauss_fit['x0'][filei,1,:],color='midnightblue')
        ax2.plot(-horizontal_separation[::-1],gauss_fit['x0'][filei,0,::-1],color='midnightblue')
        ax2.set_ylim(-3,3)
        ax2.set_ylabel('centroid in px', fontsize=18)

        #sigma
        ax3.plot(horizontal_separation,gauss_fit['sigma'][filei,1,:],color='darkgreen')
        ax3.plot(-horizontal_separation[::-1],gauss_fit['sigma'][filei,0,::-1],color='darkgreen')
        ax3.set_xlabel('Separation in px', fontsize=18)
        ax3.set_ylabel('$\sigma$ in px', fontsize=18)

        for ax in [ax1,ax2,ax3]:
            ax.grid(True)

        if 'convolved' in f:
            path_fig=os.path.join(root_path,'disk-analysis/spine_fit',f.split('/')[-1])
            #plt.savefig(path_fig.replace('.','_PA{pa}_fit.').format(pa=90-pa[filei]))

        else:
            path_fig=f.replace(data_path,data_path+'/disk_fit')
            path_fig=path_fig.replace(FILTER+'_',FILTER+'_pc{p}_'.format(p=pcs[pci(filei)]))
            if 'fake_disk' in path_fig:
                path_fig=path_fig.replace('/fake_disk','')

        plt.savefig(path_fig.replace('.fits','_fit_result.png')) ##combination of amp, cent, sig



def PlotFit(horizontal_separation, gauss_fit, files, algo_description, algo_master, nb_algo):
    colors = iter(cm.rainbow(np.linspace(0, 1, nb_algo)))

    plt.close(1)
    fig = plt.figure(1, figsize=(18,21))
    gs = gridspec.GridSpec(3,1, height_ratios=[1,1,1], width_ratios=[1])
    gs.update(left=0.1, right=0.95, bottom=0.1, top=0.96, wspace=0.2, hspace=0.1)
    ax1 = plt.subplot(gs[0,0]) # Area for the first plot
    ax2 = plt.subplot(gs[1,0]) # Area for the second plot
    ax3 = plt.subplot(gs[2,0]) # Area for the colorbar

    fig.suptitle(TARGET_NAME + ' ' + algo_master, fontsize=20, fontweight='bold')

    for filei,f in enumerate(files):
        c=next(colors)
        # amp
        ax1.plot(horizontal_separation,gauss_fit['A'][filei,1,:],color=c,label=algo_description[filei])#NW
        ax1.plot(-horizontal_separation[::-1],gauss_fit['A'][filei,0,::-1],color=c)#SE
        ax1.set_ylabel('amplitude in ADU', fontsize=18)
        ax1.legend(frameon=False,loc='upper right', fontsize=18)

        #x0
        ax2.plot(horizontal_separation,gauss_fit['x0'][filei,1,:],color=c)
        ax2.plot(-horizontal_separation[::-1],gauss_fit['x0'][filei,0,::-1],color=c)
        ax2.set_ylim(-15,15)
        ax2.set_ylabel('centroid in px', fontsize=18)

        #sigma
        ax3.plot(horizontal_separation,gauss_fit['sigma'][filei,1,:],color=c)
        ax3.plot(-horizontal_separation[::-1],gauss_fit['sigma'][filei,0,::-1],color=c)
        ax3.set_xlabel('Separation in px', fontsize=18)
        ax3.set_ylabel('$\sigma$ in px', fontsize=18)

    for ax in [ax1,ax2,ax3]:
        ax.grid(True)
    if 'convolved' in algo_master:
        plt.savefig(os.path.join(root_path,'disk-analysis/spine_fit',save_name+'_fit_result.png'))
    else:
        plt.savefig(os.path.join(data_path,'disk_fit','final',save_name+'_fit_result.png'))

#%%
def PlotPA(fit, err, horizontal_separation, data_path, pa_eq, pa_tmp, pa_err):
    if not os.path.exists(data_path + '/spine_fit'): os.makedirs(data_path + '/spine_fit')

    from ADI_reduction import FormatMethods
    method_tmp = FormatMethods(methods, norms)
    nb_mn, nb_pc = len(method_tmp), len(pcs)

    i = {}
    for b,band in enumerate(bands):
        i[band]=np.zeros((nb_mn, nb_pc),dtype=int)
        for m in range(nb_mn):
            for p in range(nb_pc):
                i[band][m,p] = b*nb_mn*nb_pc + m*nb_pc + p

    plot_bands = ['Y','J','H','YJH']
    sides=['SE','NW']
    for p in range(nb_pc):
        colors = iter(cm.rainbow(np.linspace(0, 1, len(plot_bands))))
        ls = iter(['--','--','--','-'])

        fig, ax = plt.subplots(1,1,figsize=(20,14))
        ax.set_title("HD110058 epoch combined YJH PA fit {1:s} PC {0:d}".format(pcs[p], method_tmp[0].replace('_',' ')),
                     fontsize=30, fontweight='bold', y=1.01)
        for band in plot_bands:
            c=next(colors)
            l=next(ls)
            if len(plot_bands) == 4 and plot_bands.index(band) == 2: c = 'yellowgreen'

            ax.set_ylabel('Centroid offset [arcsec]', fontsize=28)
            ax.set_xlabel('Separation [arcsec]', fontsize=28)
            ax.tick_params(labelsize=24)

            ax.errorbar(horizontal_separation, fit[i[band][0,p],1,:], err[i[band][0,p],1,:], color=c, ls=l, capsize=8, lw=3.5, label=band,alpha=0.7)
            ax.errorbar(-horizontal_separation[::-1], fit[i[band][0,p],0,::-1], err[i[band][0,p],0,::-1], color=c, ls=l, capsize=8, lw=3.5,alpha=0.7)

        for sidei,side in enumerate(sides):
            ax.plot((-1+(sidei*2))*np.insert(horizontal_separation,0, 0),
                     pa_eq[sidei][0]*(-1+(sidei*2))*np.insert(horizontal_separation,0, 0) + pa_eq[sidei][1],
                     label='{0:s} YJH PA={1:.2f}$^\circ$$\pm$ {2:.2f}'.format(side,90-(pa[0]-pa_tmp[sidei]),pa_err[sidei]),lw=3.5,ls='-',zorder=100)


        ax.legend(frameon=True, loc='upper left', fontsize=28)
        ax.grid(True)

        plt.savefig(data_path + '/spine_fit/broad_band_pa_sides_pc{0:d}_{1:s}.png'.format(pcs[p],method_tmp[0]))
        plt.show()

#PlotPA((pxscale/1000)*gauss_fit['x0'], (pxscale/1000)*gauss_error['x0'], (pxscale/1000)*horizontal_separation, data_path, pa_eq[0][-1]/(1,1000), pa_offset[0][:,-1], pa_error[0][:,-1])
#%%
def PlotSpineBands(fit, err, horizontal_separation, data_path):
    if not os.path.exists(data_path + '/spine_fit'): os.makedirs(data_path + '/spine_fit')

    from ADI_reduction import FormatMethods
    method_tmp = FormatMethods(methods, norms)
    nb_mn, nb_pc = len(method_tmp), len(pcs)

    i = {}
    for b,band in enumerate(bands):
        i[band]=np.zeros((nb_mn, nb_pc),dtype=int)
        for m in range(nb_mn):
            for p in range(nb_pc):
                i[band][m,p] = b*nb_mn*nb_pc + m*nb_pc + p

    plot_bands = ['Y','J','H','K']
    for p in range(nb_pc):

        colors = iter(cm.rainbow(np.linspace(0, 1, len(plot_bands))))

        fig = plt.figure(1, figsize=(12,16))
        gs = gridspec.GridSpec(4,1, height_ratios=[1,1,1,1], width_ratios=[1])
        gs.update(left=0.1, right=0.95, bottom=0.1, top=0.94, wspace=0.2, hspace=0.2)
        ax1 = plt.subplot(gs[0,0])
        ax2 = plt.subplot(gs[1,0])
        ax3 = plt.subplot(gs[2,0])
        ax4 = plt.subplot(gs[3,0])

        fig.suptitle("HD110058 epoch combined broad band spine fits PC {0:d}".format(pcs[p]),
                     fontsize=16, fontweight='bold', x=0.1, horizontalalignment='left')
        for band in plot_bands:
            c=next(colors)
            if len(plot_bands) == 4 and plot_bands.index(band) == 2: c = 'yellowgreen'

            ax1.set_title(method_tmp[0].replace('_',' '), fontsize=16,loc='left')
            ax1.set_ylabel('Centroid in px', fontsize=16)
            ax1.set_ylim(-6,6)
            ax1.errorbar(horizontal_separation, fit[i[band][0,p],1,:], err[i[band][0,p],1,:], color=c, ls='-', capsize=4, label=band)
            ax1.errorbar(-horizontal_separation[::-1], fit[i[band][0,p],0,::-1], err[i[band][0,p],0,::-1], color=c, ls='-', capsize=4)
            ax1.legend(frameon=True, loc='upper left', ncol=4, fontsize=14)


            ax2.set_title(method_tmp[1].replace('_',' '), fontsize=16,loc='left')
            ax2.set_ylabel('Centroid in px', fontsize=16)
            ax2.set_ylim(-6,6)
            ax2.errorbar(horizontal_separation, fit[i[band][1,p],1,:], err[i[band][1,p],1,:], color=c, ls='-', capsize=4, label=band)
            ax2.errorbar(-horizontal_separation[::-1], fit[i[band][1,p],0,::-1], err[i[band][1,p],0,::-1], color=c, ls='-', capsize=4)
            ax2.legend(frameon=True, loc='upper left', ncol=4, fontsize=14)

            ax3.set_title(method_tmp[2].replace('_',' '), fontsize=16,loc='left')
            ax3.set_ylabel('Centroid in px', fontsize=16)
            ax3.set_ylim(-6,6)
            ax3.errorbar(horizontal_separation, fit[i[band][2,p],1,:], err[i[band][2,p],1,:], color=c, ls='-', capsize=4, label=band)
            ax3.errorbar(-horizontal_separation[::-1], fit[i[band][2,p],0,::-1], err[i[band][2,p],0,::-1], color=c, ls='-', capsize=4)
            ax3.legend(frameon=True, loc='upper left', ncol=4, fontsize=14)

            ax4.set_title(method_tmp[3].replace('_',' '), fontsize=16,loc='left')
            ax4.set_ylabel('Centroid in px', fontsize=16)
            ax4.set_xlabel('Separation in px', fontsize=16)
            ax4.set_ylim(-6,6)
            ax4.errorbar(horizontal_separation, fit[i[band][3,p],1,:], err[i[band][3,p],1,:], color=c, ls='-', capsize=4, label=band)
            ax4.errorbar(-horizontal_separation[::-1], fit[i[band][3,p],0,::-1], err[i[band][3,p],0,::-1], color=c, ls='-', capsize=4)
            ax4.legend(frameon=True, loc='upper left', ncol=4, fontsize=14)

        for ax in [ax1,ax2,ax3,ax4]:
            ax.grid(True)

        plt.savefig(data_path + '/spine_fit/broad_band_inner_spine_fit_pc{0:d}.png'.format(pcs[p]))
        plt.show()

#PlotSpineBands(gauss_fit['x0'], gauss_error['x0'], horizontal_separation, data_path)
#%%
def PlotOverlay(aligned_images, fit, itr_fit, horizontal_separation, pci, data_path,
                target_info, files, algo_master, algo_description, sigmas, ext):
    if not os.path.exists(data_path + '/spine_overlay'): os.makedirs(data_path + '/spine_overlay')

    #import matplotlib.patches as patches
    nband = len(bands)

    for i,f in enumerate(files):
        fig,ax=plt.subplots(1,1, figsize=(20,10))
        lim = np.quantile(aligned_images[i%int(nb_algo/nband)],(0.1,0.993))
        #lim = (-0.03,0.09)
        extent = [-ext,ext,-ext,ext]
        ax.imshow(aligned_images[i], vmin=lim[0], vmax=lim[1], origin='lower', extent=extent)

        ax.plot(horizontal_separation, fit[i,1,:], color='tomato', label='centroid fit',lw=4)
        ax.plot(-horizontal_separation[::-1], fit[i,0,::-1], color='tomato', lw=4)

        if np.isnan(itr_fit[i]).any():
            ax.plot(horizontal_separation, itr_fit[i,1,:], ls=':', color='tomato', label='fit failed (interpolation)')
            ax.plot(-horizontal_separation[::-1], itr_fit[i,0,::-1], ls=':', color='tomato')

        if sigmas[i] !=0:
            ax.text(0.025, 0.05, 'filter sigma={0:.2f} px'.format(sigmas[i]), horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, color='w',size=28)

        ax.set_title(target_info.replace('_',' ') + ' ' + algo_master + ' ' + algo_description[i], loc='left',fontsize=30,y=1.01)
        ax.set_ylabel('Centroid offset [arcsec]', fontsize=28)
        ax.set_xlabel('Separation [arcsec]', fontsize=28)

        #rect = patches.Rectangle((31, -4), 12, 8, linewidth=1, edgecolor='r', facecolor='none')
        #ax.add_patch(rect)
        #rect = patches.Rectangle((-43, -4), 12, 8, linewidth=1, edgecolor='r', facecolor='none')
        #ax.add_patch(rect)

        #ax.annotate('PA fit region', xy=(-38, -4),  xycoords='data',
        #    xytext=(0.4, 0.15), textcoords='axes fraction', fontsize=12,
        #    arrowprops=dict(color='red', arrowstyle='-'),
        #    horizontalalignment='right', verticalalignment='top', color='red')
        #ax.annotate('', xy=(38, -4),  xycoords='data',
        #    xytext=(0.4, 0.15), textcoords='axes fraction', fontsize=12,
        #    arrowprops=dict(color='red', arrowstyle='-'),
        #    horizontalalignment='right', verticalalignment='top', color='red')

        ax.plot((-ext,ext),(0,0),color='white',ls='--')
        ax.plot((0,0),(-ext,ext),color='white',ls='--')

        ax.legend(frameon=False,loc='upper left',labelcolor='w',fontsize=28)
        ax.set_ylim(-(pxscale/1000)*40,(pxscale/1000)*40)
        ax.tick_params(labelsize=24)

        path_fig = f.replace(data_path, data_path + '/spine_overlay')
        plt.savefig(path_fig.replace('.fits','_pc_{0:d}_spine_fit_overlay.png'.format(pcs[pci(i)])))
        plt.show()

#PlotOverlay(aligned_images, (pxscale/1000)*gauss_fit['x0'], (pxscale/1000)*gauss_fit['x0itr'], (pxscale/1000)*horizontal_separation, pci, data_path, target_info, files, algo_master, algo_description, sigmas, (pxscale/1000)*(size//2))
#%%
"""
-------------------------------------------- MAIN CODE --------------------------------------------
"""
files = []
algo_description = []
nb_algo = 0

if FILTER != "":
    bands = [file_prefix]

for band in bands:
    res = FormatDescription(methods,pcs,norms,pa,fake,band)
    files += FileNames(res[0], res[2], res[1], norms, pcs, methods, mthdi, band)
    algo_description += res[0]
    nb_algo += res[1]
    algo_master = res[2]

#%%
images, noise_images = InData(nb_algo,files, pcs, pci, size)
if 'high pass filter' in algo_master:
    hpf = list(np.arange(nb_algo)); hpf_files = files
else:
    hpf = [i for i,x in enumerate(algo_description) if 'high pass filter' in x]
    hpf_files = [x for i,x in enumerate(files) if i in hpf]

sigmas = np.zeros(nb_algo)
images[hpf], noise_images[hpf], sigmas[hpf]= HighPass(images[hpf], noise_images[hpf], hpf_files, len(hpf))

aligned_images = vip.preproc.cube_derotate(images,np.ones((nb_algo))*(pa),imlib='opencv')
aligned_noise = vip.preproc.cube_derotate(noise_images,np.ones((nb_algo))*(pa),imlib='opencv')

# check visually the images
ds9.display(aligned_images)

for i in hpf: files[i] = files[i].replace(methods[mthdi(i)].split('_')[-1],methods[mthdi(i)])
sides=['SE','NW']

rad = {'inner': px_resize(33), #28
         'outer': px_resize(80), ##not detected beyond that
         'bright': px_resize(50), ##separation after which larger binning needed
         'warp': px_resize(60)}

horizontal_bin=px_resize(5)
vertical_length=np.array((px_resize(18),px_resize(50)))

buffer = vertical_length%2
vertical_length += buffer

res = pfit.Separation(horizontal_bin, vertical_length, rad)

horizontal_separation = res[0]
horizontal_bin = res[1]
vertical_displacement = res[2]

if not os.path.exists(data_path+'/disk_profile'): os.makedirs(data_path+'/disk_profile')

gauss_fit, gauss_error = pfit.FitProfiles(aligned_images, aligned_noise, horizontal_separation,
                                          horizontal_bin, vertical_displacement, vertical_length,
                                          rad, files, pcs, pci, sides, pxscale, dstar, data_path,
                                          roll=False)

if TARGET_EPOCH == "":
    target_info = TARGET_NAME + ' multi-epoch broad band ' + FILTER
elif INSTRUMENT == "":
    target_info = TARGET_NAME + ' ' + TARGET_EPOCH
else:
    target_info = TARGET_NAME + ' ' + TARGET_EPOCH + ' ' + INSTRUMENT.upper() + ' ' + FILTER

if target_info[-1] == ' ': target_info = target_info[:-1]

PlotOverlay(aligned_images, (pxscale/1000)*gauss_fit['x0'], (pxscale/1000)*gauss_fit['x0itr'], (pxscale/1000)*horizontal_separation, pci, data_path, target_info, files, algo_master, algo_description, sigmas, (pxscale/1000)*(size//2))
#PlotSpineBands(gauss_fit['x0'], gauss_error['x0'], horizontal_separation, data_path)
#%%

pa_fit_range = np.array((px_resize(33),px_resize(45))) #28

if fit_sides_indv == True:
    full = False
else:
    full = True

if iterate_pa == True and nb_algo == 1:
    pa_offset = np.zeros(nb_algo)
    pa_error = np.zeros_like(pa_offset)
    for i in range(nb_algo):
        res = fpa.IteratePA(images, noise_images, gauss_fit, gauss_error, horizontal_separation,
                            horizontal_bin, vertical_displacement, vertical_length, algo_master,
                            algo_description, files, pcs, pci, pa, pa_fit_range, sides, save_name,
                            target_info, pxscale, dstar, data_path, full, save_table=True)
        gauss_fit[i] = res[0]
        pa_offset[i] = res[1]
        pa_error[i] = res[2]
        aligned_images[i] = res[3]
        aligned_noise[i] = res[4]
pa_eq=[]
pa_offset, pa_error = [],[]
for i,full in enumerate(set([True,False])):
    #if full == False: pa_fit_range = np.array((px_resize(30),px_resize(60)))
    #else: pa_fit_range = np.array((px_resize(30),px_resize(45)))
    res = fpa.FindPA(horizontal_separation, nb_algo, gauss_fit['x0'], gauss_error['x0'],
                     algo_master, algo_description, files, pcs, pci, pa, pa_fit_range,
                     sides, data_path, save_name, pxscale, target_info, full,
                     save_table=False, return_list=True)
    pa_offset.append(res[0][0])
    pa_error.append(res[0][1])
    pa_eq.append(res[1])
    for n in range(nb_algo):
        if full == False:
            sides_offset=pa_offset[i][:,n].mean()
            sides_error=np.sqrt(np.sum(np.square(pa_error[i][:,n]))) #pa_error[:,i].std()
            print('{0:s} mean PA = {1:f} +/- {2:f} [deg]'.format(algo_description[n],90-(pa[0]-sides_offset),sides_error))
        else:
            print('{0:s} measured PA = {1:f} +/- {2:f} [deg]'.format(algo_description[n],90-(pa[0]-pa_offset[i][n]),pa_error[i][n]))

#PlotPA((pxscale/1000)*gauss_fit['x0'], (pxscale/1000)*gauss_error['x0'], (pxscale/1000)*horizontal_separation, data_path, pa_eq[0][-1]/(1,1000), pa_offset[0][:,-1], pa_error[0][:,-1])
#PlotSingleFit(horizontal_separation, gauss_fit, files, algo_description, algo_master, pcs)
#PlotFit(horizontal_separation, gauss_fit, files, algo_description, algo_master, nb_algo)
