#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: stasevis

split disk images into series of vertical profiles + fit a gaussian to each profile
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.functional_models import Gaussian1D #,Lorentz1D,Moffat1D
import mpfit
import os



def Separation(horizontal_bin, vertical_length, rad):

    if type(vertical_length) == np.ndarray:
        vertical_length = vertical_length[-1]

    if rad['bright'] is not None:
        hsep_inner = np.arange(rad['inner'],rad['bright'],horizontal_bin)
        hsep_outer = np.arange(hsep_inner[-1]+horizontal_bin+2,rad['outer'],horizontal_bin+2)

        horizontal_separation = np.concatenate((hsep_inner,hsep_outer))
        horizontal_bin = np.concatenate((np.ones(len(hsep_inner),dtype=int)*horizontal_bin,
                                         np.ones(len(hsep_outer),dtype=int)*(horizontal_bin+2)))
    else:
        horizontal_separation = np.arange(rad['inner'],rad['outer'],horizontal_bin)
        horizontal_bin = np.ones(len(hsep_inner), dtype=int)*horizontal_bin

    vertical_displacement = np.arange(-vertical_length//2,vertical_length//2)

    return horizontal_separation, horizontal_bin, vertical_displacement



def Profiles(aligned_images, aligned_noise, horizontal_bin, horizontal_separation, vertical_length,
             rad, sides, roll=False):
    nb_algo = aligned_images.shape[0]

    if horizontal_bin[0] % 2: ##allow for even or odd binning factors
        buffer=1
    else:
        buffer=0

    if type(vertical_length) == np.ndarray:
        mask_length, vertical_length = vertical_length

    nb_profiles= len(horizontal_separation)
    profiles = np.ndarray((nb_algo, 2, vertical_length, nb_profiles)) ##axis 1 is for the sides (0:SE, 1:NW)
    noise_profiles = np.zeros_like(profiles)

    noise_check=np.zeros(nb_algo)
    for i in range(nb_algo):
        noise_check[i]=aligned_noise[i].std()
    non=np.where(noise_check==0)[0] ##need to take noise from aligned image
    noise_range = vertical_length//2 ##no. px to use from top and bottom of the vertical profile for noise estimation

    size=aligned_images.shape[-1]
    for sepi, sep in enumerate(horizontal_separation):
        #lim = lambda lbin: size//2 + lbin//2
        vmin = size//2-vertical_length//2
        vmax = size//2+vertical_length//2
        hmin = {0: size//2-sep-horizontal_bin[sepi]//2,
                1: size//2+sep-horizontal_bin[sepi]//2}
        hmax = {0: size//2-sep+horizontal_bin[sepi]//2 + buffer,
                1: size//2+sep+horizontal_bin[sepi]//2 + buffer}

        for sidei in range(len(sides)):
            if sep > rad['bright'] and roll:
                prof_tmp = np.mean(aligned_images[:,vmin-1:vmax+1, hmin[sidei]:hmax[sidei]],axis=2)
                for v in range(vertical_length):
                    profiles[:,sidei,v,sepi]=np.mean(prof_tmp[:,v:v+3],axis=1)
            else:
                profiles[:,sidei,:,sepi] = np.mean(aligned_images[:,vmin:vmax, hmin[sidei]:hmax[sidei]],axis=2)

            if len(non)-nb_algo!=0:
                algo = [x not in non for x in range(0, nb_algo)]
                noise_profiles[algo,sidei,:,sepi] = np.mean(aligned_noise[algo, vmin:vmax, hmin[sidei]:hmax[sidei]],axis=2)

            if len(non)!=0:
                noise_upper = aligned_images[non, vmax:vmax+noise_range, hmin[sidei]:hmax[sidei]]
                noise_lower = aligned_images[non, vmin-noise_range:vmin, hmin[sidei]:hmax[sidei]]
                algo = np.concatenate((noise_upper, noise_lower),axis=1)
                noise_profiles[non,sidei,:,sepi] = np.mean(algo, axis=2)

    if rad['warp'] is not None and 'mask_length' in locals():
        profiles = profile_mask(profiles, horizontal_separation, mask_length, vertical_length, rad['warp'])
        noise_profiles = profile_mask(noise_profiles, horizontal_separation, mask_length, vertical_length, rad['warp'])

    return profiles, noise_profiles



def profile_mask(profiles, horizontal_separation, mask_length, vertical_length, r):
    mask_sep = np.where(horizontal_separation < r)[0]
    vmin_mask = vertical_length//2-mask_length//2
    vmax_mask = vertical_length//2+mask_length//2

    profiles[:,:,:vmin_mask,mask_sep] = -1e-6
    profiles[:,:,vmax_mask:,mask_sep] = -1e-6

    return profiles



def gauss1D_fit_erf(p, fjac=None, x=None, y=None, err=None):
    '''
    Computes the residuals to be minimized by mpfit, given a model and data.
    '''
    model = Gaussian1D(p[0],p[1],p[2])(x)
    status = 0
    return ([status, ((y-model)/err).ravel()])



def GaussDict(nb_algo, nb_profiles):

    gauss_fit = {'A':np.ones((nb_algo,2,nb_profiles))*np.nan,
                 'x0':np.ones((nb_algo,2,nb_profiles))*np.nan,
                 'sigma':np.ones((nb_algo,2,nb_profiles))*np.nan,
                 'x0itr':np.ones((nb_algo,2,nb_profiles))*np.nan}

    gauss_error = {'A':np.ones((nb_algo,2,nb_profiles))*np.nan,
                   'x0':np.ones((nb_algo,2,nb_profiles))*np.nan,
                   'sigma':np.ones((nb_algo,2,nb_profiles))*np.nan}

    return gauss_fit, gauss_error



def FitProfiles(aligned_images, aligned_noise, horizontal_separation, horizontal_bin,
                vertical_displacement, vertical_length, rad, files, pcs, pci, sides,
                pxscale, dstar, data_path, roll=False):

    if not os.path.exists(data_path+'/disk_profile'): os.makedirs(data_path+'/disk_profile')

    profiles, noise_profiles = Profiles(aligned_images, aligned_noise, horizontal_bin,
                                        horizontal_separation, vertical_length, rad, sides, roll)

    nb_algo, nb_profiles= len(files), len(horizontal_separation)
    gauss_fit, gauss_error = GaussDict(nb_algo, nb_profiles)
    itr = lambda z: z.nonzero()[0]

    for filei,f in enumerate(files):
        print('Processing {0:s} pc {1:d}'.format(str(f),pcs[pci(filei)]))
        for sepi,sep in enumerate(horizontal_separation):
            print('Separation of {0:d}px'.format(sep))
            for sidei,side in enumerate(sides):
                prof = profiles[filei,sidei,:,sepi]
                noise_array = noise_profiles[filei,sidei,:,sepi]
#%%
                max_prof = np.max(prof)
                maxima=np.where(prof==max_prof)[0]
                minima=[]
                for i in range(1,len(prof)-1):
                    x=prof[i]
                    if x< prof[i-1] and x< prof[i+1] and x< max(prof[i-1],prof[i+1])/1.1: ##buffer, may need changing
                        minima.append(i)

                try:
                    upper=minima[min(np.where(minima>maxima)[0])]
                except ValueError:
                    upper=len(prof)-1
                try:
                    lower=minima[max(np.where(minima<maxima)[0])]
                except ValueError:
                    lower=0

                try:
                    while prof[lower]<0 and lower<maxima:
                        lower+=1
                    while prof[upper]<0 and upper>maxima:
                        upper-=1
                except ValueError:
                    print('Fit failed')
                    continue

                #TODO: discontinuity in gradient of outer fitting points to truncate before
                #but only if the gradient between points lying between the maxima and outer
                #points is steady
                #if maxima-lower>3:
                #    lower = maxima-3
                #if upper-maxima>3:
                #    upper = maxima+3
                ran=np.arange(lower,upper+1)

                prof_4fit = prof[ran]
                vert_disp4fit = vertical_displacement[ran]

                fa = {'x': vert_disp4fit, 'y':prof_4fit , 'err':np.std(noise_array)*np.ones_like(prof_4fit)}
                parinfo = [{'fixed':0, 'limited':[1,1], 'limits':[0.,2*max_prof]}, # Force the amplitude to be >0 and smaller than2*max
                           {'fixed':0, 'limited':[1,1], 'limits':[np.min(vert_disp4fit),np.max(vert_disp4fit)]}, # vertical offset of Gaussian
                           {'fixed':0, 'limited':[1,1], 'limits':[2,10.]}] # Force the sigma to be > 2 and <10
                p0 = [max_prof,vert_disp4fit[np.argmax(prof_4fit)],3]
                m = mpfit.mpfit(gauss1D_fit_erf, p0, functkw=fa, parinfo=parinfo, quiet=1) # Set quiet=0 for debugging
                if m.status==1:
                    gauss_fit['A'][filei,sidei,sepi]=m.params[0]
                    gauss_fit['x0'][filei,sidei,sepi]=m.params[1]
                    gauss_fit['sigma'][filei,sidei,sepi]=m.params[2]
                    gauss_error['A'][filei,sidei,sepi]=m.perror[0]
                    gauss_error['x0'][filei,sidei,sepi]=m.perror[1]
                    gauss_error['sigma'][filei,sidei,sepi]=m.perror[2]
                    plt.close(1)
                    plt.figure(1, figsize=(6.93,7))
                    plt.clf()
                    plt.plot(vertical_displacement,prof,label='profile')
                    plt.plot(vert_disp4fit,prof_4fit,'rx',label='fitted profile')
                    plt.plot(vertical_displacement,Gaussian1D(m.params[0],m.params[1],m.params[2])(vertical_displacement),label='fit $x_0=${0:.1f} $\sigma$={1:.1f}'.format(m.params[1],m.params[2]))
                    plt.xlabel('Separation in pixel')
                    plt.ylabel('Intensity in ADU')
                    plt.title('Vertical {0:s} profile at {1:d}px={2:.0f}mas={3:.0f}au'.format(side,sep,sep*pxscale,sep*pxscale/1000*dstar))
                    plt.legend()
                    path_fig=f.replace(data_path, data_path+'/disk_profile')
                    #path_fig=path_fig.replace('pc50','pc{p}'.format(p=pcs[pci(filei)]))
#%%
                    if 'fake_disk' in path_fig:
                        path_fig=path_fig.replace('/fake_disk/','/')

                    plt.savefig(path_fig.replace('.fits','_pc_{0:d}_profile_{1:s}_sep{2:d}px.png'.format(pcs[pci(filei)],side,sep)))
                    #plt.show()

                else:
                    print('Fit failed')

        for sidei,side in enumerate(sides):
            y = gauss_fit['x0'][filei,sidei].copy()
            nans = np.isnan(y)
            #ends = np.where(nans == False)[0] ##only interpolate between two fitted points
            #nans[:ends[0]], nans[ends[-1]:] = False,False
            y[nans] = np.interp(itr(nans), itr(~nans), y[~nans])
            gauss_fit['x0itr'][filei,sidei] = y

    return gauss_fit, gauss_error