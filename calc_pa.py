#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: stasevis

fit straight line to disk centroid (find_pa = True )
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import vip_hci as vip
import profile_fit as pfit


def SidesOffsetPA(pa_fit_inner, pa_fit_outer, horizontal_separation, nb_algorithms, centroid_fit,
                  centroid_error, algorithm_master, algorithms_description, files, pcs, pci, pa, root_path):

    pa_fit=np.array(list(set(np.where(horizontal_separation>=pa_fit_inner)[0])&set(np.where(horizontal_separation<=pa_fit_outer)[0])))
    pa_fit_sep=horizontal_separation[pa_fit]

    pa_centroid_fit=centroid_fit[:,:,pa_fit]
    pa_centroid_error=centroid_error[:,:,pa_fit]

    sides=pa_centroid_fit.shape[1]

    pa_offset=np.zeros((nb_algorithms,sides,2)) ##value AND error
    x=np.array((-pa_fit_sep[::-1],pa_fit_sep))

    for i in range(nb_algorithms):
        y=np.array((pa_centroid_fit[i,0,::-1],pa_centroid_fit[i,1,:]))
        nnan=np.isfinite(y)

        sigma=np.array((pa_centroid_error[i,0,::-1],pa_centroid_error[i,1,:]))
        sigma[np.where(sigma==0)[0]]=1.
        weights=np.ones_like(sigma)/sigma

        fig, ax = plt.subplots(1,1,figsize=(10,7))
        ax.plot(horizontal_separation, centroid_fit[i,1,:], 'r-',label='fitted spine')
        ax.plot(-horizontal_separation[::-1], centroid_fit[i,0,::-1], 'r-')
        ax.errorbar(x[0], y[0], sigma[0], color='blue', marker='o', ls='', capsize=4, label='PA fitting points')
        ax.errorbar(x[1], y[1], sigma[1], color='blue', marker='o', ls='', capsize=4)
        ax.set_title(algorithm_master + ' ' + algorithms_description[i] + 'Gaussian profile fit', fontsize=16,loc='left')
        ax.set_ylabel('Centroid in px', fontsize=16)
        ax.set_xlabel('Separation in px', fontsize=16)

        try:

            for sidei in range(sides):
                snan=nnan[sidei]
                p,v=np.polyfit(x[sidei,snan],y[sidei,snan],deg=1,cov=True,w=weights[sidei,snan])
                pa_offset[i,sidei,:]=[math.atan(p[0]), math.atan(np.sqrt(v[0][0]))]
                pa_tmp = 90-(pa[i]-np.rad2deg(pa_offset[i,sidei,0]))
                pa_err = np.rad2deg(pa_offset[i,sidei,1])

                ax.plot((-1+(sidei*2))*horizontal_separation, p[0]*(-1+(sidei*2))*horizontal_separation + p[1], label='PA={0:.2f}$^\circ$$\pm$ {1:.2f}'.format(pa_tmp,pa_err))

            ax.legend(frameon=False,loc='best', fontsize=14)
            ax.grid(True)
            plt.show()

            path=os.path.join(root_path,'disk-analysis/PA_calculations',files[i].split('/')[-1])
            fig.savefig(path.replace('.fits','_pc{p}_sides_fit.png').format(p=pcs[pci(i)]))

        except np.linalg.LinAlgError:

            ax.legend(frameon=False,loc='best', fontsize=14)
            ax.grid(True)
            plt.show()

            pa_offset[i]=np.nan
            print('Line fit failed')

    return pa_offset


def OffsetPA(pa_fit_inner, pa_fit_outer, horizontal_separation, nb_algorithms, centroid_fit, centroid_error,
             algorithm_master, algorithms_description, files, pcs, pci, pa, root_path):

    pa_fit=np.array(list(set(np.where(horizontal_separation>=pa_fit_inner)[0])&set(np.where(horizontal_separation<=pa_fit_outer)[0])))
    pa_fit_sep=horizontal_separation[pa_fit]

    pa_centroid_fit=centroid_fit[:,:,pa_fit]
    pa_centroid_error=centroid_error[:,:,pa_fit]

    pa_offset=np.zeros((nb_algorithms,2)) ##value AND error
    x=np.concatenate((-pa_fit_sep[::-1],pa_fit_sep))

    for i in range(nb_algorithms):
        y=np.concatenate((pa_centroid_fit[i,0,::-1],pa_centroid_fit[i,1,:]))
        nnan=np.isfinite(y)

        sigma=np.concatenate((pa_centroid_error[i,0,::-1],pa_centroid_error[i,1,:]))
        sigma[np.where(sigma==1)[0]]=1.
        weights=np.ones_like(sigma)/sigma

        fig, ax = plt.subplots(1,1,figsize=(10,7))
        ax.plot(horizontal_separation, centroid_fit[i,1,:], 'r-',label='fitted spine')
        ax.plot(-horizontal_separation[::-1], centroid_fit[i,0,::-1], 'r-')
        ax.errorbar(x, y, sigma, marker='o', ls='', capsize=4, label='PA fitting points')
        ax.set_title(algorithm_master + ' ' + algorithms_description[i] + 'Gaussian profile fit', fontsize=16,loc='left')
        ax.set_ylabel('Centroid in px', fontsize=16)
        ax.set_xlabel('Separation in px', fontsize=16)

        try:

            p,v=np.polyfit(x[nnan],y[nnan],deg=1,cov=True,w=weights[nnan])

            ##error is square root of slope element of covarience matrix
            pa_offset[i]=[math.atan(p[0]), math.atan(np.sqrt(v[0][0]))]
            pa_tmp = 90-(pa[i]-np.rad2deg(pa_offset[i,0]))
            pa_err = np.rad2deg(pa_offset[i,1])
            ax.plot(x, p[0]*x + p[1],label='PA={0:.2f}$^\circ$$\pm$ {1:.2f}'.format(pa_tmp,pa_err))
            ax.legend(frameon=False,loc='best', fontsize=14)
            ax.grid(True)
            plt.show()

            path=os.path.join(root_path,'disk-analysis/PA_calculations',files[i].split('/')[-1])
            fig.savefig(path.replace('.fits','_pc{p}_fit.png').format(p=pcs[pci(i)]))

        except np.linalg.LinAlgError:

            pa_offset[i]=np.nan
            ax.legend(frameon=False,loc='best', fontsize=14)
            ax.grid(True)
            plt.show()

            print('Line fit failed')

    return pa_offset



def FindPA(horizontal_separation, nb_algorithms, centroid_fit, centroid_error,
           algorithm_master, algorithms_description, files, pcs, pci,
           pa, pa_fit_inner, pa_fit_outer, sides, root_path, save_name, target_info,
           full=True, save_table=False):

    initial_pa=90-(np.ones(nb_algorithms)*pa) ##multiplied by an array of shape nb_algs in case of single input PA
    if full:
        pa_offset=np.rad2deg(OffsetPA(pa_fit_inner, pa_fit_outer, horizontal_separation,
                                      nb_algorithms, centroid_fit, centroid_error, algorithm_master,
                                      algorithms_description, files, pcs, pci, pa, root_path))
        measured_pa=np.stack((90-(pa-pa_offset[:,0]),pa_offset[:,1],initial_pa)).T ##PA from fit, error of fit, initial PA guess

    else:
        pa_offset=np.rad2deg(SidesOffsetPA(pa_fit_inner, pa_fit_outer, horizontal_separation,
                                      nb_algorithms, centroid_fit, centroid_error, algorithm_master,
                                      algorithms_description, files, pcs, pci, pa, root_path))

        measured_pa=np.stack((90-(pa-pa_offset[:,0,0]),pa_offset[:,0,1],90-(pa-pa_offset[:,1,0]),pa_offset[:,1,1],initial_pa)).T

    if save_table == True:

        algo=algorithms_description[0]
        for i in range(1,len(algorithms_description)):
            algo=algo + ', ' + algorithms_description[i]

        header = target_info + ' \n' + algorithm_master + ' \n' + algo + '\nMean offset={pm} +/- {pe}\
                [deg]'.format(pm=np.nanmean(pa_offset[[0,2]]),pe=np.nanstd(pa_offset[[0,2]]))

        if full:
            header+= '\nPA [deg] \terr [deg] \tinitial input PA [deg]'
        else:
            header+='\n{0:s} PA [deg] \terr [deg]  \t{1:s} PA[deg] \terr [deg] \tinitial input PA [deg]'.format(sides[0],sides[1])

        path=os.path.join(root_path,'disk-analysis/PA_calculations',save_name+'_measured_PA.txt')
        np.savetxt(path, measured_pa, fmt='%.8f', delimiter='\t', header=header)

    return pa_offset[0].T


def IteratePA(nb_algorithms, images, noise_images, gauss_fit, gauss_error, horizontal_separation,
              horizontal_binning_factor, vertical_profile_length, inner_radius, outer_radius,
              bright_radius, algorithm_master, algorithms_description, files, pcs, pci,
              pa, pa_fit_inner, pa_fit_outer, sides, root_path, save_name, target_info,
              full=True, save=False):
    from disk_profiles import Profiles, FitProfiles
    if full == True:
        pa_offset, pa_error = FindPA(horizontal_separation, nb_algorithms, gauss_fit['x0'], gauss_error['x0'], algorithm_master,
                                     algorithms_description, files, pcs, pci, pa, pa_fit_inner, pa_fit_outer, sides, full, save)
        count=0
        while abs(pa_offset)>=pa_error and pa_error>0.1:

            pa[0]-=pa_offset

            aligned_images = vip.preproc.cube_derotate(images,np.ones((nb_algorithms))*(pa),imlib='opencv')
            aligned_noise = vip.preproc.cube_derotate(noise_images,np.ones((nb_algorithms))*(pa),imlib='opencv')

            gauss_fit, gauss_error = FitProfiles(*Profiles(aligned_images, aligned_noise,
                                                            nb_algorithms, files, inner_radius, outer_radius,
                                                            bright_radius, horizontal_binning_factor, vertical_profile_length),
                                                            nb_algorithms, files, pcs, pci, sides)
            pa_offset, pa_error = FindPA(horizontal_separation, nb_algorithms, gauss_fit['x0'], gauss_error['x0'], algorithm_master,
                                         algorithms_description, files, pcs, pci, pa, pa_fit_inner, pa_fit_outer, sides, full, save)
            count+=1

            if count>10: print("Horizontal PA not found after 10 iterations"); break

        pa[0]-=pa_offset
        aligned_images = vip.preproc.cube_derotate(images,np.ones((nb_algorithms))*(pa),imlib='opencv')
        aligned_noise = vip.preproc.cube_derotate(noise_images,np.ones((nb_algorithms))*(pa),imlib='opencv')

        gauss_fit, gauss_error = FitProfiles(*Profiles(aligned_images, aligned_noise,
                                                        nb_algorithms, files, inner_radius, outer_radius,
                                                        bright_radius, horizontal_binning_factor, vertical_profile_length),
                                                        nb_algorithms, files, pcs, pci, sides)
        pa_offset, pa_error = FindPA(horizontal_separation, nb_algorithms, gauss_fit['x0'], gauss_error['x0'], algorithm_master,
                                     algorithms_description, files, pcs, pci, pa, pa_fit_inner, pa_fit_outer, sides, full, save)
        pa[0]-=pa_offset

        print("Measured PA =",90-(pa[0]),'+/-',pa_error)

        return horizontal_separation, gauss_fit, pa_offset, pa_error, aligned_images, aligned_noise

    if full == False:
        sep_offset, sep_error = FindPA(horizontal_separation, nb_algorithms, gauss_fit['x0'], gauss_error['x0'], algorithm_master,
                                     algorithms_description, files, pcs, pci, pa, pa_fit_inner, pa_fit_outer, sides, full, save)
        pa_offset=sep_offset.mean()
        pa_error=sep_offset.std()
        count=0

        while abs(pa_offset)>=pa_error and pa_error>0.5:
            pa[0]-=pa_offset

            aligned_images = vip.preproc.cube_derotate(images,np.ones((nb_algorithms))*(pa),imlib='opencv')
            aligned_noise = vip.preproc.cube_derotate(noise_images,np.ones((nb_algorithms))*(pa),imlib='opencv')

            gauss_fit, gauss_error = FitProfiles(*Profiles(aligned_images, aligned_noise,
                                                            nb_algorithms, files, inner_radius, outer_radius,
                                                            bright_radius, horizontal_binning_factor, vertical_profile_length),
                                                            nb_algorithms, files, pcs, pci, sides)
            sep_offset, sep_error = FindPA(horizontal_separation, nb_algorithms, gauss_fit['x0'], gauss_error['x0'], algorithm_master,
                                         algorithms_description, files, pcs, pci, pa, pa_fit_inner, pa_fit_outer, sides, full, save)
            pa_offset=sep_offset.mean()
            pa_error=sep_offset.std()

            count+=1
            if count>10: print("Horizontal PA not found after 10 iterations"); break

        pa[0]-=pa_offset

        aligned_images = vip.preproc.cube_derotate(images,np.ones((nb_algorithms))*(pa),imlib='opencv')
        aligned_noise = vip.preproc.cube_derotate(noise_images,np.ones((nb_algorithms))*(pa),imlib='opencv')

        gauss_fit, gauss_error = FitProfiles(*Profiles(aligned_images, aligned_noise,
                                                        nb_algorithms, files, inner_radius, outer_radius,
                                                        bright_radius, horizontal_binning_factor, vertical_profile_length),
                                                        nb_algorithms, files, pcs, pci, sides)
        sep_offset, sep_error = FindPA(horizontal_separation, nb_algorithms, gauss_fit['x0'], gauss_error['x0'], algorithm_master,
                                     algorithms_description, files, pcs, pci, pa, pa_fit_inner, pa_fit_outer, sides, full, save)
        pa_offset=sep_offset.mean()
        pa_error=sep_offset.std()

        for i in range(len(sides)):
            print('Measured {s} PA ='.format(s=sides[i]),90-(pa[0]-sep_offset[i]),'+/-',sep_error[i],'\n')

        return gauss_fit, pa_offset, pa_error, aligned_images, aligned_noise