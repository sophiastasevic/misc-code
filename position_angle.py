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
from astropy.io import fits


def SidesOffsetPA(pa_fit_range, horizontal_separation, nb_algo, centroid_fit, centroid_error,
                  algo_master, algo_description, files, pcs, pci, pa, data_path):

    pa_fit = np.array(list(set(np.where(horizontal_separation>=pa_fit_range[0])[0])&
                           set(np.where(horizontal_separation<=pa_fit_range[1])[0])))
    pa_fit_sep = horizontal_separation[pa_fit]

    pa_centroid_fit = centroid_fit[:,:,pa_fit]
    pa_centroid_error = centroid_error[:,:,pa_fit]

    sides = pa_centroid_fit.shape[1]

    pa_line_eq = np.zeros((nb_algo,2,sides))
    pa_offset = np.zeros((nb_algo,2,sides)) ##value AND error
    x = np.array((-pa_fit_sep[::-1], pa_fit_sep))

    vert = np.nanmax(abs(centroid_fit)+centroid_error)
    vert = float('{0:.2g}'.format(vert + vert/10))

    for i,f in enumerate(files):
        y = np.array((pa_centroid_fit[i,0,::-1], pa_centroid_fit[i,1,:]))
        nnan = np.isfinite(y)

        sigma = np.array((pa_centroid_error[i,0,::-1], pa_centroid_error[i,1,:]))
        sigma[np.where(sigma==0)[0]]=1.
        weights = np.ones_like(sigma)/sigma

        fig, ax = plt.subplots(1,1,figsize=(20,14))

        ax.errorbar(horizontal_separation, centroid_fit[i,1,:], centroid_error[i,1,:], color='r', marker='', capsize=8, label='fitted spine',lw=3.5)
        ax.errorbar(-horizontal_separation[::-1], centroid_fit[i,0,::-1], centroid_error[i,0,::-1], color='r', marker='', capsize=8,lw=3.5)

        ax.errorbar(x[0], y[0], sigma[0], color='blue', marker='o', ls='', capsize=8, label='PA fitting points',lw=3.5,markersize=6)
        ax.errorbar(x[1], y[1], sigma[1], color='blue', marker='o', ls='', capsize=8,lw=3.5)
        ax.set_title(algo_master + ' ' + algo_description[i] + ' Gaussian profile fit', fontsize=30,loc='left')
        ax.set_ylabel('Centroid offset [arcsec]', fontsize=28)
        ax.set_xlabel('Separation [arcsec]', fontsize=28)
        ax.set_ylim(-vert,vert)
        #ax.set_ylim(-10,10)
        ax.tick_params(labelsize=24)

        try:
            for sidei in range(sides):
                snan = nnan[sidei]
                p,v = np.polyfit(x[sidei,snan], y[sidei,snan], deg=1, cov=True, w=weights[sidei,snan])
                pa_line_eq[i,sidei]=p
                pa_offset[i,sidei,:] = np.rad2deg([math.atan(p[0]), math.atan(np.sqrt(v[0][0]))])
                pa_tmp = 90-(pa[0]-pa_offset[i,sidei,0])
                pa_err = pa_offset[i,sidei,1]

                ax.plot((-1+(sidei*2))*np.insert(horizontal_separation,0, 0), p[0]*(-1+(sidei*2))*np.insert(horizontal_separation,0, 0) + p[1], label='PA={0:.2f}$^\circ$$\pm$ {1:.2f}'.format(pa_tmp,pa_err),lw=3.5,ls='--')

            ax.legend(frameon=False, loc='upper left', fontsize=28)
            ax.grid(True)
            plt.show()

            path = f.replace(data_path, data_path + '/PA_calculations')
            fig.savefig(path.replace('.fits', '_pc{p}_sides_fit.png').format(p=pcs[pci(i)]))

        except np.linalg.LinAlgError:
            pa_offset[i] = np.nan
            ax.legend(frameon=False, loc='upper left', fontsize=28)
            ax.grid(True)
            plt.show()

            print('Line fit failed')

    return pa_offset, pa_line_eq



def OffsetPA(pa_fit_range, horizontal_separation, nb_algo, centroid_fit, centroid_error,
             algo_master, algo_description, files, pcs, pci, pa, data_path):

    pa_fit = np.array(list(set(np.where(horizontal_separation>=pa_fit_range[0])[0])&
                           set(np.where(horizontal_separation<=pa_fit_range[1])[0])))
    pa_fit_sep = horizontal_separation[pa_fit]

    pa_centroid_fit = centroid_fit[:,:,pa_fit]
    pa_centroid_error = centroid_error[:,:,pa_fit]

    pa_offset = np.zeros((nb_algo,2)) ##value AND error
    x = np.concatenate((-pa_fit_sep[::-1],pa_fit_sep))
    sep = np.concatenate((-horizontal_separation[::-1],horizontal_separation))

    pa_line_eq = np.zeros((nb_algo,2))

    vert = np.nanmax(abs(centroid_fit)+centroid_error)
    vert = float('{0:.2g}'.format(vert + vert/10))

    for i,f in enumerate(files):
        y = np.concatenate((pa_centroid_fit[i,0,::-1], pa_centroid_fit[i,1,:]))
        nnan = np.isfinite(y)

        sigma = np.concatenate((pa_centroid_error[i,0,::-1], pa_centroid_error[i,1,:]))
        sigma[np.where(sigma==0)[0]] = 1.
        weights = np.ones_like(sigma)/sigma

        fig, ax = plt.subplots(1,1,figsize=(20,14))
        ax.errorbar(horizontal_separation, centroid_fit[i,1,:], centroid_error[i,1,:], color='r', marker='', capsize=8, label='fitted spine',lw=3.5)
        ax.errorbar(-horizontal_separation[::-1], centroid_fit[i,0,::-1], centroid_error[i,0,::-1], color='r', marker='', capsize=8,lw=3.5)
        ax.errorbar(x, y, sigma, marker='o', ls='', capsize=8, label='PA fitting points',lw=3.5,markersize=6)
        ax.set_title(algo_master + ' ' + algo_description[i] + ' Gaussian profile fit', fontsize=30,loc='left')
        ax.set_ylabel('Centroid offset [arcsec]', fontsize=28)
        ax.set_xlabel('Separation [arcsec]', fontsize=28)
        ax.set_ylim(-vert,vert)
        #ax.set_ylim(-7.46/1000*10,7.46/1000*10)
        ax.tick_params(labelsize=24)

        try:
            p,v = np.polyfit(x[nnan],y[nnan],deg=1,cov=True,w=weights[nnan])
            pa_line_eq[i]=p

            ##error is square root of slope element of covarience matrix
            pa_offset[i] = np.rad2deg([math.atan(p[0]), math.atan(np.sqrt(v[0][0]))])
            pa_tmp = 90-(pa[0]-pa_offset[i,0])
            pa_err = pa_offset[i,1]
            ax.plot(sep, p[0]*sep + p[1], label='PA={0:.2f}$^\circ$$\pm$ {1:.2f}'.format(pa_tmp,pa_err),lw=3.5,ls='--')
            ax.legend(frameon=False, loc='upper left', fontsize=28)
            #ax.set_ylim(-2,2)
            ax.grid(True)
            plt.show()

            path = f.replace(data_path, data_path + '/PA_calculations')
            fig.savefig(path.replace('.fits','_pc{p}_fit.png').format(p=pcs[pci(i)]))

        except np.linalg.LinAlgError:
            pa_offset[i] = np.nan
            ax.legend(frameon=False,loc='upper left', fontsize=28)
            ax.grid(True)
            plt.show()

            print('Line fit failed')

    return pa_offset, pa_line_eq



def FindPA(horizontal_separation, nb_algo, centroid_fit, centroid_error, algo_master,
           algo_description, files, pcs, pci, pa, pa_fit_range, sides, data_path, save_name,
           pxscale, target_info, full=True, save_table=False, return_list=False):

    if not os.path.exists(data_path + '/PA_calculations/data'):
        os.makedirs(data_path + '/PA_calculations/data')

    ##multiplied by an array of shape nb_algs in case of single input PA
    initial_pa = 90-(np.ones(nb_algo)*pa)
    if full:
        pa_offset, pa_line_eq = OffsetPA((pxscale/1000)*pa_fit_range, (pxscale/1000)*horizontal_separation, nb_algo,
                                         (pxscale/1000)*centroid_fit, (pxscale/1000)*centroid_error, algo_master,
                                         algo_description, files, pcs, pci, pa, data_path)
        measured_pa = 90-(pa-pa_offset[:,0])

    else:
        pa_offset, pa_line_eq = SidesOffsetPA((pxscale/1000)*pa_fit_range, (pxscale/1000)*horizontal_separation, nb_algo,
                                              (pxscale/1000)*centroid_fit, (pxscale/1000)*centroid_error, algo_master,
                                              algo_description, files, pcs, pci, pa, data_path)
        measured_pa = 90-(pa-pa_offset[:,:,0])

    if save_table == True:

        col_init = fits.Column(name='Initial_PA', format='D', unit='deg', array=initial_pa)
        col_algo = fits.Column(name='Description', format='50A', array=[x.split(' PA')[0] for x in algo_description])

        if full:
            col1 = fits.Column(name='PA', format='D', unit='deg', array=measured_pa)
            col2 = fits.Column(name='err', format='D', unit='deg', array=pa_offset[:,1])
            coldefs = fits.ColDefs([col1, col2, col_init, col_algo])
            save_name+= '_measured_PA.fits'

        else:
            col1 = fits.Column(name='{0:s}_PA'.format(sides[0]), format='D', unit='deg', array=pa_offset[:,0,0])
            col2 = fits.Column(name='{0:s}_err'.format(sides[0]), format='D', unit='deg', array=pa_offset[:,0,1])
            col3 = fits.Column(name='{0:s}_PA'.format(sides[1]), format='D', unit='deg', array=pa_offset[:,1,0])
            col4 = fits.Column(name='{0:s}_err'.format(sides[1]), format='D', unit='deg', array=pa_offset[:,1,1])
            coldefs = fits.ColDefs([col1, col2, col3, col4, col_init, col_algo])
            save_name+= '_measured_sides_PA.fits'

        hdr = fits.Header()
        hdr['TARGET'] = target_info.split(' ')[0]
        hdr['EPOCH'] = target_info.split(' ')[1]
        hdr['FILTER'] = target_info.split(' ')[2]
        hdr['MASTER'] = algo_master
        hdr['RANG_PX'] = str(pa_fit_range)
        hdr['RANG_MAS'] = str(pa_fit_range*pxscale)

        table_hdu = fits.BinTableHDU.from_columns(coldefs, header=hdr)
        table_hdu.writeto(os.path.join(data_path,'PA_calculations/data',save_name),overwrite=True)

    if return_list == False:
        pa_offset = pa_offset[0]

    return pa_offset.T, pa_line_eq


def IteratePA(images, noise_images, gauss_fit, gauss_error, horizontal_separation, horizontal_bin,
              vertical_displacement, vertical_length, rad, algo_master, algo_description, files,
              pcs, pci, pa, pa_fit_range, sides, save_name, target_info, pxscale, dstar, data_path,
              full=True, save_table=False):

    nb_algo = len(files)
    if full == True:
        pa_offset, pa_error = FindPA(horizontal_separation, nb_algo, gauss_fit['x0'], gauss_error['x0'],
                                     algo_master, algo_description, files, pcs, pci, pa, pa_fit_range,
                                     sides, data_path, save_name, target_info, full, save_table)

        count=0
        while abs(pa_offset)>=pa_error and pa_error>0.1:
            print('\n--------- Iteration {0:d} ---------\n'.format(count+1))
            pa[0]-=pa_offset

            aligned_images = vip.preproc.cube_derotate(images,np.ones((nb_algo))*(pa),imlib='opencv')
            aligned_noise = vip.preproc.cube_derotate(noise_images,np.ones((nb_algo))*(pa),imlib='opencv')

            gauss_fit, gauss_error = pfit.FitProfiles(aligned_images, aligned_noise, nb_algo,
                                                      horizontal_separation, vertical_displacement,
                                                      horizontal_bin, vertical_length, rad, files,
                                                      pcs, pci, sides, pxscale, dstar, data_path)

            pa_offset, pa_error = FindPA(horizontal_separation, nb_algo, gauss_fit['x0'], gauss_error['x0'],
                                         algo_master, algo_description, files, pcs, pci, pa, pa_fit_range,
                                         sides, data_path, save_name, pxscale, target_info, full, save_table)
            count+=1
            if count>10: print("Horizontal PA not found after 10 iterations"); break

        pa[0]-=pa_offset
        aligned_images = vip.preproc.cube_derotate(images,np.ones((nb_algo))*(pa),imlib='opencv')
        aligned_noise = vip.preproc.cube_derotate(noise_images,np.ones((nb_algo))*(pa),imlib='opencv')

        gauss_fit, gauss_error = pfit.FitProfiles(aligned_images, aligned_noise, nb_algo,
                                                  horizontal_separation, vertical_displacement,
                                                  horizontal_bin, vertical_length, rad, files,
                                                  pcs, pci, sides, pxscale, dstar, data_path)

        pa_offset, pa_error = FindPA(horizontal_separation, nb_algo, gauss_fit['x0'], gauss_error['x0'],
                                     algo_master, algo_description, files, pcs, pci, pa, pa_fit_range,
                                     sides, data_path, save_name, pxscale, target_info, full, save_table)
        pa[0]-=pa_offset

        print("\nMeasured PA =",90-(pa[0]),'+/-',pa_error)

        return gauss_fit, pa_offset, pa_error, aligned_images, aligned_noise

    if full == False:
        sep_offset, sep_error = FindPA(horizontal_separation, nb_algo, gauss_fit['x0'], gauss_error['x0'],
                                       algo_master, algo_description, files, pcs, pci, pa, pa_fit_range,
                                       sides, data_path, save_name, pxscale, target_info, full, save_table)
        pa_offset=sep_offset.mean()
        pa_error=sep_offset.std()

        count=0
        while abs(pa_offset)>=pa_error and pa_error>0.5:
            pa[0]-=pa_offset

            aligned_images = vip.preproc.cube_derotate(images,np.ones((nb_algo))*(pa),imlib='opencv')
            aligned_noise = vip.preproc.cube_derotate(noise_images,np.ones((nb_algo))*(pa),imlib='opencv')

            gauss_fit, gauss_error = pfit.FitProfiles(aligned_images, aligned_noise, nb_algo,
                                                      horizontal_separation, vertical_displacement,
                                                      horizontal_bin, vertical_length, rad, files,
                                                      pcs, pci, sides, pxscale, dstar, data_path)

            sep_offset, sep_error = FindPA(horizontal_separation, nb_algo, gauss_fit['x0'], gauss_error['x0'],
                                           algo_master, algo_description, files, pcs, pci, pa, pa_fit_range,
                                           sides, data_path, save_name, pxscale, target_info, full, save_table)
            pa_offset=sep_offset.mean()
            pa_error=np.sqrt(np.sum(np.square(pa_error))) #sep_offset.std()

            count+=1
            if count>10: print("Horizontal PA not found after 10 iterations"); break

        pa[0]-=pa_offset
        aligned_images = vip.preproc.cube_derotate(images,np.ones((nb_algo))*(pa),imlib='opencv')
        aligned_noise = vip.preproc.cube_derotate(noise_images,np.ones((nb_algo))*(pa),imlib='opencv')

        gauss_fit, gauss_error = pfit.FitProfiles(aligned_images, aligned_noise, nb_algo,
                                                  horizontal_separation, vertical_displacement,
                                                  horizontal_bin, vertical_length, rad, files,
                                                  pcs, pci, sides, pxscale, dstar, data_path)

        sep_offset, sep_error = FindPA(horizontal_separation, nb_algo, gauss_fit['x0'], gauss_error['x0'],
                                       algo_master, algo_description, files, pcs, pci, pa, pa_fit_range,
                                       sides, data_path, save_name, pxscale, target_info, full, save_table)
        pa_offset=sep_offset.mean()
        pa_error=np.sqrt(np.sum(np.square(pa_error)))

        for i in range(len(sides)):
            print('\nMeasured {s} PA ='.format(s=sides[i]),90-(pa[0]-sep_offset[i]),'+/-',sep_error[i],'\n')

        return gauss_fit, pa_offset, pa_error, aligned_images, aligned_noise