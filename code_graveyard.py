#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unused functions
"""
#%%
##remove bad pixels

import os
import vip_hci as vip
import astropy.io as fits
from ADI_reduction import FileIn

remove_badpx = False #set to True to remove bad pixels from cube, or load cube
                    #with bad pixels removed if already exists

#removes bad pixels in science cube + saves processed cube
def RemoveBadPixels(cube, channels, r_mask, save_path):
    for i in range(channels):
        cube[i] = vip.preproc.badpixremoval.cube_fix_badpix_isolated(cube[i], bpm_mask=None,
                                sigma_clip=3, num_neig=5, size=5, frame_by_frame=False,
                                protect_mask=False, radius=r_mask, verbose=False)
    hdu_new = fits.PrimaryHDU(data=cube)
    hdu_new.writeto(save_path, overwrite=True)

    return cube

def FormatCube(channels, r_mask, cube_path, remove_badpx, data_dir):
    path_badpxrm = os.path.join(data_dir,cube_path+'_badpxrm.fits')
    if os.path.isfile(path_badpxrm) == True and remove_badpx == True:
        cube = FileIn(data_dir,cube_path+'_badpxrm.fits')
        badpxrm_check = True
    else:
        cube = FileIn(data_dir,cube_path)
        badpxrm_check = False

    #if specified to remove bad pixels and file does not already exist
    if remove_badpx == True and badpxrm_check == False:
        cube = RemoveBadPixels(cube, channels, r_mask, data_dir,cube_path+'_badpxrm.fits')

#%%
##change methods to be processed

def mod_methods(method, available_methods):
    method_tmp=input('Add or remove a method to reprocessed. [press enter to skip]: ')
    while len(method_tmp)!=0:
        if method_tmp in method:
            method.remove(method_tmp)
            print(method_tmp, 'removed!')
        elif method_tmp in available_methods:
            method.append(method_tmp)
            print(method_tmp, 'added!')
        else:
            print('Input not recognised. \nAvailable reduction methods are: ',
                  ', '.join('{0:s}'.format(m) for i,m in enumerate(available_methods)))
        method_tmp=input('Add or remove a method to be reprocess. [press enter to skip]: ')

#%%