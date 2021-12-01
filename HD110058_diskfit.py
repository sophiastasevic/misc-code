#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 14:17:46 2021

@author: sstas
"""

#HD110058 preliminary model

import os
import numpy as np
from astropy.io import fits
import vip_hci as vip
ds9=vip.Ds9Window()

pixel_scale=0.01225 # pixel scale in arcsec/px

nx = 191 # number of pixels of your image in X
ny = 191 # number of pixels of your image in Y

#PA = #-90+15.
#itilt=#80.
e=0.0
omega=0.
aniso_g=0
gamma=2.
ksi0=1.
dstar=130.
#r0=#24.*pixel_scale*dstar
ain=10.
aout=-4.

r0_HD110058 = 39. #inner planetessimal belt radius 0.3", adjusted to new Gaia distance
PA_HD110058 = 155.
itilt_HD110058 = 88.5 #'near edge on'

#alphaout_HD110058 = #-4.9849274
# flux_HD110058 = #965.16694895

#%%
fake_disk_HD110058 = vip.metrics.scattered_light_disk.ScatteredLightDisk(\
                        nx=nx,ny=ny,distance=dstar,\
                        itilt=itilt_HD110058,omega=omega,pxInArcsec=pixel_scale,pa=PA_HD110058,\
                        density_dico={'name':'2PowerLaws','ain':ain,'aout':aout,\
                        'a':r0_HD110058,'e':0.0,'ksi0':ksi0,'gamma':gamma,'beta':1.},\
                        spf_dico={'name':'HG', 'g':aniso_g, 'polar':False}) #add flux_max parameter --> adi tends to reduce flux by half for looking at sphere image

#fake_disk_HD114082.print_info()

fake_disk_HD110058_map = fake_disk_HD110058.compute_scattered_light()
ds9.display(fake_disk_HD110058_map)

#%%
fake_disk = vip.metrics.scattered_light_disk.ScatteredLightDisk(\
                        nx=nx,ny=ny,distance=dstar,\
                        itilt=itilt_HD110058,omega=omega,pxInArcsec=pixel_scale,pa=PA_HD110058,\
                        density_dico={'name':'2PowerLaws','ain':ain,'aout':aout,\
                        'a':r0_HD110058,'e':0.0,'ksi0':ksi0,'gamma':gamma,'beta':1.},\
                        spf_dico={'name':'HG', 'g':aniso_g, 'polar':False})
    

fake_disk_map = fake_disk.compute_scattered_light()


ds9.display(fake_disk_HD110058_map,fake_disk_map)


#%%

# we create a synthetic cube of 30 images
#nframes = 30
# actual: -155.468, -121.221
derotation_angles = np.arange(-155,-121,1)

# we create a cube with the disk injected at the correct parallactic angles:
#cube_fake_disk1 = vip.metrics.cube_inject_fakedisk(fake_disk1_map,derotation_angles)
cube_fake_disk_HD110058 = vip.metrics.cube_inject_fakedisk(fake_disk_HD110058_map,-derotation_angles)

cadi_fakedisk_HD110058 = vip.medsub.median_sub(cube_fake_disk_HD110058,derotation_angles)

#cube_fake_disk = vip.metrics.cube_inject_fakedisk(fake_disk_map,-derotation_angles)
#cadi_fakedisk = vip.medsub.median_sub(cube_fake_disk,derotation_angles)

ds9.display(fake_disk_HD110058_map,cadi_fakedisk_HD110058)
#ds9.display(cadi_fakedisk_HD110058, cadi_fakedisk)


#%%
# Now if have a PSF and want to also convolve the disk with the PSF, we can do that ! 

data_dir='/mnt/c/Users/sophi/Documents/PhD/sphere_data/SPHERE_DC_DATA'
psf_path='HIP 61782_DB_H23_2015-04-12_ird_convert_recenter_dc5_90436/ird_convert_recenter_dc5-IRD_SCIENCE_PSF_MASTER_CUBE-median_unsat.fits'

path=os.path.join(data_dir,psf_path)

with fits.open(path,ignore_missing_end=True, verbose=False) as hdul:
    psf = hdul[0].data

psf=psf[0]

# Then we inject the disk in the cube and convolve by the PSF
cube_fake_disk_HD110058_convolved = vip.metrics.cube_inject_fakedisk(fake_disk_HD110058_map,-derotation_angles,psf=psf)
cadi_fakedisk_HD110058_convolved = vip.medsub.median_sub(cube_fake_disk_HD110058_convolved,derotation_angles)

#cube_fake_disk_convolved = vip.metrics.cube_inject_fakedisk(fake_disk_map,-derotation_angles,psf=psf)
#cadi_fakedisk_convolved = vip.medsub.median_sub(cube_fake_disk_convolved,derotation_angles)

#ds9.display(cadi_fakedisk_HD110058_convolved,cadi_fakedisk_convolved)
ds9.display(fake_disk_HD110058_map,cadi_fakedisk_HD110058,cadi_fakedisk_HD110058_convolved)