#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 11:05:30 2022

@author: millij
"""
#from image_tools import distance_array
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import vip_hci as vip
import os
ds9=vip.Ds9Window()

TARGET_EPOCH="2015-04-12"
TARGET_NAME="HD110058"
INSTRUMENT="ird"
FILTER="H23"

sphere_dir ='/mnt/c/Users/stasevis/Documents/sphere_data'
data_path = TARGET_NAME + '/' + TARGET_NAME + '_' + FILTER+ '_' + TARGET_EPOCH + '_' + INSTRUMENT

parang_path = os.path.join(sphere_dir, data_path + '_convert_recenter/SCIENCE_PARA_ROTATION_CUBE-rotnth.fits')
cube_path = os.path.join(sphere_dir, data_path + '_convert_recenter/SCIENCE_REDUCED_MASTER_CUBE-center_im.fits')
frame_path = os.path.join(sphere_dir, data_path + '_convert_recenter/FRAME_SELECTION_VECTOR-frame_selection_vector.fits')
psf_path = os.path.join(sphere_dir, data_path + '_convert_recenter/SCIENCE_PSF_MASTER_CUBE-median_unsat.fits')

fake_disk_path = os.path.join(sphere_dir, data_path + '_fake_disk_injection')

cube = fits.getdata(cube_path)
parang = fits.getdata(parang_path)
psf = fits.getdata(psf_path)
header = fits.getheader(cube_path)
frame_vector = fits.getdata(frame_path)

if not os.path.exists(fake_disk_path):
    os.makedirs(fake_disk_path)

# ds9.display(psf)

case = 'faint'
#case = 'bright'

if case=='bright':
    flux_max=100
elif case=='faint':
    flux_max=10

pixel_scale=0.01225 # pixel scale in arcsec/px
dstar= 129.9849 # distance to the star in pc
nx = 200 # number of pixels of your image in X
ny = 200 # number of pixels of your image in Y

reduced_cube_path= '/mnt/c/Users/stasevis/Documents/RDI/ADI/output/HD110058/2015-04-12/ird_H_PCA_temp-mean_stack.fits'
reduced_img= fits.getdata(reduced_cube_path)
im_x,im_y=reduced_img.shape[1:]
reduced_crop=reduced_img[3, (im_x-nx)//2:(im_x+nx)//2, (im_y-ny)//2:(im_y+ny)//2]
#%%
# Warning:
#        The star is assumed to be centered at the frame center as defined in
#        the vip_hci.var.frame_center function, e.g. at pixel [ny//2,nx//2]
#        (if odd: nx/2-0.5 e.g. the middle of the central pixel
#        if even: nx/2)

itilt = 87 # inclination of your disk in degreess
pa= 157  # position angle of the disk in degrees (from north to east)
# a = 55 # semimajoraxis of the disk in au : Sma: 55.0au or   423mas
a = 36 # semimajoraxis of the disk in au : Sma: 50.0au or   385mas


print('Sma: {0:3.1f}au or {1:5.0f}mas'.format(a,a/dstar*1000))

fake_disk_cropped = vip.fm.scattered_light_disk.ScatteredLightDisk(\
                        nx=nx,ny=ny,distance=dstar,\
                        itilt=itilt,omega=0,pxInArcsec=pixel_scale,pa=pa,\
                        density_dico={'name':'2PowerLaws','ain':12,'aout':-3.8,\
                        'a':a,'e':0.0,'ksi0':3.,'gamma':3.,'beta':1.},\
                        spf_dico={'name':'DoubleHG','g':[0.5,0.2],'weight':0.4,'polar':False},\
                        flux_max=flux_max)
fake_disk_cropped_map = fake_disk_cropped.compute_scattered_light()
ds9.display(fake_disk_cropped_map,reduced_crop)
"""
fake_disk_cropped1 = vip.fm.scattered_light_disk.ScatteredLightDisk(\
                        nx=nx,ny=ny,distance=dstar,\
                        itilt=itilt,omega=0,pxInArcsec=pixel_scale,pa=pa,\
                        density_dico={'name':'2PowerLaws','ain':12,'aout':-4,\
                        'a':a,'e':0.0,'ksi0':3.,'gamma':3.,'beta':1.},\
                        spf_dico={'name':'DoubleHG','g':[0.5,0.2],'weight':0.5,'polar':False},\
                        flux_max=flux_max)
fake_disk_cropped1_map = fake_disk_cropped1.compute_scattered_light()

ds9.display(fake_disk_cropped_map,fake_disk_cropped1_map)
ds9.display(fake_disk_cropped_map,reduced_crop)
#%%
"""
cube_w_fd_fullframe = np.copy(cube)
fakedisk_fullframe_convolved = np.zeros((2,1024,1024))
fakedisk_fullframe_unconvolved = np.zeros((1024,1024))
pca_filtered = np.zeros((2,200,200))
pca_filtered_fd = np.zeros((2,200,200))

# ichannel=0
# ichannel=1
def distance_array(shape):
    nx=shape[0]
    ny=shape[1]
    x=np.abs(np.arange(-nx//2,nx//2))
    y=np.abs(np.arange(-ny//2,ny//2))
    grid=np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            grid[i][j]=np.sqrt((x[i]**2)+(y[j]**2))

    return grid
#%%

for ichannel in range(2):
    cube_cropped = cube[ichannel,:,1024//2-100:1024//2+100,1024//2-100:1024//2+100]
    pca_img = vip.psfsub.pca(cube_cropped, -parang , ncomp=5, verbose=False,mask_center_px=10,imlib='opencv')
    ds9.display(pca_img)

    t,_,_,res,_= vip.psfsub.pca(cube_cropped, -parang , ncomp=5, verbose=False,mask_center_px=10,imlib='opencv',full_output=True)
    ds9.display(res)

    distarr = distance_array(pca_img.shape)
    nb_frames = len(parang)
    residual_rms = np.zeros((len(parang)))
    for j in range(nb_frames):
        tmp = res[j,:,:]
        residual_rms[j] = np.std(tmp[distarr<30])
    plt.plot(residual_rms)
    threshold = np.percentile(residual_rms,85)
    plt.axhline(threshold)
    plt.plot(np.arange(nb_frames)[residual_rms>threshold],residual_rms[residual_rms>threshold],'ro')


    good_frames = np.arange(nb_frames)[residual_rms<threshold]
    pca_img_filtered = vip.psfsub.pca(cube_cropped[good_frames], -parang[good_frames] , ncomp=5, verbose=False,mask_center_px=10,imlib='opencv')
    ds9.display(pca_img,pca_img_filtered)
    pca_filtered[ichannel,:,:] = pca_img_filtered

    psf_channel = psf[ichannel,:,:]
    psf_channel = psf_channel/np.sum(psf_channel)

    cube_fake_disk_convolved = vip.fm.cube_inject_fakedisk(fake_disk_cropped_map,parang,psf=psf_channel,imlib='opencv')

    cube_w_fd = cube_cropped+cube_fake_disk_convolved

    pca_img_fd = vip.psfsub.pca(cube_w_fd, -parang , ncomp=5, verbose=False,mask_center_px=10,imlib='opencv')
    pca_filtered_fd[ichannel,:,:] = vip.psfsub.pca(cube_w_fd[good_frames], -parang[good_frames] , ncomp=5, verbose=False,mask_center_px=10,imlib='opencv')


    ds9.display(pca_img,pca_img_fd)

    cube_w_fd_fullframe[ichannel,:,1024//2-100:1024//2+100,1024//2-100:1024//2+100] += cube_fake_disk_convolved

    fakedisk_convolved = np.median(vip.preproc.cube_derotate(cube_fake_disk_convolved,-parang,imlib='opencv'),axis=0)
    fakedisk_fullframe_convolved[ichannel,1024//2-100:1024//2+100,1024//2-100:1024//2+100] = fakedisk_convolved

pca_img_fd_rot=vip.preproc.frame_rotate(pca_img_fd,-101)
ds9.display(pca_img,pca_img_fd_rot)

fakedisk_fullframe_unconvolved[1024//2-100:1024//2+100,1024//2-100:1024//2+100] = fake_disk_cropped_map
#%%

fits.writeto(os.path.join(fake_disk_path, 'SCIENCE_REDUCED_MASTER_CUBE-center_im_process_{0:s}_disk.fits'.format(case)), cube_w_fd_fullframe, header, overwrite=True)
fits.writeto(os.path.join(fake_disk_path, 'SCIENCE_PARA_ROTATION_CUBE-rotnth.fits'), parang, header, overwrite=True)
fits.writeto(os.path.join(fake_disk_path, 'SCIENCE_PSF_MASTER_CUBE-median_unsat.fits'), psf, header, overwrite=True)
fits.writeto(os.path.join(fake_disk_path, 'FRAME_SELECTION_VECTOR-frame_selection_vector.fits'), frame_vector, header, overwrite=True)

fits.writeto(os.path.join(fake_disk_path, 'fake_disk_{0:s}_unconvolved.fits'.format(case)), fakedisk_fullframe_unconvolved, header, overwrite=True)
fits.writeto(os.path.join(fake_disk_path, 'fake_disk_{0:s}_convolved.fits'.format(case)), fakedisk_fullframe_convolved, header, overwrite=True)

fits.writeto(os.path.join(fake_disk_path, 'pca_klip5_no_fakedisk.fits'), pca_filtered, header, overwrite=True)
fits.writeto(os.path.join(fake_disk_path, 'pca_klip5_with_{0:s}_fakedisk.fits'.format(case)), pca_filtered_fd, header, overwrite=True)


