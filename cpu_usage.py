#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:49:25 2022

@author: stasevis
"""

import psutil
import numpy as np
from astropy.io import fits
from datetime import datetime

def distance(x1, y1, x2, y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

def create_mask(crop_size, inner_radius, outer_radius):
    count = 0
    res = np.full((crop_size, crop_size), True)
    x = crop_size//2
    y = crop_size//2
    for i in range(crop_size):
        for j in range(crop_size):
            if distance(i, j, x, y) >= outer_radius or distance(i, j, x, y) <= inner_radius:
                res[i,j] = False
                count = count + 1
    return res

p=psutil.Process()
print('Init CPU usage', p.cpu_percent(1),'\nMemory',p.memory_percent())
inner_radius=20
outer_radius=60
crop_size = 2*outer_radius
mask = create_mask(crop_size, inner_radius, outer_radius)

path="/mnt/c/Users/stasevis/Documents/sphere_data/HD110058/HD110058_H23_2015-04-12_ird_convert_recenter/SCIENCE_REDUCED_MASTER_CUBE-center_im.fits"

start_time=datetime.now()
#cpusage=[(datetime.now(),psutil.cpu_percent())]
cube=fits.getdata(path)
ny,nx=cube.shape[-2:]
if (ny+crop_size)%2:
    ny+=1
border_l = (ny - crop_size)//2
border_r = (ny + crop_size)//2
ref_frames = cube[..., border_l:border_r, border_l:border_r]
wl_ref, nb_ref_frames, ref_x, ref_y = ref_frames.shape
print("\nLoad time:", str(datetime.now()-start_time))
#%%
print('CPU usage', p.cpu_percent(),'\nMemory',p.memory_percent())
start_time=datetime.now()
res = np.zeros((nb_ref_frames,nb_ref_frames))
#cpusage.append((datetime.now(),psutil.cpu_percent()))
corr_mat = []
for i in range(nb_ref_frames):
    for j in range(nb_ref_frames):
        res[i,j]= np.corrcoef(np.reshape(ref_frames[0,i]*mask, ref_x*ref_y),np.reshape(ref_frames[0,j]*mask, ref_x*ref_y))[0,1]
        #cpusage.append((datetime.now(),psutil.cpu_percent()))

corr_mat.append(res)
matrix=np.concatenate(np.array(corr_mat),axis=-1)
print("\nCalculation time:", str(datetime.now()-start_time))
print('CPU usage', p.cpu_percent(),'\nMemory',p.memory_percent())