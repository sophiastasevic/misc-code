#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test vip rotation
"""

import numpy as np
import vip_hci as vip
import cv2
from cv2 import getRotationMatrix2D, warpAffine

ds9=vip.Ds9Window()

#%%
x,y = 200,200
cx, cy = x//2,y//2

test_img = np.zeros((x,y))
test_img[:,cy] = 1
test_img[cx,:] = 1
derotation_angles = np.arange(-90,91,1)

#%%
cube1 = vip.fm.cube_inject_fakedisk(test_img,derotation_angles,imlib='vip-fft')
cube2 = vip.fm.cube_inject_fakedisk(test_img,derotation_angles,imlib='opencv')
cube3 = vip.fm.cube_inject_fakedisk(test_img,derotation_angles,imlib='skimage')

cube4 = np.zeros((len(derotation_angles),x,y))
for i in range(len(derotation_angles)):
    rot = getRotationMatrix2D((cx,cy),derotation_angles[i],1)
    cube4[i] = warpAffine(test_img, rot, (x,y) ,flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT)

ds9.display(cube1,cube2,cube3,cube4)

#%%
scale_img = cv2.resize(test_img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
test_img2 = np.zeros((round(0.5*x),round(0.5*x)))
test_img2[:,round(0.5*y)//2] = 1
test_img2[round(0.5*x)//2,:] = 1

ds9.display(test_img2, scale_img)
#%%
x,y=(10,10)
new_x, new_y=(9,9)

##centre = centre of pixel
a = np.zeros((x,y))
a[:,x//2] = 1
a[x//2,:] = 1

if new_x%2:
    x+=1
if new_y%2:
    y+=1
xmin, xmax, ymin, ymax=((x-new_x)//2, (x+new_x)//2, (y-new_y)//2, (y+new_y)//2)

b = a[xmin:xmax,ymin:ymax]

a_rot = vip.metrics.cube_inject_fakedisk(a,np.array((-90,-45,0,45,90)),imlib='opencv')
b_rot = vip.metrics.cube_inject_fakedisk(b,np.array((-90,-45,0,45,90)),imlib='opencv')

##centre = top left of pixel
a[:,x//2-1] = 1
a[x//2-1,:] = 1

b = a[xmin:xmax,ymin:ymax]

cxy=(x//2-.5,y//2-.5)
a_rot = vip.metrics.cube_inject_fakedisk(a,np.array((-90,-45,0,45,90)),cxy=cxy,imlib='opencv')
cxy=(new_x//2-.5,new_y//2-.5)
b_rot = vip.metrics.cube_inject_fakedisk(b,np.array((-90,-45,0,45,90)),cxy=cxy,imlib='opencv')
