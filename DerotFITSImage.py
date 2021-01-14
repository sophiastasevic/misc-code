#script to derotate data using angles from 'rotdest' in image header
#create file called 'starnames.txt' with names of stars needing to be derotated
#with the name in the form 'HIP_#####' 

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def ReadIn(name):
    angles=np.loadtxt('{name}_rotation.txt'.format(name=name),dtype=float)
    images=np.loadtxt('{name}_filenames.txt'.format(name=name),dtype=str)
    return angles,images

def RotImage(angles,images):
    for n in range(0,angles.size):
        if angles.size!=1:
            angles=angles[n]
            images=images[n]
        with fits.open('{file}'.format(file=images),ignore_missing_end=True, verbose=False) as hdul:
            img = hdul[0].data
        img_rot=sp.ndimage.rotate(img,angles)
        hdu_new = fits.PrimaryHDU(img_rot)
        hdul_new = fits.HDUList([hdu_new])
        hdul_new.writeto('derot_{name}'.format(name=images))
        plt.imshow(img_rot, origin='lower')

star_name=np.loadtxt('starnames.txt',dtype=str)
for i in range(0,star_name.size):
    if star_name.size==1:
        name=star_name
    else:
        name=star_name[i]
    data=ReadIn(name)
    angles=data[0]
    images=data[1]
    RotImage(angles,images)