#script to derotate data using angles from image header
#create file called 'starnames.txt' with names of stars needing to be derotated
#with the name in the form 'HIP_#####' 

import scipy 
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import warnings
warnings.simplefilter("ignore")

#reads in the names of the images
def ReadIn(name):
    
    file_names=np.loadtxt('{name}_filenames.txt'.format(name=name),dtype=str)
    for i in range(0,file_names.size):
        file_names[i] = file_names[i].rstrip()
    return file_names

#rotates each image in the centred cube by the angle taken from the header
def RotCube(name,angles):
    
    with fits.open('centeredcube_{name}.fits'.format(name=name),ignore_missing_end=True, verbose=False) as hdul:
        cube = hdul[0].data #creates 3D array of centred cube
    n_images=cube[:,0][:,0].size
    n_pixels=cube[0][:,0].size #assuming image is square
    rot_cube=np.empty((n_images,n_pixels,n_pixels))
    for n in range(0,n_images):
        rot_cube[n]=scipy.ndimage.rotate(cube[n],angles[n])
        
    hdu_new = fits.PrimaryHDU(rot_cube)
    hdul_new = fits.HDUList([hdu_new])
    hdul_new.writeto('derot_centeredcube_{name}'.format(name=name))
    #plt.imshow(img_rot, origin='lower')

#reads in images and takes the angles from their headers to calculate the correction angle
def FindAngleCorrections(name):
    
    file_names = ReadIn(name)
    n_images= file_names.size
    
    angles = np.zeros(n_images)
    for i in range(0, n_images):
        with fits.open('{file}'.format(file=file_names[i]),ignore_missing_end=True, verbose=False) as hdul:
            img_data = hdul[0].header
        
        angles[i] = img_data['ROTPOSN']-img_data['INSTANGL']
    
    np.savetxt('{name}_rotation.txt'.format(name=name), angles)
    return angles

#finding angles and derotating centred cube for each star in the file
def FindAnglesAndDerot():
    
    star_name=np.loadtxt('starnames.txt',dtype=str)
    
    n_stars=star_name.size
    for i in range(0,n_stars):
        if n_stars==1:
            name=star_name
        else:
            name=star_name[i]
        angles=FindAngleCorrections(name)
        RotCube(name,angles)
        
FindAnglesAndDerot()

"""
#function for rotating single frames
def RotImage(angles,images):
    
    for n in range(0,angles.size):
        with fits.open('{file}'.format(file=images),ignore_missing_end=True, verbose=False) as hdul:
            img = hdul[0].data
        img_rot=scipy.ndimage.rotate(img,angles[n])
        hdu_new = fits.PrimaryHDU(img_rot)
        hdul_new = fits.HDUList([hdu_new])
        hdul_new.writeto('derot_{name}'.format(name=images[n]))
        plt.imshow(img_rot, origin='lower')
"""
      