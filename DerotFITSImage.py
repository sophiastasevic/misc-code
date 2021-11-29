#script to derotate data using angles from image header
#create test file called 'starnames.txt' with names of stars needing to be derotated with the name in the form 'HIP_#####' 

import scipy 
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import warnings
warnings.simplefilter("ignore")

#reads in list containing the file names of the individual science cube frames
def ReadIn(name):
    
    file_names=np.loadtxt('{name}_filenames.txt'.format(name=name),dtype=str)
    for i in range(0,file_names.size):
        file_names[i] = file_names[i].rstrip() #removes trailing characters
    return file_names

#reads in centered cube for the star and correct each frame using the angle calculated from the header
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
    hdul_new.writeto('derot_centeredcube_{name}.fits'.format(name=name)) #saves new derotated cube
    #plt.imshow(img_rot, origin='lower')
    
#same as above but for derotating only a single image
def RotImage(name,angle):
    
    with fits.open('{name}.fits'.format(name=name),ignore_missing_end=True, verbose=False) as hdul:
        img = hdul[0].data
    img_rot=scipy.ndimage.rotate(img,angle)
    hdu_new = fits.PrimaryHDU(img_rot)
    hdul_new = fits.HDUList([hdu_new])
    hdul_new.writeto('derot_{name}.fits'.format(name=name)
    plt.imshow(img_rot, origin='lower')

#reads in header of each individual science cube frame and calculates the correction angle
def FindAngleCorrections(name):
    
    file_names = ReadIn(name)
    n_images= file_names.size
    
    angles = np.zeros(n_images)
    for i in range(0, n_images):
        with fits.open('{file}'.format(file=file_names[i]),ignore_missing_end=True, verbose=False) as hdul:
            img_data = hdul[0].header
       
        angles[i] = img_data['ROTPOSN']-img_data['INSTANGL'] #ROTPOSN: rotator position angle, including offset INSTANGL: instrument angle
    
    #for single image, comment above 7 lines and uncomment lines below:
                     
    #with fits.open('{file}.fits'.format(file=name),ignore_missing_end=True, verbose=False) as hdul:
        #img_data = hdul[0].header
    #angles = img_data['ROTPOSN']-img_data['INSTANGL']
    
    np.savetxt('{name}_rotation.txt'.format(name=name), angles) #saves list of correction angles
    return angles

#main function --> loads file 'starnames.txt' containing list of stars + calls processing functions
def FindAnglesAndDerot():
    
    star_name=np.loadtxt('starnames.txt',dtype=str) #for single image, change to fits file name (not including '.fits' at end)
    
    n_stars=star_name.size
    for i in range(0,n_stars):
        if n_stars==1:
            name=star_name
        else:
            name=star_name[i]
        angles=FindAngleCorrections(name) #reads in fits header angle info
        RotCube(name,angles) #derotates fits image cube by correction angle
        #RotImage(name,angles) #derotates single fits image by correction angle
        
FindAnglesAndDerot() 
