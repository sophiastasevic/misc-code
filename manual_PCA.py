# -*- coding: utf-8 -*-
"""
Calculates eigenvectors+values for science data to perform PCA reduction using
different number of principal components, + performs classical ADI on the PCA
reduced cube to return a final reduced image + no-disk image for noise estimate
"""

from astropy.io import fits
import numpy as np
import os
from datetime import datetime
import cv2
from cv2 import getRotationMatrix2D, warpAffine
from skimage.draw import disk
import matplotlib.pyplot as plt

TARGET_EPOCH='2015-04-12'
TARGET_NAME='HD110058'
INSTRUMENT='ird'
FILTER='H23'


SAVE_DIR = '/mnt/c/Users/stasevis/Documents/RDI/ADI/output'
SPHERE_DIR = '/mnt/c/Users/stasevis/Documents/sphere_data'

use_fake = True
fake_lum = 'faint'

path_end = {False: '_convert_recenter', True: '_fake_disk_injection'}
cube_path_end = {False: '', True: '_{0:s}_disk'.format(fake_lum)}
save_dir_end = {False: '', True: '/fake_disk'}

data_path = TARGET_NAME + '/' + TARGET_NAME + '_' + FILTER+ '_' + TARGET_EPOCH + '_' + INSTRUMENT
parang_path = os.path.join(data_path + path_end[use_fake], 'SCIENCE_PARA_ROTATION_CUBE-rotnth')
cube_path = os.path.join(data_path + path_end[use_fake], 'SCIENCE_REDUCED_MASTER_CUBE-center_im' + cube_path_end[use_fake])
frame_path = os.path.join(data_path + path_end[use_fake], 'FRAME_SELECTION_VECTOR-frame_selection_vector')

ref_path = os.path.join(data_path + path_end[False], 'IRD_REFERENCE_CUBE-reference_cube_wl0')
#ref_path = None

output_dir =  os.path.join(SAVE_DIR, TARGET_NAME, TARGET_EPOCH + save_dir_end[use_fake])
save_path = os.path.join(output_dir, INSTRUMENT + '_' + FILTER[0:2] + cube_path_end[use_fake] + '_PCA_')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

NORM_METHOD=['spat-mean', 'temp-mean', 'spat-standard', 'temp-standard', 'none']
MAX_PC=10
SPAT_AXIS=1
TEMP_AXIS=0

do_RDI=False
save_residuals=True

in_x, in_y=(256,256)
if INSTRUMENT=='ird':
    pxscale=0.01225 ##arcsec/px
else: ##ifs
    pxscale=0.00746 ##arcsec/px
r_mask=round(0.092/pxscale) ##integer radius of coronagraphic mask in px


class ScienceCube:
    def __init__(self,axis,method,data):
        self.n_frames,self.x_px,self.y_px=data.shape
        self.data=np.reshape(data,(self.n_frames,self.x_px*self.y_px))
        self.mean,self.stdev=(Normalise(self.data,method,axis))
        self.data=MaskCube(self.data,self.x_px,self.y_px,self.n_frames)


##reads in SPHERE data required for reduction + removes bad frames
def GetScienceData():
    try:
        frames=FileIn(SPHERE_DIR,frame_path)
        if len(frames.shape)>1:
            frames=frames[:,0]
        good_frames=np.where(frames==1.)[0]

    except FileNotFoundError:
        good_frames=None

    cube,x,y=TrimCube(FileIn(SPHERE_DIR,cube_path),in_x,in_y,good_frames)
    parang=FileIn(SPHERE_DIR,parang_path)[good_frames]
    channels=cube.shape[0]

    return cube,parang,channels,x,y


def FileIn(dir_path,file_path):
    path=os.path.join(dir_path,file_path)
    data=fits.getdata(path + '.fits')

    return data


def MaskCube(data,x,y,frames):
    cxy=(x//2,y//2)
    mask=disk(cxy,r_mask,shape=(x,y))
    data=np.reshape(data,(frames,x,y))
    data[:,mask[0],mask[1]]=0
    data=np.reshape(data,(frames,x*y))

    return data

##reshapes cube if new size is provided, maintaining center of rotation
def TrimCube(cube,new_x,new_y,good_frames=None):
    x,y=(cube.shape[-2],cube.shape[-1])

    if good_frames is None:
        good_frames=np.arange(0,cube.shape[-3])

    if x <= new_x or y <= new_y:
        return cube[:,good_frames],x,y
    try:
        if (x+new_x)%2:
            x+=1
        if (y+new_y)%2:
            y+=1

        xmin, xmax, ymin, ymax=((x-new_x)//2, (x+new_x)//2, (y-new_y)//2, (y+new_y)//2)
        cube=cube[:,good_frames,xmin:xmax,ymin:ymax]
        x,y=(new_x,new_y)

    except NameError: ##no resizing
        cube=cube[:,good_frames]

    return cube,x,y


##apply centering and/or scaling about the spatial or temporal axis to the data
def Normalise(data,method,axis):
    def NormaliseData(data,axis,get_mean,get_stdev):
        mean=np.nanmean(data,axis)
        stdev=np.nanstd(data,axis)
        zeros=np.where(stdev==0.0)[0]
        stdev[zeros]=1.0 ##avoid dividing by 0 --> =1 to not scale constant features

        ##match shape of arrays for applying normalisation without changing shape of original
        data_ax=np.moveaxis(data,axis,0)

        if get_mean == True:
            data_ax-=mean

        if get_stdev == True:
            data_ax/=stdev

        return data,mean,stdev

    if method =='mean':
        data,mean,stdev=(NormaliseData(data,axis,get_mean=True,get_stdev=False))
    elif method =='standard':
        data,mean,stdev=(NormaliseData(data,axis,get_mean=True,get_stdev=True))
    elif method =='none':
        mean,stdev=(0,1) ##still need to be set since returned by function
    else:
        raise ValueError("Unrecognised normalisation method, please choose from:"
                         "spat-mean, temp-mean, spat-standard, temp-standard, or none")
    return mean,stdev


##carries out PCA reduction on data for each no. PCs up to the specified MAX_PC
def PcaReduction(data,ref_data,x_px,y_px,n_frames,axis,method,mean,stdev):
    def CalculateEigen(data):
        ##covariance matrix of data creates px x px sized array - not enough memory
        ##instead compute covarience as A.A^T and then carry out additional linear algebra
        cov=np.dot(data,data.T)

        eigen_val, eigen_vect=np.linalg.eig(cov) ##"column v[:,i] is the eigenvector
                                                ##corresponding to the eigenvalue w[i]"
        sort=np.argsort(-eigen_val)
        eigen_val=eigen_val[sort]

        pc = np.dot(eigen_vect.T,data)[sort] ##go from eigenvectors of A.A^T to transpose
                                            ##of eigenvectors of A^T.A (=(A^T.ev)^T)
        pc_vals = np.sqrt(abs(eigen_val)) ##'eigen values' for non square matrix
        for i in range(pc.shape[1]):
            pc[:,i]/=pc_vals ##renormalise values after dot product with data

        return pc[:MAX_PC], eigen_val[:MAX_PC]

##TODO: use residuals in additional frame selection (after frame selection, could recompute PCA also,
##most deviant frames probably removed using frame selection vector anyway so may not be the important to do this)
    eigen_vect, eigen_val = CalculateEigen(ref_data) #for RDI, rather than using science matrix here, use ref matrix
    pca_cube=np.zeros((MAX_PC,n_frames,x_px,y_px))
    for i in range(MAX_PC):
        science_data=data.copy()

        eigen_vect_klip=eigen_vect[:(i+1)].copy()
        pca_transf=np.dot(eigen_vect_klip,science_data.T)
        pca_recon=np.dot(pca_transf.T,eigen_vect_klip)

        final=science_data-pca_recon

        ##undo normalisation
        final_ax=np.moveaxis(final,axis,0)
        if method =='standard':
            final_ax*=stdev

        pca_cube[i]=np.reshape(final,(n_frames,x_px,y_px))

    return pca_cube, eigen_val


##performs classical ADI on the PCA reduced cube in order to produce the final image
def Cadi(cube,x,y,n_frames,parang):
    median_combined=np.nanmedian(cube,axis=0)
    cxy=(x//2,y//2)

    cube_reduced=np.subtract(cube,median_combined)
    cube_derot = np.zeros((n_frames,x,y))

    for n in range(0,n_frames):
        rot=getRotationMatrix2D(cxy,parang[n],1)
        cube_derot[n]=warpAffine(cube_reduced[n],rot,(x,y),flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_CONSTANT)

    cadi_cube=np.nanmedian(cube_derot,axis=0)

    return cadi_cube


##creates fits header with relevant parameter for the reduced fits file
def AddHeader(norm_method):
    hdr = fits.getheader(os.path.join(SPHERE_DIR,cube_path)+'.fits')
    hdr['REDUCTN'] = 'PCA'
    hdr['PCA-NORM'] = norm_method

    return hdr


##formats input normalisation method, sends data to PCA and CADI functions,
##and saves final image
def RunPCA(cube,parang,channels,x,y,norm_method,ref_cube=None):
    if norm_method != 'none':
        axis,method=norm_method.split('-')
        if axis =='spat':
            axis=SPAT_AXIS
        elif axis =='temp':
            axis=TEMP_AXIS
    elif norm_method =='none':
        method='none'
        axis=0
    else:
        raise ValueError("Unrecognised normalisation method, please choose from:"
                         "spat-mean, temp-mean, spat-standard, temp-standard, or none")

    reduced_cube=np.zeros((channels,MAX_PC,x,y))
    neg_cube=np.zeros_like(reduced_cube)
    pca_cube=np.zeros((channels,MAX_PC,)+cube.shape[1:])

    for i in range(channels):
        science_cube = ScienceCube(axis,method,cube[i])
        if do_RDI:
            ref_lib = ScienceCube(axis,method,ref_cube).data
        else:
            ref_lib = science_cube.data
        print("Performing {n} PCA on channel".format(n=norm_method),i+1)
        pca_cube[i], eigen_val = PcaReduction(science_cube.data,ref_lib,science_cube.x_px,science_cube.y_px,
                              science_cube.n_frames,axis,method,science_cube.mean,science_cube.stdev)

        eigen_sum=np.sum(eigen_val)
        c=np.divide(eigen_val,eigen_sum)
        plt.plot(np.arange(1,MAX_PC+1),c)
        plt.xlabel('PCs')
        plt.ylabel('eigenvalue [normalised]')
        plt.title('{0:s} H{1:d}'.format(norm_method, [2,3][i]))
        plt.savefig(os.path.join(SAVE_DIR, TARGET_NAME, TARGET_EPOCH,'norm_test',
                                 norm_method+'_H{0:d}_pc_eigenvalues.png'.format([2,3][i])))
        plt.show()

        for j in range(MAX_PC):
            print("CADI reduction with {pc} principal components".format(pc=j+1))
            reduced_cube[i][j] = Cadi(pca_cube[i][j],science_cube.x_px,science_cube.y_px,
                           science_cube.n_frames,parang)
            neg_cube[i][j] = Cadi(pca_cube[i][j],science_cube.x_px,science_cube.y_px,
                           science_cube.n_frames,-parang)

    save_file=save_path + norm_method

    if channels==1 and FILTER=='H23':
        save_file=save_file.replace(FILTER,FILTER[:-1])

    if do_RDI:
        save_file=save_file.replace('PCA','RDI-PCA')

    hdr = AddHeader(norm_method)
    hdu_cube = fits.PrimaryHDU(data=reduced_cube,header=hdr)
    hdu_cube.writeto(save_file+'.fits', overwrite=True)

    if save_residuals:
        hdu_res = fits.PrimaryHDU(data=pca_cube,header=hdr)
        hdu_res.writeto(save_file+'_residuals_cube.fits', overwrite=True)

    hdr['BG_NOTE'] = 'cube reduced using -ve parang'
    hdu_bg = fits.PrimaryHDU(data=neg_cube,header=hdr)
    hdu_bg.writeto(save_file+'_neg_parang.fits', overwrite=True)

    #return reduced_cube

#%%
if __name__=='__main__':

    start_time=datetime.now()
    print("Start time:",start_time.isoformat(' ',timespec='seconds'))

    science_cube,parang,channels,x,y=GetScienceData()

    if do_RDI:
        ref_cube=FileIn(SPHERE_DIR,ref_path)
        ref_x,ref_y=ref_cube.shape[-2:]

        if do_RDI:
            if ref_x < x or ref_y < y:
                science_cube,x,y = TrimCube(science_cube,ref_x,ref_y)
            elif ref_x > x or ref_y > y:
                ref_cube = TrimCube(ref_cube,x,y)[0]

        for i in range(len(NORM_METHOD)):
            RunPCA(science_cube.copy(),parang,channels,x,y,NORM_METHOD[i],ref_cube.copy())

    else:
        for i in range(len(NORM_METHOD)):
            RunPCA(science_cube.copy(),parang,channels,x,y,NORM_METHOD[i])

    print("Total run time:", str(datetime.now()-start_time))

