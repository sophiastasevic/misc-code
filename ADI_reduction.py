"""
ADI recuction of IFS and IRDIS data
    ADI methods: VIP PCA, VIP LLSG, CADI
        - For PCA: normalisation methods: spat-mean, temp-mean, 
          spat-standard, temp-standard, none
    Inputs: centered science cube (axes: [wavelength,frames,x,y]), psf, parang, frame selection
    Outputs: reduced image + no-disk image for each wavelength channel
    
Uses the negative of the parallactic angles to produce a no-disk image for
noise estimation

"""
import numpy as np
import os
from astropy.io import fits
import vip_hci as vip
from datetime import datetime
import cv2
from cv2 import getRotationMatrix2D, warpAffine

TARGET_EPOCH="2015-04-12" 
TARGET_NAME="HD110058"
INSTRUMENT="ird"
FILTER="H23"

SAVE_DIR='/mnt/c/Users/stasevis/Documents/RDI/ADI/output'
SPHERE_DIR='/mnt/c/Users/stasevis/Documents/sphere_data'
DATA_PATH= TARGET_NAME + "/" + TARGET_NAME + "_" + FILTER+ "_" + TARGET_EPOCH +  INSTRUMENT #"_fake_disk_injection/" +
PSF_PATH= DATA_PATH + '_convert_recenter_dc2021/SCIENCE_PSF_MASTER_CUBE-median_unsat'
PARANG_PATH= DATA_PATH + '_convert_recenter_dc2021/SCIENCE_PARA_ROTATION_CUBE-rotnth'
CUBE_PATH= DATA_PATH + '_convert_recenter_dc2021/SCIENCE_REDUCED_MASTER_CUBE-center_im'
FRAME_PATH= DATA_PATH + '_sortframes_vector_dc4/FRAME_SELECTION_VECTOR-frame_selection_vector'


if INSTRUMENT=="ird":
    pxscale=0.01225 ##arcsec/px
else: ##ifs
    pxscale=0.00746 ##arcsec/px 
mask=round(0.092/pxscale) ##int radius of mask in px

new_x, new_y=(256,256) ##new frame size
max_pcs=10
max_ranks=6


##if new frame size is given, reshape cube
def TrimCube(cube,good_frames):
    try:
        x,y=(cube.shape[2],cube.shape[3])
        ##to preserve correct center of rotation of original even cube if new size is odd
        if new_x%2:
            x+=1
        if new_y%2:
            y+=1
        xmin, xmax, ymin, ymax=((x-new_x)//2, (x+new_x)//2, (y-new_y)//2, (y+new_y)//2)
        cube=cube[:,good_frames,xmin:xmax,ymin:ymax]
    except NameError:
        cube=cube[:,good_frames,:,:]
    
    return cube
    

##reads in data + reshapes science cube + snr source positions
def DataInitialisation(method,norm,redo_cube=False,redo_negcube=False):
    output_path= os.path.join(SAVE_DIR, TARGET_NAME, TARGET_EPOCH) \
                + INSTRUMENT + "_" + FILTER + "_" #fake_disk_" 
    
    if 'PCA' in method: 
        for i in range(len(norm)):
            try:
                method.append('PCA_'+norm[i])
            except TypeError:
                method.append('PCA_none')
        method.remove('PCA')        
    
    ##checking if anything needs to be processed before spending time on loading cube
    if int(redo_cube)+int(redo_negcube) == 0: ##only check for existing files if no reprocessing wanted
        for i in range(len(method)):
            output_filename = output_path + method[i]
            
            if os.path.isfile(output_filename+"_neg_parang.fits")==False or \
               os.path.isfile(output_filename+".fits")==False:
                break ##at least one method needs processing --> break loop + continue script
            
            redo=input("{m} reduced disk already exists, reprocess anyway? [yes/no]: ".format(m=method[i]))
            while redo.lower() not in set(['y','yes']) and redo.lower() not in set(['n','no']):
                redo=input("Answer not recognised, please input yes(y) or no(n): ")
            if redo.lower() in set(['y','yes']):
                redo_cube,redo_negcube=(True,True)
                break
            elif i<len(method)-1:
                continue ##still remaining methods to check so moves to next one
            
            print("\nNo methods to be reprocessed, exiting program.")
            return ##exits function + ends script
    
    
    frames=vip.fits.open_fits(os.path.join(SPHERE_DIR,FRAME_PATH))
    if len(frames.shape)>1:
        frames=frames[:,0]
    good_frames=np.where(frames==1.)[0]
    
    parang=vip.fits.open_fits(os.path.join(SPHERE_DIR,PARANG_PATH))[good_frames]
    
    cube=TrimCube(vip.fits.open_fits(os.path.join(SPHERE_DIR,CUBE_PATH)),good_frames) 
    
    psf=vip.fits.open_fits(os.path.join(SPHERE_DIR,PSF_PATH))
    if len(psf.shape)>3:
        psf=np.nanmean(psf,axis=1) ##psf before + after obs axis
        #psf=psf[:,1] ##2015-04-03 1st psf measurement not good
    
    
    for i in range(len(method)):
        output_filename = output_path + method[i]
        
        if os.path.isfile(output_filename+".fits") == False or redo_cube == True:
            print("Running {m} reduction.".format(m=method[i]))
            Reduction(cube,psf,parang,method[i],output_filename)
        else:
            print("{m} reduced data already exists. \nChecking for background reduction".format(m=method[i]))
            
        ##for SNR calculation
        if os.path.isfile(output_filename+"_neg_parang.fits") == False or redo_negcube == True:
            print("Running {m} reduction with negative parang for SNR calculation".format(m=method[i]))
            Reduction(cube,psf,-parang,method[i],output_filename+"_neg_parang")
        else:
            print("{m} background reduction already exists. \nContinuing to next method".format(m=method[i]))
            continue
         
      
##calculates fwhm + sends to function that runs specified redution method for
##each filter band in the science cube + saves processed data cube   
def Reduction(cube,psf,parang,method,output_filename):
    bands,x_px,y_px=(cube.shape[0],cube.shape[2],cube.shape[3])
    norm=None
        
    if method.split('_')[0] == 'PCA':
        method,norm=method.split('_')
        if norm=='no-norm':
            norm=None
        reduced_cube = np.zeros((bands,max_pcs,x_px,y_px))
    elif method == 'LLSG':
        reduced_cube = np.zeros((bands,max_ranks,x_px,y_px))
    elif method == 'CADI':
        reduced_cube = np.zeros((bands,x_px,y_px))  
    else:
        raise ValueError("Unrecognised reduction method: {m}. \
                         \nAvailable methods: PCA, LLSG, CADI".format(m=method))
    
    cxy=(x_px//2,y_px//2) ##opencv rotates about top left of pixel rather than center, so if cube is odd, 
                          ##(size/2 -0.5)=(size//2) is needed as the center in order to rotate about the
                          ##center of the original even sized cube
    
    try:
        for i in range(bands):
            print("Processing wavelength channel",(i+1))
            DF_fit=vip.var.fit_2dgaussian(psf[i], crop =True, cropsize=22, debug=False, full_output=True)
            fwhm=np.mean([DF_fit['fwhm_x'],DF_fit['fwhm_y']])
            
            if method == 'PCA':
                reduced_cube[i] = RunPCA(cube[i],parang,fwhm,output_filename,cxy,norm)
            elif method == 'LLSG':
                reduced_cube[i] = RunLLSG(cube[i],parang,fwhm,output_filename,x_px,y_px,cxy)
            elif method == 'CADI': ##ValueError not needed as would have already been found
                reduced_cube[i] = RunCADI(cube[i],parang,fwhm,output_filename,x_px,y_px,cxy)
        
        ##saves output w/ data info in header
        hdr = AddHeader(method,norm)
        hdu_new = fits.PrimaryHDU(data=reduced_cube,header=hdr)
        hdu_new.writeto(output_filename + ".fits", overwrite=True)
        
    except KeyboardInterrupt: 
        print("Program interrupted. Saving current reduced data without header.")
        hdu_new = fits.PrimaryHDU(data=reduced_cube)
        hdu_new.writeto(output_filename + "_autosave.fits", overwrite=True)
        
        return
        
##creates fits header with relevant parameter for the reduced fits file    
def AddHeader(method,norm):
    hdr = fits.Header()
    hdr['Instrument'] = INSTRUMENT
    hdr['Filter'] = FILTER
    hdr['Object'] = TARGET_NAME
    hdr['Epoch'] = TARGET_EPOCH
    hdr['Reduction Method'] = method
    if method == 'PCA':
        hdr['PCA Normalisation'] = norm
    hdr['Mask Size [px]'] = mask
    hdr['Pixel Scale [arcsec/px]'] = pxscale
    
    return hdr
        

##performs pca for each value of pc up to specified max_pcs
def RunPCA(science_cube,parang,fwhm,output_filename,cxy,norm):
    ##grid method in vip iterates through different pc values --> for loop not needed
    pca_cube=vip.pca.pca_fullfr.pca(cube=science_cube, angle_list=-parang, cxy=cxy,
                                    batch=None, cube_ref=None, ncomp=(1,max_pcs), 
                                    scaling=norm, mask_center_px=mask, imlib='opencv',
                                    fwhm=fwhm, full_output=False, verbose=True, nproc=1)    
    return pca_cube


##performs llsg for each rank up to specified max_rank
def RunLLSG(science_cube,parang,fwhm,output_filename,x_px,y_px,cxy):
    llsg_cube = np.zeros((max_ranks, x_px, y_px))
    
    ##looping through each rank
    for i in range(max_ranks):
        print("Rank:",i+1)
        llsg_cube[i] = vip.llsg.llsg(science_cube, -parang, fwhm=fwhm, rank = i+1, 
                                    thresh = 1, max_iter = 10, random_seed = 10, 
                                    full_output=False, verbose=True, imlib='opencv')     
    return llsg_cube
 
   
##performs classical adi (i.e. no dimensionality reduction)
def RunCADI(cube,parang,fwhm,output_filename,x_px,y_px,cxy):
    ##median combining science cube along temporal axis to get speckle image 
    ## + subtracting speckles from science cube
    median_combined=np.nanmedian(cube,axis=0)
    cube_reduced=np.subtract(cube,median_combined)
    
    ##derotating each science frame by its parallactic angle + median combining
    cube_derot = np.zeros((len(parang), x_px, y_px))
    for i in range(0,len(parang)):
        rot=getRotationMatrix2D(cxy,parang[i],1)
        cube_derot[i]=warpAffine(cube_reduced[i],rot,(x_px,y_px),flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_CONSTANT)
    
    cadi_cube=np.nanmedian(cube_derot,axis=0)
    
    return cadi_cube


if __name__=='__main__':
    
    start_time=datetime.now()
    
    method=['PCA','CADI']
    norm=['spat-mean','temp-mean','spat-standard','temp-standard',None]
    
    DataInitialisation(method,norm,redo_cube= False,redo_negcube= False) 
    
    #ds9=vip.Ds9Window()
    #ds9.display(data_cube[0])
    print("\nTotal run time:", str(datetime.now()-start_time))

"""
------------------------------Unused Functions------------------------------

##remove bad pixels    

remove_badpx= False #set to True to remove bad pixels from cube, or load cube 
                    #with bad pixels removed if already exists

#removes bad pixels in science cube + saves processed cube
def RemoveBadPixels(cube,bands):
    for i in range(bands):
        cube[i]=vip.preproc.badpixremoval.cube_fix_badpix_isolated(cube[i], bpm_mask=None, 
                                sigma_clip=3, num_neig=5, size=5, frame_by_frame=False, 
                                protect_mask=False, radius=mask, verbose=False)
    hdu_new = fits.PrimaryHDU(data=cube)
    hdu_new.writeto(os.path.join(SPHERE_DIR,CUBE_PATH+"_badpxrm.fits"), overwrite=True)
    
    return cube

path_badpxrm=os.path.join(SPHERE_DIR,CUBE_PATH+"_badpxrm.fits")
if os.path.isfile(path_badpxrm) == True and remove_badpx == True:
    cube=vip.fits.open_fits(path_badpxrm)
    badpxrm_check=True
else:
    cube=vip.fits.open_fits(os.path.join(SPHERE_DIR,CUBE_PATH))
    badpxrm_check=False
    
#if specified to remove bad pixels and file does not already exist
if remove_badpx == True and badpxrm_check == False:
    cube=RemoveBadPixels(cube,bands)
"""