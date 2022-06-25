"""
ADI/RDI reduction of IFS and IRDIS data using CADI, PCA, VIP-PCA, and VIP-LLSG function

Scripts:
    - ADI_reduction.py
    - pca.py
    - residuals_frame_selection.py
    - rescore_good_frames.py

Reduction methods:
    - (VIP-)PCA
    - LLSG
    - CADI
    - RDI-(VIP-)PCA

PCA normalisation methods:
    - spat-mean
    - temp-mean
    - spat-standard
    - temp-standard
    - none

Inputs:
    - centered science cube [wavelength,frames,x,y]
    - star psf
    - parallactic angles list
    - frame selection vector
    - (if RDI-PCA) reference library

Outputs:
    - reduced cube [wavelength, (pcs/ranks,) x, y]
    - (if NEG_DISK = True) reduction using -ve of the parallactic angles
    - (if SAVE_RESIDUALS = True) cube of PCA residuals before CADI combination
    - (if RESIDUALS_FRAME_SELECT = True) frame selection vector for residual frames

Options:
    - USE_FAKE: use cube with fake disk injection (changes cube path)
    - NEG_CUBE: use the negative of the parallactic angles to produce a no-disk image for
                background noise estimation
    - SAVE_RESIDUALS: saves the PCA subtracted cube for each wavelength in a new folder
    - RESIDUALS_FRAME_SELECT: calculates corrolation between residual frames + redoes PCA
                              reduction using only frames with PCC > 0
    - REDO_(NEG)CUBE: if requested reducton already exists;
                    True: proceeds with reduction + overwrites existing file
                    False: removes any existing reductions from process list
                    None: asks user input whether to redo reduction for each existing file

"""
import numpy as np
import os
import sys
from astropy.io import fits
import vip_hci as vip
from datetime import datetime
import cv2
from cv2 import getRotationMatrix2D, warpAffine
import warnings
import pca
import residuals_frame_selection as rfs
import rescore_good_frames as refscore
warnings.simplefilter("ignore")

EPOCH = '2015-04-12'
TARGET = 'HD110058'
INSTRUMENT = 'ird'
FILTER = 'H23'

SAVE_DIR = '/mnt/c/Users/stasevis/Documents/RDI/ADI/output' ##directory to save reductions in
SPHERE_DIR = '/mnt/c/Users/stasevis/Documents/sphere_data'

USE_FAKE = False
FAKE_LUM = 'faint'

PATH_END = {False: '_convert_recenter', True: '_fake_disk_injection'}
CUBE_PATH_END = {False: '', True: '_{0:s}_disk'.format(FAKE_LUM)}
SAVE_DIR_END = {False: '', True: '/fake_disk'}
FRAME_PATH_END = {False: '', True: '_residuals'}

data_path = TARGET + '/' + TARGET + '_' + FILTER + '_' + EPOCH + '_' + INSTRUMENT  ##input data path
#%%
psf_path = os.path.join(data_path+PATH_END[USE_FAKE], 'SCIENCE_PSF_MASTER_CUBE-median_unsat')
parang_path = os.path.join(data_path + PATH_END[USE_FAKE], 'SCIENCE_PARA_ROTATION_CUBE-rotnth')
cube_path = os.path.join(data_path + PATH_END[USE_FAKE], 'SCIENCE_REDUCED_MASTER_CUBE-center_im' + CUBE_PATH_END[USE_FAKE])
frame_path = os.path.join(data_path + PATH_END[USE_FAKE], 'FRAME_SELECTION_VECTOR-frame_selection_vector')

ref_path = os.path.join(data_path + PATH_END[False], 'REFERENCE_CUBE-reference_cube_wl')
#ref_path = None

output_dir =  os.path.join(SAVE_DIR, TARGET, EPOCH + SAVE_DIR_END[USE_FAKE])
output_path = os.path.join(output_dir, INSTRUMENT + '_' + FILTER + CUBE_PATH_END[USE_FAKE]+ '_')# + '_outer_disk_')

if not os.path.exists(output_dir): os.makedirs(output_dir)

AVAILABLE_METHODS = ['CADI', 'LLSG', 'VIP-PCA', 'PCA', 'RDI-VIP-PCA', 'RDI-PCA']
AVAILABLE_NORMS = ['spat-mean','temp-mean','spat-standard','temp-standard','none']

if INSTRUMENT=='ird':
    PXSCALE = 0.01225 ##arcsec/px
else: ##ifs
    PXSCALE = 0.00746 ##arcsec/px
r_mask = round(0.092/PXSCALE) ##integer radius of coronagraphic mask in px
#r_mask = 40
corr_in, corr_out = (10,50)

crop_x, crop_y = (256,256) ##new frame size
max_pcs = 10
max_ranks = 6
pc_list = (1,5,10,15,20,25,35,50,75,100)
if len(pc_list) != max_pcs:
    max_pcs = len(pc_list)
    Warning('PC list for RDI must be same length as value for max PCs, changing max_pcs to', max_pcs)

NEG_DISK = True
SAVE_RESIDUALS = False
RESIDUALS_FRAME_SELECT = True

REDO_CUBE, REDO_NEGCUBE = (False,False)

method = ['RDI-VIP-PCA','RDI-PCA']
norm = ['temp-mean','spat-mean']

"""
-------------------------------------------- FILE HANDLING --------------------------------------------
"""
def FileIn(dir_path, file_path=None):
    if file_path is not None:
        path=os.path.join(dir_path,file_path)
    else:
        path = dir_path
    data=fits.getdata(path + '.fits')

    return data


def SaveCube(data, hdr, path):
    directory = path.replace(path.split('/')[-1],'')
    if not os.path.exists(directory):
        os.makedirs(directory)

    hdu_new = fits.PrimaryHDU(data=data,header=hdr)
    hdu_new.writeto(path + '.fits', overwrite=True)

    print('> file saved as', path.split('/')[-1]+'.fits')


## adds reduction information to science cube header for the output fits file
def AddHeader(method, norm):
    hdr = fits.getheader(os.path.join(SPHERE_DIR,cube_path) + '.fits')
    hdr['REDUCTN'] = method.replace('-',' ')
    if 'PCA' in method:
        if norm == None: norm = 'none'
        hdr['PCA-NORM'] = norm
        hdr['PCA_MASK'] = r_mask
    if 'RDI' in method: hdr['PCS'] = str(pc_list)
    else: hdr['PCS'] = str(np.arange(1,max_pcs+1))
    if RESIDUALS_FRAME_SELECT: hdr['R_CORR'] = str((corr_in, corr_out))

    return hdr

"""
-------------------------------------------- FORMATTING --------------------------------------------
"""
## if PCA is in the method list, appends methods joining 'PCA' and the normalisation methods input
def FormatMethods(method,norm):
    method_tmp = []
    for m in range(len(method)):
        if 'PCA' in method[m]:
            for i in range(len(norm)):
                try:
                    method_tmp.append(method[m] + '_' + norm[i])
                except TypeError:
                    method_tmp.append(method[m] + '_none') #just in case None was put for norm rather than 'none'
        else:
            method_tmp.append(method[m])

    return method_tmp


## check if data has been processed using a certain method before loading the cube + prompting user
## if all processing methods already exist, whether they want to reprocess
def CheckReprocess(method, norm, output_path, REDO_CUBE=None, REDO_NEGCUBE=None):
    valid_m = []
    valid_n = []
    for i in range(len(method)):
        if method[i] not in AVAILABLE_METHODS:
            print('Unrecognised reduction method: {m} \nRemoving from list'.format(m=method[i]))
        else:
            valid_m.append(method[i])
    for i in range(len(norm)):
        if norm[i] not in AVAILABLE_NORMS:
            print('Unrecognised normalisation method: {n} \nRemoving from list'.format(n=norm[i]))
        else:
            valid_n.append(norm[i])

    if len(valid_m)==0:
        raise ValueError('No recognised reduction methods input. Available methods: ' +
                         ', '.join('{0:s}'.format(m) for i,m in enumerate(AVAILABLE_METHODS)) +
                         ', norms: ' + ', '.join('{0:s}'.format(n) for i,n in enumerate(AVAILABLE_NORMS)))
    elif len(valid_n)==0 and any('PCA' in x for x in valid_m):
        print('No normalisation specified for PCA reduction, defaulting to spat-mean')
        valid_n.append('spat-mean')

    method = FormatMethods(valid_m,valid_n)
    file_end = {0: '.fits', 1: '_neg_parang.fits'}

    method_tmp = []
    redo_process = []
    for i in range(len(method)):
        output_filename = output_path + method[i]
        if RESIDUALS_FRAME_SELECT: output_filename += '_best_frames'

        redo_tmp = [False,False]
        for r, redo in enumerate((REDO_CUBE, REDO_NEGCUBE)):
            if NEG_DISK == False and r == 1:
                continue
            elif redo == True or (os.path.isfile(output_filename + file_end[r]))==False:
                redo_tmp[r] = True
            elif redo == None:
                print('\n> {0:s} reduction already exists for method: {1:s}'.format(('Standard','Negative parang')[r], method[i]))
                res = input('Reprocess anyway? [yes/no]: ')
                while res.lower() not in set(['y','yes','n','no']):
                    res = input('Answer not recognised, please input yes(y) or no(n): ')
                if res.lower() in set(['y','yes']):
                    redo_tmp[r] = True
        if True in redo_tmp:
            method_tmp.append(method[i])
            redo_process.append(redo_tmp)

    if len(method_tmp)==0:
        print("\nNo methods to be reprocessed, exiting program.")
        sys.exit()
    else:
        method = method_tmp

    return method, redo_process

"""
-------------------------------------------- DATA HANDLING --------------------------------------------
"""
## returns frames whose PCA residuals have a +ve frame corrolation with the median of the cube
def ResFrameSelect(cube, parang, fwhm, x, y, norm, r_mask, pcs, method, wl, output_filename, hdr):
    print('Getting residuals for wavelength channel {0:d}.'.format(wl))
    res_path = output_filename + '_residuals/' + 'residuals_cube_wl{0:d}'.format(wl)
    try:
        pca_residuals = FileIn(res_path)
        r_corr_tmp = fits.getheader(res_path + '.fits')['R_CORR']
    except FileNotFoundError:
        pca_residuals = None

    if pca_residuals is None or r_corr_tmp != str((corr_in, corr_out)):
        cube = cube[...,x//2-corr_out:x//2+corr_out, y//2-corr_out:y//2+corr_out]
        x,y = corr_out*2, corr_out*2
        if 'VIP' in method:
            pca_residuals = RunPCA(cube, parang, fwhm, x, y, norm, pcs)[1]
        else:
            pca_residuals = pca.RunPCA(cube, x, y, norm, r_mask, pcs)
        if SAVE_RESIDUALS:
            SaveCube(pca_residuals, hdr, output_filename + '_residuals/' + 'residuals_cube_wl{0:d}'.format(wl))

    good_frames=[]
    print('..Calculating correlation between residual frames..')
    for n in range(max_pcs):
        good_frames.append(rfs.residuals_frame_select(pca_residuals[n], corr_in, corr_out))

    return good_frames


## trims frames + removes bad frames using the frame selection vector
def TrimCube(cube, new_x, new_y, good_frames=None):
    x,y = cube.shape[-2:]

    if good_frames is None:
        good_frames = np.arange(0,cube.shape[-3])

    new_x, new_y = [t if t!=None else (x,y)[i] for i,t in enumerate((new_x,new_y))]
    if x <= new_x and y <= new_y:
        cube = cube[:,good_frames]

    else:
        ##preserves center of rotation of original (even) cube if new size is odd
        if (x+new_x)%2: x+=1
        if (y+new_y)%2: y+=1

        xmin, xmax, ymin, ymax = (x-new_x)//2, (x+new_x)//2, (y-new_y)//2, (y+new_y)//2
        cube = cube[:,good_frames,xmin:xmax,ymin:ymax]
        x,y = (new_x,new_y)

    return cube,x,y,good_frames


## reads in data + trims science(+ref) cube(s)
def DataInitialisation(method, norm, REDO_CUBE=None, REDO_NEGCUBE=None):
    method, redo_process = CheckReprocess(method, norm, output_path, REDO_CUBE, REDO_NEGCUBE)

    if RESIDUALS_FRAME_SELECT:
        good_frames = None ##frame selection from residuals frames
    else:
        try:
            frames = FileIn(SPHERE_DIR,frame_path)
            if len(frames.shape)>1:
                frames = frames[:,0]
            good_frames = np.where(frames==1.)[0]
        except FileNotFoundError: good_frames = None

    print("..Loading data cube..")
    cube,x,y,good_frames = TrimCube(FileIn(SPHERE_DIR,cube_path), crop_x, crop_y, good_frames)
    parang = FileIn(SPHERE_DIR,parang_path)[good_frames]

    psf = FileIn(SPHERE_DIR,psf_path)
    if len(psf.shape)>3:
        if INSTRUMENT=='ifs' and EPOCH=='2015-04-03':
            psf = psf[:,1] ##1st psf measurement for these data not good
        else:
            psf = np.nanmean(psf,axis=1) ##psf before + after obs sequence

    channels = cube.shape[0]
    fwhm = np.zeros(channels)
    for i in range(channels):
        DF_fit = vip.var.fit_2dgaussian(psf[i], crop =True, cropsize=22, debug=False, full_output=True)
        fwhm[i] = np.mean([DF_fit['fwhm_x'],DF_fit['fwhm_y']])

    if any('RDI' in x for x in method):
        print("..Loading reference cube..")
        ref_cube = []
        for wl in [0,1]:
            ref_cube_tmp = FileIn(SPHERE_DIR,ref_path+'{0:d}'.format(wl)) ##keep np.array until ref cube has 4 axes
            ref_x, ref_y = ref_cube_tmp.shape[-2:]

            if ref_x < x or ref_y < y:
                cube,x,y = TrimCube(cube,ref_x,ref_y)[:3]
            elif ref_x > x or ref_y > y:
                ref_cube_tmp = TrimCube(ref_cube_tmp,x,y)[0]
            ref_cube.append(ref_cube_tmp)
    else:
        ref_cube = np.full(channels,None)

    return cube, x, y, channels, ref_cube, parang, fwhm, method, redo_process

"""
-------------------------------------------- REDUCTION --------------------------------------------
"""
## iterates through input list of reduction methods
def ADI_RDI(cube, x, y, channels, ref_cube, parang, fwhm, method, redo_process):
    for i in range(len(method)):
        try: method_tmp, norm = method[i].split('_')
        except ValueError: method_tmp, norm = (method[i], None)

        output_filename = output_path + method[i]
        hdr = AddHeader(method_tmp,norm)

        if 'RDI' in method_tmp: pcs = pc_list
        elif SAVE_RESIDUALS or RESIDUALS_FRAME_SELECT or 'VIP' not in method_tmp: pcs = tuple(np.arange(max_pcs)+1)
        else: pcs = max_pcs

        good_frames = []
        rdi_frames = None
        if RESIDUALS_FRAME_SELECT:
            res_frame_path = os.path.join(SPHERE_DIR, data_path+PATH_END[False], 'residuals_frame_selection',
                                          'frame_selection_vector_' + method[i].replace('RDI-',''))
            try:
                frame_selection = FileIn(res_frame_path)
                r_corr_tmp = fits.getheader(res_frame_path + '.fits')['R_CORR']

            except FileNotFoundError:
                frame_selection = None

            if frame_selection is not None and r_corr_tmp == str((corr_in, corr_out)):
                for j in range(channels):
                    good_frames_tmp = []
                    for p in range(max_pcs):
                        good_frames_tmp.append(np.where(frame_selection[j,:,p]==1.)[0])
                    good_frames.append(good_frames_tmp)

            else:
                frame_selection = np.zeros((channels,len(parang),max_pcs))
                for j in range(channels):
                    good_frames.append(ResFrameSelect(cube[j], parang, fwhm[j], x, y, norm, r_mask, pcs, method_tmp, j, output_filename, hdr))
                    for p in range(max_pcs):
                        frame_selection[j,good_frames[j][p],p] = 1.
                SaveCube(frame_selection, hdr, res_frame_path)

            output_filename += '_best_frames'

            if 'RDI' in method_tmp:
                rdi_frames = refscore.rescore(frame_selection, os.path.join(SPHERE_DIR,ref_path))

        if redo_process[i][0]:
            print("\nRunning {m} reduction of data cube.".format(m=method_tmp))
            reduced_cube = Reduction(cube, x, y, channels, ref_cube, parang, fwhm, method_tmp, norm, pcs, output_filename, hdr, good_frames, rdi_frames)

        if redo_process[i][1]: ##for SNR calculation
            print("\nRunning {m} reduction of data cube using negative parang.".format(m=method[i]))
            Reduction(cube, x, y, channels, ref_cube, -parang, fwhm, method_tmp, norm, pcs, output_filename+'_neg_parang', hdr, good_frames, rdi_frames)

    return reduced_cube, frame_selection


## sends data to relevant reduction method function + saves result
def Reduction(cube, x, y, channels, ref_cube, parang, fwhm, method, norm, pcs, output_filename, hdr, good_frames, rdi_frames):
    if 'PCA' in method:
        reduced_cube = np.zeros((channels,max_pcs,x,y))
    elif method == 'LLSG':
        reduced_cube = np.zeros((channels,max_ranks,x,y))
    elif method == 'CADI':
        reduced_cube = np.zeros((channels,x,y))

    try:
        for i in range(channels):
            print("Processing wavelength channel",(i))

            if 'PCA' in method:
                if RESIDUALS_FRAME_SELECT:
                    if rdi_frames is not None:
                        ref_cube_tmp = ref_cube[i][rdi_frames[i]]
                    else:
                        ref_cube_tmp = None
                    reduced_cube[i] = ResFramePCA(cube[i], parang, fwhm[i], x, y, norm, r_mask, pcs, method, good_frames[i], ref_cube_tmp)

                elif 'VIP' in method:
                    reduced_cube[i], pca_residuals = RunPCA(cube[i], parang, fwhm[i], x, y, norm, pcs, ref_cube[i])

                else:
                    pca_residuals = pca.RunPCA(cube[i], x, y, norm, r_mask, pcs, ref_cube[i])
                    for j in range(max_pcs):
                        reduced_cube[i][j] = RunCADI(pca_residuals[j], parang, x, y)

                if SAVE_RESIDUALS and not RESIDUALS_FRAME_SELECT and 'neg_parang' not in output_filename:
                    SaveCube(pca_residuals, hdr, output_filename + '_residuals/' + 'residuals_cube_wl{0:d}'.format(i))

            elif method == 'LLSG':
                reduced_cube[i] = RunLLSG(cube[i], parang, fwhm[i], x, y)

            elif method == 'CADI':
                reduced_cube[i] = RunCADI(cube[i], parang, x, y)

        ## saves output w/ data info in header
        SaveCube(reduced_cube, hdr, output_filename)
        return reduced_cube

    except KeyboardInterrupt:
        print("Program interrupted. Saving any reduced data")
        SaveCube(reduced_cube, hdr, output_filename  + '_autosave.fits')

        sys.exit()

"""-------------------------------------------- PCA --------------------------------------------"""
## performs pca for each value of pc up to specified max_pcs using either ADI or RDI
def RunPCA(cube, parang, fwhm, x, y, norm, pcs, ref_cube=None):
    if norm == 'none':
        norm = None

    if type(pcs) != tuple:
        ## when ncomp = tuple: iterates between interval of the specified pcs
        pca_cube = vip.psfsub.pca_fullfr.pca(cube=cube, angle_list=-parang,
                                        batch=None, cube_ref=ref_cube, ncomp=(1,pcs),
                                        scaling=norm, mask_center_px=r_mask, imlib='opencv', ##vip-fft better but MUCH slower
                                        fwhm=fwhm, full_output=False, verbose=True)
        pca_residuals = None

    else:
        pca_cube = np.zeros((max_pcs, x, y))
        pca_residuals = [] ##stores residuals frames for each PC
        for i,pc in enumerate(pcs):
            pc=int(pc)
            print("Processing PC:",pc)

            pca_out = vip.psfsub.pca_fullfr.pca(cube=cube, angle_list=-parang,
                                            batch=None, cube_ref=ref_cube, ncomp=pc,
                                            scaling=norm, mask_center_px=r_mask, imlib='opencv',
                                            fwhm=fwhm, full_output=True, verbose=False)
            pca_cube[i] = pca_out[0]
            pca_residuals.append(pca_out[3])

    return pca_cube, pca_residuals


## PCA reduction for residuals frame selection cubes
def ResFramePCA(cube, parang, fwhm, x, y, norm, r_mask, pcs, method, good_frames, ref_cube):
    reduced_cube = np.zeros((max_pcs,x,y))
    for i,pc in enumerate(pcs):
        if 'VIP' in method:
            pc=int(pc)
            reduced_cube[i] = vip.psfsub.pca_fullfr.pca(cube=cube[good_frames[i]],
                                            angle_list=-parang[good_frames[i]],
                                            batch=None, cube_ref=ref_cube, ncomp=pc,
                                            scaling=norm, mask_center_px=r_mask, imlib='opencv',
                                            fwhm=fwhm, full_output=False, verbose=False)
        else:
            pca_residuals = pca.RunPCA(cube[good_frames[i]], x, y, norm, r_mask, [pc], ref_cube)[0]
            reduced_cube[i] = RunCADI(pca_residuals, parang[good_frames[i]], x, y)

    return reduced_cube

"""-------------------------------------------- LLSG --------------------------------------------"""
## performs llsg for each rank up to specified max_rank
def RunLLSG(cube,parang,fwhm,x,y):
    llsg_cube = np.zeros((max_ranks, x, y))

    for i in range(max_ranks):
        print("Rank:",i+1)
        llsg_cube[i] = vip.psfsub.llsg(cube, -parang, fwhm=fwhm, rank = i+1,
                                    thresh = 1, max_iter = 10, random_seed = 10,
                                    full_output=False, verbose=True, imlib='opencv')
    return llsg_cube

"""-------------------------------------------- CADI --------------------------------------------"""
## performs classical adi (i.e. no dimensionality reduction)
def RunCADI(cube,parang,x,y):
    ## speckles from temporal median of science cube
    median_combined = np.nanmedian(cube,axis=0)
    cube_reduced = np.subtract(cube,median_combined)
    cxy = (x//2,y//2)

    cube_derot = np.zeros((len(parang),x,y))
    for i in range(0,len(parang)):
        rot = getRotationMatrix2D(cxy,parang[i],1)
        cube_derot[i] = warpAffine(cube_reduced[i],rot,(x,y),flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_CONSTANT)

    cadi_cube = np.nanmedian(cube_derot,axis=0)

    return cadi_cube


#%%
if __name__=='__main__':

    start_time=datetime.now()

    ADI_RDI(*DataInitialisation(method, norm, REDO_CUBE, REDO_NEGCUBE))

    print("\nTotal run time:", str(datetime.now()-start_time))

#cube, x, y, channels, ref_cube, parang, fwhm, method_tmp, redo_process = DataInitialisation(method, norm, REDO_CUBE, REDO_NEGCUBE)
#res_cube, frame_select = ADI_RDI(cube, x, y, channels, ref_cube, parang, fwhm, method_tmp, redo_process)