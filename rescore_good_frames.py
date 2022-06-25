#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:47:21 2022

@author: stasevis
"""
import numpy as np
import warnings
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter('ignore', category=AstropyWarning)

def matrix_frame_index(ref_path, ref_hdr):
    ref_table_path = ref_path.replace('REFERENCE_CUBE-reference_cube','REFERENCE_TABLE-target_info_table')
    ref_table = fits.getdata(ref_table_path + '.fits')

    ncube = len(ref_table['Object'])-1
    frame_start = np.zeros(ncube, dtype=int)
    for i in range(1,ncube):
        frame_start[i]= ref_table['Cube_Nframes'][i]+frame_start[i-1]

    nref = ref_hdr['NCUBE']
    og_framei = []
    for i in range(nref):
        og_framei.append(np.array([int(x) for x in ref_hdr['INIT_{0:03}'.format(i)].split(', ') if x.isdigit()]))

    refs = np.where(ref_table['Frame_type']=='reference')[0]-1
    corr_frames = np.concatenate(og_framei + frame_start[refs])

    return corr_frames


def rescore(frame_selection, ref_path):
    corr_path = ref_path.replace(ref_path.split('/')[-1],'CORR_MATRIX-pcc_matrix')
    corr_matrix = fits.getdata(corr_path + '.fits')
    channels, nb_fr_ref = corr_matrix.shape[0], corr_matrix.shape[-1]

    mean_good_frames=np.mean(frame_selection,axis=2)
    ref_cube_rescore = []

    for wli in range(channels):
        ref_hdr = fits.getheader(ref_path + '{0:d}.fits'.format(wli))
        ncorr=ref_hdr['NCORR']
        score=ref_hdr['SCORE']

        good_frames = np.where(mean_good_frames[wli]>0.5)[0]
        nb_fr_sci = good_frames.shape[0]

        corr_tmp = corr_matrix[wli,good_frames,:]
        ref_scores = np.zeros((nb_fr_ref),dtype=int)

        for i in range(nb_fr_sci):
            best = np.argsort(corr_tmp[i])[::-1][:ncorr]
            ref_scores[best] +=1

        ref_frame_index = np.where(ref_scores>=score)[0]

        corr_frames = matrix_frame_index(ref_path + '{0:d}'.format(wli), ref_hdr)
        keep = corr_frames.searchsorted(ref_frame_index)
        ref_cube_rescore.append(keep)

    return ref_cube_rescore



"""
TARGET_EPOCH='2015-04-12'
TARGET_NAME='HD110058'
INSTRUMENT='ird'
FILTER='H23'

SPHERE_DIR='/mnt/c/Users/stasevis/Documents/sphere_data'
DATA_PATH= TARGET_NAME + '/' + TARGET_NAME + '_' + FILTER+ '_' + TARGET_EPOCH + '_' + INSTRUMENT
FRAME_PATH= DATA_PATH + '__convert_recenter/FRAME_SELECTION_VECTOR-frame_selection_vector_new.fits'

corr_matrix_path=os.path.join(SPHERE_DIR, DATA_PATH.replace('H23','H2') + '_rdi_corr_matrix','IRD_CORR_MATRIX-pcc_matrix.fits')
science_cube_path=os.path.join(SPHERE_DIR,DATA_PATH + '_convert_recenter','SCIENCE_REDUCED_MASTER_CUBE-center_im.fits')
frame_select_path=os.path.join(SPHERE_DIR, DATA_PATH +'_convert_recenter', 'FRAME_SELECTION_VECTOR-frame_selection_vector_new.fits')

crop=256
ncorr=100
score=20
max_frames=None
use_select_vect=True
ref_per_frame=None

#%%
frames=fits.getdata(os.path.join(SPHERE_DIR,FRAME_PATH) + '.fits')
if len(frames.shape)>1:
    frames=frames[:,0]
good_frames=np.where(frames==1.)[0]

ref_path=DATA_PATH.replace('H23','H2') + '_rdi_ref_cube/IRD_SELECTED_REFERENCE_FRAMES-reference_cube_s100'
ref_cube=fits.getdata(os.path.join(SPHERE_DIR,ref_path) + '.fits')#.replace('.fits','_good_frames.fits')))

score_path=DATA_PATH.replace('H23','H2') + '_rdi_ref_cube/frame_correlation_score_s100.txt'
scores=np.loadtxt(os.path.join(SPHERE_DIR,score_path), dtype=int, delimiter='\t')

corr_path=DATA_PATH.replace('H23','H2') + '_rdi_corr_matrix/IRD_CORR_MATRIX-pcc_matrix'
corr_matrix=fits.getdata(os.path.join(SPHERE_DIR,corr_path) + '.fits')

#%%
nb_fr_ref=corr_matrix.shape[-1]
nb_fr_sci=good_frames.shape[0]

corr_matrix=corr_matrix[0,good_frames,:]

ref_scores = np.zeros((nb_fr_ref),dtype=int)
for i in range(nb_fr_sci):
    best = np.argsort(corr_matrix[i])[::-1][:ncorr]
    ref_scores[best] +=1

if max_frames is None:
    ref_frame_index = np.where(ref_scores>=score)[0]
else:
    best_set = np.argsort(ref_scores)[::-1][:max_frames]
    ref_frame_index = ref_scores[best_set]

keep=scores[:,1].searchsorted(ref_frame_index)
ref_cube_tmp=ref_cube[keep]

#%%
hdr=fits.getheader(os.path.join(SPHERE_DIR,ref_path))
hdu_new = fits.PrimaryHDU(data=ref_cube_tmp,header=hdr)
hdu_new.writeto(os.path.join(SPHERE_DIR,ref_path.replace('.fits','_good_frames.fits')))

scores_tmp=scores[keep]
np.savetxt(os.path.join(SPHERE_DIR,score_path.replace('.txt','_good_frames.txt')),scores_tmp, delimiter='\t', fmt='%d', header='ref_frame corr_frame	 score')

#sort=np.argsort(scores_tmp[:,2])[:500]
"""