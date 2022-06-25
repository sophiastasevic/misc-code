"""
Created on Mon Apr 25 15:00:42 2022

@author: stasevis

Find best correlated reference frames for the science cube using correlation matrix and
save reference cube containing these frames

sof file containing: correlation matrix, science target, frame selection vector [OPTIONAL]
--score: score above which ref frame must be in order to be used
--ncorr: no. top correlated ref frames per science frame to have score increased by +1
--max_frames [OPTIONAL]: no. frames in final ref cube
--use_select_vect [OPTIONAL]: flag to use frame selection vector for science cube

outputs:
    'reference_cube_wl*.fits'
    'target_info_table_wl*.fits'

[2022-05-03] added option to crop reference frames
[2022-05-06] saves target + observation information of all data cubes to fits table
"""

import argparse
import warnings
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter('ignore', category=AstropyWarning)

## gets ref target information from its header + adds to new ref cube header
## adds information on frame index positions in original ref target cube, new ref
## cube, and calculated frame score
def add_header_ref_info(cube_hdr, i, init_framei, frame_start, ref_hdr, frame_score):
    ref_hdr['OBJ_{0:03d}'.format(i)] = cube_hdr['OBJECT']
    ref_hdr['RA_{0:03d}'.format(i)] = cube_hdr['RA']
    ref_hdr['DEC_{0:03d}'.format(i)] = cube_hdr['DEC']
    ref_hdr['OBS_{0:03d}'.format(i)] = cube_hdr['DATE-OBS']
    ref_hdr.set('PRO_{0:03d}'.format(i), cube_hdr['DATE'], 'master cube file created')
    ref_hdr.set('EXPT_{0:03d}'.format(i), cube_hdr['EXPTIME'], '[s] exposure time')

    nframes = len(init_framei)
    fin_framei = np.arange(frame_start, frame_start + nframes)

    ref_hdr['INIT_{0:03d}'.format(i)] = ', '.join([str(int) for int in init_framei])
    ref_hdr['REFI_{0:03d}'.format(i)] = ', '.join([str(int) for int in fin_framei])
    ref_hdr['SCOR_{0:03d}'.format(i)] = ', '.join([str(int) for int in frame_score])

    return ref_hdr


## finds best [ncorr] correlated reference frames for each science frame and
## increases score of frame by +1
## all frames with a score above [score] are selected for the reference cube
## OPT. PARAM [good_frames]: applies frame selection vector to science frames before scoring
## OPT. PARAM [max_frames]: takes that set number of highest scoring frames only
def score_frames(corr_mat, score, ncorr, max_frames=None, good_frames=None):
    if good_frames is not None:
        corr_mat = corr_mat[good_frames]

    nframes_ref = corr_mat.shape[-1]
    nframes_sci = corr_mat.shape[0]

    ref_scores = np.zeros((nframes_ref),dtype=int)
    for i in range(nframes_sci):
        best_corr = np.argsort(corr_mat[i])[::-1][:ncorr]
        ref_scores[best_corr] +=1

    if max_frames is None:
        ref_frame_index = np.where(ref_scores>=score)[0]
    else:
        best_set = np.argsort(ref_scores)[::-1][:max_frames]
        ref_frame_index = np.sort(best_set)

    return ref_frame_index, ref_scores[ref_frame_index]


## find which ref target cubes contain best correlated frames + returns index
## positions of those cube in the original list of ref targets + an array
## containing the new cube position, index positions of its corresponding ref
## frames, and the frame scores
def update_ref_cubes(frame_index, ref_cube_range, ref_scores):
    nframes = len(frame_index)
    ncubes = len(ref_cube_range)-1
    ref_cube_index = []
    frame_posn = np.zeros(frame_index.shape + (3,), dtype=int) ##store frame index within each ref target cube
    for i in range(nframes):
        for j in range(ncubes):
            if frame_index[i]>=ref_cube_range[j] and frame_index[i]<ref_cube_range[j+1]:
                if j not in ref_cube_index:
                    ref_cube_index.append(j)
                frame_posn[i] = (ref_cube_index.index(j), frame_index[i]-ref_cube_range[j], ref_scores[i])
                break

    return np.array(ref_cube_index), frame_posn


## loads reference target cube, copies ref frames to new variable, and adds
## the reference target information to the final header
def make_ref_cube(ref_frame_posn, ref_cube_path, wl, crop, ref_hdr):
    ref_cube = []
    frame_count = 0

    ref_hdr[''] = ''
    ref_hdr['COMMENT'] = 'INIT = index of frames in original reference target cube'
    ref_hdr['COMMENT'] = 'REFI = new index of frames in this reference cube '
    ref_hdr['COMMENT'] = 'SCOR = score of frames calculated using correlation matrix'

    ref_info = np.zeros((len(ref_cube_path),2)) #saving no. frames used, and sum of frame scores of ref target cubes

    for i, cube_path in enumerate(ref_cube_path):
        try:
            cube_tmp = fits.getdata(cube_path)
            hdr_tmp = fits.getheader(cube_path)

            cubei = np.where(ref_frame_posn[:,0]==i)[0]
            framei = ref_frame_posn[cubei,1]
            frame_score = ref_frame_posn[cubei,2]

            if crop is not None:
                x,y = cube_tmp.shape[-2:]
                if (x+crop)%2:
                    x+=1
                fmin, fmax = ((x-crop)//2, (x+crop)//2)
                cube_tmp = cube_tmp[...,fmin:fmax,fmin:fmax]

            ref_cube.append(cube_tmp[wl,framei])
            ref_hdr = add_header_ref_info(hdr_tmp, i, framei, frame_count, ref_hdr, frame_score)
            frame_count += len(framei)

            ref_info[i] = (len(framei),frame_score.sum())

        except FileNotFoundError:
            print('[Warning] File not found with path {p}, \n.. Continuing to next reference target .. \n'
                  .format(p=cube_path))
            continue

    ref_hdr.set('CHANNEL', wl, after='SEL-VECT')
    ref_hdr.set('NCUBE', len(ref_cube), 'total number of ref targets used', after='CHANNEL')
    ref_hdr.set('NFRAME', frame_count, 'total number of ref frames', after='NCUBE')
    ref_hdr.set('', '----------------', after='NFRAME')##line break between science + ref cube info

    print('> Reference library built using {0:d} reference targets and {1:d} frames.'
          .format(len(ref_cube),frame_count))

    return np.concatenate(ref_cube), ref_hdr, ref_info


## creates header for ref cube + adds header information from science cube + recipe
## input parameters
def make_header_sci_info(science_hdr, ncorr, score, use_select_vect):
    hdr = fits.Header()

    hdr.set('OBJECT', science_hdr['OBJECT'], 'science target name')
    hdr.set('RA', science_hdr['RA'], '[deg] science target RA')
    hdr.set('DEC', science_hdr['DEC'], '[deg] science target Dec')
    hdr.set('DATE-OBS', science_hdr['DATE-OBS'], 'observation date of cube')
    hdr.set('DATE-PRO', science_hdr['DATE'], 'processed cube file creation date (UT)')
    hdr.set('EXPTIME', science_hdr['EXPTIME'], 'observation exposure time')
    hdr.set('PIXSCAL', science_hdr['PIXSCAL'],'[mas/pixel] IRDIS pixel scale')

    hdr.set('ESO INS COMB IFLT', science_hdr['ESO INS COMB IFLT'], 'wavelength filter')
    hdr.set('ESO INS COMB ICOR', science_hdr['ESO INS COMB ICOR'], 'coronagraph')
    hdr.set('ESO INS4 COMB IND', science_hdr['ESO INS4 COMB IND'], 'neutral density assembly')

    hdr['ESO OBS ID'] = science_hdr['ESO OBS ID']

    hdr['NCORR'] = ncorr
    hdr['SCORE'] = score
    hdr.set('SEL-VECT', use_select_vect, 'frame selection vector used for science cube')

    return hdr


##updating dict with information for each data cube
def add_target_info(target_info, cube_hdr, i, ftype, rframes, tscore):

    target_info['Obj'].append(cube_hdr['OBJECT'])
    target_info['Type'].append(ftype)
    target_info['RA'][i] = cube_hdr['RA']
    target_info['Dec'][i] = cube_hdr['DEC']
    target_info['Epoch'].append(cube_hdr['DATE-OBS'])

    target_info['Nframes'][i] = cube_hdr['NAXIS3']
    target_info['Rframes'][i] = rframes
    target_info['Tscore'][i] = tscore

    target_info['Expt'][i] = cube_hdr['EXPTIME']
    target_info['Alt'][i] = cube_hdr['ESO TEL ALT']
    target_info['Az'][i] = cube_hdr['ESO TEL AZ']

    try:
        target_info['SR_avg'][i] = cube_hdr['SR_AVG']
        target_info['SR_min'][i] = cube_hdr['SR_MIN']
        target_info['SR_max'][i] = cube_hdr['SR_MAX']
        target_info['Seei_avg'][i] = cube_hdr['SEEI_AVG']
        target_info['Seei_min'][i] = cube_hdr['SEEI_MIN']
        target_info['Seei_max'][i] = cube_hdr['SEEI_MAX']
        target_info['Wind_avg'][i] = cube_hdr['WIND_AVG']
        target_info['Wind_min'][i] = cube_hdr['WIND_MIN']
        target_info['Wind_max'][i] = cube_hdr['WIND_MAX']
    except:
        print('[Warning] Problem with cube header for',cube_hdr['OBJECT'],cube_hdr['DATE-OBS'])

    target_info['WINDSP'][i] = cube_hdr['ESO TEL AMBI WINDSP']
    target_info['AIRM_start'][i] = cube_hdr['ESO TEL AIRM END']
    target_info['AIRM_end'][i] = cube_hdr['ESO TEL AIRM START']
    target_info['FWHM_start'][i] = cube_hdr['ESO TEL AMBI FWHM START']
    target_info['FWHM_end'][i] = cube_hdr['ESO TEL AMBI FWHM END']
    target_info['TAU0'][i] = cube_hdr['ESO TEL AMBI TAU0']

    try:
        target_info['AIRM_mean'][i] = cube_hdr['ESO TEL AIRM MEAN']
        target_info['FWHM_mean'][i] = cube_hdr['ESO TEL AMBI FWHM MEAN']
        target_info['TAU0_mean'][i] = cube_hdr['ESO TEL TAU0 MEAN']
    except KeyError:
        target_info['AIRM_mean'][i] = -1
        target_info['FWHM_mean'][i] = -1
        target_info['TAU0_mean'][i] = -1

    return target_info

## creating fits table containing information for each data cube, including
## science, reference, and unused reference targets
def get_target_info(science_hdr, ncubes, ref_path, ref_cube_index, ref_info):
    target_info = add_target_info(init_info_dict(ncubes), science_hdr, 0, 'science', 0, -1)

    for i in range(ncubes-1):
        try:
            hdr_tmp = fits.getheader(ref_path[i])
        except FileNotFoundError:
            target_info['Obj'].append('-')
            target_info['Epoch'].append('-')
            target_info['Type'].append('[file missing]')

            continue

        if i in ref_cube_index:
            j = np.where(ref_cube_index == i)[0][0]
            target_info=add_target_info(target_info, hdr_tmp, i+1, 'reference', ref_info[j][0], ref_info[j][1])
        else:
            target_info=add_target_info(target_info, hdr_tmp, i+1, 'unused', 0, 0) ##for testing; change to save ref frame info only later

    return make_info_table(target_info)


def make_info_table(target_info):

    col1 = fits.Column(name='Object', format='30A', array=np.array(target_info['Obj']))
    col2 = fits.Column(name='Frame_type', format='10A', array=np.array(target_info['Type']))
    col3 = fits.Column(name='RA', format='D', unit='deg', array=target_info['RA'])
    col4 = fits.Column(name='Dec', format='D', unit='deg', array=target_info['Dec'])
    col5 = fits.Column(name='Obs_date', format='30A', array=np.array(target_info['Epoch']))

    col6 = fits.Column(name='Cube_Nframes', format='K', array=target_info['Nframes'])
    col7 = fits.Column(name='Ref_Nframes', format='K', array=target_info['Rframes'])
    col8 = fits.Column(name='Frame_score_sum', format='K', array=target_info['Tscore'])
    col9 = fits.Column(name='Exptime', format='D', unit='s', array=target_info['Expt'])
    col10 = fits.Column(name='Alt', format='D', unit='deg', array=target_info['Alt'])
    col11 = fits.Column(name='Az', format='D', unit='deg', array=target_info['Az'])

    col12 = fits.Column(name='Air_mass_mean_DIMM', format='D', array=target_info['AIRM_mean'])
    col13 = fits.Column(name='FWHM_mean_DIMM', format='D', unit='arcsec', array=target_info['FWHM_mean'])
    col14 = fits.Column(name='Tau0_mean_DIMM', format='D', unit='s', array=target_info['TAU0_mean'])

    col15 = fits.Column(name='V_wind_tel_ambi', format='D', unit='m/s', array=target_info['WINDSP'])
    col16 = fits.Column(name='Air_mass_tel_start', format='D', array=target_info['AIRM_start'])
    col17 = fits.Column(name='Air_mass_tel_end', format='D', array=target_info['AIRM_end'])
    col18 = fits.Column(name='FWHM_tel_ambi_start', format='D', unit='arcsec', array=target_info['FWHM_start'])
    col19 = fits.Column(name='FWHM_tel_ambi_end', format='D', unit='arcsec', array=target_info['FWHM_end'])
    col20 = fits.Column(name='Tau0_tel_ambi', format='D', unit='s', array=target_info['TAU0'])

    col21 = fits.Column(name='Strehl_avg_SPARTA', format='D', array=target_info['SR_avg'])
    col22 = fits.Column(name='Strehl_min_SPARTA', format='D', array=target_info['SR_min'])
    col23 = fits.Column(name='Strehl_max_SPARTA', format='D', array=target_info['SR_max'])
    col24 = fits.Column(name='FWHM_avg_SPARTA', format='D', unit='arcsec', array=target_info['Seei_avg'])
    col25 = fits.Column(name='FWHM_min_SPARTA', format='D', unit='arcsec', array=target_info['Seei_min'])
    col26 = fits.Column(name='FWHM_max_SPARTA', format='D', unit='arcsec', array=target_info['Seei_max'])
    col27 = fits.Column(name='V_wind_avg_SPARTA', format='D', unit='m/s', array=target_info['Wind_avg'])
    col28 = fits.Column(name='V_wind_min_SPARTA', format='D', unit='m/s', array=target_info['Wind_min'])
    col29 = fits.Column(name='V_wind_max_SPARTA', format='D', unit='m/s', array=target_info['Wind_max'])

    coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11,
                            col12, col13, col14, col15, col16, col17, col18, col19, col20,
                            col21, col22, col23, col24, col25, col26, col27, col28, col29])

    return coldefs

##initialise dict for storing data cube information
def init_info_dict(ncube):
    target_info = {'Obj':[],
                   'Type':[],
                   'RA':np.full(ncube,-1,float),
                   'Dec':np.full(ncube,-1,float),
                   'Epoch':[],
                   'Nframes':np.zeros(ncube,int),
                   'Rframes':np.zeros(ncube,int),
                   'Tscore':np.zeros(ncube,int),
                   'Expt':np.full(ncube,-1,float),
                   'Alt':np.full(ncube,-1,float),
                   'Az':np.full(ncube,-1,float),
                   'AIRM_mean':np.full(ncube,-1,float),
                   'FWHM_mean':np.full(ncube,-1,float),
                   'TAU0_mean':np.full(ncube,-1,float),
                   'WINDSP':np.full(ncube,-1,float),
                   'AIRM_start':np.full(ncube,-1,float),
                   'AIRM_end':np.full(ncube,-1,float),
                   'FWHM_start':np.full(ncube,-1,float),
                   'FWHM_end':np.full(ncube,-1,float),
                   'TAU0':np.full(ncube,-1,float),
                   'SR_avg':np.full(ncube,-1,float),
                   'SR_min':np.full(ncube,-1,float),
                   'SR_max':np.full(ncube,-1,float),
                   'Seei_avg':np.full(ncube,-1,float),
                   'Seei_min':np.full(ncube,-1,float),
                   'Seei_max':np.full(ncube,-1,float),
                   'Wind_avg':np.full(ncube,-1,float),
                   'Wind_min':np.full(ncube,-1,float),
                   'Wind_max':np.full(ncube,-1,float)
                   }

    return target_info

#%%
parser = argparse.ArgumentParser()
parser.add_argument('sof', help='name of sof file', type=str)
parser.add_argument('--score', help='score frame must be above in order to be selected', type=int, default=10)
parser.add_argument('--ncorr', help='no. of top best correlated ref frames scored', type=int, default=100)
parser.add_argument('--crop', help='px size to crop frames to', type=int)
parser.add_argument('--max_frames', help='set no. frames to make ref cube', type=int)
parser.add_argument('--use_select_vect', action='store_true', help='take frame selection vector into account')
#parser.add_argument('--per_frame', action='store_true', help='make ref cube for each frame')
## perhaps when making a ref cube per science frame, rather than storing a [x,y,nframes-ref,nframes-sci]
## cube, which would need to impose large restriction on max nframe-ref allowed, instead store
## as a 3D cube with ref frames common to all science frames only stored once + save additional
## ref frame selection vector for the science frames
args = parser.parse_args()

sof = args.sof
score = args.score
ncorr = args.ncorr
crop = args.crop
max_frames = args.max_frames
use_select_vect = args.use_select_vect
#ref_per_frame = args.per_frame

data = np.loadtxt(sof,dtype=str)
fnames = data[:,0]
ftypes = data[:,1]

## getting required file names for science cube, correlation matrix, and frame selection vector
science_cube_path = fnames[np.where(ftypes=='IRD_SCIENCE_REDUCED_MASTER_CUBE')[0]]
if len(science_cube_path)!=1:
    raise Exception('[Error] Input must contain one reduced science cube.')
science_cube_path = science_cube_path[0]

corr_matrix_path = fnames[np.where(ftypes=='IRD_CORR_MATRIX')[0]]
if len(corr_matrix_path)!=1:
    raise Exception('[Error] Input must contain one correlation matrix.')
corr_matrix_path = corr_matrix_path[0]

if use_select_vect:
    frame_select_path = fnames[np.where(ftypes=='IRD_FRAME_SELECTION_VECTOR')[0]]
    if len(frame_select_path)!=1:
        raise Exception('[Error] Input must contain one frame selection vector when --select_frames set to True.')
    frame_select_path = frame_select_path[0]
#%%
print('.. Reading in science data ..')

science_hdr = fits.getheader(science_cube_path)
x = science_hdr['NAXIS1']
y = science_hdr['NAXIS2']
nframes = science_hdr['NAXIS3']

corr_matrix = fits.getdata(corr_matrix_path)
corr_matrix_hdr = fits.getheader(corr_matrix_path)
corr_channels = corr_matrix.shape[0]

wl_convert = {0: [0], 1: [1], 2: [0,1]}
wl_channel = wl_convert[corr_matrix_hdr['WL_CHOSE']]

if use_select_vect:
    frame_select = fits.getdata(frame_select_path)
    if len(frame_select.shape)>1:
        frame_select=frame_select[:,0]
    good_frames = np.where(frame_select==1.)[0]
    print('> Reference frame selection using correlation of best {0:d} science frames'.format(len(good_frames)))
else:
    good_frames = None
    print('> Reference frame selection using correlation of all {0:d} science frames'.format(nframes))

print('.. Finished reading in science data ..')

nb_ref_cube = int(corr_matrix_hdr["NB_REF_CUBES"])
ref_cube_path_tmp = []
ref_cube_nframes = []
ref_cube_range = []

## storing information about the reference targets used
for i in range(nb_ref_cube):
    nb_str = "{0:06d}".format(i)
    ref_cube_path_tmp.append(corr_matrix_hdr["RN"+nb_str])
    ref_cube_nframes.append(int(corr_matrix_hdr["RF"+nb_str]))
    ref_cube_range.append(int(corr_matrix_hdr["RS"+nb_str]))
ref_cube_range.append(ref_cube_range[-1]+ref_cube_nframes[-1])
print('> Finding best correlated frames from {0:d} reference target cube(s)'.format(nb_ref_cube))

## scoring reference frames + making cube for each wl channel of the correlation matrix
for wl in wl_channel:
    print('.. Scoring reference frames for channel {0:d} ..'.format(wl))
    ref_frame_index, ref_scores = score_frames(corr_matrix[wl], score, ncorr, max_frames, good_frames)
    ref_cube_index, ref_frame_posn = update_ref_cubes(ref_frame_index, ref_cube_range, ref_scores)
    ref_cube_path = [ref_cube_path_tmp[p] for p in ref_cube_index]

    print('.. Building reference cube ..')
    ref_cube, ref_hdr, ref_info = make_ref_cube(ref_frame_posn, ref_cube_path, wl, crop,
                        make_header_sci_info(science_hdr, ncorr, score, use_select_vect))

    coldefs = get_target_info(science_hdr, nb_ref_cube + 1, ref_cube_path_tmp, ref_cube_index, ref_info)
    table_hdu = fits.BinTableHDU.from_columns(coldefs, header=make_header_sci_info(science_hdr, ncorr, score, use_select_vect))

    table_hdu.writeto('target_info_table_wl{0:d}.fits'.format(wl))
#%%
    hdu = fits.PrimaryHDU(data = ref_cube, header = ref_hdr)
    hdu.writeto('reference_cube_wl{0:d}.fits'.format(wl))

"""
from skimage.draw import disk
## create mask for correlation area
def Mask(size,inner_radius,outer_radius):
    cxy=(size//2,size//2)
    mask_in=disk(cxy,inner_radius,shape=(size,size))
    mask_out=disk(cxy,outer_radius,shape=(size,size))
    mask=np.full((size,size),False)
    mask[mask_out]=True
    mask[mask_in]=False

    return mask
"""