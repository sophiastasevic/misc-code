'''
Reduce a IRD_SCIENCE_REDUCED_MASTER_CUBE

@Author : Yuchen BAI
@Date   : 17/07/2021
@Contact: yuchenbai@hotmail.com

20/04/22 [SS] Updated execution of loop to read in + calculate correlation one cube at a time
25/04/22 [SS] Subtract spatial mean from frames before calculating correlation
'''

import argparse
import warnings
import numpy as np
#import vip_hci as vip
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from skimage.draw import disk

from datetime import datetime, timedelta

###########
# setting #
###########

warnings.simplefilter('ignore', category=AstropyWarning)

############
# function #
############

# add information into the header
def complete_header(science_header, reference_cube_names, ref_nb_frames):
    '''
    This function is for adding some additional information on science header.
    Useful for the next stage.
    Args:
        science_header : a fits.header. The science header.
        reference_cube_names : a list of reference cube names.
        ref_nb_frames : a list of number of cubes.
    Return:
        None.
    '''
    nb_ref_cube = len(reference_cube_names)
    science_header['NB_REF_CUBES'] = nb_ref_cube
    ind = 0
    for i in range(nb_ref_cube):
        nb_str = '{0:06d}'.format(i)
        science_header['RN'+nb_str] = reference_cube_names[i]
        science_header['RF'+nb_str] = ref_nb_frames[i]
        science_header['RS'+nb_str] = ind
        ind = ind + ref_nb_frames[i]


## create boolean mask for annulus within which correlation is calculated
def create_mask(size, inner_radius, outer_radius):
    '''
    Args:
        crop_size : an integer. The size of frame/image.
        inner_radius : an integer.
        outer_radius : an integer.
    Return:
        mask : bool array of dim. (size, size); True when inner_radius < r < outer radius, False otherwise
    '''
    cxy=(size//2,size//2)
    mask_in=disk(cxy,inner_radius,shape=(size,size))
    mask_out=disk(cxy,outer_radius,shape=(size,size))
    mask=np.full((size,size),False)
    mask[mask_out]=True
    mask[mask_in]=False

    return mask

# print cueb info
def print_cube_info(science_header, name):
    '''
    Arg:
        science_header: a fits header.
        name : a string. What we display here.
    Return:
        None.
    '''
    print('\n------')
    print('> This is', name)
    print('>> DATE-OBS:', science_header['DATE-OBS'])
    print('>> OBJECT:', science_header['OBJECT'])
    print('>> EXPTIME:', science_header['EXPTIME'])
    print('>> ESO INS COMB ICOR:', science_header['ESO INS COMB ICOR'])
    print('>> ESO INS COMB IFLT:', science_header['ESO INS COMB IFLT'])
    print('------\n')

    return None

#%%
#############
# main code #
#############
print('######### Start program : ird_rdi_corr_matrix.py #########')
parser = argparse.ArgumentParser(description='For build the Pearson Correlation Coefficient matrix for the science target and the reference master cubes, we need the following parameters.')
parser.add_argument('sof', help='file name of the sof file',type=str)
parser.add_argument('--inner_radius',help='inner radius where the reduction starts', type=int, default=10)
parser.add_argument('--outer_radius',help='outer radius where the reduction starts', type=int, default=100)
##TODO:change so that science + ref frames go in the sof file
parser.add_argument('--science_object', help='the OBJECT keyword of the science target', type=str, default='unspecified')
parser.add_argument('--wl_channels', help='Spectral channel to use (to choose between 0 for channel 0, 1 for channel 1, 2 for both channels)', type=int, choices=[0,1,2], default=0)

# handle args
args = parser.parse_args()

# sof
sofname=args.sof

# --science_object
science_object = args.science_object

# --wl_channels
dict_conversion_wl_channels = {0 : [0], 1 : [1], 2 : [0,1]}
wl_channels = dict_conversion_wl_channels[args.wl_channels]
nb_wl = len(wl_channels)

# --crop_size and inner/outer radius
inner_radius = args.inner_radius
outer_radius = args.outer_radius

# start the program

if outer_radius<=10:
    outer_radius=10
    print('Warning: outer radius too small. Value set to 10')

if outer_radius <= inner_radius:
    print('Warning: outer_radius <= inner_radius. Inner radius set to {0:d}'.format(outer_radius-1))

crop_size = 2*outer_radius+1

if type(crop_size) != int:
    crop_size = int(crop_size)

# Reading the sof file
data=np.loadtxt(sofname, dtype=str)
filenames=data[:,0]
datatypes=data[:,1]

cube_names = filenames[np.where(datatypes == 'IRD_SCIENCE_REDUCED_MASTER_CUBE')[0]]
nb_cubes = len(cube_names)

if nb_cubes < 2:
    raise Exception('The sof file must contain at least 2 IRD_SCIENCE_REDUCED_MASTER_CUBE (science and reference)')

# except one science cube, the rest are reference cubes
nb_reference_cubes = nb_cubes - 1 ##TODO:option to not do this

##TODO: change this to not use object header, but instead ra + dec vizier match
if science_object != 'unspecified': ##why does this option exist?
    for i,cube_name in enumerate(cube_names):
        header =fits.getheader(cube_name)
        if header['OBJECT'].strip() == science_object.strip():
            science_cube_name = cube_name
            reference_cube_names = [cube_name for cube_name in cube_names if cube_name != science_cube_name]
            science_object_final = header['OBJECT']
            break
    try:
        print('\nScience OBJECT set to {0:s}'.format(science_object_final))
    except:
        print('Unable to find IRD_SCIENCE_REDUCED_MASTER_CUBE for science object. Using by default option the first cube as science')
        science_cube_name = cube_names[0]
        reference_cube_names = cube_names[1:]
else:
    science_cube_name = cube_names[0]
    reference_cube_names = cube_names[1:]

print('> science cube :', science_cube_name)

# take science cube
science_cube = fits.getdata(science_cube_name)
science_header = fits.getheader(science_cube_name)
print_cube_info(science_header, 'science cube header')
nb_wl_channels, nb_science_frames, nx, ny = science_cube.shape

print('>> science_cube.shape =', science_cube.shape)

if (nx+crop_size)%2:
    nx+=1
border_l = (nx - crop_size)//2
border_r = (nx + crop_size)//2

science_cube_cropped = science_cube[..., border_l:border_r, border_l:border_r]

mask = create_mask(crop_size, inner_radius, outer_radius)
print('> The mask has been created, crop_size=', crop_size, 'inner_radius=', inner_radius, 'outer_radius=', outer_radius)

# sort reference cube names
reference_cube_names.sort()
print('> Library contains', len(reference_cube_names), 'reference stars.')
print('>> The reference cube library has been sorted')

ref_cube_nb_frames = []
corr_mat = []
problem_ref = []

load_time=timedelta(0)
comp_time=timedelta(0)
for name in reference_cube_names:
    try:
        #get reference cube data
        start_time=datetime.now()
        ref_frames = fits.getdata(name)[..., border_l:border_r, border_l:border_r]
        load_time+=datetime.now()-start_time
        wl_ref, nb_ref_frames, ref_x, ref_y = ref_frames.shape
        ref_cube_nb_frames.append(nb_ref_frames)
        res = np.zeros((nb_wl, nb_science_frames, nb_ref_frames))

        #calculate correlation between reference frames + science frames
        start_time=datetime.now()
        for w in range(nb_wl):
            wl = wl_channels[w]
            for i in range(nb_science_frames):
                sci_frame = science_cube_cropped[wl,i]-np.nanmean(science_cube_cropped[wl,i]) ##subtract spat mean
                for j in range(nb_ref_frames):
                    ref_frame = ref_frames[wl, j]-np.nanmean(ref_frames[wl, j]) ##subtract spat mean
                    res[wl, i, j] = np.corrcoef(np.reshape(sci_frame*mask, ref_x*ref_y),
                                                np.reshape(ref_frame*mask, ref_x*ref_y))[0,1]
        comp_time+=datetime.now()-start_time
        corr_mat.append(res)

    except FileNotFoundError:
        print('> Warning: Could not find file:', name,'\nContinuing to next reference target')
        problem_ref.append(name)
        continue

corr_mat=np.concatenate(corr_mat,axis=-1)
reference_cube_names=[name for name in reference_cube_names if name not in problem_ref]
print('For {t} reference targets, load time = {lt}, calculation time = {ct}'
      .format(t=len(reference_cube_names),lt=load_time,ct=comp_time))

# compelte header
science_header['PATH_TAR'] = science_cube_name
science_header['CROPSIZE'] = crop_size
science_header['INNER_R'] = inner_radius
science_header['OUTER_R'] = outer_radius
science_header['WL_CHOSE'] = args.wl_channels
complete_header(science_header, reference_cube_names, ref_cube_nb_frames)

file_name = 'pcc_matrix.fits'
hdu = fits.PrimaryHDU(data=corr_mat, header=science_header)
hdu.writeto(file_name)
print('######### End program : no error #########')


