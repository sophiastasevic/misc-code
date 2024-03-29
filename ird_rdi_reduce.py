"""
Reduce a IRD_SCIENCE_REDUCED_MASTER_CUBE

@Author : Yuchen BAI
@Date   : 22/07/2021
@Contact: yuchenbai@hotmail.com
@Modifications
2021-12-20: JMi bug fix when no fake cube is specified
2021-12-21: add the option to save the reference cube
2022-02-23: [SS] added print outputs (original ref frame score + array index) + fixed counter in ref header
"""

import argparse
import warnings
import datetime
import numpy as np
import vip_hci as vip
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

###########
# setting #
###########

warnings.simplefilter('ignore', category=AstropyWarning)

############
# function #
############

# generate a list of tuple
def get_coords_of_ref_frames(nth, nb_frames):
    '''
    This function can help to preserve the information of frame coordinates.
    Can help us find which image is used to process the target frame.
    Args:
        nth : a integer. The nth cube in the reference library.
        nb_frames : a integer. The nth frame number of a cube.
    Return:
        res : a list of tuple. (nth cube, nth frame of the cube)
    '''
    res = []
    for i in range(nb_frames):
        res.append((nth, i, nb_frames))
    return res

# collect reference data from datacenter
def collect_frames(files_path, crop_size, full_output=True):
    '''
    Args:
        files_path : a list of string. files path contain keyword.
        crop_size : a integer. The size in center that we want process.
        full_output : a boolean. Default value is False. If it is true, we will return 3 result, if not, we inly return ref_frames.
    Rrturn:
        ref_frames : ndarray, 4 dimensions. Return (wavelengths, nb_frames, x, y)
        ref_frames_coords : a list of tuple. [(0,0), (0,1), ...(nth cube, nth frame of the cube)]
        ref_cube_nb_frames : a list of integer. The list contains all frame numbers of the reference cube.
    '''

    if not files_path:
        raise Exception("In the function collect_frames, there is nothing in the files_path which means no reference cube!!!")

    ref_frames_coords = []
    hd = fits.getdata(files_path[0])

    # frames in the first wavelength and second wavelength
    # K1/K2, H2/H3, etc...
    wl, nb_fr, w, h = hd.shape
    start = int((w-crop_size)//2)
    end = start + crop_size

    ref_frames = hd[..., start:end, start:end]
    ref_frames_coords = ref_frames_coords + get_coords_of_ref_frames(0, nb_fr)
    ref_cube_nb_frames = []
    ref_cube_nb_frames.append(nb_fr)

    for i in range(1,len(files_path)):
        hd = fits.getdata(files_path[i])
        wl, nb_fr, w, h = hd.shape
        ref_frames =np.append(ref_frames, hd[..., start:end, start:end], axis=1)
        ref_frames_coords = ref_frames_coords + get_coords_of_ref_frames(i, nb_fr)
        ref_cube_nb_frames.append(nb_fr)

    if full_output is False:
        return ref_frames, ref_frames_coords

    return ref_frames, ref_frames_coords, ref_cube_nb_frames

# get histogram of reference stars
def get_histogram_of_ref_stars_score(ref_star_scores, ref_cube_nb_frames):
    '''
    This function will count how many frames we use for each star in the reference library.
    Args:
        ref_star_scores : a list of integer. The list of indice, nth frame in the reference frame library.
        ref_cube_nb_frames : a list of integer. Each element is the frame number of a reference star.
    Return:
        res : a ndarray list of integer. The number of integer for each reference star we use.
    '''
    l = len(ref_cube_nb_frames)
    res = np.zeros(l,dtype=int)
    for i in ref_star_scores:
        # indice plus 1, then we can deal with it with the length of
        i = i
        for n in range(l):
            i = i - ref_cube_nb_frames[n]
            if i<=0:
                res[n] += 1
                break

    return res

# frame based version selection but with score system
def selection_frame_based_score(corr_matrix, target, nb_best_frame, ref_frames, ref_cube_nb_frames, score, wave_length):
    '''
    Args:
        corr_matrix : a numpy.ndarray, 2 dims. The correlation matrix.
        target : a numpy.ndarray, 4 dims. The science target cube, (wavelengths, nb_frames, x, y).
        nb_best : a integer. How many best frames fo the references stars array we want for each target frame.
        ref_frames : a numpy.ndarry, 4 dims. The reference stars data we have.
        ref_cube_nb_frames : a list of integer. Each element is the frame number of a reference star.
        score : a integer. We will pick all the reference stars which has higher or equal score.
        wave_length : a integer. Wave length of the reference cube.
    Rrturn:
        res : a ndarray, 3 dimensions. Return (nb_frames, x, y).
    '''
    # target shape
    wl_t, nb_fr_t, w, h = target.shape
    wl_ref, nb_fr_ref, w_ref, h_ref = ref_frames.shape

    # score_system
    ref_scores = np.zeros((nb_fr_ref),dtype=int)

    for i in range(nb_fr_t):
        tmp = {}
        for j in range(nb_fr_ref):
            tmp[j] = corr_matrix[i,j]

        if nb_best_frame > len(tmp):
            raise Exception("!!! inside the function selection_frame_based, tmp", len(tmp),"is samller than nb_best_frame", nb_best_frame)

        res_tmp = sorted(tmp.items(),key = lambda r:(r[1],r[0]), reverse=True)[0:nb_best_frame]

        for (ind, pcc) in res_tmp:
            ref_scores[ind] += 1

    res_coords = np.where(ref_scores>=score)[0]
    print("ref frame, corr frame, score: ")
    print('\n'.join('{0:d} \t{1:d} \t{2:d}'.format(k,index,score) for k,(index,score) in enumerate(zip(res_coords,ref_scores[res_coords]))))
    res = ref_frames[wave_length][res_coords]
    print("res.shape =", res.shape)

    return res, get_histogram_of_ref_stars_score(res_coords, ref_cube_nb_frames)

# make a dictionary from two list
def get_dict(key, value):
    '''
    This function will count how many frames we use for each star in the reference library.
    Args:
        key : a list of element, string or integer... Input key is the ref_files.
        value : a list of element, string or integer...
    Return:
        res : a dict. For drawing a barplot.
    '''
    res = {}

    for i in range(len(key)):

        k = fits.open(key[i])[0].header['OBJECT']
        v = value[i]

        if v == 0:
            continue

        if k not in res.keys():
            res[k] = value[i]
        else:
            res[k] = res[k] + value[i]

    return res

# distance between two points
def distance(x1, y1, x2, y2):
    '''
    Args:
        x1 : an integer. object 1 - coordinate X
        y1 : an integer. object 1 - coordinate Y
        x2 : an integer. object 2 - coordinate X
        y2 : an integer. object 2 - coordinate Y
    Return:
        res : an integer. The distance between two points.
    '''
    return ((x1-x2)**2+(y1-y2)**2)**0.5

# use inner mask and outer mask to calculate the pcc(pearson correlation coeffient)
def create_mask(crop_size, inner_radius, outer_radius):
    '''
    Args:
        crop_size : an integer. The size of frame/image.
        inner_radius : an integer.
        outer_radius : an integer.
    Return:
        res : a numpy.ndarray, 2 dimens. Ex. (256, 256) but the center is all 0.
    '''
    count = 0
    res = np.full((crop_size, crop_size), True)
    x = crop_size//2
    y = crop_size//2
    for i in range(crop_size):
        for j in range(crop_size):
            if distance(i, j, x, y) >= outer_radius or distance(i, j, x, y) <= inner_radius:
                res[i,j] = False
                count = count + 1
    return res

# print cueb info
def print_cube_info(science_header, name):
    '''
    Arg:
        science_header: a fits header.
        name : a string. What we display here.
    Return:
        None.
    '''
    print("\n------")
    print("> This is", name)
    print(">> DATE-OBS:", science_header["DATE-OBS"])
    print(">> OBJECT:", science_header["OBJECT"])
    print(">> EXPTIME:", science_header["EXPTIME"])
    print(">> ESO INS COMB ICOR:", science_header["ESO INS COMB ICOR"])
    print(">> ESO INS COMB IFLT:", science_header["ESO INS COMB IFLT"])
    print("------\n")

    return None

# do rdi
def do_rdi(corr_matrix_wl, science_cube_croped, pct, n_corr, ref_frames, ref_cube_nb_frames, score, wl, ref_cube_path, derotation_angles, ncomp, inner_radius, mask, scaling, ref_x, ref_y):
    '''
    We will do selection and rdi in this function
    Arg:
        corr_matrix_wl : a 2d-array. One big matrix which contains pearson correlation coefficient value.
        science_cube_croped : a 3d-array. A croped science cube.
        pct : a list of float. See parser.
        n_corr : a integer. See parser.
        ncomp : a integer. See parser.
        ref_frames : a 2d array. The referene frames.
        ref_cube_nb_frames : a list of integer. Sotre the frame number for each reference target.
        score : a integer. See parser.
        wl : a integer. 0 or 1, the wave length we will handle.
        ref_cube_path : a dictionary.
        derotation_angles : a fits file.
        inner_radius : a integer.
        mask : a 2-d array. Mask to cover the no-interesting region.
        scaling : a string. The scaling.
        ref_x : a integer. Length of ref frame.
        ref_y : a integer. Length of ref frame.
    Return:
        res : a n-darray. The result of algo RDI.
    '''
    l_pct = len(pct)
    l_n_corr = len(n_corr)
    l_ncomp = len(ncomp)
    res = np.zeros((l_pct, l_n_corr, l_ncomp, ref_x, ref_y))
    for i in range(l_pct):
        nb_ref_frame_tmp = int(nb_ref_frames*pct[i])
        for j in range(l_n_corr):
            print(">> nb_ref_frames =", nb_ref_frame_tmp,"n_corr =", n_corr[j])
            ref_frames_selected, target_ref_coords = selection_frame_based_score(corr_matrix_wl, science_cube_croped, n_corr[j], ref_frames[:,0:nb_ref_frame_tmp], ref_cube_nb_frames, score, wave_length=wl)
            dict_ref_in_target = get_dict(ref_cube_path, target_ref_coords)
            print(">>> wave length channel=", wl_final, "\nref star, nframes:")
            print('\n'.join('{0:s} \t{1:d}'.format(star,frame) for star,frame in dict_ref_in_target.items()))
            print(">>> ref_frames_selected_fake.shape =", ref_frames_selected.shape)
            if save_lib:
                if sc_exist:
                    header_ref_frames_selected = science_header
                if sc_f_exist:
                    header_ref_frames_selected = science_header_fake
                counter = 1
                for k,v in dict_ref_in_target.items():
                    header_ref_frames_selected['REF CUBE {0:05d}'.format(counter)] = '{0:s} ({1:d})'.format(k,v)
                    counter += 1
                fits.writeto('reference_cube.fits', ref_frames_selected,header_ref_frames_selected)
            for k in range(l_ncomp):
                print(">>>> ncomp =", ncomp[k])
                res[i,j,k] = vip.pca.pca_fullfr.pca(science_cube_croped[wl]*mask, -derotation_angles, ncomp=ncomp[k], mask_center_px=inner_radius, cube_ref=ref_frames_selected*mask, scaling=scaling)

    return res

#############
# main code #
#############

print("############ recipe : ird_rdi_reduce.py is running ############")
start_time = datetime.datetime.now()
parser = argparse.ArgumentParser(description="Do the RDI reduction with help of big reference library.")
# file .sof whille contain the CORRELATION_MATRIX, SCIENCE TARGET, PARALLACTIC ANGLE
parser.add_argument("sof", help="file name of the sof file", type=str)
parser.add_argument("--score", help="which decide how we choose the reference frame (>=1)", type=int, default=1)
parser.add_argument("--n_corr", help="the number of best correalted frames for each frame of science target, a list of integer", type=str, default="empty")
parser.add_argument("--ncomp",help="number of principal components to remove, a list of integer", type=str, default="empty")
parser.add_argument("--pct", help="the percentage we want to use the reference library, a list of float >0 and <=1. If you don't know what this param means, use 1 !", type=str, default="empty")
parser.add_argument("--scaling", help="scaling for the PCA (to choose between 0 for spat-mean, 1 for spat-standard, 2 for temp-mean, 3 for temp-standard or 4 for None)",\
                    type=int, choices=[0,1,2,3,4], default=0)
parser.add_argument('--save_lib', action='store_true',help="if True save the library of reference frames. In this case only one value for pct, n_corr and score must be used.")

# handle args
args = parser.parse_args()

# sof
sofname=args.sof

# --score
score = args.score

# --n_corr, a list of integer
n_corr = [int(e) for e in args.n_corr.split(' ')]

# --ncomp, a list of integer
ncomp = [int(e) for e in args.ncomp.split(' ')]

# --pct, a list of float
pct = [float(e) for e in args.pct.split(' ')]

# --scaling
scaling_dict = {0 : 'spat-mean', 1 : 'spat-standard', 2 : 'temp-mean', 3 : 'temp-standard', 4 : None}
scaling = scaling_dict[args.scaling]

# --save_lib
save_lib = args.save_lib

# Step-1 Reading the sof file
data=np.loadtxt(sofname,dtype=str)
filenames=data[:,0]
datatypes=data[:,1]

# sof - master cube, raw science cube
print("\n------------ reading sof file ------------\n")
science_cube_paths = filenames[np.where(datatypes == 'IRD_SCIENCE_REDUCED_MASTER_CUBE')[0]]
sc_exist = False
if len(science_cube_paths) < 1:
    print("> Well, we don't have the raw data/raw science cube this time")
elif (len(science_cube_paths) == 1):
    sc_exist = True
    print("> 1 raw data/raw science master cube - found")
    science_cube_path = science_cube_paths[0]
else:
    raise Exception('The sof file must contain only one IRD_SCIENCE_PSF_MASTER_CUBE file or none')

# sof - master cube, science cube with fake injection
science_cube_paths_fake = filenames[np.where(datatypes == 'IRD_SCIENCE_REDUCED_MASTER_CUBE_FAKE_COMP')[0]]
sc_f_exist = False
if len(science_cube_paths_fake) < 1:
    print("> Well, we don't have the science cube with the fake injection this time")
elif len(science_cube_paths_fake) == 1:
    sc_f_exist = True
    print("> 1 science master cube with the fake companion injection cube - found")
    science_cube_path_fake = science_cube_paths_fake[0]
else:
    raise Exception('The sof file must contain only one IRD_SCIENCE_REDUCED_MASTER_CUBE_FAKE_COMP file')


if not sc_exist and not sc_f_exist:
    raise Exception('The sof file must contain one IRD_SCIENCE_REDUCED_MASTER_CUBE or one IRD_SCIENCE_REDUCED_MASTER_CUBE_FAKE_COMP')

# sof - correlation matrix
corr_matrix_path = filenames[np.where(datatypes == "IRD_CORR_MATRIX")[0]]
if len(corr_matrix_path) > 1:
    raise Exception("The sof file must contain exactly one IRD_CORR_MATRIX file")

# sof - parallactic angle
anglenames = filenames[np.where(datatypes == 'IRD_SCIENCE_PARA_ROTATION_CUBE')[0]]
if len(anglenames) != 1:
    raise Exception('The sof file must contain exactly one IRD_SCIENCE_PARA_ROTATION_CUBE file')

print("\n------------ reading end ------------\n")

# Step-2 take science cube
print(">> corr_matrix_path", corr_matrix_path)
print(">> it's type", type(corr_matrix_path))
corr_matrix_path = corr_matrix_path[0]
corr_matrix = fits.getdata(corr_matrix_path)
corr_matrix_header = fits.getheader(corr_matrix_path)
print_cube_info(corr_matrix_header, "corr_matrix header")
print("> The corr_matrix.shape=", corr_matrix.shape)

# --wl_channels
dico_conversion_wl_channels = {0 : [0], 1 : [1], 2 : [0,1]}
wl_channels = dico_conversion_wl_channels[corr_matrix_header["WL_CHOSE"]]
nb_wl = len(wl_channels)
print("> We will investigate wave length :", wl_channels)

if sc_exist:
    science_cube = fits.getdata(science_cube_path)
    science_header = fits.getheader(science_cube_path)
    print_cube_info(science_header, "science cube header")
    nb_science_wl, nb_science_frames, nx, ny = science_cube.shape
    print(">> science_cube.shape =", science_cube.shape)


if sc_f_exist:
    science_cube_fake = fits.getdata(science_cube_path_fake)
    science_header_fake = fits.getheader(science_cube_path_fake)
    print_cube_info(science_header_fake, "science cube with fake injection header")
    print(">> science_cube_fake.shape =", science_cube_fake.shape)

# parrallactic angle
anglename = anglenames[0]
derotation_angles = fits.getdata(anglename)
derotation_angles_header = fits.getheader(anglename)
print_cube_info(derotation_angles_header, "parallactic angle header")

if len(derotation_angles) != nb_science_frames:
    raise Exception('The science cube IRD_SCIENCE_REDUCED_MASTER_CUBE contains {0:d} frames while the list IRD_SCIENCE_PARA_ROTATION_CUBE contains {1:d} angles'.format(nb_science_frames,len(derotation_angles)))

# the number of reference cube we have, take the necessary
nb_ref_cube = int(corr_matrix_header["NB_REF_CUBES"])
ref_cube_path = []
ref_cube_nb_frames = []
ref_cube_start = []

for i in range(nb_ref_cube):
    nb_str = "{0:06d}".format(i)
    ref_cube_path.append(corr_matrix_header["RN"+nb_str])
    ref_cube_nb_frames.append(int(corr_matrix_header["RF"+nb_str]))
    ref_cube_start.append(int(corr_matrix_header["RS"+nb_str]))

# crop_size
crop_size = int(corr_matrix_header["CROPSIZE"])

# collect data, with pct we can use different size ref_frames
# TODO(yuchen): there is a smarter way to do it
ref_frames, ref_frames_coords, ref_cube_nb_frames_check = collect_frames(ref_cube_path, crop_size)
if ref_cube_nb_frames_check != ref_cube_nb_frames:
    print("Warning! There is something wrong about the ref_cube_nb_frames, check it")

print("> ref_frames.shape =", ref_frames.shape)
wl_ref, nb_ref_frames, ref_x, ref_y = ref_frames.shape

# add information to the header
pct_str = ' '.join(str(int(e*nb_ref_frames)) for e in pct)
n_corr_str = ' '.join(str(e) for e in n_corr)
ncomp_str = ' '.join(str(e) for e in ncomp)

science_header["D_PCT"] = pct_str
science_header["D_N_COR"] = n_corr_str
science_header["D_NCOMP"] = ncomp_str

if sc_f_exist:
    science_header_fake["D_PCT"] = pct_str
    science_header_fake["D_N_COR"] = n_corr_str
    science_header_fake["D_NCOMP"] = ncomp_str

print(">> pct_str :", pct_str)
print(">> n_corr_str :", n_corr_str)
print(">> ncomp_str :", ncomp_str)

# crop science cube
start = int((nx-crop_size)//2)
end = start + crop_size

if sc_exist:
    science_cube_croped = science_cube[..., start:end, start:end]
if sc_f_exist:
    science_cube_croped_fake = science_cube_fake[..., start:end, start:end]

# correlation matrix
inner_radius = int(corr_matrix_header["INNER_R"])
outer_radius = int(corr_matrix_header["OUTER_R"])
mask = create_mask(crop_size, inner_radius, outer_radius)

# corr_matrix
print("> corr_matrix.shape", corr_matrix.shape)
corr_matrix_0 = corr_matrix[0]
corr_matrix_1 = None
if len(wl_channels)>1:
    corr_matrix_1 = corr_matrix[1]

# set values
wl_final = wl_channels[0]

if sc_exist:
    print("------------ raw data type IRD_SCIENCE_REDUCED_MASTER_CUBE exists, do RDI on wl=", wl_final," ------------")
    res_0 = do_rdi(
        corr_matrix_wl=corr_matrix_0,\
        science_cube_croped=science_cube_croped,\
        ref_frames=ref_frames,\
        pct=pct,\
        n_corr=n_corr,\
        ncomp=ncomp,\
        ref_cube_nb_frames=ref_cube_nb_frames,\
        score=score,\
        wl=wl_final,\
        ref_cube_path=ref_cube_path,\
        derotation_angles=derotation_angles,\
        inner_radius=inner_radius,\
        mask=mask,\
        scaling=scaling,\
        ref_x=ref_x,\
        ref_y= ref_y)

    file_name = "rdi_real_res_0.fits"
    print("> The result will be stored in :", file_name)
    science_header["WL_CHOSE"] = wl_final
    science_header["NB_REF"] = nb_ref_cube
    science_header["Fake"] = 0
    hdu = fits.PrimaryHDU(data=res_0, header=science_header)
    hdu.writeto(file_name)

if sc_f_exist:
    print("------------ data type IRD_SCIENCE_REDUCED_MASTER_CUBE_FAKE_COMP exists, do RDI on wl=", wl_final," ------------")
    res_0_fake = do_rdi(
        corr_matrix_wl=corr_matrix_0,\
        science_cube_croped=science_cube_croped_fake,\
        ref_frames=ref_frames,\
        pct=pct,\
        n_corr=n_corr,\
        ncomp=ncomp,\
        ref_cube_nb_frames=ref_cube_nb_frames,\
        score=score,\
        wl=wl_final,\
        ref_cube_path=ref_cube_path,\
        derotation_angles=derotation_angles,\
        inner_radius=inner_radius,\
        mask=mask,\
        scaling=scaling,\
        ref_x=ref_x,\
        ref_y=ref_y)

    file_name_fake = "rdi_fake_res_0.fits"
    print("> The result of fake companion injection will be stored in :", file_name_fake)
    science_header_fake["WL_CHOSE"] = wl_final
    science_header_fake["NB_REF"] = nb_ref_cube
    science_header_fake["Fake"] = 1
    hdu = fits.PrimaryHDU(data=res_0_fake, header=science_header_fake)
    hdu.writeto(file_name_fake)

ref_frames_selected_bis = []
target_ref_coords_bis = []

if nb_wl>1:
    if sc_exist:
        print("------------ raw data type IRD_SCIENCE_REDUCED_MASTER_CUBE exists, do RDI on wl=1 ------------")
        # real science cube on wl=1
        res_1 = do_rdi(
            corr_matrix_wl=corr_matrix_1,\
            science_cube_croped=science_cube_croped,\
            ref_frames=ref_frames,\
            pct=pct,\
            n_corr=n_corr,\
            ncomp=ncomp,\
            ref_cube_nb_frames=ref_cube_nb_frames,\
            score=score,\
            wl=wl_channels[1],\
            ref_cube_path=ref_cube_path,\
            derotation_angles=derotation_angles,\
            inner_radius=inner_radius,\
            mask=mask,\
            scaling=scaling,\
            ref_x=ref_x,\
            ref_y= ref_y)

        file_name = "rdi_real_res_1.fits"
        print("> The result will be stored in :", file_name)
        science_header["WL_CHOSE"] = wl_channels[1]

        hdu = fits.PrimaryHDU(data=res_1, header=science_header)
        hdu.writeto(file_name)

    if sc_f_exist:
        print("------------ data type IRD_SCIENCE_REDUCED_MASTER_CUBE_FAKE_COMP exists, do RDI on wl=1 ------------")
        # science cube with fake comp injection on wl=1
        res_1_fake = do_rdi(
            corr_matrix_wl=corr_matrix_1,\
            science_cube_croped=science_cube_croped_fake,\
            ref_frames=ref_frames,\
            pct=pct,\
            n_corr=n_corr,\
            ncomp=ncomp,\
            ref_cube_nb_frames=ref_cube_nb_frames,\
            score=score,\
            wl=wl_channels[1],\
            ref_cube_path=ref_cube_path,\
            derotation_angles=derotation_angles,\
            inner_radius=inner_radius,\
            mask=mask,\
            scaling=scaling,\
            ref_x=ref_x,\
            ref_y= ref_y)

        file_name_fake = "rdi_fake_res_1.fits"
        print("> The result will be stored in :", file_name)
        science_header_fake["WL_CHOSE"] = 1

        hdu = fits.PrimaryHDU(data=res_1_fake, header=science_header_fake)
        hdu.writeto(file_name_fake)

end_time = datetime.datetime.now()
print("######### End program : no error! Take:", end_time - start_time, "#########")
