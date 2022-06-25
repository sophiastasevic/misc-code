"""
calculate correlation between frames in a matrix
"""

import numpy as np
from skimage.draw import disk

def create_mask(x, y, inner_radius, outer_radius):
    cxy=(x//2,y//2)
    mask_in=disk(cxy,inner_radius,shape=(x,y))
    mask_out=disk(cxy,outer_radius,shape=(x,y))
    mask=np.full((x,y),False)
    mask[mask_out]=True
    mask[mask_in]=False

    return mask


def corr_matrix(cube, inner_radius, outer_radius):
    nframes,x,y = cube.shape
    median_cube = np.nanmedian(cube, axis=0)
    matrix = np.zeros(nframes)
    mask = create_mask(x, y, inner_radius, outer_radius)
    for i in range(nframes):
        matrix[i] = np.corrcoef(np.reshape(median_cube*mask, x*y),
                                np.reshape(cube[i]*mask, x*y))[0,1]
    return matrix


def residuals_frame_select(cube, inner_radius, outer_radius):
    corr_mean = corr_matrix(cube, inner_radius, outer_radius)

    good_frames = np.where(corr_mean > 0)[0]

    return good_frames
