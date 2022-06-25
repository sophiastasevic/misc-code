"""
Function for PCA reduction of datacube
"""

import numpy as np
from skimage.draw import disk

class ScienceCube:
    def __init__(self, axis, method, data, r_mask):
        self.n_frames, self.x_px, self.y_px = data.shape
        self.data = np.reshape(data, (self.n_frames, self.x_px*self.y_px))
        self.mean, self.stdev = (Normalise(self.data, method, axis))
        self.data = MaskCube(self.data, self.x_px, self.y_px, r_mask, self.n_frames)


def MaskCube(data, x, y, r_mask, frames):
    cxy = (x//2, y//2)
    mask = disk(cxy, r_mask, shape=(x,y))
    data = np.reshape(data, (frames, x, y))
    data[:, mask[0], mask[1]] = 0
    data = np.reshape(data, (frames, x*y))

    return data

##apply centering and/or scaling about the spatial or temporal axis to the data
def Normalise(data, method, axis):
    def ApplyNorm(data, axis, get_mean, get_stdev):
        mean = np.nanmean(data,axis)
        stdev = np.nanstd(data,axis)
        zeros = np.where(stdev==0.0)[0]
        stdev[zeros] = 1.0 ##avoid dividing by 0 --> =1 to not scale constant features

        data_ax = np.moveaxis(data,axis,0)

        if get_mean == True:
            data_ax-= mean

        if get_stdev == True:
            data_ax/= stdev

        return data, mean, stdev

    if method =='mean':
        data, mean, stdev = (ApplyNorm(data, axis, get_mean=True, get_stdev=False))
    elif method =='standard':
        data, mean, stdev = (ApplyNorm(data, axis, get_mean=True, get_stdev=True))
    elif method =='none':
        mean, stdev = (0,1) ##still need to be set since returned by function
    else:
        raise ValueError("Unrecognised normalisation method, please choose from:"
                         "spat-mean, temp-mean, spat-standard, temp-standard, or none")
    return mean, stdev


##carries out PCA reduction on data for each no. PCs up to the specified MAX_PC
def SubtractPCs(data, ref_data, x_px, y_px, n_frames, axis, method, mean, stdev, pcs):

    eigen_vect = CalcEigen(ref_data, max(pcs))
    pca_cube = np.zeros((len(pcs), n_frames, x_px, y_px))
    for i, pc in enumerate(pcs):
        print("Processing PC:",pc)
        science_data = data.copy()

        eigen_vect_klip = eigen_vect[:pc].copy()
        pca_transf = np.dot(eigen_vect_klip, science_data.T)
        pca_recon = np.dot(pca_transf.T, eigen_vect_klip)

        final = science_data-pca_recon

        ##undo normalisation
        final_ax=  np.moveaxis(final,axis,0)
        if method =='standard':
            final_ax*= stdev

        pca_cube[i] = np.reshape(final, (n_frames,x_px,y_px))

    return pca_cube


def CalcEigen(data, max_pcs):
    cov = np.dot(data,data.T)

    eigen_val, eigen_vect = np.linalg.eig(cov) ##"column v[:,i] is the eigenvector
                                            ##corresponding to the eigenvalue w[i]"
    sort = np.argsort(-eigen_val)
    eigen_val = eigen_val[sort]

    pc = np.dot(eigen_vect.T, data)[sort] ##go from eigenvectors of A.A^T to transpose
                                        ##of eigenvectors of A^T.A (=(A^T.ev)^T)
    pc_vals = np.sqrt(abs(eigen_val)) ##'eigen values' for non square matrix
    for i in range(pc.shape[1]):
        pc[:,i] /= pc_vals ##renormalise values after dot product with data

    return pc[:max_pcs]


##formats input normalisation method, sends data to PCA and CADI functions,
##and saves final image
def RunPCA(cube, x, y, norm_method, r_mask, pcs, ref_cube=None):

    axes = {'spat':1, 'temp':0}
    if norm_method != 'none':
        axis, method = norm_method.split('-')
        axis = axes[axis]
    elif norm_method =='none':
        method = 'none'
        axis = 0
    else:
        raise ValueError("Unrecognised normalisation method, please choose from:"
                         "spat-mean, temp-mean, spat-standard, temp-standard, or none")


    science_cube = ScienceCube(axis, method, cube.copy(), r_mask)
    if ref_cube is not None:
        ref_lib = ScienceCube(axis, method, ref_cube.copy(), r_mask).data
    else:
        ref_lib = science_cube.data
    pca_cube = SubtractPCs(science_cube.data, ref_lib, science_cube.x_px,
                           science_cube.y_px, science_cube.n_frames, axis,
                           method, science_cube.mean, science_cube.stdev, pcs)

    return pca_cube