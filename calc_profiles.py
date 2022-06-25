#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: stasevis

split disk images into series of vertical profiles + fit a gaussian to each profile
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.functional_models import Gaussian1D #,Lorentz1D,Moffat1D
import mpfit

def Separation(horizontal_binning_factor, vertical_profile_length,
                inner_radius, outer_radius, bright_radius):
    ##TODO: could define different lengths depending on separation to avoid inc. speckles close in + cutting off warp far out
    hsep_inner = np.arange(inner_radius,bright_radius,horizontal_binning_factor)
    hsep_outer = np.arange(hsep_inner[-1]+horizontal_binning_factor+2,outer_radius,horizontal_binning_factor+2)

    horizontal_separation = np.concatenate((hsep_inner,hsep_outer))
    #horizontal_separation = np.arange(inner_radius,outer_radius,horizontal_binning_factor)
    horizontal_binning_factor = np.concatenate((np.ones(len(hsep_inner),dtype=int)*horizontal_binning_factor,
                                                np.ones(len(hsep_outer),dtype=int)*(horizontal_binning_factor+2)))

    vertical_displacement = np.arange(-vertical_profile_length//2,vertical_profile_length//2)
    
    return horizontal_separation, horizontal_binning_factor, vertical_displacement


def Profiles(images, noise, nb_algorithms, horizontal_binning_factor, horizontal_separation, vertical_profile_length, sides):
    if horizontal_binning_factor % 2: ##allow for even or odd binning factors
        buffer=1
    else:
        buffer=0
        
    nb_vertical_profiles= len(horizontal_separation)
    profiles = np.ndarray((nb_algorithms, 2, vertical_profile_length, nb_vertical_profiles)) ##axis 1 is for the sides (0:SE, 1:NW)
    noise_profiles = np.zeros_like(profiles)
    
    noise_check=np.zeros(nb_algorithms)
    for i in range(nb_algorithms):
        noise_check[i]=noise[i].std()
    non=np.where(noise_check==0)[0] ##need to take noise from aligned image   
    noise_range = vertical_profile_length//2 ##no. px to use from top and bottom of the vertical profile for noise estimation
    
    size=images.shape[-1]
    for i, sep in enumerate(horizontal_separation):
        lim = lambda lbin: size//2 + lbin//2
        vmin = size//2-vertical_profile_length//2
        vmax = size//2+vertical_profile_length//2
        hmin = {0: size//2-sep-horizontal_binning_factor[i]//2, 
                1: size//2+sep-horizontal_binning_factor[i]//2}
        hmax = {0: size//2-sep+horizontal_binning_factor[i]//2 + buffer, 
                1: size//2+sep+horizontal_binning_factor[i]//2 + buffer}
        
        for sidei in range(len(sides)):
            profiles[:,sidei,:,i] = np.mean(images[:,vmin:vmax, hmin[sidei]:hmax[sidei]],axis=2)
            
            if len(non)-nb_algorithms!=0:
                algo = [x not in non for x in range(0, nb_algorithms)]
                noise_profiles[algo,sidei,:,i] = np.mean(noise[algo, vmin:vmax, hmin[sidei]:hmax[sidei]],axis=2)
                
            if len(non)!=0:
                noise_upper = images[non, vmax:vmax+noise_range, hmin[sidei]:hmax[sidei]]
                noise_lower = images[non, vmin-noise_range:vmin, hmin[sidei]:hmax[sidei]]
                algo = np.concatenate((noise_upper, noise_lower),axis=1)
                noise_profiles[non,sidei,:,i] = np.mean(algo, axis=2)

    return profiles, noise_profiles



def gauss1D_fit_erf(p,fjac=None, x=None,y=None,err=None):
    '''
    Computes the residuals to be minimized by mpfit, given a model and data.
    '''
    model = Gaussian1D(p[0],p[1],p[2])(x) ##TODO: perhaps add asymmetry to gaussian?
    status = 0
    return ([status, ((y-model)/err).ravel()])



def GaussDict(nb_algorithms, nb_vertical_profiles):

    gauss_fit = {'A':np.ones((nb_algorithms,2,nb_vertical_profiles))*np.nan,
                 'x0':np.ones((nb_algorithms,2,nb_vertical_profiles))*np.nan,
                 'sigma':np.ones((nb_algorithms,2,nb_vertical_profiles))*np.nan}

    gauss_error = {'A':np.ones((nb_algorithms,2,nb_vertical_profiles))*np.nan,
                   'x0':np.ones((nb_algorithms,2,nb_vertical_profiles))*np.nan,
                   'sigma':np.ones((nb_algorithms,2,nb_vertical_profiles))*np.nan}

    return gauss_fit, gauss_error



def FitProfiles(profiles, noise_profiles, nb_vertical_profiles, horizontal_separation,
                vertical_displacement, nb_algorithms, files, pcs, pci, sides,
                px_size, dstar, data_path):

    gauss_fit, gauss_error = GaussDict(nb_algorithms, nb_vertical_profiles)

    for filei,f in enumerate(files):
        print('Processing {0:s}'.format(str(f)))
        for sepi,sep in enumerate(horizontal_separation):
            print('Separation of {0:d}px'.format(sep))
            for sidei,side in enumerate(sides):
                prof = profiles[filei,sidei,:,sepi]
                noise_array = noise_profiles[filei,sidei,:,sepi]

                max_prof = np.max(prof)
                maxima=np.where(prof==max_prof)[0]
                minima=[]
                for i in range(1,len(prof)-1):
                    x=prof[i]
                    if x< prof[i-1] and x< prof[i+1] and x< max(prof[i-1],prof[i+1])/1.2: ##buffer, may need changing
                        minima.append(i)

                try:
                    upper=minima[min(np.where(minima>maxima)[0])]
                except ValueError:
                    upper=len(prof)-1
                try:
                    lower=minima[max(np.where(minima<maxima)[0])]
                except ValueError:
                    lower=0

                while prof[lower]<0 and lower<maxima:
                    lower+=1
                while prof[upper]<0 and upper>maxima:
                    upper-=1
                ran=np.arange(lower,upper+1)

                #if len(ran)<3:
                #    prof_4fit = prof[prof>max_prof/1.5] ##cutting out outer noise, might need to adjust
                #    vert_disp4fit = vertical_displacement[prof>np.max(prof)/1.5] ##change dividing no. here as above

                #else:
                prof_4fit = prof[ran]
                vert_disp4fit = vertical_displacement[ran]

                fa = {'x': vert_disp4fit, 'y':prof_4fit , 'err':np.std(noise_array)*np.ones_like(prof_4fit)}
                parinfo = [{'fixed':0, 'limited':[1,1], 'limits':[0.,2*max_prof]}, # Force the amplitude to be >0 and smaller than2*max
                           {'fixed':0, 'limited':[1,1], 'limits':[np.min(vert_disp4fit),np.max(vert_disp4fit)]}, # vertical offset of Gaussian
                           {'fixed':0, 'limited':[1,1], 'limits':[2,10.]}] # Force the sigma to be > 2 and <10
                p0 = [max_prof,vert_disp4fit[np.argmax(prof_4fit)],3]
                m = mpfit.mpfit(gauss1D_fit_erf, p0, functkw=fa, parinfo=parinfo, quiet=1) # Set quiet=0 for debugging
                if m.status==1:
                    gauss_fit['A'][filei,sidei,sepi]=m.params[0]
                    gauss_fit['x0'][filei,sidei,sepi]=m.params[1]
                    gauss_fit['sigma'][filei,sidei,sepi]=m.params[2]
                    gauss_error['A'][filei,sidei,sepi]=m.perror[0]
                    gauss_error['x0'][filei,sidei,sepi]=m.perror[1]
                    gauss_error['sigma'][filei,sidei,sepi]=m.perror[2]
                    plt.close(1)
                    plt.figure(1, figsize=(6.93,7))
                    plt.clf()
                    plt.plot(vertical_displacement,prof,label='profile')
                    plt.plot(vert_disp4fit,prof_4fit,'rx',label='fitted profile')
                    plt.plot(vertical_displacement,Gaussian1D(m.params[0],m.params[1],m.params[2])(vertical_displacement),label='fit $x_0=${0:.1f} $\sigma$={1:.1f}'.format(m.params[1],m.params[2]))
                    plt.xlabel('Separation in pixel')
                    plt.ylabel('Intensity in ADU')
                    plt.title('Vertical {0:s} profile at {1:d}px={2:.0f}mas={3:.0f}au'.format(side,sep,sep*px_size,sep*px_size/1000*dstar))
                    plt.legend()
                    #plt.show()
                    path_fig=f.replace(data_path, data_path+'/disk_profile')
                    #path_fig=path_fig.replace('pc50','pc{p}'.format(p=pcs[pci(filei)]))

                    if 'fake_disk' in path_fig:
                        path_fig=path_fig.replace('/fake_disk/','/')

                    plt.savefig(path_fig.replace('.fits','_pc_{0:d}_profile_{1:s}_sep{2:d}px.png'.format(pcs[pci(filei)],side,sep)))

                else:
                    print('Fit failed')

    return gauss_fit, gauss_error