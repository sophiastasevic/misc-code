#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 17:45:54 2022

@author: stasevis
"""

import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

root_path = '/mnt/c/Users/stasevis/Documents/RDI/ADI/output/HD110058'
path=os.path.join(root_path,'all_data_VIP-PCA_temp-mean_measured_PA.fits')

with fits.open(path) as hdul:
    pa_table=hdul[1].data

pc_loc = np.zeros((10,7),dtype=int)
for i in range(10):
    pc_loc[i]=np.where(pa_table['Description']=='PC {0:d}'.format(i+1))[0]

algo_descriptions=[]
for i in range(7):
    algo_descriptions.append(pa_table['Epoch'][i*10]+' '+pa_table['Band'][i*10])

pa=np.zeros(pc_loc.shape)
pa_err=np.zeros(pc_loc.shape)
for i in range(10):
    pa[i]=pa_table['PA'][pc_loc[i]]
    pa_err[i]=pa_table['err'][pc_loc[i]]
pcs=np.arange(1,11)

pa_bb_mean=np.mean(pa,axis=0)
pa_pc_mean=np.mean(pa,axis=1)

pa_bb_err=np.sqrt(np.sum(np.square(pa_err),axis=0))
pa_pc_err=np.sqrt(np.sum(np.square(pa_err),axis=1))
#%%
import itertools
def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

colors = iter(cm.rainbow(np.linspace(0, 1, 7)))
fig, ax = plt.subplots(1,1,figsize=(12,9))

ax.set_title('HD110058 temp-mean VIP-PCA disk PA measured between 224-336 mas', fontsize=16, y=1.1)
ax.set_ylabel('PA [deg]', fontsize=16)
ax.set_xlabel('PCs', fontsize=16)

for i,algo in enumerate(algo_descriptions):
    c=next(colors)
    ax.errorbar(pcs, pa[:,i], pa_err[:,i], color=c, marker='o', capsize=4, label=algo)

ax.set_xticks(np.arange(1,11))
ax.tick_params(labelsize=14)
box = ax.get_position()
ax.set_position([box.x0, box.y0 - box.height * 0.03, box.width, box.height * 0.97])
handles, labels = ax.get_legend_handles_labels()
ax.legend(flip(handles, 4), flip(labels, 4), frameon=True, loc='lower center', bbox_to_anchor=(0.5,0.97), borderaxespad=0, ncol=4, fontsize=14)
#ax.legend(frameon=True, loc='lower center', bbox_to_anchor=(0.5,0.97), borderaxespad=0, ncol=4, fontsize=14)

ax.grid(True)

plt.savefig(path.replace('.fits','_comp.png'))

#%%
fig, ax = plt.subplots(1,1,figsize=(12,5))

ax.set_title('HD110058 temp-mean VIP-PCA PC mean disk PA measured between 224-336 mas', fontsize=16)
ax.set_ylabel('PA [deg]', fontsize=16)
#ax.set_xlabel('data', fontsize=16)

params=np.arange(1,8)
ax.errorbar(params, pa_bb_mean, pa_bb_err, color='red', marker='o', ls='', capsize=4)

ax.set_xticks(params)
ax.set_xticklabels(algo_descriptions,rotation=15)
ax.tick_params(labelsize=14)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.05, box.width, box.height * 0.95])

ax.grid(True)

plt.savefig(path.replace('.fits','_pc_mean_comp.png'))

#%%
fig, ax = plt.subplots(1,1,figsize=(12,5))

ax.set_title('HD110058 temp-mean VIP-PCA broadband mean disk PA measured between 224-336 mas', fontsize=16)
ax.set_ylabel('PA [deg]', fontsize=16)
ax.set_xlabel('PCs', fontsize=16)

params=np.arange(1,11)
ax.errorbar(params, pa_pc_mean, pa_pc_err, color='blue', marker='o', ls='', capsize=4)

ax.set_xticks(params)
#ax.set_xticklabels(algo_descriptions,rotation=45)
ax.tick_params(labelsize=14)

ax.grid(True)

plt.savefig(path.replace('.fits','_bb_mean_comp.png'))