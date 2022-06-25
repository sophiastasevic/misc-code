#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 18:49:53 2022

@author: stasevis

Mass-separation relationship for a planetary disk perturber using equation from
Augereau+ 2001
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

R_w = [60, 170, 200, 250, 300] ##warp radius in AU
t_unit = 5.2 ##time unit in y (Augereau+ 2001)
t = 17e6 ##system age in y
M_s = 2.1*1047.57 #star mass in M_Jup
D_s = 1/0.0076932

D = lambda M, R_w: 10*math.sqrt(((t_unit*M_s)/(t*M))*math.exp((0.2+math.log(R_w/10))/0.29))
M = lambda D, R_w: ((t_unit*M_s)/(t*(D/10)**2))*math.exp((0.2+math.log(R_w/10))/0.29)

r_warp = lambda M, D: 10*math.exp(0.29*math.log((M/M_s)*pow((D/10),2)*(t/t_unit))-0.2)

separations = np.arange(0.01,1.6,0.02)*D_s ##planet separation in AU
masses = np.zeros((len(R_w),len(separations))) ##planet mass in M_Jup

for i,r in enumerate(R_w):
    for j,d in enumerate(separations):
        masses[i,j] = M(d,r)

sensitivity_data = pd.read_csv('sensitivity_IFS_2015-04-03.csv')
#sensitivity_data = sensitivity_data.drop(sensitivity_data.columns[0], axis=1)

#%%
fig,ax=plt.subplots(1,1, figsize=(20,12))

colors = iter(cm.rainbow(np.linspace(0, 1, len(R_w))))

#ax.set_title('HD 110058 planetary perturber mass-separation relation', fontsize=30, y=1.05)
ax.set_ylabel('Planet mass [M$_{Jup}$]', fontsize=28)
ax.set_xlabel('Separation [arcsec]', fontsize=28)

for i,r in enumerate(R_w):
    c=next(colors)
    ax.plot(separations/D_s, masses[i], color=c, lw=3, label = 'R$_W$={0:d} AU'.format(r))
ax.plot(sensitivity_data['separation [arcsec]'],sensitivity_data['sensitivity [MJup]'], lw=3, color='black', label = 'IFS 2015-04-03 sensitivity')
ax.plot((0.0925,0.0925),(-0.09,100),ls='--', lw=2, color='black', label='mask')
ax.legend(frameon=False, loc='upper right', fontsize=28)
ax.tick_params(labelsize=24)
ax.set_xlim(0,1.5)
ax.set_ylim(-0.09,10.09)
ax.grid(True)

path_fig = 'HD110058_perturber_mass_separation.png'
plt.savefig(path_fig)
plt.show()