import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.gridspec as gridspec


def NormOffset(horizontal_separation,algorithm_master,gauss_fit,pcs,norms,convolved):

	colors = iter(cm.rainbow(np.linspace(0, 1, len(norms)+1)))

	plt.close(1)
	fig = plt.figure(1, figsize=(21,29.7))
	gs = gridspec.GridSpec(5,1, height_ratios=[1,1,1,1,1], width_ratios=[1])
	gs.update(left=0.1, right=0.95, bottom=0.1, top=0.96, wspace=0.2, hspace=0.1)
	ax1 = plt.subplot(gs[0,0]) # Area for the first plot
	ax2 = plt.subplot(gs[1,0]) # Area for the second plot
	ax3 = plt.subplot(gs[2,0]) # Area for the colorbar
	ax4 = plt.subplot(gs[3,0])
	ax5 = plt.subplot(gs[4,0])

	fig.suptitle(algorithm_master + 'VIP PCA NW-SE offset', fontsize=20, fontweight='bold')

	for i,n in enumerate(norms):
	    c=next(colors)
	    # pc1
	    ax1.plot(horizontal_separation,gauss_fit[i][0,1,:]-gauss_fit[i][0,0,:],color=c,label=norms[i])
	    #ax1.plot(-horizontal_separation[::-1],gauss_fit[i][0,0,::-1],color=c)
	    ax1.set_title('PC {p}'.format(p=pcs[0]),fontsize=18,loc='left')
	    ax1.set_ylim(-1.5,2.5)
	    ax1.set_ylabel('centroid in px', fontsize=18)
	    ax1.legend(frameon=False,loc='upper right', fontsize=18)

	    # pc2
	    ax2.plot(horizontal_separation,gauss_fit[i][1,1,:]-gauss_fit[i][1,0,:],color=c,label=norms[i])
	    #ax2.plot(-horizontal_separation[::-1],gauss_fit[i][1,0,::-1],color=c)
	    ax2.set_title('PC {p}'.format(p=pcs[1]),fontsize=18,loc='left')
	    ax2.set_ylim(-1.5,2.5)
	    ax2.set_ylabel('centroid in px', fontsize=18)
	    ax2.legend(frameon=False,loc='upper right', fontsize=18)

	    # pc3
	    ax3.plot(horizontal_separation,gauss_fit[i][2,1,:]-gauss_fit[i][1,0,:],color=c,label=norms[i])
	    #ax3.plot(-horizontal_separation[::-1],gauss_fit[i][2,0,::-1],color=c)
	    ax3.set_title('PC {p}'.format(p=pcs[2]),fontsize=18,loc='left')
	    ax3.set_ylim(-1.5,2.5)
	    ax3.set_ylabel('centroid in px', fontsize=18)
	    ax3.legend(frameon=False,loc='upper right', fontsize=18)

	    # pc5
	    ax4.plot(horizontal_separation,gauss_fit[i][3,1,:]-gauss_fit[i][1,0,:],color=c,label=norms[i])
	    #ax4.plot(-horizontal_separation[::-1],gauss_fit[i][3,0,::-1],color=c)
	    ax4.set_title('PC {p}'.format(p=pcs[3]),fontsize=18,loc='left')
	    ax4.set_ylim(-1.5,2.5)
	    ax4.set_ylabel('centroid in px', fontsize=18)
	    ax4.legend(frameon=False,loc='upper right', fontsize=18)

	    # pc9
	    ax5.plot(horizontal_separation,gauss_fit[i][4,1,:]-gauss_fit[i][1,0,:],color=c,label=norms[i])
	    #ax5.plot(-horizontal_separation[::-1],gauss_fit[i][4,0,::-1],color=c)
	    ax5.set_title('PC {p}'.format(p=pcs[4]),fontsize=18,loc='left')
	    ax5.set_ylim(-1.5,2.5)
	    ax5.set_ylabel('centroid in px', fontsize=18)
	    ax5.legend(frameon=False,loc='upper right', fontsize=18)
	    ax5.set_xlabel('Separation in px', fontsize=18)

	c=next(colors)
	# model
	ax1.plot(horizontal_separation,convolved[1,:]-convolved[0,:],color=c,label='convolved disk',ls='--',lw=2,zorder=-1,alpha=0.8)
	#ax1.plot(-horizontal_separation[::-1],convolved[0,::-1],color=c,ls='--',lw=2,zorder=-1,alpha=0.8)
	ax1.set_title('PC {p}'.format(p=pcs[0]),fontsize=18,loc='left')
	ax1.set_ylim(-1.5,2.5)
	ax1.set_ylabel('centroid in px', fontsize=18)
	ax1.legend(frameon=False,loc='upper right', fontsize=18)

	# temp-mean
	ax2.plot(horizontal_separation,convolved[1,:]-convolved[0,:],color=c,label='convolved disk',ls='--',lw=2,zorder=-1,alpha=0.8)
	#ax2.plot(-horizontal_separation[::-1],convolved[0,::-1],color=c,ls='--',lw=2,zorder=-1,alpha=0.8)
	ax2.set_title('PC {p}'.format(p=pcs[1]),fontsize=18,loc='left')
	ax2.set_ylim(-1.5,2.5)
	ax2.set_ylabel('centroid in px', fontsize=18)
	ax2.legend(frameon=False,loc='upper right', fontsize=18)

	# spat-standard
	ax3.plot(horizontal_separation,convolved[1,:]-convolved[0,:],color=c,label='convolved disk',ls='--',lw=2,zorder=-1,alpha=0.8)
	#ax3.plot(-horizontal_separation[::-1],convolved[0,::-1],color=c,ls='--',lw=2,zorder=-1,alpha=0.8)
	ax3.set_title('PC {p}'.format(p=pcs[2]),fontsize=18,loc='left')
	ax3.set_ylim(-1.5,2.5)
	ax3.set_ylabel('centroid in px', fontsize=18)
	ax3.legend(frameon=False,loc='upper right', fontsize=18)

	# temp-standard
	ax4.plot(horizontal_separation,convolved[1,:]-convolved[0,:],color=c,label='convolved disk',ls='--',lw=2,zorder=-1,alpha=0.8)
	#ax4.plot(-horizontal_separation[::-1],convolved[0,::-1],color=c,ls='--',lw=2,zorder=-1,alpha=0.8)
	ax4.set_title('PC {p}'.format(p=pcs[3]),fontsize=18,loc='left')
	ax4.set_ylim(-1.5,2.5)
	ax4.set_ylabel('centroid in px', fontsize=18)
	ax4.legend(frameon=False,loc='upper right', fontsize=18)

	# none
	ax5.plot(horizontal_separation,convolved[1,:]-convolved[0,:],color=c,label='convolved disk',ls='--',lw=2,zorder=-1,alpha=0.8)
	#ax5.plot(-horizontal_separation[::-1],convolved[0,::-1],color=c,ls='--',lw=2,zorder=-1,alpha=0.8)
	ax5.set_title('PC {p}'.format(p=pcs[4]),fontsize=18,loc='left')
	ax5.set_ylim(-1.5,2.5)
	ax5.set_ylabel('centroid in px', fontsize=18)
	ax5.legend(frameon=False,loc='upper right', fontsize=18)
	ax5.set_xlabel('Separation in px', fontsize=18)

	for ax in [ax1,ax2,ax3,ax4,ax5]:
	    ax.grid(True)
	plt.savefig('bright_disk_vip_pca_spine_side_offset.pdf') ##combination of reductions

#%%

def CalcOffset(gauss_fit,nb_norms,nb_pcs,nb_vertical_profiles):
	side_offset=np.zeros((nb_norms,nb_pcs,nb_vertical_profiles))
	for n in range(nb_norms): #norm
	    for i in range(nb_pcs): #pc
	        side_offset[n,i,:]=gauss_fit[i][n,1,:]-gauss_fit[i][n,0,:]
	np.nanmean(side_offset,axis=2)

"""
mean NW-SE offset

model: -0.09370438350123805 +/- 0.09031880562325986

VIP
    spat-mean      temp-mean   spat-standard temp-standard  none
pc1:([0.18921974, 0.04133734, 0.07690169, 0.016622  , 0.20233401])
pc2:([-0.06726172, -0.09833752,  0.09699041,  0.31814338, -0.03353863])
pc3:([-0.12254113, -0.22787337, -0.24130962,  0.10060159, -0.14818815])
pc5:([-0.04644593, -0.12831287,  0.3607819 , -0.46130842, -0.03652695])
pc9:([-0.09427566, -0.08690346, -0.58845081,  0.02194334, -0.13369087])

NW-SE offset stdev
    spat-mean      temp-mean   spat-standard temp-standard  none
pc1:([0.17604833, 0.15284743, 0.18798694, 0.19419103, 0.21368421])
pc2:([0.25440771, 0.16137494, 0.20614575, 0.28954648, 0.33213086])
pc3:([0.17434877, 0.1675769 , 0.25285956, 0.20245298, 0.1425512 ])
pc5:([0.21486424, 0.18513572, 0.31630883, 0.33700716, 0.21311861])
pc9:([0.20822947, 0.24163328, 0.25394192, 0.27870118, 0.23213704])

ssPCA
    ([[ 0.23434098,  0.11718534,  0.27715491,  0.03205266, -0.05701879],
       [ 0.02315644,  0.0903234 , -0.12822381, -0.12444929, -0.28299682],
       [-0.04533757, -0.05212031, -0.27729267, -0.14219922, -0.26659557],
       [-0.07646155,  0.00380617, -0.51516336, -0.53755078, -0.65552537],
       [-0.01607782, -0.11244138, -0.73452605, -0.31849304, -0.42637489]])

stdev
    ([[0.12918438, 0.16197974, 0.25335965, 0.18243792, 0.17886307],
       [0.25678352, 0.20690537, 0.19032399, 0.18553641, 0.19661527],
       [0.15840459, 0.16845626, 0.16121139, 0.19945474, 0.2867786 ],
       [0.19431969, 0.21519053, 0.30509899, 0.40488635, 0.43011904],
       [0.18520157, 0.25760149, 0.2934193 , 0.3796446 , 0.40606169]])
"""
#%%

def FormatOffset(pcs,gauss_fit,convolved):
	algorithms_description=[]
	for i in range(len(pcs)):
	    algorithms_description.append('PC {p}'.format(p=pcs[i]))
	x0=gauss_fit['x0']
	x0=np.array(x0)
	x0_con=convolved
	norms=['spat-mean','temp-mean','spat-standard','temp-standard','none']
	pcs=np.array(pcs)
	x0_offset=x0[:,:,1,:]-x0[:,:,0,:]
	x0_offset_mean=np.nanmean(x0_offset,axis=2)
	x0_offset_std=np.nanstd(x0_offset,axis=2)
	con_offset=x0_con[0,1,:]-x0_con[0,0,:]
	con_offset_mean=np.nanmean(con_offset)
	con_offset_std=np.nanstd(con_offset)
	y=np.zeros_like(pcs)+con_offset_mean
	x=np.arange(1.,11.)
	x[0]-=.25
	x[-1]+=.25
	y1=y+con_offset_std
	y2=y-con_offset_std

	#x0_offset_mean=np.nanmean(x0_offset[:,:,1:14],axis=2)
	#x0_offset_std=np.nanstd(x0_offset[:,:,1:14],axis=2)
	#con_offset_mean=np.nanmean(con_offset[1:14])
	#con_offset_std=np.nanstd(con_offset[1:14])

	return norms,pcs,algorithms_description,y1,y2,x,x0_offset_mean,x0_offset_std

def MeanNormOffset(gauss_fit,pcs,convolved):
	norms=['spat-mean','temp-mean','spat-standard','temp-standard','none']

	algorithms_description=[]
	for i in range(len(pcs)):
	    algorithms_description.append('PC {p}'.format(p=pcs[i]))
	x0=gauss_fit['x0']
	x0=np.array(x0)
	x0_model=np.array(convolved)

	x0_offset=x0[:,0,:]-x0[:,1,:]
	x0_offset_mean=np.nanmean(x0_offset,axis=1)
	x0_offset_std=np.nanstd(x0_offset,axis=1)

	model_offset=x0_model[0,0,:]-x0_model[0,1,:]
	model_offset_mean=np.nanmean(model_offset)
	model_offset_std=np.nanstd(model_offset)

	pcs=np.array(pcs)

	plt.close(1)
	colors = iter(cm.rainbow(np.linspace(0, 1, len(norms)+1)))
	#fig, axes = plt.subplots(2,1,figsize=(10,14))
	#ax=axes[0]
	fig, ax = plt.subplots(1,1,figsize=(11,6))
	ax.set_title('Bright disk PCA reduction',fontsize=12,loc='left')
	for i in range(len(norms)):
		c=next(colors)
	    #ax.errorbar(pcs+(0.2-i/10),x0_offset_mean[i],x0_offset_std[i],color=c,marker='o',markersize=5,ls='',capsize=3,label=norms[i])
		ax.plot(pcs,x0_offset_mean[i],color=c,marker='o',markersize=5,label=norms[i])
		ax.legend(frameon=False,loc='upper right', fontsize=12)
	ax.set_xlabel('PCs', fontsize=12)
	ax.set_ylabel('mean NW-SE centroid offset [px]', fontsize=12)
	c=next(colors)
	#x=[0.7,29.3]
	x=[1.,30.]
	y=[model_offset_mean,model_offset_mean]
	y1=[model_offset_mean+model_offset_std,model_offset_mean+model_offset_std]
	y2=[model_offset_mean-model_offset_std,model_offset_mean-model_offset_std]
	ax.fill_between(x,y1,y2,color=c,alpha=0.2,zorder=0)
	ax.plot(x,y1,color=c,alpha=0.3,zorder=0)
	ax.plot(x,y2,color=c,alpha=0.3,zorder=0)
	ax.plot(x,y,color=c,alpha=0.8,ls='--',zorder=0,label='convolved disk')
	#ax.set_ylim(-1.2,0.7)
	ax.legend(frameon=False,loc='upper left', fontsize=12)
	#ax.set_xticks(np.arange(1,30))
	ax.grid(True)

	fig.show()
	plt.savefig('bright_disk_vip_pca_NW-SE_spine_offset_pc30-no_err.png')

	"""
	plt.close(1)
	colors = iter(cm.rainbow(np.linspace(0, 1, len(norms)+1)))
	ax=axes[1]

	ax.set_title('Bright disk VIP PCA reduction',fontsize=12,loc='left')
	for i in range(len(norms)):
	    c=next(colors)
	    ax.errorbar(pcs+(0.2-i/10),x0_offset_mean[:,i],x0_offset_std[:,i],color=c,marker='o',ls='',capsize=4,label=norms[i])
	    ax.legend(frameon=False,loc='upper right', fontsize=12)
	ax.set_xlabel('PCs', fontsize=12)
	ax.set_ylabel('mean NW-SE centroid offset [px]', fontsize=12)
	c=next(colors)
	ax.fill_between(x,y1,y2,color=c,alpha=0.2,zorder=0)
	ax.plot(x,y1,color=c,alpha=0.3,zorder=0)
	ax.plot(x,y2,color=c,alpha=0.3,zorder=0)
	ax.plot(x,y,color=c,alpha=0.8,ls='--',zorder=0,label='convolved disk')
	ax.set_ylim(-1.2,0.7)
	ax.legend(frameon=False,loc='lower left', fontsize=12)
	ax.set_xticks(np.arange(1,10))
	ax.grid(True)
	"""

#%%

def PlotSpine(nb_algorithms,horizontal_separation,x0,algorithms_description):
	fig, ax = plt.subplots(figsize=(10,7))
	ax.set_title('Bright fake disk spine spat-mean VIP PCA',fontsize=12,loc='left')
	colors = iter(cm.rainbow(np.linspace(0, 1, nb_algorithms)))
	for i in range(nb_algorithms):
	    c=next(colors)

	    ax.plot(horizontal_separation, x0[0,i,1,:], color=c,label=algorithms_description[i])
	    ax.plot(-horizontal_separation[::-1], x0[0,i,0,::-1], color=c)

	    ax.set_ylim(-0.5,3.5)
	    ax.set_ylabel('centroid [px]', fontsize=12)
	    ax.set_xlabel('separation [px]', fontsize=12)
	    ax.legend(frameon=False,loc='upper right', fontsize=12)
	ax.grid(True)
	fig.show()


"""
#%%
colors = iter(cm.rainbow(np.linspace(0, 1, nb_algorithms)))

plt.close(1)
fig = plt.figure(1, figsize=(21,10))
gs = gridspec.GridSpec(1,1, height_ratios=[1], width_ratios=[1])
gs.update(left=0.1, right=0.95, bottom=0.1, top=0.93, wspace=0.2, hspace=0.1)
ax = plt.subplot(gs[0,0])
for filei,f in enumerate(files):
    #ax = plt.subplot(gs[filei,0])
    c=next(colors)
    ax.plot(horizontal_separation,gauss_fit['x0'][filei,1,:]-gauss_fit['x0'][filei,0,:],color=c,label=algorithms_description[filei])
    ax.set_ylim(-0.7,0.5)
    ax.set_ylabel('centroid in px', fontsize=18)
    ax.legend(frameon=False,loc='upper right', fontsize=18)
ax.grid(True)
ax.set_xlabel('Separation in px', fontsize=18)
plt.savefig(os.path.join(data_path,'disk_fit','final',save_name+'_offset.png')) ##combination of reductions


data_path='/mnt/c/Users/stasevis/Documents/disk-analysis/PA_calculations'
save_name = file_prefix + '_convolution_test'
algorithm_master='Convolution offset test'
algorithms_description=['unconvolved', 'convolved + no rotation', 'convolved']
pa =35.

nb_algorithms = len(algorithms_description)
images = np.ndarray((nb_algorithms,size,size))

files=[]
path='/mnt/c/Users/stasevis/Documents/sphere_data/HD110058/HD110058_H23_2015-04-12_fake_disk_injection/fake_disk_bright_unconvolved.fits'
files.append(path)
unconvolved_disk= fits.getdata(path)

path='/mnt/c/Users/stasevis/Documents/sphere_data/HD110058/HD110058_H23_2015-04-12_fake_disk_injection/fake_disk_bright_convolved.fits'
files.append(path.replace('bright','bright_no_rotation'))
files.append(path)
convolved_disk= fits.getdata(path)
convolved_disk=convolved_disk[0]

path="/mnt/c/Users/stasevis/Documents/sphere_data/HD110058/HD110058_H23_2015-04-12_fake_disk_injection/ird_convert_recenter_dc2021-SCIENCE_PSF_MASTER_CUBE-median_unsat.fits"
psf= fits.getdata(path)
psf=psf[0]
psf = psf/np.sum(psf)
from scipy import signal
convolved_frame=signal.fftconvolve(unconvolved_disk, psf, mode='same')

size_tmp = convolved.shape[-1]
if size+size_tmp%2:
    size_tmp+1
size_min,size_max=((size_tmp-size)//2, (size_tmp+size)//2)

images[0,:,:] = unconvolved_disk[size_min:size_max,size_min:size_max]
images[2,:,:] = convolved_disk[size_min:size_max,size_min:size_max]
images[1,:,:] = convolved_frame[size_min:size_max,size_min:size_max]

aligned_images = vip.preproc.cube_derotate(images,np.ones((nb_algorithms))*(pa),imlib='opencv')
aligned_noise = np.zeros_like(aligned_images)

#%%

data_path='/mnt/c/Users/stasevis/Documents/disk-analysis/PA_calculations'
save_name = file_prefix + '_initial_PA'
algorithm_master='Initial PA measurement test'
pa =[33.,34.,35.,36.,37.]
algorithms_description=Descriptions(pa=pa)

nb_algorithms = len(algorithms_description)
images = np.ndarray((nb_algorithms,size,size))

files=[]
path='/mnt/c/Users/stasevis/Documents/sphere_data/HD110058/HD110058_H23_2015-04-12_fake_disk_injection/fake_disk_bright_unconvolved.fits'
for i in range(nb_algorithms):
    files.append(path)
    image_tmp=fits.getdata(path)

    size_tmp = image_tmp.shape[-1]
    if size+size_tmp%2:
        size_tmp+1
    size_min,size_max=((size_tmp-size)//2, (size_tmp+size)//2)

    images[i,:,:] = image_tmp[size_min:size_max,size_min:size_max]

aligned_images = vip.preproc.cube_derotate(images,np.ones((nb_algorithms))*(pa),imlib='opencv')
aligned_noise = np.zeros_like(aligned_images)

"""