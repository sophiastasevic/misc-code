#This script will measure the position of a candidate using a 2D gaussian fit
#It will then give the user the option to do aperture photometry on this candidate

import numpy as np
import os
from photutils import aperture_photometry
from photutils import CircularAperture
from photutils import CircularAnnulus
from photutils.utils import calc_total_error
from photutils import centroid_2dg
from photutils.utils import calc_total_error
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")


#Loads data and defines as a numpy array
print('Hello!\n')
image_file = input('Please enter the name of the image (of the form .fits):')

# Sanity check to ensure the image inputted actually exists.
# If not, it will continue to loop until the correct name is inputted
while (not os.path.isfile(image_file)):
    print("Image '%s' not found" % image_file)
    image_file = input('Please enter the name of the image (of the form .fits):')

hdu_list = fits.open(image_file, ignore_missing_end=True)
image=hdu_list[0].data
print(image.shape)

print('Please enter the x and y bounds for a box surrounding the approximate candidate position\n')
xmin = int(input('Lower x value:'))
xmax = int(input('Upper x value:'))
ymin = int(input('Lower y value:'))
ymax = int(input('Upper y value:'))

#note y is first, x is second below
snapshot = image[ymin:ymax, xmin:xmax]

x, y = centroid_2dg(snapshot)
aa=x+xmin+1
bb=y+ymin+1
print('Position is:\n')
print((aa, bb))

#plt.figure()
plt.imshow(snapshot, cmap='gray')
plt.gca().invert_yaxis()
marker = '+'
ms, mew = 15, 2.
plt.plot(x,y,color='red',marker=marker, ms=ms, mew=mew)
plt.show()

#Start of aperture photometry
print('Would you like to do aperture photometry on this source?\n')
m = float(input('1 for yes, any other number for no:'))

if m == 1:
	a = float(input('Please enter the desired radius of the aperture:'))
	b = float(input('Please enter the desired inner radius of the annulus:'))
	c = float(input('Please enter the desired outer radius of the annulus:'))

	#defines positions of the aperture as well as the size of the aperture and annulus
	positions=[(aa-1,bb-1)]
	data = image
	aperture = CircularAperture(positions, r=a)
	annulus_aperture = CircularAnnulus(positions, r_in=b, r_out=c)

	error=1*data
	apers=[aperture,annulus_aperture]
	phot_table = aperture_photometry(data, apers, error=error, method='exact')
	for col in phot_table.colnames:
		phot_table[col].info.format = '%.8g'  # for consistent table output
	print(phot_table)

	annulus_masks=annulus_aperture.to_mask(method='center')
	annulus_data = annulus_masks[0].multiply(data)
	mask = annulus_masks[0].data
	annulus_data_1d = annulus_data[mask > 0]
	_, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
	background = median_sigclip * aperture.area

	final_sum = phot_table['aperture_sum_0'] - background
	phot_table['residual_aperture_sum'] = final_sum
	phot_table['total_err'] = phot_table['aperture_sum_err_0']*2
	phot_table['residual_aperture_sum'].info.format = '%.8g'  # for consistent table output
	print(phot_table['residual_aperture_sum','total_err'])


	p=a/100
	h = [None]*100
	aperture1 = [None]*100
	apers1 = [None]*100
	f = [None]*100
	g = [None]*100
	bkg = [None]*100

	for i in range(100):
		h[i]=p+p*i
		aperture1[i] = CircularAperture(positions, r=h[i])
		apers1[i]=[aperture1[i]]
		f[i] = aperture_photometry(data, apers1[i], method='exact')
		bkg[i] = median_sigclip * aperture1[i].area
		g[i] = f[i]['aperture_sum_0'] - bkg[i]


	#plots figure with apertures
	plt.imshow(data, cmap='gray')
	plt.gca().invert_yaxis()
	aperture.plot(color='red', lw=2)
	annulus_aperture.plot(color='green',lw=2)
	plt.xlim(aa-c-10,aa+c+10)
	plt.ylim(bb-c-10,bb+c+10)
	plt.show()


	#plots curve of growth
	plt.plot(h,g,'k-')
	plt.title('Curve of Growth for aperture')
	plt.xlabel('Radius (pix)')
	plt.ylabel('Counts')
	plt.show()
	print('End of aperture photometry, thank you!')

else:
	print('End of CPM position measurement, thank you!')