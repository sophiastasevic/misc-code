import matplotlib.pyplot as plt
import numpy as np
import os
import vip_hci as vip
import math
import pandas as pd
from math import cos
from math import sin
from astropy.io import fits
import matplotlib.patches as mpatches
from pylab import figure
import warnings

"""	
Buildcube initialises an empty array and loads the images into 
this array, removing the badpixels and also subtracting the flatfield.
"""	

def Buildcube(Number_images,name_input):
	
	file_names = ReadImageFilenames(name_input)
	Number_images = len(file_names)
	#Counts the length of the list.
	
	#Initialises two cubes to loop through.
	cube0 = np.zeros((Number_images, 1024, 1024))				
	cube = np.zeros((Number_images, 1024, 1024))				
	
	flatfield0 = './flat_Kp.fits'	#Loads in the flat-field.	
		
	flatfield = vip.fits.open_fits(
		flatfield0, n=0, header=False, 
		ignore_missing_end=True, verbose=True)
	#Opens the flat_field.
	
	Images_loaded =0	#Initialises a counter.
	
	#Loop opens the images and loads them into the data cube format required.
	#Loop also divides by the flatfield. 
	for i in range (0, Number_images):
	
		Images_loaded = Images_loaded + 1
			#Counter.
		print( "Processing {:d}/{:d}.".format(Images_loaded, Number_images) )
		
		cube0 = vip.fits.open_fits(
			file_names[i], n=0, header=False,
			ignore_missing_end=True, verbose=False)
			#Opens the fits file using VIP.
		
		for j in range(0, 1024):
			for k in range(0, 1024):
				cube[i][j][k] = cube0[j][k] / flatfield[j][k]
		
		if Images_loaded == Number_images:
			print( "\nImages loaded in.\n" )
	
	#Removes the 'bad pixels' in the image using a VIP function to do so.
	print( "Removing the badpixels in the Cube...\n" )
	
	cube = vip.preproc.badpixremoval.cube_fix_badpix_isolated(
		cube, bpm_mask=None, sigma_clip=3,
		num_neig=5, size=5, protect_mask=False,
		radius=30, verbose=True, debug=False)		
		
	#Writes the created cube to a fits file.	
	vip.fits.write_fits('Newcube.fits', cube,  verbose=True)	
	
	#Loads the angles and appends them to the cube, then overwrites the previous cube
	#with the appended one.
	   
	angles = np.loadtxt('{name}_angles.txt'.format(name=name_input))
	vip.fits.info_fits('Newcube.fits')
	
"""
Removeimage removes the bad images by erasing 
the name of the file from the list of filenames 
"""

def Removeimage(Number_images,name_input):

	#Loads in the filenames.
	file_names = ReadImageFilenames(name_input)
	user_input = input(
		"Type the file numbers (eg. 1,4,5.. ) of the images you would like to remove\n Seperated by single commas\n")
	
	Bad_images = [int(i) for i in user_input.split(',') if i.isdigit()]
	Bad_images = sorted(Bad_images, reverse =True)	#Sorts the numbers in descending order.
	
	#Loop removes the filenames corresponding to the numbers entered from the list of filenames.
	for i in range(0,len(Bad_images)):
		file_names = np.delete(file_names, (Bad_images[i]-1))
	
	return file_names

"""		
Checkfunction allows for yes no answers and prompts the 
user to re-enter if initially enetered incorrectly.

Yes = 0
No = 1

"""

def CenteringCube (name_input,psf_xy):

	while Centring_loop == 0:
		#Loads the filenames into python and counts the number of files.
		file_names = ReadImageFilenames(name_input)
		Number_images = len(file_names)
		
		
		"""
		------ 2 - CREATE DATA CUBE
		"""
		build_check = 0
		#Initialises the variable which is used to build the cube (or not).
		print( "Would you like to create the data cube?")
		build_check = Checkfunction()
	
		#Loop runs the Readangles and Buildcube functions, building the data cube.
		if build_check == 0:
			file_names = ReadImageFilenames(name_input)
			Number_images = len(file_names)
			Readangles(Number_images,name_input)
			Buildcube(Number_images,name_input)
	
		
		print( "Open DS9 and look through the created data cube at each frame, taking note of the bad frames" )
		remove_check = 0
		
		
		"""
		------ 3 - REMOVE BAD IMAGES
		"""
		print( "Would you like to remove any image?" )
		remove_check = Checkfunction()
	
		#Loop runs the Removeimage function and then overwrites the new shortened list to
		#the original text file. 
		if remove_check == 0:
			
			file_names = Removeimage(Number_images,name_input)
			file_names = file_names.tofile('{name}_filenames.txt'.format(name=name_input), sep ="\n", format='%s')
			file_names = ReadImageFilenames(name_input)
			Number_images = len(file_names)
			Readangles(Number_images,name_input)
			Buildcube(Number_images,name_input)
		
	
		"""
		------ 4 - RECENTERING
		"""
		#Loads the psf and the cube fits into python.
		initialpsf = './psf.fits'			
		cube = './Newcube.fits'
		
		
		print("Recentering Cube")
		print(" There are two ways to recenter the cube")
		print(" 1. Use a 2D Gaussian Fit=")
		print(" 2. Recenter manually. (Use if Gaussian fit doesn't work. Works if all images are already aligned but not in the center of the image)")
		
		while True:
			print(" Choose 1 (Gaussian fit) or 2 (manual fit): ")
			fit_method = input()
			if fit_method == '1' or fit_method == '2':
				break
			else:
				print("Option not recognised")
		
		print(" Fit method = ", fit_method)
		
		
		star_xy = [0.1, 0.1]
		
		print("Using DS9, open any image in Newcube.fits")
		star_xy[0] = input("Input the x coordinate of the central position of the star: ")
		star_xy[0] = int(star_xy[0])
			
		star_xy[1]=input("Input the x coordinate of the central position of the star: ")
		star_xy[1] = int(star_xy[1])
		
		
		#Opens the cube (using VIP) with cube_orig as HDU:0 and calls it cube_orig, 
		#and the parallactic angles from a text file.
		#Uses VIP to also open the point spread function previously loaded in.
			
		cube_orig = vip.fits.open_fits(cube)
		angs = np.loadtxt( '{name}_angles.txt'.format(name = name_input) )
		psf = vip.fits.open_fits(initialpsf, n=0, header=False, ignore_missing_end=True, verbose=True)
		
		cube1 = cube_orig
		
		# Gaussian Fit
		if fit_method == '1':
			print(" --2D Gaussian Fit")
			print( "Fitting a 2D gaussian to centre the images..." )
			#Uses VIP's 2D gaussian fitting algorithm to centre the cube.
			cube1, shy1, shx1, fwhm = Gaussian_2d_Fit(psf, cube_orig, star_xy)
		
		# Manual Fit
		elif fit_method == '2':
			print(" --Manual Fit")
			# Calculate shifts here
			image_centre = [512, 512]
			print(" Image centre is at", image_centre)
			shift_x = image_centre[0] - star_xy[0]
			shift_y = image_centre[1] - star_xy[1]
			cube1 = vip.preproc.recentering.cube_shift(cube_orig, shift_y, shift_x)
			fwhm = Calculate_fwhm(psf,psf_xy)
			
		
		#Writes the values of the centered cube into a fits file.
		vip.fits.write_fits('centeredcube_{name}.fits'.format(name=name_input), cube1, verbose=True)	
	
		cube = cube1
		#Loads up the centered cube.
		#Plots the original cube vs the new centered cube.
	
		im1 = vip.preproc.cosmetics.frame_crop(cube_orig[0], 1000, verbose=False)
		im2 = vip.preproc.cosmetics.frame_crop(cube[0], 1000, verbose=False)
	
		hciplot.plot_frames( 
			(im1, im2), 
			label = ('Original first frame', 'First frame after recentering'), 
			grid = True, 
			size_factor = 4
			)
		
		print( "Open DS9 and look through the centred data cube at each frame making sure it is centred.")
		print( "If you're not happy with it, redo centring" )
		print( "Redo Centring?" )
		Centring_loop = Checkfunction()

	return cube,fwhm, angs, psf
