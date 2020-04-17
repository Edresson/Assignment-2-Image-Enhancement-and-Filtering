# Image Processing
# Author : Gabriel Cavallari 
# Asignment 1: Intensity Transformations
#
import numpy as np
import imageio

def inversion(img):
"""Function to perform inversion of intensity values"""    
    return 255-img

def contrast_modulation(img, c, d):
"""Function to perform contrast modulation
Arguments:
	img -- input image
	c -- target minimum intensity
	d -- target maximum intensity
"""    
    modified_img = np.zeros(img.shape)
    a = float(np.min(img))
    b = float(np.max(img))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):            
            modified_img[x,y] = ((float(img[x,y])) - a)*((d-c)/(b-a)) + float(c)
    
    return modified_img

def logarithmic(img):
"""Function to perform logarithmic intensity transformation"""    
    modified_img = np.zeros(img.shape)
    c = 255.0 / np.log2(1.0 + np.max(img))
    modified_img = c * np.log2(1.0 + img)
    
    return modified_img

def gamma(img, W, g):
"""Function to perform Gamma adjustment
Arguments:
	g - gamma parameter
"""    
    modified_img = np.zeros(img.shape)
    modified_img = W*(img**g)
    
    return modified_img

def calc_error(m,r):
"""Computing error from image reference to processed image"""    
    erro = np.sqrt(np.sum(np.square(m.astype(float) - r.astype(float))))
    
    return erro


# parameter input
filename = str(input()).rstrip()
input_img = imageio.imread(filename)
method = int(input())
save = int(input())

if method == 1:
    output_img = inversion(input_img)

if method == 2:
    c = int(input())
    d = int(input())

    output_img = contrast_modulation(input_img, c, d)

if method == 3:
    output_img = logarithmic(input_img)

if method == 4:
    W = int(input())
    lambd = float(input())
    output_img = gamma(input_img, W, lambd)

if save == 1:
    imageio.imwrite('output_img.png',output_img)

print(np.round(calc_error(output_img,input_img),4))

