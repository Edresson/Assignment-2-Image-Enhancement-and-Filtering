
# -*- coding: utf-8 -*-
"""
Autors Infos:
    Name: Edresson Casanova    
        USP Number: 11572715
        Course: SCC5830-3/3 
        Year: 1º year

    Name: Pedro Rocha    
        USP Number: 
        Course: SCC5830-3/3 
        Year: 

    Program: Computer Science and Computational Mathematics
    
Assignment 2:  Image Enhancement and Filtering
"""
import imageio # improt imageio library
import numpy as np # import numpy library

def RSE(imgA,imgB): 
     """This function calculate Root Squared Error between Image A and Image B
    Parameter:
        imgA: is Image A source
        imgB: is Image B source
    """
    # np.subtract calcute the error between pixels, after np.square square the error, after np.sum sum all error and finnaly np.sqrt calculate the root square
    return np.sqrt(np.sum(np.square(np.subtract(imgA.astype("float"), imgB.astype("float"))))) 


def bilateral_filter(input_img, sigma_s, sigma_r):
    pass

def unsharp_mask_laplacian(input_img, c, kernel):
    pass

def vignette_filter(sigma_row, sigma_col):
    pass

def main(): 
    filename = str(input()).rstrip() # input filename
    input_img = np.array(imageio.imread(filename)).astype(np.uint8) # load image using imageio library
    method = int(input()) # input method parameter
    save = int(input()) # input save parameter

    if method == 1: # if method 1
        n = int(input())
        sigma_s = float(input())
        sigma_r = float(input())
        output_img = bilateral_filter(input_img, sigma_s, sigma_r)

    if method == 2: # if method 2
        c = float(input()) # its float c <= 1
        kernel = int(input()) # its 1 or 2
        output_img = unsharp_mask_laplacian(input_img, c, kernel)

    if method == 3: #if method 3
        sigma_row = float(input())
        sigma_col = float(input())
        output_img = vignette_filter(sigma_row, sigma_col)

    if save == 1: #if save parameter
        imageio.imwrite('output_img.png',output_img) # save output image

    print("%.4f" %RSE(input_img,output_img))


if __name__ == '__main__':
    main()