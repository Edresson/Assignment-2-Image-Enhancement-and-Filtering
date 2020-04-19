
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
    Parameters:
        imgA: is Image A source
        imgB: is Image B source
    """
    # np.subtract calcute the error between pixels, after np.square square the error, after np.sum sum all error and finnaly np.sqrt calculate the root square
    return np.sqrt(np.sum(np.square(np.subtract(imgA.astype("float"), imgB.astype("float"))))) 

def unpad(img, pad_len):
    '''This function unpad  2D  Array padded with same lenght in the all axes
    Parameters:
        img: 2D source image
        pad_len: padding factor used for padding Array
    '''
    return img[pad_len:-pad_len,pad_len:-pad_len]

def euclidian_distance(x,y):
    ''' This functions calculate the euclidian distance between point x and point y
    Parameters:
        x: point x
        y: point y
    '''
    return np.sqrt(x**2 + y**2)

def gaussian_kernel(fil, sig):
    ''' This Function calculate a gaussian filter used on Bilateral Filter
    Parameters:
        fil: is the filter/image
        sig: is the sigma
    '''
    return (1.0 / ( 2 *np.pi * (sig ** 2))) * np.exp(- (np.square(fil) / (2 *  (sig ** 2))))

def bilateral_filter(input_img, n, sigma_s, sigma_r):
    '''This Function implement the bilateral filter. The Bilateral Filter 
        is a nonlinear filtering technique to smooth images while 
        preserving edges
        
    '''
    pad_len = n-(int(n/2)+1)
    # padding image
    input_img = np.pad(input_img, pad_len, mode='constant')

    #copy input image to output image
    output_img = input_img.copy()

    # make a centred origin filter
    w = np.zeros((n,n))
    img_shape = input_img.shape
    filter_shape = w.shape
    # get points and aplied euclidian distance
    for i in range(n): 
        for j in range(n):
            x = int(-(n-1)/2)+i
            y = int(-(n-1)/2)+j
            w[i][j] = euclidian_distance(x,y)  

    # get filter with kernel gaussian transformation
    w = gaussian_kernel(w, sigma_s).astype(np.float)

    x = int((filter_shape[0]-1)/2)
    y = int((filter_shape[1]-1)/2)  

    # Apply the convolution
    for i in range(x,img_shape[0]-x):
        for j in range(y,img_shape[1]-y):
            Wp = 0 
            I_f = 0
            # step 1: compute the value of the range Gaussian component, using the Gaussian kernel equation, with σ=σr and x as the diference between the intensity ofthe neighbor pixel and the intensity of the center pixel
            neighbors = input_img[i-x : i+(x+1) , j-y : j+(y+1)].astype(np.float)
            for k in range(filter_shape[0]):
                for l in range(filter_shape[1]):
                    #print(input_img[i][j],neighbors[k][l])
                    g_neighbor = gaussian_kernel(neighbors[k][l]-input_img[i][j],sigma_r).astype(np.float)
                    # step 2: compute the total value of the filter, on this neighbor, by multiplying the value of the range component by the value of the spatial component gsi(in the corresponding position)
                    g_w = (g_neighbor * w[k][l])
                    # step 3
                    Wp = Wp + g_w
                    # step 4
                    I_f = I_f + np.multiply(g_w,neighbors[k][l])
            # end step        
            I_f = I_f/Wp
            output_img[i][j] = (I_f).astype(np.uint8)

    #unpad image
    output_img = unpad(output_img, pad_len)
    return output_img

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
        output_img = bilateral_filter(input_img, n, sigma_s, sigma_r)
        
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