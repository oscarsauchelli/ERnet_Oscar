import glob
from PIL import Image
import random
import parser
import os
import scipy.ndimage as ndimage
import numpy as np
import pickle
import cv2
from skimage.measure import compare_ssim
from skimage import io
from scipy.fft import fft, ifft
from scipy.ndimage import convolve
import scipy


def PsfOtf(w,scale):    # STEP 1: GENERATES A PSF WHICH YOU DON'T USE AND AN OTF WHICH YOU DOU USE, RETURNS THEM AS A PAIR
    # AIM: To generate PSF and OTF using Bessel function
    # INPUT VARIABLES
    # w: image size
    # scale: a parameter used to adjust PSF/OTF width
    # OUTPUT VRAIBLES
    # yyo: system PSF
    # OTF2dc: system OTF   
    eps = np.finfo(np.float64).eps
    x = np.linspace(0,w-1,w)
    y = np.linspace(0,w-1,w)
    X,Y = np.meshgrid(x,y)
    
    #Generation of the PSF with BesselJ 
    R = np.sqrt(np.minimum(X, np.abs(X-w))**2 + np.minimum(Y, np.abs(Y-w))**2)
    yy = np.abs(2*scipy.special.jv(1, scale*R + eps)/(scale*R + eps))**2 # 0.5 is introduced to make PSF wider
    yy0 = np.fft.fftshift(yy)
        
    # Generate 2D OTF.
    OTF2d = np.fft.fft2(yy)
    OTF2dmax = np.max([np.abs(OTF2d)])
    OTF2d = OTF2d/OTF2dmax
    OTF2dc = np.abs(np.fft.fftshift(OTF2d))
    

    
    
    return (yy0,OTF2dc)

def onechanto3chan(imggray,imggraywithnoise,img):
    (h,w,d) = img.shape
    img3chanwithnoise = np.zeros(img.shape,img.dtype)
    for i in range(0,h-1):
        for j in range(0,w-1):
            noisefactor = imggraywithnoise[i,j]/imggray[i,j]
            img3chanwithnoise[i,j,:] = img[i,j,:]*noisefactor
    return img3chanwithnoise    
                         
def PSFnoise(image,scale):  # INPUT IS THE PNG IMAGE IN NUMPY FORMAT WITH n channels and scale value for OTF generation
   
# img is made to point to one channel version of input image 
    imggray = image.mean(2)
    img = imggray  

# img is made to point to fourier shift of one channel version of input image
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    img = fshift   # img is now the the fourier transform of what it previously was
    
# img is made to point to elementwise product of fourier transform with OTF of given scale
    (a,b) = img.shape
    imagesize = a   # it is assumed here that the input image is such that a = b, otherwise the product thing wouldn't work
    (PSF,OTF) = PsfOtf(imagesize, scale)
    imgXOTF = img*OTF
    img = imgXOTF
    
# img is made to point to inverse FFT of fourier plane result
    funshift = np.fft.ifftshift(img)
    finvcomplex = np.fft.ifft2(funshift)
    finv = abs(finvcomplex)
    img = finv

    img = onechanto3chan(imggray,img,image) # image3 is already 128by128 : option for scaling image back to 3 channels
    return img






def onechanCLIP(fsX,width,height):
    
    clip = np.zeros(fsX.shape,fsX.dtype) 
    (a,b) = clip.shape
    
    a = int(a)
    b = int(b)
    ahalf = int(int(a)/2-0.5)
    bhalf = int(int(b)/2-0.5)
    (chalf,dhalf) = (int((width/2-0.5)),int((height/2-0.5)))
    
    for i in range(0,a):
        for j in range(0,b):
            if abs(ahalf-i)<chalf and abs(bhalf-j)<dhalf:
               # print((chalf,abs(ahalf-i),dhalf,abs(bhalf-j)))
                fsX[i,j] = fsX[i,j]    
            else:
                fsX[i,j] = 0           
    return fsX
















 
  