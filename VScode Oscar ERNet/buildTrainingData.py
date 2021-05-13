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
import extraOperations


def noisy(noise_typ,image,opts=[0.0,0.1]): # opts is mean and var
   
    if noise_typ == "gauss":
        mean = opts[0]
        var = opts[1]
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,image.shape) 
        gauss = gauss.reshape(image.shape)
        noisy = image + gauss*image 
        return noisy
    
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.2
        amount = 0.004
        out = np.copy(image) # make a copy otherwise you are just re-adjusting a pointer
        
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]

        out[coords] = 0
        
        return out
    
    elif noise_typ == "poisson":
        
        image = image.astype(float)
        PSFimage = extraOperations.PSFnoise(image,0.8)
        #print("PSF noise added")
        #row,col,ch = image.shape
        #lam = (opts[1])**0.5
        #noisy = (np.random.poisson(lam,(row,col))).astype(float)
        #noisy = (noisy)/np.max(noisy)
        #print(noisy)
        #poisson_image = np.ones((row, col, ch),dtype = float)
        
        #j = 0
        #while j < ch:
        #    poisson_image[:,:,j] = (image[:,:,j].astype(float))*(noisy)
        #    j = j + 1
            
        #brightness_ratio = np.mean(image)/np.mean(poisson_image)
        #poisson_image = poisson_image*brightness_ratio
        #poisson_image = poisson_image.astype(int)
        #poisson_image = np.clip(poisson_image, 0, 255)
        #return poisson_image
        return PSFimage

    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)     
        noisy = image + image*gauss
        return noisy

def partitionDataset(imgs,outdir,nreps,dim):
    try:
        os.makedirs(outdir)
    except OSError:
        pass
    for i in range(0,len(imgs),1):
        
        img = Image.open(imgs[i])
        img = np.array(img)
        img = img
        imgGT = np.copy(img)
        h,w,ch = img.shape  # (height, width, channels)
       
        j = 0
        while j < nreps:
            r_rand = np.random.randint(0,h-dim)
            c_rand = np.random.randint(0,w-dim)
            sub_img = img[r_rand:r_rand+dim,c_rand:c_rand+dim]
            sub_imgGT = imgGT[r_rand:r_rand+dim,c_rand:c_rand+dim] # (to be used as corresponding ground truth for training)


            if np.mean(sub_imgGT) < 0.1*255:
            # print('redoing')
                continue
            # adding random brightness
            brightness = 1 + 0.1*np.random.randn()
            sub_img = np.clip(sub_img* brightness,0,255)

            if  np.random.rand() > 0: #use combination of poisson noise and gaussian noise (former dominates)
                 poisson_param = 100 # variance of poisson distribution
                 sub_img = sub_img.astype(float)
                 sub_img = noisy('poisson',sub_img,[0,poisson_param])
                
                 gauss_param = abs(0.002*np.random.randn() + 0.005)
                 sub_img = noisy('gauss',sub_img,[0,gauss_param])
            else:
                 sigma_param = np.random.rand()
                 sub_img = ndimage.gaussian_filter(sub_img, sigma=(sigma_param,sigma_param,0), order=0)


  
            filename = '%s/%d-%d.npy' % (outdir,i,j) 

            print(i,j,r_rand,c_rand,sub_img.shape,sub_imgGT.shape)

            sub_img = Image.fromarray(sub_img.astype('uint8'))
            sub_imgGT = Image.fromarray(sub_imgGT.astype('uint8'))
            pickle.dump((sub_img,sub_imgGT), open(filename,'wb'))
            combined = np.concatenate((np.array(sub_img),np.array(sub_imgGT)),axis=1)
            io.imsave(filename.replace(".npy",".png"),combined)
            j += 1

        print('[%d/%d]' % (i+1,len(imgs)))


allimgs = []

for i in range (1,10,1):
    dirname = "trainingdata/samples/000" + str(i) + ".png" 
    #dir = input(dirname)
    #allimgs.append(dir)
    allimgs.append(dirname)

for i in range (10,100,1):
    dirname = "trainingdata/samples/00" + str(i) + ".png" 
    #dir = input(dirname)
    #allimgs.append(dir)
    allimgs.append(dirname)

for i in range (100,301,1):
    dirname = "trainingdata/samples/0" + str(i) + ".png" 
    #dir = input(dirname)
    #allimgs.append(dir)
    allimgs.append(dirname)

nreps = 5
dim = 128



outdir = 'trainingdata/testpartitioned_' + str(dim)

# commented out for model evaluation triplets
#partitionDataset(allimgs,outdir,nreps,dim)

