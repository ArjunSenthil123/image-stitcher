import numpy as np
from skimage.color import rgb2gray
import utils
from utils import *

#This code is part of:
#
#  CMPSCI 370: Computer Vision, Spring 2021
#  University of Massachusetts, Amherst
#  Instructor: Subhransu Maji
#
#  Homework 4

def extractFeatures(im, c, patch_radius):
    
    h,w = c.shape
    d = ((2*patch_radius)+1)**2
    feat_arr = np.zeros(shape = (d,w))
    grayim = rgb2gray(im)
    grayim = np.pad(grayim, (patch_radius,patch_radius), 'constant', constant_values=(0, 0))
    for x in range(w):

        xpos = round(c[0,x])
        ypos = round(c[1,x])

        #(patch_radius + xpos - patch_radius) : (patch_radius + xpos + patch_radius + 1)

        temparr = grayim[ (patch_radius + ypos - patch_radius) : (patch_radius + ypos + patch_radius + 1), 
        (patch_radius + xpos - patch_radius) : (patch_radius + xpos + patch_radius + 1)]

     
        temparr = np.reshape(temparr,(d,1))
       
        feat_arr[0:d,x] = temparr[0:d,0]
        



    return feat_arr
    



