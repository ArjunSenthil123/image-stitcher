import numpy as np
import time
import random
import math

# This code is part of:
#
#   CMPSCI 370: Computer Vision, Spring 2021
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Homework 4

def ransac(matches, c1, c2):
    
    '''w = c1.shape[1]
    l = round(w/8)
    
    matrix = np.zeros(shape=(2*l,3))
    matrix[0:l,0] = 1
    matrix[l:2*l,0] = 0
    matrix[0:l,1] = 0
    matrix[l:2*l,1] = 1
    

    m2 = np.zeros(shape=(2*l,1))


    for n in range(0,l):
        z = int(matches[n])
        matrix[n,2] = c2[0,z]

        m2[n,0] = c1[0,n]
   
    for n in range(l,2*l):
        z = int(matches[n])
        matrix[n,2] = c2[1,z]

        m2[n,0] = c1[1,n-l]

    A = matrix
    AT = matrix.transpose()
    ATA = np.matmul(AT,A)
    ATAinv = np.linalg.inv(ATA)
    y = m2

    ATAinvAT = np.matmul(ATAinv, AT)
    x = np.matmul(ATAinvAT ,y)
    print(x)'''


    numpts = c1.shape[1]
    numiters = 1000
    np.random.seed(0)
    max_inlier = 0
    transf = [0,0,1]
    inliers = []

    for n in range(numiters):
        r = random.randint(0,numpts-1)

        x1 = c1[0,r]
        y1 = c1[1,r]

        z = int(matches[r])
        x1p = c2[0,z]
        y1p = c2[1,z]

        r = random.randint(0,c1.shape[1]-1)

        x2 = c1[0,r]
        y2 = c1[1,r]

        z = int(matches[r])
        x2p = c2[0,z]
        y2p = c2[1,z]

        s_num = ((((x1p - x2p)**2) + ((y1p - y2p)**2)) **(1/2))
        s_den = ((((x1 - x2)**2) + ((y1 - y2)**2)) **(1/2))

        if(math.isclose(s_num,0) or (math.isclose(s_den,0))):
            continue

        s = s_num / s_den
    
        tx = x1p - (s*x1)
        ty = y1p - (s*y1)

        num_inliers = 0
        inlier_thresh = 10
        #find inliers
        inliers_n = []
        for m in range(numpts):
            x1 = c1[0,m]
            y1 = c1[1,m]

            z = int(matches[m])
            x1p = c2[0,z]
            y1p = c2[1,z]


            Tx1p = (x1p - tx)/s
            Ty1p = (y1p - ty)/s
        
            d = ((x1 - Tx1p)**2) + ((y1 - Ty1p)**2)
            if(d < inlier_thresh):
                num_inliers = num_inliers + 1
                inliers_n.append(m)
            
        #print(inliers)
        if(num_inliers > max_inlier):
            max_inlier = num_inliers
            transf = [tx,ty,s]
            inliers = inliers_n
        
        #print(n,tx,ty,s,num_inliers)




    print("final: ",transf, max_inlier)


    return(inliers,transf)


   

