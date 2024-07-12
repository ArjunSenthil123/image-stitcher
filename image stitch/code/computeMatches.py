import numpy as np

# This code is part of:
#
#   CMPSCI 370: Computer Vision, Spring 2021
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Homework 4

def computeMatches(f1, f2):

    print(f1.shape)
    print(f2.shape)


    n1 = f1.shape[1]
    n2 = f2.shape[1]

    matches = np.zeros(n1)
    sqdiffarr = np.zeros(shape = (n1,n2))
    for y in range(n1):
        for x in range(n2):
            diff = f1[:,y] - f2[:,x]
            sqr = diff**2
            sqdiffarr[y,x] = np.sum(sqr)

    
    for x in range(n1):
        slice = sqdiffarr[x,:]
        matches[x] = slice.argmin()
        
    #matches = np.reshape(matches,(n1,1))
        
    return matches




