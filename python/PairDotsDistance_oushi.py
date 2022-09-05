import numpy as np

def PairDotsDistance_oushi(Dots, NDim, numOfDots):
    Len = numOfDots*(numOfDots-1)/2
    DistanceMat = np.zeros((3, Len))

    matIndex=1

    for i in range(1, numOfDots-1):
        for j in range(i+1, numOfDots):
            DistanceMat[0, matIndex] = i
            DistanceMat[1, matIndex] = j

        DistanceMat[2,matIndex] = np.sqrt(np.sum(np.square(Dots[:,i] - Dots[:,j])))
    
    matIndex = matIndex + 1

    return matIndex