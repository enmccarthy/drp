import scipy.io as scio
import numpy as np
import math 
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sklearn.externals import joblib
def rowsToColumns(mat):
    (width, length) = mat.shape
    newmat = np.empty((length, width))
    for (x, row) in enumerate(mat):
        for (y, i) in enumerate(row):
            newmat[y][x] = i
    return newmat
def moving_average(a, n=3) :
    avg = np.zeros((a.shape))
    for rowInd, row in enumerate(a):
        for itemInd, item in enumerate(row):
            if itemInd < (n):
                avg[rowInd][itemInd] = sum(row[0:(itemInd+n)])/(itemInd+n)
            elif itemInd < len(a)-n:
                avg[rowInd][itemInd] = sum(row[(itemInd-n):(itemInd+n)])/(2*(itemInd+n))
            else:
                avg[rowInd][itemInd] = sum(row[(itemInd-n):])/((itemInd+n)+(len(a)-itemInd))
    return avg
def calcDff(video, winSz, hz):
    dt = 1/hz
    winSZ = int(winSz/dt)
    print(winSZ)
    (length, y, x) = video.shape
    shapedvideo = video.reshape(length,(y*x))
    fmean = np.zeros(shapedvideo.shape, dtype=np.float32)
    for rowind, row in enumerate(shapedvideo):
        fmean[rowind] = np.convolve(row, np.ones((winSZ,))/winSZ, mode='same')

#     fmean = moving_average(shapedvideo, winSZ)
    # fmean = shapedvideo
    print(fmean.shape)
    fmean = fmean.reshape(length, (y*x))
    print("new")
    #calculate moving mean
    f0 = np.percentile(fmean, 10, axis=0, interpolation='midpoint')

    # get percentile using prctile
    #fmean -mean(fmean,2)./f_o (percentile) ./ is array right division 
    print(f0)
    tot = np.ndarray.astype(np.divide(np.subtract(fmean,f0),f0), np.float16)
    print(tot.dtype)
    fin = tot.reshape((length, y, x))
    return fin
def otherDff(video, t0=int(0.2/.0333), t1=int(0.75/.0333), t2=int(3/.0333)):
    minMean = 0
    baseline = np.zeros((len(video)))
    videoMean = np.average(video)
    basetemp = []
    for find, frame in enumerate(video):
        def avg(q):
            # print(q)
            return np.average(video[int(q)])
        if find == 0:
            print("here")
            for c in range(find-t2+1, find):
                arg1 = (c-t1)/2
                if arg1 <0:
                    arg1 =0
                basetemp.append(np.multiply((1/t1),quad(avg, arg1,(c+t1)/2)[0]))
        else: 
            c= find-t2+1
            arg1 = (c-t1)/2
            if arg1 <0:
                arg1 =0
            basetemp = basetemp[1:]
            basetemp.append(np.multiply((1/t1),quad(avg, arg1,(c+t1)/2)[0]))
        baseline[find] = (min(basetemp))
    def rt(x, tau):
        return np.divide(np.subtract(videoMean[x]-baseline[x]), baseline[x])*np.exp(-(tau)/t0)
    dff = np.zeros(len(video))
    for ind, curFrame in enumerate(video):
        dff[ind] = quad(rt, 0, ind, args={ind})
    return dff
def bin_ndarray(ndarray, new_shape, operation='average'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.
    Number of output dimensions must match number of input dimensions.
    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)
    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]
    """
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation {} not supported.".format(operation))
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray
def find_nearest(array,value, func="none"):
    if func == "lick":
        if len(array) >1: 
            idx = np.searchsorted(array, value, side="left")
            if idx > 0 and (idx == len(array)):
                return idx-1
            elif array[idx] < value:
                return idx+1
            else:
                return idx
        else:
            return 0
    else:
        if len(array) >1: 
            idx = np.searchsorted(array, value, side="left")
            if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
                return idx-1
            elif array[idx] > value:
                return idx-1
            else:
                return idx
        else:
            return 0
def getData(data, rate, length, time, ts, func='add'):
    # find the index in the data that is closest to the ts
    # rate of collection vs the amount of time you want to apply
    # the function to
    # .250 30 hz how many frames
    #  .250/(1/30)
    index = find_nearest(ts, time)
    numFrames = int(length*rate)
    total = np.zeros(numFrames)
    if func == 'change':
        total = data[index+numFrames]-data[index]
    else:
        for x in range(numFrames):
            point = data[index+x]
            if np.isnan(point):
                total[x] = 0
            else:
                total[x] = point
        if func == 'avg':
            totalret = np.divide((np.sum(total)),numFrames)
        elif func == 'arr':
            totalret = total
        else:
            totalret = (np.sum(total))
    return totalret
def rowsToColumns(mat):
    (width, length) = mat.shape
    newmat = np.empty((length, width))
    for (x, row) in enumerate(mat):
        for (y, i) in enumerate(row):
            newmat[y][x] = i
    return newmat
def vizWeights(columns, modP):
    # modPath = open(modP, "rb")
    modelWeights = np.load(modP)
    modelWeightRot = rowsToColumns(modelWeights)

    for colInd, column in enumerate(columns):
        # (319, 320)
        # (638, 640)
        imc = plt.imshow(modelWeightRot[colInd].reshape(638, 640), cmap='hot', interpolation='gaussian')
        photoPath = './weight_pictures/AV_5_dff'+column+'.png'
        plt.colorbar(label='weights')
        plt.savefig(photoPath)
        plt.gcf().clear()
    return modelWeightRot
def vizPerm(orig,name, columns):
    print(columns)
    allA = np.empty((len(columns)+1, 300*300))
    original = np.empty(300*300)
    perm = np.empty(300*300)
    original = joblib.load(orig)
    original[original<0] = 0
    for colIndex, colName in enumerate(columns):
        filename = name+colName+'.pkl'
        perm = joblib.load(filename)
        perm[perm<0] = 0
        im = np.subtract(original, perm)
        allA[colIndex] = im
        imc = plt.imshow(im.reshape(300,300), cmap='hot', interpolation='gaussian')
        plt.colorbar(label='weights')
        photoPath ='./stim_pictures/'+str(colName)+'.png'
        plt.savefig(photoPath)
        plt.gcf().clear()
    allA[len(columns)] = original
    imc = plt.imshow(original.reshape(300,300), cmap='hot', interpolation='gaussian')
    plt.colorbar(label='weights')
    photoPath = './stim_pictures/total.png'
    plt.savefig(photoPath)
    return allA
