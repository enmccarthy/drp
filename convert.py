import numpy as np
import av
v = av.open(('./data/av5_1.avi'))
video = np.zeros((10000, 746,922), dtype=np.uint8)
for ind, packet in enumerate(v.demux()):
    for frame in packet.decode():
        truFrame = np.zeros((746,922))
        im = frame.to_image()
        arr = np.asarray(im)
        for i, val in enumerate(arr):
            for j, valj in enumerate(val):
                truFrame[i][j] = valj[0]
        video[ind] = truFrame
        if(np.sum(video[ind])==0):
            break
np.save('./data/AV_5_1.npy', video)