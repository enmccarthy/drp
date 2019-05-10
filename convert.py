import numpy as np
import av
v = av.open(('./r3m1170210s2.avi'))
video = np.zeros((30000, 516,388))
for ind, packet in enumerate(v.demux()):
    for frame in packet.decode():
        truFrame = np.zeros((516,388))
        im = frame.to_image()
        arr = np.asarray(im)
        for i, val in enumerate(arr):
            for j, valj in enumerate(val):
                truFrame[i][j] = valj[0]
        video[ind] = truFrame
        if(np.sum(video[ind])==0):
            break
np.save('./r3m1170210s2AVI.npy', video)
