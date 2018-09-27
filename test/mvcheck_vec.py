from __future__ import print_function
import numpy as np
import time


def doWarp(src, mvec, width, height):
    absoluteMvec = np.round(mvec + np.stack(np.split(np.indices((height, width), dtype='float32'), 2), axis=-1)[0]).astype('int32')

    output = np.zeros((height, width, 3))

    for y in range(0, height):
        for x in range(0, width):
            ref = absoluteMvec[y, x]

            if np.isfinite(mvec[y, x, 0]) and np.isfinite(mvec[y, x, 1]):
                ry = np.clip(ref[0], 0, height-1)
                rx = np.clip(ref[1], 0, width-1)

                output[y, x, :] = src[ry, rx, 0:3]
            else:
                output[y, x, :] = [1.0, 0.0, 0.0]

    return output


def doWarp_vec(src, mvec, width, height):
    absoluteMvec = np.round(mvec + np.stack(np.split(np.indices((height, width), dtype='float32'), 2), axis=-1)[0]).astype('int32')
    #print(absoluteMvec[:5, :5, :5])
    output = np.zeros((height, width, 3))
    output[..., 0] = 1.0

    ry_idx = np.clip(absoluteMvec[..., 0], 0, height-1)
    rx_idx = np.clip(absoluteMvec[..., 1], 0, width-1)
    output[...] = src[ry_idx, rx_idx, :3]
    #output[...] = src[ry_idx, rx_idx] is this ok?
    finite_mask = np.isfinite(mvec[..., 0]) & np.isfinite(mvec[..., 1])
    output[~finite_mask] = [1.0, 0.0, 0.0]

    return output

height = 100
width = 200

src = np.random.randint(0, 10, size=(height, width, 3)).astype('float32')
mvec = np.random.randint(0, 1000, size=(height, width, 2)).astype('float32')
assert np.allclose(doWarp(src, mvec, width, height), doWarp_vec(src, mvec, width, height))

t0 = time.time()
o1 = doWarp(src, mvec, width, height)
t1 = time.time()
o2 = doWarp_vec(src, mvec, width, height)
end = time.time()
print("Time taken by doWarp vs doWarp_vec: {}-{}".format(t1-t0, end-t1))
print(o1[:2, :2, :2])
print(o2[:2, :2, :2])


#\\dcg-zfs-04\dlaa.cosmos420\


#sudo mount -t cifs -o username=ericx,sec=ntlm,domain=NVIDIA.COM,dir_mode=0777,file_mode=0777 //dcg-zfs-04.nvidia.com/dlaa.cosmos420 /private/dlaa_capture

/usr/bin/python mvcheck/mvcheck.py --color "~/Desktop/nvidia_work/pytorch_examples/test/capture_data/hdr_in/BFV.%0.4d_apicaccum_1spp.pfm" --mvec "~/Desktop/nvidia_work/pytorch_examples/test/capture_data/mvec/BFV.%0.4d.pfm" --lookback 7 --frame 7




