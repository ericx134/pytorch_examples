
import os, sys, numpy as np
from PIL import Image
from pfm_helper import load_pfm, save_pfm
from exr_helper import *
import argparse


def load_mvec(file, mvscalex, mvscaley):
    if file.endswith('.pfm'):
        mv_img = load_pfm(file)
        mvx = mv_img[...,0:1] * mvscalex
        mvy = mv_img[...,1:2] * mvscaley

        # mvx = np.clip(mvx, float(-512), float(512))
        # mvy = np.clip(mvy, float(-512), float(512))
        return np.concatenate([mvy, mvx], axis = -1)
    else:
        mv_img = load_exr(file)
        mvx = mv_img[...,0:1] * mvscalex
        mvy = mv_img[...,1:2] * mvscaley

        # save_pfm(file.replace(".exr", "_test.pfm"), mv_img)

        # mvx = np.clip(mvx, float(-512), float(512))
        # mvy = np.clip(mvy, float(-512), float(512))
        return np.concatenate([mvy, mvx], axis = -1)


# Stack a list of images vertically
def StackVertical(images):
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)
    newImage = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for im in images:
      newImage.paste(im, (0,y_offset))
      y_offset += im.size[1]
    return newImage

# Stack a list of images horizontally
def StackHorizontal(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    newImage = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      newImage.paste(im, (x_offset,0))
      x_offset += im.size[0]
    return newImage

def doWarp(src, mvec, width, height):
    absoluteMvec = np.round(mvec + np.stack(np.split(np.indices((height,width), dtype='float32'), 2), axis=-1)[0]).astype('int32')

    output = np.zeros((height,width,3))
    
    for y in range(0, height):
        for x in range(0, width):
            ref = absoluteMvec[y,x]

            if np.isfinite(mvec[y,x,0]) and np.isfinite(mvec[y,x,1]):
                ry = np.clip(ref[0], 0, height-1)
                rx = np.clip(ref[1], 0, width-1)

                output[y,x,:] = src[ry, rx, 0:3]
            else:
                output[y,x,:] = [1.0, 0.0, 0.0]

    return output


# ---- main ---- 


def parseCommandLine():

    parser = argparse.ArgumentParser()
    parser.add_argument("--color",    type=str, action='store', dest='color',    required=True, help="color images (use %%d-like formatter for frame ID)")
    parser.add_argument("--mvec",     type=str, action='store', dest='mvec',     required=True, help="motion images (use %%d-like formatter for frame ID)")
    parser.add_argument("--frame",    type=int, action='store', dest='frame',    default=7,    help="which frame ID to use")
    parser.add_argument("--lookback", type=int, action='store', dest='lookback', default=1,     help="how many frames to look back")
    parser.add_argument("--mvscalex", type=float, action='store', dest='mvscalex', default=1.0,     help="mv scale x")
    parser.add_argument("--mvscaley", type=float, action='store', dest='mvscaley', default=1.0,     help="mv scale y")

    return parser.parse_args()

def load_image(file):
    if file.endswith('.exr'):        
        return load_exr(file)   
    elif file.endswith('.pfm'):
        return load_pfm(file)     
    else:
        return np.array(Image.open(file)).astype('float32') * 1.0/255.0

def save_image(arr, file):
    Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype('uint8')).save(file)

def mvcheck(color, mv, frame, lookback, mvscalex, mvscaley):
    
    getcolor = lambda idx : (color % idx)
    getmvec  = lambda idx : (mv  % idx)

    tgt = load_image(getcolor(frame))
    src = load_image(getcolor(frame-lookback))

    width  = src.shape[1]
    height = src.shape[0]

    print("Warping %s to %s" % (color % (frame-lookback), color % (frame)))

    warp = src.copy()
    for tgtFrame in range(frame-lookback+1,frame+1):
        print(" - %s" % getmvec(tgtFrame))
        mvec = load_mvec(getmvec(tgtFrame), mvscalex, mvscaley)
        
        warp = doWarp(warp, mvec, width, height)
  

    meanval = np.mean (np.concatenate([src,tgt,warp], axis=-1))
    stdev   = np.std  (np.concatenate([src,tgt,warp], axis=-1))

    save_image(src  / (meanval + stdev * 2),  "source.png")
    save_image(tgt  / (meanval + stdev * 2),  "target.png")
    save_image(warp / (meanval + stdev * 2),  "warped.png")

    print("saved source.png, target.png and warped.png")
    
cmd = parseCommandLine()
color = cmd.color
mvec = cmd.mvec
frame = cmd.frame
lookback = cmd.lookback
mvscalex = cmd.mvscalex
mvscaley = cmd.mvscaley

mvcheck(color, mvec, frame, lookback, mvscalex, mvscaley)