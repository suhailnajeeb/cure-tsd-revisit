import os
import cv2
import numpy as np
from pathlib import Path

def ensure_dir(file_path):
    Path(file_path).mkdir(parents=True, exist_ok=True)

def extract_frames(vid_path, out_folder):
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 1

    while success:
        out_path = os.path.join(out_folder, "%03d.jpg" % count)
        cv2.imwrite(out_path, image)     # save frame as JPEG file
        success, image = vidcap.read()
        #print('Read a new frame: ', success)
        count += 1

def crop_frames(src_path, tgt_path, crop = (128, 128), n = 10):
    src = cv2.imread(src_path)
    tgt = cv2.imread(tgt_path)

    dx = int(crop[0]/2)
    dy = int(crop[1]/2)

    width = tgt.shape[0]
    height = tgt.shape[1]

    xrands = np.random.randint(dx, width - dx, n)
    yrands = np.random.randint(dy, height - dy, n)

    src_crop = []
    tgt_crop = []

    for xrand, yrand in zip(xrands, yrands):
        src_crop.append(src[xrand-dx:xrand+dx, yrand-dx:yrand+dx])
        tgt_crop.append(tgt[xrand-dx:xrand+dx, yrand-dx:yrand+dx])
    
    return src_crop, tgt_crop

# depreciated function
def get_names(out_folder, challenge, real=1, seqs=np.arange(1, 50), chlngSrc=1, level=1, frames=300):
    srcnames = []
    tgtnames = []

    for seq in seqs:
        srcname = "%02d_%02d_%02d_%02d_%02d" % (
            real, seq, chlngSrc, challenge, level)
        tgtname = "%02d_%02d_00_00_00" % (real, seq)
        for i in range(1, frames + 1):
            frame = "%03d.jpg" % i
            srcnames.append(os.path.join(out_folder, srcname, frame))
            tgtnames.append(os.path.join(out_folder, tgtname, frame))
    return srcnames, tgtnames

def data_generator(h5file, indexes, batch_size = 1):
    X = []
    y = []
    idx = 0
    while True:
        for index in indexes:
            if(idx == 0):
                X = []
                y = []
            src = h5file["source"][index]
            tgt = h5file["target"][index]

            X.append(src)
            y.append(tgt)
            idx = idx + 1

            if(idx >= batch_size):
                idx = 0
                yield np.asarray(X)/255, np.asarray(y)/255