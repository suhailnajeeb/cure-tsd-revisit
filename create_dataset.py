import h5py
from utils import get_names, ensure_dir, crop_frames
import numpy as np
import os
from alive_progress import alive_bar

real = 1
seqs = np.arange(1, 50)
chlngSrc = 1
challenge = 9
level = 1
frames = 300
n_crops = 8
n_imgs = int(len(seqs)*frames*n_crops)


out_folder = "C:\\Data\\CURE-TSD\\output"
db_folder = "C:\\Data\\CURE-TSD\\h5"
db_path = "C:\\Data\\CURE-TSD\\h5\\db_%02d.h5"%challenge

ensure_dir(db_folder)

f = h5py.File(db_path,'w')

img_shape = (128, 128, 3)
img_chunk = (1,) + img_shape
nimg_shape = (n_imgs, ) + img_shape

f.create_dataset('source', nimg_shape, chunks = img_chunk, dtype = 'uint8', compression = 'gzip')
print("Creating Source Dataset with size: ", nimg_shape)

f.create_dataset('target', nimg_shape, chunks = img_chunk, dtype = 'uint8', compression = 'gzip')
print("Creating Target Dataset with size: ", nimg_shape)

idx = 0

with alive_bar(n_imgs) as bar:
    for seq in seqs:
        print("Processing Sequence: %d"%seq)
        srcname = "%02d_%02d_%02d_%02d_%02d" % (
            real, seq, chlngSrc, challenge, level)
        tgtname = "%02d_%02d_00_00_00" % (real, seq)
        for i in range(1, frames + 1):
            #print("\tProcessing Frame %d" % i)
            frame = "%03d.jpg" % i
            src_path = os.path.join(out_folder, srcname, frame)
            tgt_path = os.path.join(out_folder, tgtname, frame)
            src_crop, tgt_crop = crop_frames(src_path, tgt_path, n = n_crops)
            for src, tgt in zip(src_crop, tgt_crop):
                f['source'][idx] = src
                f['target'][idx] = tgt
                idx += 1
                bar()

f.close()
