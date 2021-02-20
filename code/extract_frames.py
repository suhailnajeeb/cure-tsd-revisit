import os
from utils import ensure_dir, extract_frames
import numpy as np
import cv2

data_folder = "E:\\CURE-TSD\\CURE-TSD\\data"
out_folder = "C:\\Data\\CURE-TSD\\output"

# 5 levels of identification

real = 1
seqs = np.arange(1, 50)
chlngSrc = 1
challenge = 12
level = 2

vidnames = ["%02d_%02d_%02d_%02d_%02d" %
            (real, s, chlngSrc, challenge, level) for s in seqs]

for vidname in vidnames:
    print("Processing Video: ", vidname)
    vid_path = os.path.join(data_folder, vidname + ".mp4")
    out_path = os.path.join(out_folder, vidname)
    ensure_dir(out_path)
    extract_frames(vid_path, out_path)