import cv2
import numpy as np
from utils import ensure_dir, crop_frames

src_path = "C:\\Data\\CURE-TSD\\output\\01_01_01_09_01\\001.jpg"
tgt_path = "C:\\Data\\CURE-TSD\\output\\01_01_00_00_00\\001.jpg"

out_path = "C:\\Data\\CURE-TSD\\test"
ensure_dir(out_path)

src, tgt = crop_frames(src_path, tgt_path)

count = 0

for s, t in zip(src, tgt):
    cv2.imwrite(out_path + "//%02d_src.jpg"%count, s)
    cv2.imwrite(out_path + "//%02d_tgt.jpg"%count, t)
    count += 1
