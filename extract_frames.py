import os
from utils import ensure_dir, extract_frames
import cv2

data_folder = "E:\\CURE-TSD\\CURE-TSD"
vid_no = "01_01_00_00_00.mp4"
out_folder = "E:\\CURE-TSD\\CURE-TSD\\output"

ensure_dir(out_folder)

vid_path = os.path.join(data_folder, vid_no)

extract_frames(vid_path, out_folder)