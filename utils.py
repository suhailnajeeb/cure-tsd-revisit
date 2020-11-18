import os
import cv2

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