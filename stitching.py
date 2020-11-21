import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from models import makeModel
import h5py
from keras.models import load_model
from metrics import ssim_loss, psnr_tf

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# dynamically grow GPU memory
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

img_path = "C:\\Data\\CURE-TSD\\output\\01_01_01_09_01\\028.jpg"
model_path = "C:\\Data\\CURE-TSD\\models\\rain\\best.hdf5"

patch = 128

img = cv2.imread(img_path)
h = img.shape[0]
w = img.shape[1]

tgt_h = math.ceil(h/patch)*patch
tgt_w = math.ceil(w/patch)*patch

nh = math.ceil(h/patch)
nw = math.ceil(w/patch)

dh = int((tgt_h - h)/2)
dw = int((tgt_w - w)/2)

image = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_DEFAULT)

model = load_model(model_path, custom_objects = {'ssim_loss': ssim_loss, 'psnr_tf': psnr_tf})

prediction = image.copy()

x = 0
for i in range(nh):
    y = 0
    for j in range(nw):
        src = image[x:x+128, y:y+128, :]/255
        src = np.expand_dims(src, axis = 0)
        pred = model.predict(src)
        pred = np.squeeze(pred)*255
        #plt.imshow(pred)
        #plt.show()
        prediction[x:x+128, y:y+128, :] = pred
        y += 128
    x += 128

fig, ax = plt.subplots(1, 2)

ax[0].imshow(image)
ax[1].imshow(prediction)

plt.show()