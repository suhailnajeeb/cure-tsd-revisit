import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# dynamically grow GPU memory
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from metrics import ssim_loss, psnr_tf
from skimage.metrics import structural_similarity
from math import log10, sqrt 
from models import makeModel
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

db_path = "C:\\Data\\CURE-TSD\\h5\\db_12_02.h5"
model_path = "C:\\Data\\CURE-TSD\\models\\haze\\best.hdf5"

model = load_model(model_path, custom_objects = {'ssim_loss': ssim_loss, 'psnr_tf': psnr_tf})

hf = h5py.File(db_path, 'r')
n_imgs = hf['source'].shape[0]

idx = np.random.randint(n_imgs)

src = hf['source'][idx]
tgt = hf['target'][idx]

x = np.expand_dims(src, axis = 0)/255
y_true = np.expand_dims(tgt, axis = 0)/255

y_pred = model.predict(x)
pred = np.squeeze(y_pred)

#ssim = structural_similarity(src, pred, multichannel = True)
#psnr = PSNR(src, pred)

fig, (ax1, ax2, ax3) = plt.subplots(1,3)

ax1.imshow(src)
ax1.set_title('Hazy Image')
ax2.imshow(tgt)
ax2.set_title('Target Image')
ax3.imshow(pred)
ax3.set_title('Output')

#ax3.text(0, 170,"SSIM: %.04f" % ssim)
#ax3.text(0, 190, "PSNR: %2.3f" % psnr)

plt.show()