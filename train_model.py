import numpy as np
import h5py
from utils import data_generator, ensure_dir
from models import makeModel
from metrics import ssim_loss, psnr_tf
#import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
from keras.callbacks import ModelCheckpoint, CSVLogger

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# dynamically grow GPU memory
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

db_path = "C:\\Data\\CURE-TSD\\h5\\db_12_02.h5"
model_path = "C:\\Data\\CURE-TSD\\models\\haze"

ensure_dir(model_path)

hf = h5py.File(db_path, 'r')
n_imgs = hf['source'].shape[0]
indexes = np.arange(n_imgs)

batch_size = 128

idx_train, idx_test = train_test_split(indexes, test_size=0.2, random_state=42)

n_train = len(idx_train)
n_test = len(idx_test)

train_gen = data_generator(hf, idx_train, batch_size=batch_size)
val_gen = data_generator(hf, idx_test, batch_size=batch_size)

#gen = data_generator(hf, indexes, batch_size = batch_size)

#x, y = next(gen)

check1 = ModelCheckpoint(os.path.join(
    model_path, "dehaze_{epoch:02d}-loss-{val_loss:.3f}.hdf5"), monitor='loss', save_best_only=True, mode='auto')
check2 = ModelCheckpoint(os.path.join(
    model_path, "best.hdf5"), monitor='loss', save_best_only=True, mode='auto')
check3 = CSVLogger(os.path.join(
    model_path, 'dehaze_trainingLog.csv'), separator=',', append=True)

model = makeModel('noiseNet001')
model.compile(optimizer='adam', loss='mae', metrics=[ssim_loss, psnr_tf])

model.fit(train_gen, steps_per_epoch=n_train // batch_size, callbacks=[check1, check2, check3],
          validation_data=val_gen, validation_steps=n_test // batch_size, epochs=10)
#model.fit(gen, steps_per_epoch = n_imgs // batch_size, epochs = 1)
