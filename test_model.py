import numpy as np
import h5py
from utils import data_generator
from models import makeModel
#import tensorflow as tf

db_path = "C:\\Data\\CURE-TSD\\h5\\db_09.h5"

hf = h5py.File(db_path,'r')
n_imgs = hf['source'].shape[0]
indexes = np.arange(n_imgs)

gen = data_generator(hf, indexes)

x, y = next(gen)

model = makeModel('dummyModel')
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x, y)


# Loss functions paper:
#https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf