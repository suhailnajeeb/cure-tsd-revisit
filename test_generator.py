import numpy as np
import h5py
from utils import data_generator

db_path = "C:\\Data\\CURE-TSD\\h5\\db_09.h5"

hf = h5py.File(db_path,'r')
n_imgs = hf['source'].shape[0]
indexes = np.arange(n_imgs)

gen = data_generator(hf, indexes)

x, y = next(gen)

print(x.shape)
print(y.shape)