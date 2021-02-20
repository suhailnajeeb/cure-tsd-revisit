import h5py
import matplotlib.pyplot as plt

db_path = "C:\\Data\\CURE-TSD\\h5\\db_09.h5"

f = h5py.File(db_path,'r')

idx = 0

src = f['source'][idx]
tgt = f['target'][idx]

plt.imshow(src)
plt.show()

plt.imshow(tgt)
plt.show()

f.close()