import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = "C:\\Data\\CURE-TSD\\output\\01_23_01_09_01\\120.jpg"

img = cv2.imread(img_path)

n = 16

crop = (128, 128)

color = (255,0,0)
thickness = 5

dx = int(crop[0]/2)
dy = int(crop[1]/2)

height = img.shape[0]
width = img.shape[1]

xrands = np.random.randint(2*dx, height - 2*dx, n)
yrands = np.random.randint(2*dy, width - 2*dy, n)

for xrand, yrand in zip(xrands, yrands):
    start = (xrand-dx, yrand-dy)
    end = (xrand+dx, yrand+dy)
    img = cv2.rectangle(img, start, end, color, thickness)

plt.imshow(img)
plt.show()