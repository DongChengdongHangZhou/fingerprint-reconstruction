import numpy as np
import cv2
import matplotlib.pyplot as plt

size = 400

phi_c = np.zeros((size,size))
phi_s = np.zeros((size,size))

for i in range(size):
    for j in range(size):
        x = (i-size*0.5)/10
        y = (j-size*0.5)/10
        phi_c[i][j] = x + y
        phi_s[i][j] = -np.arctan2(x-0,y-0)


plt.imshow(np.cos(phi_s+phi_c))
plt.show()