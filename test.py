import numpy as np
import cv2
import matplotlib.pyplot as plt

size = 480



phi_c = np.zeros((size,size))
phi_s = np.zeros((size,size))
phi_s_temp = np.zeros((size,size))

with open('1_1.txt') as f:
    data = f.readlines()

for i in range(4,len(data)):
    minutiae_x = int(data[i].split(' ')[0])
    minutiae_y = int(data[i].split(' ')[1])

    minutiae_x = (minutiae_x - size*0.5)/1
    minutiae_y = (minutiae_y - size*0.5)/1

    for j in range(size):
        for k in range(size):
            x = (j-size*0.5)/1
            y = (k-size*0.5)/1
            phi_c[j][k] = x
            phi_s_temp[j][k] = np.arctan2(x-minutiae_x,y-minutiae_y) + np.pi

    phi_s = (phi_s + phi_s_temp) % (2*np.pi)

phi_s = phi_s - np.pi
plt.imshow(np.cos(phi_c + phi_s))
plt.show()


plt.imshow(phi_s)
plt.show()