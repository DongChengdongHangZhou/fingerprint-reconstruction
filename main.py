import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
Feng, Jianjiang, and Anil K. Jain. "Fingerprint reconstruction: from minutiae to phase." IEEE transactions on pattern analysis and machine intelligence 33.2 (2010): 209-223.
'''

size = 480

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

plt.imshow(phi_s)
plt.show()

def drawOrientation(ori, background, mask=None, block_size=16, color=(255,0, 0), thickness=1, is_block_ori=False):
    h = ori.shape[0]
    w = ori.shape[1]
    if is_block_ori:
        draw_step = 1
        offset = 0
    else:
        draw_step = block_size
        offset = int(block_size / 2)
    for x in range(0, w - offset, draw_step):
        for y in range(0, h - offset, draw_step):
            if mask is not None and mask[y + offset, x + offset] == 0:
                continue
            th = ori[y + offset, x + offset]
            print(th)
            if is_block_ori:
                x0 = x * block_size + block_size / 2
                y0 = y * block_size + block_size / 2
            else:
                x0 = x + block_size / 2
                y0 = y + block_size / 2
            x1 = int(x0 + 0.4 * block_size * np.cos(th))
            y1 = int(y0 + 0.4 * block_size * np.sin(th))
            x2 = int(x0 - 0.4 * block_size * np.cos(th))
            y2 = int(y0 - 0.4 * block_size * np.sin(th))
            cv2.line(background, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(background,(x2,y2),radius=thickness*2,color=color,thickness=thickness)
    return background


grad = np.zeros((size,size))
g1 = np.gradient(phi_s)[0]
g2 = np.gradient(phi_s)[1]
for i in range(size):
    for j in range(size):
        grad[i][j] = np.arctan2(g1[i][j],g2[i][j])+np.pi

vis = drawOrientation(grad,(np.ones((size,size,3))*255).astype(np.uint8))
plt.imshow(vis)
plt.show()