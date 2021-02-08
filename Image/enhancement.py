import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import math

image = Image.open('F:\Awinter\MCM\C\postive\\14_img1.jpg')
img = np.array(image)

img2 = img[::-1, :, :]
img_2 = Image.fromarray(img2)
img_2.save('F:\Awinter\MCM\C\postive\\14_img2.jpg')

img3 = mg[:, ::-1, :]
img_3 = Image.fromarray(img3)
img_3.save('F:\Awinter\MCM\C\postive\\14_img3.jpg')

H, W = img.shape[0], img.shape[1]

H1 = H * 2 // 3
img4 = img[0:H1, :, :]

img_4 = Image.fromarray(img4)
img_4.save('F:\Awinter\MCM\C\postive\\14_img4.jpg')

W1 = W // 10
W2 = W // 3 * 2
img5 = img[:, W1:W2, :]

img_5 = Image.fromarray(img5)
img_5.save('F:\Awinter\MCM\C\postive\\14_img5.jpg')

img6 = img * 0.5
img6 = img6.astype('uint8')

img_6 = Image.fromarray(img6)
img_6.save('F:\Awinter\MCM\C\postive\\14_img6.jpg')

img7 = img * 2.5
img7 = np.clip(img7, a_min=None, a_max=255.)
img7 = img7.astype('uint8')
img_7 = Image.fromarray(img7)
img_7.save('F:\Awinter\MCM\C\postive\\14_img7.jpg')

img8 = img[::2, ::2, :]
img_8 = Image.fromarray(img8)
img_8.save('F:\Awinter\MCM\C\postive\\14_img8.jpg')

img9 = image.transpose(Image.ROTATE_90)
img10 = image.transpose(Image.ROTATE_180)
img11 = image.transpose(Image.ROTATE_270)
img_9 = img9
img_9.save('F:\Awinter\MCM\C\postive\\14_img9.jpg')
img_10 = img10
img_10.save('F:\Awinter\MCM\C\postive\\14_img10.jpg')
img_11 = img11
img_11.save('F:\Awinter\MCM\C\postive\\14_img11.jpg')

H, W = img.shape[0], img.shape[1]
img_float = img.astype("float")
Gauss_noise = np.random.normal(0, 50, (H, W, 3))
img_Gauss = img_float + Gauss_noise
img_12 = np.where(img_Gauss < 0, 0, np.where(img_Gauss > 255, 255, img_Gauss))
img_12 = img_12.astype("uint8")
img_12 = Image.fromarray(img_12)
img_12.save('F:\Awinter\MCM\C\postive\\14_img12.jpg')

