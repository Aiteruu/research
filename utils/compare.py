import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import resize


true = np.squeeze(np.array(Image.open("true.png"), dtype='float32'))
#pred = np.squeeze(np.array(Image.open("test.png"), dtype='float32'))



true /= np.max(true)
true = 1000 / np.clip(true * 1000, 10, 1000)
true = resize(true, (240, 320))
error = []

for i in range(30):
    pred = np.load("sample/{}.npy".format(i + 1))
    k = np.mean(true/pred)
    pred = k * pred
    error.append(np.mean(np.abs(pred - true) / true))
    #error.append((np.maximum(pred/true, true/pred) < 1.25).mean())

plt.plot(error)
plt.show()