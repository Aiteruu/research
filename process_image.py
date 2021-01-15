import tensorflow as tf
import os
from model import Depth
from loss import loss
import numpy as np
import sys
import matplotlib.pyplot as plt 

rgb_shape = (480, 640, 3)
depth_shape = (240, 320, 1)

def parse(filename):
    image = tf.image.decode_image(tf.io.read_file(filename))
    image = tf.image.convert_image_dtype(image, dtype = tf.float32)
    return image

def normalize(x, max_depth):
    return max_depth/x

def predict(model, images, min_depth=10, max_depth=1000, batch_size=1):
    if len(images.shape) < 3: images = np.stack((images,images,images), axis=2)
    if len(images.shape) < 4: images = np.reshape(images, (1, images.shape[0], images.shape[1], images.shape[2]))
    predictions = model.predict(images, batch_size=batch_size)
    return predictions

filename = sys.argv[1].split('.')[0]

for i in range(30):
    path = "./ckpt2/{0:0=3d}.ckpt".format(i + 1)
    print(path)
    model = Depth()
    model.load_weights(path)
    y_pred = predict(model, parse('test/in/' + sys.argv[1]))
    y_pred = np.squeeze(y_pred)
    plt.imsave("sample/{}.png".format(i + 1), y_pred)
    np.save("sample/{}.npy".format(i + 1), y_pred)

#np.save("pred.npy", tf.squeeze(y_pred))
#plt.imsave('test/out/' + filename + '.png', tf.squeeze(y_pred))