import tensorflow as tf
import os
from model import Depth
from loss import loss
import numpy as np

rgb_shape = (480, 640, 3)
depth_shape = (240, 320, 1)

def parse(filename, label):
    image = tf.image.decode_image(tf.io.read_file(filename))
    depth = tf.image.resize(tf.image.decode_image(tf.io.read_file(label)), [depth_shape[0], depth_shape[1]])

    image = tf.image.convert_image_dtype(image, dtype = tf.float32)
    depth = tf.image.convert_image_dtype(depth / 255.0, dtype = tf.float32)
    depth = 1000 / tf.clip_by_value(depth * 1000, 10, 1000)
    return image, depth

def normalize(x, max_depth):
    return max_depth/x

def predict(model, images, min_depth=10, max_depth=1000, batch_size=2):

    if len(images.shape) < 3: images = np.stack((images,images,images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))

    predictions = model.predict(images, batch_size=batch_size)

    return np.clip(normalize(predictions, max_depth=1000), min_depth, max_depth) / max_depth

def error(y_true, y_pred):
    threshold = np.maximum((y_true / y_pred), (y_pred / y_true))
    print(threshold)
    a1 = (threshold < 1.25     ).mean()
    a2 = (threshold < 1.25 ** 2).mean()
    a3 = (threshold < 1.25 ** 3).mean()
    return (a1, a2, a3)

def absolute_rel_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.math.abs(y_true - y_pred) / y_true)

csv = open('data/nyu2_test.csv', 'r').read()
nyu2 = list((row.split(',') for row in (csv).split('\n') if len(row) > 0))
filenames = [i[0] for i in nyu2]

labels = [i[1] for i in nyu2]
length = len(labels)

parsed = [parse(filenames[i], labels[i]) for i in range(length)]
x = tf.convert_to_tensor([i[0] for i in parsed])
y = tf.convert_to_tensor([i[1] for i in parsed])

for i in range(30):
    i=20
    path = "./ckpt2/{0:0=3d}.ckpt".format(i + 1)
    print(path)
    model = Depth()
    model.load_weights(path)
    y_pred = predict(model, x)
    print(absolute_rel_error(y, y_pred))
    print(error(y, y_pred))
    break