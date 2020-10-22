import tensorflow as tf

rgb_shape = (480, 640, 3)
depth_shape = (240, 320, 1)

csv = open('data/nyu2_train.csv', 'r').read()
nyu2 = list((row.split(',') for row in (csv).split('\n') if len(row) > 0))
filenames = [i[0] for i in nyu2]
labels = [i[1] for i in nyu2]
length = len(labels)
dataset =  tf.data.Dataset.from_tensor_slices((filenames, labels))

def parse(filename, label):
    image = tf.image.decode_jpeg(tf.io.read_file(filename))
    depth = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(label)), [depth_shape[0], depth_shape[1]])

    image = tf.image.convert_image_dtype(image, dtype = tf.float32)
    depth = tf.image.convert_image_dtype(depth / 255.0, dtype = tf.float32)
    depth = 1000 / tf.clip_by_value(depth * 1000, 10, 1000)
    return image, depth

def batch(batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(buffer_size=length, reshuffle_each_iteration=True).repeat()
    dataset = dataset.map(map_func=parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return length, dataset.batch(batch_size=batch_size)
