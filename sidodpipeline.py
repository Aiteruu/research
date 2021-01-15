import tensorflow as tf

csv = open('dataset/train.csv', 'r').read()
sidod = list((row.split(',') for row in (csv).split('\n') if len(row) > 0))

rgb_shape = (960, 540, 3)
depth_shape = (480, 270, 1)

filenames = [i[0] for i in sidod]
labels = [i[1] for i in sidod]

length = len(labels)

def parse(filename, label):
    image = tf.image.decode_jpeg(tf.io.read_file(filename, channels=3))
    depth = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(label)), [depth_shape[0], depth_shape[1]])

    image = tf.image.convert_image_dtype(image, dtype = tf.float32)
    depth = tf.div(
        tf.subtract(
            depth, 
            tf.reduce_min(depth)
        ), 
        tf.subtract(
            tf.reduce_max(depth), 
            tf.reduce_min(depth)
        )
    )
    depth = tf.image.convert_image_dtype(depth, dtype = tf.float32)

    return image, depth

def batch(batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(buffer_size=length, reshuffle_each_iteration=True).repeat()
    dataset = dataset.map(map_func=parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return length, dataset.batch(batch_size=batch_size)
    