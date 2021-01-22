import tensorflow as tf

csv = open('dataset/train.csv', 'r').read()
sidod = list((row.split(',') for row in (csv).split('\n') if len(row) > 0))

rgb_shape = (960, 540, 3)
depth_shape = (480, 270, 1)

filenames = [i[0] for i in sidod]
labels = [i[1] for i in sidod]

length = len(labels)

def parse(filename, label):
    image = tf.image.decode_png(tf.io.read_file('dataset/' + filename), channels=3)
    depth = tf.image.resize(tf.image.decode_png(tf.io.read_file('dataset/' + label), channels=1), [depth_shape[0], depth_shape[1]])

    image = tf.image.convert_image_dtype(image, dtype = tf.float32)
    depth = tf.image.convert_image_dtype(depth, dtype = tf.float32)
    depth = tf.divide(
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
    split = int(.8 * len(filenames))
    dataset = tf.random.shuffle(dataset)

    filenames_train, filenames_val = filenames[:split], filenames[split:]
    labels_train, labels_val = labels[:split], labels[split:]

    dataset_train = tf.data.Dataset.from_tensor_slices((filenames_train, labels_train))
    dataset_val = tf.data.Dataset.from_tensor_slices((filenames_val, labels_val))

    dataset_train = dataset_train.shuffle(buffer_size=length, reshuffle_each_iteration=True).repeat()
    dataset_val = dataset_val.shuffle(buffer_size=length, reshuffle_each_iteration=True).repeat()

    dataset_train = dataset_train.map(map_func=parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_val = dataset_val.map(map_func=parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return length, split, dataset_train.batch(batch_size=batch_size), dataset_val.batch(batch_size=batch_size)
    
