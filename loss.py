import tensorflow as tf

def loss(y_true, y_pred, maxDepth=100.0):
    depth = tf.math.reduce_mean(tf.math.abs(y_true - y_pred), axis=-1)
    ssim = tf.clip_by_value(1 - tf.image.ssim(y_true, y_pred, maxDepth) * 0.5, 0, 1)
    dy, dx = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    edges = tf.math.reduce_mean(tf.math.abs(dy_pred - dy)) + tf.math.reduce_mean(tf.math.abs(dx_pred - dx))

    return ssim + tf.reduce_mean(edges) + .1 * tf.reduce_mean(depth)