from model import Depth
from sidodpipeline import batch
from loss import loss
import tensorflow as tf 
import os
import time

batch_size = 20
learning_rate = 0.0001
epochs = 30
NAME = "depth-sidod-cnn{}".format(int(time.time()))


length, data_generator = batch(batch_size)

def get_compiled_model():
    model = Depth()
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=learning_rate, amsgrad=True))
    return model
    
checkpoint_dir = "./checkpoint/{}".format(NAME)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

model = get_compiled_model()

checkpoint_filepath = '/best/{}'.format(NAME)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_dir + "/ckpt-loss={epoch:03d} {val_loss:.2f}", save_freq=10, verbose=1 
    ),
    tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NAME)),
    model_checkpoint_callback
]


model.fit(data_generator, epochs=epochs, callbacks=callbacks, validation_split=.2)
