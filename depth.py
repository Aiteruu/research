from model import Depth
from sidodpipeline import batch
from loss import loss
import tensorflow as tf 
import os

batch_size = 20
learning_rate = 0.0001
epochs = 10

length, data_generator = batch(batch_size)

def get_compiled_model():
    model = Depth()
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=learning_rate, amsgrad=True))
    return model
    
checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

model = get_compiled_model()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_dir + "/ckpt-loss={epoch:03d} {loss:.2f}.h5", save_freq=10, verbose=1 
    )
]


model.fit(data_generator, epochs=epochs, steps_per_epoch=length//batch_size, callbacks=callbacks)
