from model import Depth
from data import batch
from loss import loss
import tensorflow as tf 
import os

batch_size = 4
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

def make_or_restore_model():
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoints:
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()


model = make_or_restore_model()
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + "/ckpt-loss={loss:.2f}", save_freq=1000, verbose=1 
    )
]


model.fit(data_generator, epochs=epochs, steps_per_epoch=length//batch_size, callbacks=callbacks)