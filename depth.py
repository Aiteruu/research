from model import Depth
from data import batch
from loss import loss
import tensorflow as tf 
import os

batch_size = 8
learning_rate = 0.0001
epochs = 10

model = Depth()
length, data_generator = batch(8)

model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=learning_rate, amsgrad=True))
outputFolder = './output'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
filepath=outputFolder+"/model-{epoch:02d}-{val_accuracy:.2f}.hdf5"

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_weights_only=True, save_frequency=1)

model.fit(data_generator, epochs=epochs, steps_per_epoch=length//batch_size, callbacks=[checkpoint_callback])