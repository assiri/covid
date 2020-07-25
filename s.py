import tensorflow as tf
from tensorflow import keras
model = keras.applications.mobilenet.MobileNet()
model.save('./model.h5')
