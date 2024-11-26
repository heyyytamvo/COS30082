import tensorflow as tf
from tensorflow.keras import layers, models
from keras.models import load_model
from PIL import Image
import numpy as np


IMAGE_SIZE=224
model = load_model('model_liveness_crop.h5')
real = Image.open("tam_real_crop.png").convert("RGB")
fake = Image.open("tam_fake_crop.png").convert("RGB")
real_resized = np.array(real.resize((IMAGE_SIZE, IMAGE_SIZE))) 
fake_resized = np.array(fake.resize((IMAGE_SIZE, IMAGE_SIZE))) 
real_resized_batch = np.expand_dims(real_resized, axis=0) / 255.0
fake_resized_batch = np.expand_dims(fake_resized, axis=0) / 255.0
result_real = model.predict(real_resized_batch)
result_fake = model.predict(fake_resized_batch)
print(result_real)
print(result_fake)