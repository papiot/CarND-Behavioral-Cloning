import csv
import cv2

import numpy as np

lines = []

with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

images = []
measurements = []

print("Getting data....")
correction = 0.8
for line in lines:
  for i in range(3):
    source_path = line[i]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
 
    image = cv2.imread(current_path)
    images.append(image)

    measurement = float(line[3])

    if (i == 1):
      measurement = measurement + correction
    
    if (i == 2):
      measurement = measurement - correction

    measurements.append(measurement)


augmentated_images, augmentated_measurements = [], []

for image, measurement in zip(images, measurements):
  augmentated_images.append(image)
  augmentated_measurements.append(measurement)
  augmentated_images.append(cv2.flip(image, 1))
  augmentated_measurements.append(measurement * -1)

X_train = np.array(augmentated_images)
y_train = np.array(augmentated_measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Nvidia's CNN architecture
# Taken from here: https://devblogs.nvidia.com/deep-learning-self-driving-cars/
# with modifications to fit our data
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=X_train[0].shape))

# Normalize the image
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# Use 0.2 dropout (tried different numbers, 0.2 worked best)
model.add(Dropout(0.2))


model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())

# Fully connected layers, with 1 the final output
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20)

model.save('model.h5')
