import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from keras import optimizers, losses

# System libraries
from pathlib import Path
import os.path

# Metrics
import itertools

train_data = '../ClassificationV2/dataFruit/train'
val_data = '../ClassificationV2/dataFruit/test'
test_data = '../ClassificationV2/dataFruit/predict'

def display_random_images(train_data_path):
    # Get the list of subdirectories (each representing a class)
    classes = os.listdir(train_data_path)

    # Create a grid of images
    plt.figure(figsize=(20, 20))

    # Loop through each class folder and display a random image
    for i, class_name in enumerate(classes):
        class_path = os.path.join(train_data_path, class_name)
        class_images = os.listdir(class_path)

        # Select a random image from the class
        random_image = random.choice(class_images)

        # Load and display the image using plt.imshow
        image_path = os.path.join(class_path, random_image)
        img = mpimg.imread(image_path)

        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.title(f"fruit: {class_name}")
        plt.axis("off")

    plt.show()

# Display a random image from each fruit
display_random_images(train_data)

# Print the amount of pictures from each fruit
for class_name in os.listdir(train_data):
    class_path = os.path.join(train_data, class_name)
    num_images = len(os.listdir(class_path))
    print(f"{class_name}: {num_images} images")


# Making image augmentators to increase the number of images in the dataset
train_dataGenerator = ImageDataGenerator(rescale=1. / 255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
validation_dataGenerator = ImageDataGenerator(rescale=1. / 255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                              shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
test_dataGenerator = ImageDataGenerator(rescale=1. / 255)

# Using the image augmentators to create new images
train_images = train_dataGenerator.flow_from_directory(directory=train_data, batch_size=32, target_size=(224, 224),
                                                       class_mode="categorical", shuffle=False)
validation_images = validation_dataGenerator.flow_from_directory(directory=val_data, batch_size=32, target_size=(224, 224),
                                                                 class_mode="categorical")
test_images = test_dataGenerator.flow_from_directory(directory=test_data, batch_size=32, target_size=(224, 224),
                                                     class_mode="categorical")

# Load pretrained Model
mobile_model = Sequential()

pretrained_model = tf.keras.applications.mobilenet.MobileNet(include_top=False,
                                                             input_shape=(224, 224, 3),
                                                             pooling='avg', classes=10,
                                                             weights='imagenet')
for layer in pretrained_model.layers:
    layer.trainable = False

mobile_model.add(pretrained_model)

mobile_model.add(Flatten())
mobile_model.add(Dense(512, activation='relu'))
mobile_model.add(Dense(128, activation='relu'))
mobile_model.add(Dropout(0.2))
mobile_model.add(Dense(10, activation='softmax'))

mobile_model.summary()

mobile_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
history = mobile_model.fit(train_images, epochs=20, steps_per_epoch=len(train_images),
                           validation_data=validation_images, validation_steps=None)

# Function to save the trained model
def save_model(model, filename):
    model.save(filename)
    print(f"Model saved as {filename}")

save_model(mobile_model, "fruit_classification_model.h5")

# Data visualizations
def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('loss')
    plt.xlabel('epochs')
    plt.legend()

    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('accuracy')
    plt.xlabel('epochs')
    plt.legend()

plot_loss_curves(history)
mobile_model.evaluate(validation_images)
