# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras import Model
from glob import glob
import os
import pathlib
import random

# %%
"""
### Step1: Becoming one with data
"""

# %%
"""
* lets understand how our data is present
"""

# %%
## walk through the data directory and list the number of files

for dirpath, dirnames, filename in os.walk("Datasets"):
    print(f"There are {len(dirnames)} directories and {len(filename)} files in {dirpath}")

# %%
# number of train images in data
audi_count = len(os.listdir("D:\\car-brand-classification\\Datasets\\Train\\audi"))
lamborghini_count = len(os.listdir("D:\\car-brand-classification\\Datasets\\Train\\lamborghini"))
mercedes_count = len(os.listdir("D:\\car-brand-classification\\Datasets\\Train\\mercedes"))
print(f"There are {audi_count} audi images")
print(f"There are {lamborghini_count} lamborghini images")
print(f"There are {mercedes_count} mercedes images")

# %%
# let's get the class names programatically
class_names = []
data_dir = pathlib.Path("D:\\car-brand-classification\\Datasets\\Train")
for item in data_dir.glob('*'):
    class_names.append(item.name)
class_names = np.array(class_names)
print(class_names)

# %%
type(class_names)

# %%
"""
### 2. visualize the data
"""


# %%
def visualize_data(train_directory, target_class):
    target_folder = train_directory + target_class

    # picking up a random image
    image = random.sample(os.listdir(target_folder), 1)
    image = image[0]

    img = plt.imread(target_folder + "/" + image)
    plt.imshow(img)
    plt.show()


# %%
for i in range(10):
    img = visualize_data("D:\\car-brand-classification\\Datasets\\Train\\", "audi")

# %%
for i in range(10):
    img = visualize_data("D:\\car-brand-classification\\Datasets\\Train\\", "mercedes")

# %%
for i in range(10):
    img = visualize_data("D:\\car-brand-classification\\Datasets\\Train\\", "lamborghini")

# %%
"""
### Data preprocessing
"""

# %%
# resize all the image to this
IMAGE_SIZE = [224, 224]
train_dir = "D:\\car-brand-classification\\Datasets\\Train"
test_dir = "D:\\car-brand-classification\\Datasets\\Test"
[224, 224] + [3]

# %%
## Impport the Resnet library as shown below and add preprocessing layer to front of vgg

resnet = ResNet50(input_shape=IMAGE_SIZE + [3],
                  weights="imagenet",
                  include_top=False)

# %%
resnet.summary()

# %%
# DOn't train existing weights
for layer in resnet.layers:
    layer.trainable = False

# %%
# getting the number of output classes
folders = glob('D:\\car-brand-classification\\Datasets\\Train\\*')
print(folders)

# %%
## adding our end layers to resnet
X = Flatten()(resnet.output)
prediction = Dense(len(folders), activation="softmax")(X)
model = Model(inputs=resnet.input, outputs=prediction)

# %%
model.summary()

# %%
"""
#### WE can observe that we have added our flatten and output Dense layer with softmax activation to the end of Resnet50 model
"""

# %%
# compiling the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# %%
# use the ImageDatagenerator to import images from dataset

train_datagen_augmented = ImageDataGenerator(featurewise_center=False,
                                             samplewise_center=False,
                                             featurewise_std_normalization=False,
                                             samplewise_std_normalization=False,
                                             zca_whitening=False,
                                             zca_epsilon=1e-06,
                                             rotation_range=0,
                                             width_shift_range=0.0,
                                             height_shift_range=0.0,
                                             brightness_range=None,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             channel_shift_range=0.0,
                                             fill_mode='nearest',
                                             cval=0.0,
                                             horizontal_flip=True,
                                             vertical_flip=False,
                                             rescale=1. / 255,
                                             preprocessing_function=None,
                                             data_format=None,
                                             validation_split=0.0,
                                             dtype=None)

# make sure to provide the same target size as initialized in the model
training_set_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                     target_size=(224, 224),
                                                                     batch_size=32,
                                                                     class_mode='categorical',
                                                                     shuffle=False)

# non augmented training data
train_datagen = ImageDataGenerator(rescale=1. / 255)
training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=False)

# %%
images, labels = training_set.next()
print(len(images))
augment_images, augment_labels = training_set_augmented.next()
print(len(images))

# %%
"""
##### We are getting a total of 32 images in each of training and augmented images because we have initialized batch size to 32 and im flow_from directory we have given shuffle = false because we want to compare augmented and original images
"""


# %%
def show_comparision():
    random_number = random.randint(0, 31)
    plt.figure(figsize=[10, 5])
    plt.subplot(1, 2, 1)
    plt.imshow(images[random_number])
    plt.title("Original_image")

    plt.subplot(1, 2, 2)
    plt.imshow(augment_images[random_number])
    plt.title("augmented_image")


for i in range(20):
    show_comparision()

# %%
### now let's prepare data to be fitted for our model
train_datagen_augmented = ImageDataGenerator(featurewise_center=False,
                                             samplewise_center=False,
                                             featurewise_std_normalization=False,
                                             samplewise_std_normalization=False,
                                             zca_whitening=False,
                                             zca_epsilon=1e-06,
                                             rotation_range=0,
                                             width_shift_range=0.0,
                                             height_shift_range=0.0,
                                             brightness_range=None,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             channel_shift_range=0.0,
                                             fill_mode='nearest',
                                             cval=0.0,
                                             horizontal_flip=True,
                                             vertical_flip=False,
                                             rescale=1. / 255,
                                             preprocessing_function=None,
                                             data_format=None,
                                             validation_split=0.0,
                                             dtype=None)

# make sure to provide the same target size as initialized in the model
training_set_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                     target_size=(224, 224),
                                                                     batch_size=32,
                                                                     class_mode='categorical',
                                                                     shuffle=True)

test_data_gen = ImageDataGenerator(rescale=1. / 255)
test_data = test_data_gen.flow_from_directory(test_dir,
                                              target_size=(224, 224),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=True)

# %%
model1 = model.fit_generator(training_set_augmented,
                             epochs=25,
                             steps_per_epoch=len(training_set_augmented))


def plot_curve(history):
    train_loss = history.history['loss']
    # valid_loss = history.history['val_loss']

    train_accuracy = history.history['accuracy']
    # test_accuracy = history.history['val_accuracy']

    epochs = range(len(train_loss))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='train_loss')
    # plt.plot(epochs, valid_loss, label='valid_loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='train_accuracy')
    # plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.legend()


plot_curve(model1)

# y_pred = model.predict(test_data)
# y_pred

# model.evaluate(test_data)
# y_pred = tf.argmax(y_pred, axis=1)
# print(y_pred)

model.save('model.h5')

'''def load_prep_image(filename, img_shape=224):
    img = tf.io.read_file(filename)
    # Decode the read file to tensor
    img = tf.image.decode_jpeg(img)
    # resize the image
    img = tf.image.resize(img, size=[img_shape, img_shape])
    # rescale the image
    img = img/255.
    return img

def pred_and_plot(model, filename, classnames=class_names):
    img = load_prep_image(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))
    pred_class = class_names[int(tf.round(pred))]
    
    # plot the image and predicted class
    plt.imshow(img)
    plt.title(f"prediction {pred_class}")
    plt.axis("off")

pred_and_plot(model, '1.jpeg')'''
