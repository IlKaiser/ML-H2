import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers

def execute():
    EPOCHS = 100
    BATCH_SIZE   = batch_size = 64
    image_height = img_height = 227
    image_width  = img_width  = 227
    train_dir = trainingset_dir = "dataset/"
    train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(image_height, image_width),
                                                        color_mode="rgb",
                                                        batch_size=BATCH_SIZE,
                                                        class_mode="categorical",
                                                        shuffle=True,
                                                        subset="training")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        trainingset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        trainingset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_batches = tf.data.experimental.cardinality(val_ds)
    #test_ds = val_ds.take(val_batches // 5)
    val_ds  = val_ds.skip(val_batches // 5)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)    
    
    num_classes = train_generator.num_classes
    input_shape = (image_height,image_width,3)
    data_augmentation = tf.keras.Sequential(
    [
      layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                  input_shape=input_shape),
      layers.experimental.preprocessing.RandomRotation(0.1),
      layers.experimental.preprocessing.RandomZoom(0.1),
    ]
  )
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        data_augmentation,
        layers.Conv2D(96, kernel_size=(11,11), strides= 4,
                        padding= 'valid', activation= 'relu',
                        input_shape= input_shape,
                        kernel_initializer= 'he_normal'),
        layers.MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None),

        layers.Conv2D(256, kernel_size=(5,5), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'),
        layers.MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None), 

        layers.Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'),

        layers.Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'),

        layers.Conv2D(256, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'),

        layers.MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None),
        
        layers.Dropout(0.2),


        layers.Flatten(),
        layers.Dense(4096, activation= 'relu'),
        layers.Dense(4096, activation= 'relu'),
        layers.Dense(1000, activation= 'relu'),
        layers.Dense(num_classes, activation= 'softmax')
    ])
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    model.summary()

    # start training
    history = model.fit(train_ds,
                        epochs=EPOCHS,
                        validation_data=val_ds)


    model_dir = "models/alex.h5" 
    model.save(model_dir)
    return history,EPOCHS