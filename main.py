import os
import sys

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten,\
                         Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,\
                         UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import optimizers
from keras import callbacks

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
import pickle 

import tensorflow as tf
import matplotlib.pyplot as plt

import alex

from keras.models import load_model


physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

######### Config #########

# New,Transfer of Load
train = "New"
# Model to load in case load is chosen
load  = "Tr" 

data_augmentation_level = 2
batch_size  = 64

if load != "alex":
  img_height = 227
  img_width  = 227
else:
  img_height = 160
  img_width  = 160

input_shape = (img_height,img_width,3)
trainingset_dir = "dataset"
epochs=32

#################

################

models_dir = 'models/'
##########################

######################### Dataset ############################
train_datagen = ImageDataGenerator(
      rescale = 1. / 255,\
      zoom_range=0.1,\
      rotation_range=20,\
      width_shift_range=0.1,\
      height_shift_range=0.1,\
      horizontal_flip=True,\
      vertical_flip=False,
      validation_split=0.2)

train_shuffle = True

train_generator = train_datagen.flow_from_directory(
    directory=trainingset_dir,
    target_size=(227, 227),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=train_shuffle)

num_samples = train_generator.n
num_classes = train_generator.num_classes

classnames = [k for k,v in train_generator.class_indices.items()]
print("Image input %s" %str(input_shape))
print("Classes: %r" %classnames)

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
test_ds = val_ds.take(val_batches // 5)
val_ds  = val_ds.skip(val_batches // 5)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential(
    [
      layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                  input_shape=input_shape),
      layers.experimental.preprocessing.RandomRotation(0.1),
      layers.experimental.preprocessing.RandomZoom(0.1),
    ]
  )
############################################################################

############################################################################
def loadmodel(problem):
    filename = os.path.join(models_dir, '%s.h5' %problem)
    try:
        model = load_model(filename)
        print("\nModel loaded successfully from file %s\n" %filename)
    except OSError:    
        print("\nModel file %s not found!!!\n" %filename)
        model = None
    return model

def savemodel(model,problem):
    filename = os.path.join(models_dir, '%s.h5' %problem)
    model.save(filename)
    print("\nModel saved on file %s\n" %filename)
###########################################################################


########################## Main function #########################
def execute(train=train,load=load):
  if train== "New":
    history,epochs = alex.execute()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and validation Loss')
    plt.show()
  
  elif train == "Transfer":
    IMG_SIZE = (160,160)
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
    #base_model.summary()
    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer=layers.Dense(num_classes,activation='softmax')

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    initial_epochs = 10
    history = model.fit(train_ds,
                      epochs=initial_epochs,
                      validation_data=val_ds)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and validation Loss')
    plt.xlabel('epoch')
    plt.show()
    

    base_model.trainable = True
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))
    # Fine-tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable =  False

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
                metrics=['accuracy'])

    model.summary()

    fine_tune_epochs = 10
    total_epochs =  initial_epochs + fine_tune_epochs

    history_fine = model.fit(train_ds,
                          epochs=total_epochs,
                          initial_epoch=history.epoch[-1],
                          validation_data=val_ds)

    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([initial_epochs-1,initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs-1,initial_epochs-1],
    plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and validation Loss')
    plt.xlabel('epoch')
    plt.show()
    savemodel(model,"Tr")
  else:
    if   load == "Cl":
      model = tf.keras.models.load_model('models/Cl.h5')
    elif load == "Tr":
      model = tf.keras.models.load_model('models/Tr.h5')
    else:
      model = tf.keras.models.load_model('models/my_model.h5')

  ################ Verify #####################

  
  #Retrieve a batch of images from the test set
  loss0, accuracy0 = model.evaluate(val_ds)
  image_batch, label_batch = test_ds.as_numpy_iterator().next()

  print("loss: {:.2f}".format(loss0))
  print("accuracy: {:.2f}".format(accuracy0))


  predictions = model.predict(image_batch)
  score = []
  for p in predictions:
    score.append(tf.nn.softmax(p))

  plt.figure(figsize=(10, 10))
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(classnames[np.argmax(score[i])]+"/"+classnames[label_batch[i]])
    plt.axis("off")


  img = tf.keras.preprocessing.image.load_img(
      "test/juice.jpg", target_size=(img_height, img_width)
  )
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])


  def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label, img
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(classnames[predicted_label],
                                  100*np.max(score),
                                  "class"),
                                  color=color)

  def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label
    plt.grid(False)
    plt.xticks(range(8))
    plt.yticks([])
    thisplot = plt.bar(range(8), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(classnames[np.argmax(score)], 100 * np.max(score))
  )

  i = 0
  plt.figure(figsize=(6,3))
  plt.subplot(1,2,1)
  plot_image(i, predictions[i], "class", img)
  plt.subplot(1,2,2)
  plot_value_array(i, predictions[i], 0)
  plt.show()