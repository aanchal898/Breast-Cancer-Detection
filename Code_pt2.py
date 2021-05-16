import tensorflow as tf
from tensorflow.keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator

#train set
train_datagen = ImageDataGenerator(
        rescale=1./255, #behaves like feature scaling
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        '...\\training_set',
        target_size=(50, 50),
        batch_size=32,
        class_mode='binary')

#test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        '...\\test_set',
        target_size=(50, 50),
        batch_size=32,
        class_mode='binary')

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import regularizers
from keras.layers import Dropout

classifier=tf.keras.models.Sequential()

#Sequencing 8 3CL+4AL+BN

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
classifier = Sequential()
#1st Conv layer
classifier.add(Convolution2D(128, (3, 3), input_shape=(50, 50, 3), strides=(2,2), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(3,3)))
#2nd Conv layer
classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#3rd Conv layer
classifier.add(Convolution2D(32, (2, 2), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(BatchNormalization())

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(BatchNormalization())

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(BatchNormalization())

classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting dataset

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('...\\training_set',
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('...\\test_set',
                                            target_size = (50, 50),
                                            batch_size = 32,
                                            class_mode = 'binary')
#steps_per_epoch = number of images in training set / batch size (which is 55839/32)
#validation_steps = number of images in test set / batch size (which is 18739/32)

classifier.fit_generator(
        training_set,
        steps_per_epoch=(47858+109248)/32,
        epochs=15,
        validation_data=test_set,
        validation_steps=(15756+39748)/32)

#Sequencing 15 5CL+6ANN

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier = Sequential()
#1st Conv layera
classifier.add(Convolution2D(128, (2, 2), input_shape = (50, 50, 3),strides=(2,2), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#2nd Conv layer
classifier.add(Convolution2D(64, (2, 2),strides=(1,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#3rd Conv layer
classifier.add(Convolution2D(32, (1, 1),strides=(1,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))
#4th Conv layer
classifier.add(Convolution2D(32, (1, 1),strides=(1,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))
#5th Conv layer
classifier.add(Convolution2D(16, (1, 1),strides=(1,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))

#Flattening
classifier.add(Flatten())

#Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting dataset
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('...\\training_set',
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('...\\test_set',
                                            target_size = (50, 50),
                                            batch_size = 32,
                                            class_mode = 'binary')
classifier.fit_generator(
    training_set,
    steps_per_epoch = (47858+109248)/32,
    epochs = 15,
    validation_data = test_set,
    validation_steps = (15756+39748)/32)

#Sequencing 14 5CL+5AL

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import regularizers
from keras.layers import Dropout

classifier = Sequential()
#1st Conv layer
classifier.add(Convolution2D(128, (2, 2), input_shape=(50, 50, 3),strides=(2,2), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#2nd Conv layer
classifier.add(Convolution2D(64, (2, 2),strides=(1,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#3rd Conv layer
classifier.add(Convolution2D(32, (2,2),strides=(1,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#4th Conv layer
classifier.add(Convolution2D(32, (2, 2),strides=(1,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))
#5th Conv layer
classifier.add(Convolution2D(16, (1, 1),strides=(1,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))

#Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting dataset

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('...\\training_set',
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('...\\test_set',
                                            target_size = (50, 50),
                                            batch_size = 32,
                                            class_mode = 'binary')
#steps_per_epoch = number of images in training set / batch size (which is 55839/32)
#validation_steps = number of images in test set / batch size (which is 18739/32)

classifier.fit_generator(
        training_set,
        steps_per_epoch=(47858+109248)/32,
        epochs=15,
        validation_data=test_set,
        validation_steps=(15756+39748)/32)

#Sequencing 12 4CL+5AL
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier = Sequential()
#1st Conv layera
classifier.add(Convolution2D(128, (3, 3), input_shape = (50, 50, 3),strides=(2,2), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#2nd Conv layer
classifier.add(Convolution2D(64, (2, 2),strides=(2,2), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))
#3rd Conv layer
classifier.add(Convolution2D(32, (1, 1),strides=(1,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))
#4th Conv layer
classifier.add(Convolution2D(16, (1, 1),strides=(1,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))

#Flattening
classifier.add(Flatten())

#Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting dataset
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('...\\training_set',
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('...\\test_set',
                                            target_size = (50,50),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(
    training_set,
    steps_per_epoch = (47858+109248)/32,
    epochs = 15,
    validation_data = test_set,
    validation_steps =(15756+39748)/32)

#Sequencing 11 4CL+5AL+L1+L2

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import regularizers

classifier = Sequential()
#1st Conv layer
classifier.add(Convolution2D(128, (3,3), input_shape = (50, 50, 3),strides=(2,2), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#2nd Conv layer
classifier.add(Convolution2D(64, (2, 2),strides=(1,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))
#3rd Conv layer
classifier.add(Convolution2D(32, (1, 1),strides=(1,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))
#4th Conv layer
classifier.add(Convolution2D(16, (1, 1),strides=(1,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))

#Flattening
classifier.add(Flatten())

#Full Connection
classifier.add(Dense(units = 128, kernel_regularizer = regularizers.l2(0.01), activation = 'relu'))
classifier.add(Dense(units = 128, activity_regularizer = regularizers.l1(0.01), activation = 'relu'))
classifier.add(Dense(units = 128, kernel_regularizer = regularizers.l2(0.01), activity_regularizer = regularizers.l1(0.01), activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting dataset
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('...\\training_set',
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('...\\test_set',
                                            target_size = (50, 50),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(
    training_set,
    steps_per_epoch =(47858+109248)/32,
    epochs = 15,
    validation_data = test_set,
    validation_steps = (15756+39748)/32)

#Sequencing 10 4CL+4ANN+DO+L1
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import regularizers
from keras.layers import Dropout

classifier = Sequential()
#1st Conv layer
classifier.add(Convolution2D(128, (2, 2), input_shape=(50, 50, 3),strides=(2,2), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#2nd Conv layer
classifier.add(Convolution2D(64, (2, 2),strides=(1,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))
#3rd Conv layer
classifier.add(Convolution2D(32, (1, 1),strides=(1,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))
#4th Conv layer
classifier.add(Convolution2D(16, (1, 1),strides=(1,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))

#Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activity_regularizer = regularizers.l1(0.01), activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 128, activity_regularizer = regularizers.l1(0.01), activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting dataset

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('...\\training_set',
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('...\\test_set',
                                            target_size = (50, 50),
                                            batch_size = 32,
                                            class_mode = 'binary')
#steps_per_epoch = number of images in training set / batch size (which is 55839/32)
#validation_steps = number of images in test set / batch size (which is 18739/32)

classifier.fit_generator(
        training_set,
        steps_per_epoch=(47858+109248)/32,
        epochs=15,
        validation_data=test_set,
        validation_steps=(15756+39748)/32)

#Sequencing 9 3CL+5AL+L2

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers

classifier = Sequential()
#1st Conv layer
classifier.add(Convolution2D(128, (3, 3), input_shape = (50, 50, 3),strides=(2,2), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#2nd Conv layer
classifier.add(Convolution2D(64, (3, 3),strides=(1,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))
#3rd Conv layer
classifier.add(Convolution2D(32, (1, 1),strides=(1,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))

#Flattening
classifier.add(Flatten())

#Full Connection
classifier.add(Dense(units = 128, kernel_regularizer = regularizers.l2(0.01), activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, kernel_regularizer = regularizers.l2(0.01), activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting dataset
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('...\\training_set',
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('...\\test_set',
                                            target_size = (50, 50),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(
    training_set,
    steps_per_epoch = (47858+109248)/32,
    epochs = 15,
    validation_data = test_set,
    validation_steps = (15756+39748)/32)
