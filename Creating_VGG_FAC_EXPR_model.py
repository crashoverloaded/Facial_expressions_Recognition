#!/usr/bin/python3


# Importing the libraries
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense , Dropout , Activation , Flatten , BatchNormalization ,  Conv2D  ,MaxPooling2D
import os

# No.of classes - angry , sad , happy ,etc....
num_classes = 5

# Image Shape
img_rows , img_cols = 48 , 48

batch_size = 32

# Training Repo Path
train_data_dir = 'PATH'
validation_data_dir = 'PATH'

# DATA AUgmentation 
# Training Data Generator
train_datagen = ImageDataGenerator(
			rescale = 1./255 , 
			rotation_range = 30,
			shear_range = 0.3 , 
			zoom_range = 0.3,
			width_shift_range = 0.4 , 
			height_shift_range = 0.4 , 
			horizontal_flip = True,
			fill_mode = "nearest")

# Validation Data
validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode = 'grayscale',
					target_size = (img_rows, img_cols),
					batch_size = batch_size,
					class_mode = 'categorical',
					shuffle =True)				


validation_generator = validation_datagen.flow_from_directory(
					validation_data_dir,
					color_mode = 'grayscale',
					target_size = (img_rows, img_cols),
					batch_size = batch_size,
					class_mode = 'categorical',
					shuffle =True)				
# Building Model

model = Sequential()

# BLOCK 1 


model.add(Conv2D(32 , (3,3) , padding='same',kernel_initializer='he_normal' , input_shape= (img_rows, img_cols , 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32 , (3,3) , padding='same',kernel_initializer='he_normal' , input_shape= (img_rows, img_cols , 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


# BLOCK 2

model.add(Conv2D(64 , (3,3) , padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64 , (3,3) , padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# BLOCK 3

model.add(Conv2D(128 , (3,3) , padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128 , (3,3) , padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# BLOCK 4

model.add(Conv2D(256 , (3,3) , padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256 , (3,3) , padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# BLOCK 5

model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block 6

model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block 7

model.add(Dense(num_classes , kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())


# COMPILING AND CALLBACKS

from keras.optimizers import RMSprop , Adam , SGD
from keras.callbacks import ModelCheckpoint , EarlyStopping , ReduceLROnPlateau

# Save the best weights in Emotion_little_vgg.h5 file

checkpoint = ModelCheckpoint('Emotion_little_vgg.h5' ,
				monitor = 'val_loss' ,
				mode = 'min' ,
				save_best_only = True ,
				verbose =1)

# If model accuracy not improving than stop
earlystop = EarlyStopping(monitor = 'val_loss' , 
				min_delta = 0 ,
				patience = 3, 
				verbose=1 ,
				restore_best_weights = True)

# Reduce the Learning Rate if training reaches the Plateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss' ,
				factor = 0.3 ,
				patience = 3 ,
				verbose = 1 , 
				min_delta = 0.0001)

# All the Callbacks
callbacks = [earlystop , checkpoint , reduce_lr]

# COmpiling Model 
model.compile(loss = 'categorical_crossentropy' , optimizers = Adam(lr=0.001) , metrics=['accuracy'])

# No.of samples

nb_train_samples = 24176
nb_validation_samples = 3006
epochs = 25


# Fitting the model 

history = model.fit_generator(
			train_generator ,
			steps_per_epoch = nb_train_samples//batch_size , 
			epochs = epochs , 
			callbacks = callbacks , 
			validation_data = validation_generator,
			validation_steps = nb_validation_samples // batch_size)
