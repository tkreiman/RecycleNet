import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D, Dropout, Flatten
from keras.applications import InceptionV3
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


# In[2]:

N_CLASSES = 10

base_model=InceptionV3(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

for layer in base_model.layers:
    layer.trainable = False

x=base_model.output
x=GlobalAveragePooling2D()(x)

x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dropout(0.5)(x)
x=Dense(1024,activation='relu')(x) #dense layer 3
x=Dropout(0.5)(x)
x=Dense(1024,activation='relu')(x) #dense layer 3
x=Dense(512,activation='relu')(x) #dense layer 4
    
preds=Dense(N_CLASSES,activation='softmax')(x) #final layer with softmax activation


# In[3]:


model=Model(inputs=base_model.input,outputs=preds)
#specify the inputs
#specify the outputs
#now a model has been created based on our architecture


# In[4]:
#print("LENGTH", len(model.layers)) - 27 layers
# Freeze first 5 layers
#for layer in model.layers[:20]:
#    layer.trainable=False
#for layer in model.layers[20:]:
 #   layer.trainable=True


# In[5]:


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory('data/TransferTrainData', # this is where you specify the path to the main data folder
                                                  target_size=(224,224),
                                                  color_mode='rgb',
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  shuffle=True)


# In[33]:


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

#checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=20)

model.save('InceptionTransfer1.h5')

