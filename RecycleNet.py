from keras.layers import Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras

# Dim of image
#img_rows = 384
#img_col = 512

# 6 classes
num_classes = 24

# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
#
# model.add(Conv2D(32, (3, 3), data_format="channels_first"))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
#
# model.add(Conv2D(64, (3, 3), data_format="channels_first"))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='rmsprop',
              metrics=['accuracy'])

batch_size = 16
epochs = 100
print("epochs", epochs)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data/recycling2_train',  # this is the target directory
        target_size=(256, 256),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/recycling2_validation',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=2400 // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=50 // batch_size)

model.save_weights('first_try.h5')  # always save your weights after training or during training
model.save('first_run.h5')


