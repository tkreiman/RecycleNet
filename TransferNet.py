from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam

HEIGHT = 224
WIDTH = 224

base_model = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(HEIGHT, WIDTH, 3))



TRAIN_DIR = "data/TransferTrainData"
BATCH_SIZE = 32

train_datagen =  ImageDataGenerator(
                                    preprocessing_function=preprocess_input,
                                    rotation_range=90,
                                    horizontal_flip=True,
                                    vertical_flip=True
                                    )

train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    target_size=(HEIGHT, WIDTH),
                                                    batch_size=BATCH_SIZE)


def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    
    model = Sequential()
    
    for layer in base_model.layers:
        layer.trainable = False
        model.add(layer)
    
    #x = base_model.output
    #x = Flatten()(x)
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
        #for fc in fc_layers:
        # New FC layer, random init
        #x = Dense(fc, activation='relu')(x)
#x = Dropout(dropout)(x)

    # New softmax layer
#   predictions = Dense(num_classes, activation='softmax')(x)

#   finetune_model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

class_list = ["cup", "bottle"]
FC_LAYERS = [1024, 1024]
dropout = 0.5

finetune_model = build_finetune_model(base_model,
                                      dropout=dropout,
                                      fc_layers=FC_LAYERS,
                                      num_classes=len(class_list))


NUM_EPOCHS = 10
num_train_images = 2520

adam = Adam(lr=0.00001)
finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

filepath="./checkpoints/" + "ResNet50" + "_model_weights.h5"
#checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
#callbacks_list = [checkpoint]

#history = finetune_model.fit_generator(train_generator, epochs=NUM_EPOCHS, steps_per_epoch=2520/32)

finetune_model.fit_generator(train_generator, epochs=NUM_EPOCHS, steps_per_epoch=2520/32)
