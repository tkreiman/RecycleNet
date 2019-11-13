from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from keras.preprocessing import image
import keras
import numpy as np

HEIGHT = 224
WIDTH = 224

with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    model = load_model('transfer1.h5')

    #img = image.load_img('data/TransferTrainData/n04557648_waterbottle/n04557648_9415.JPEG', target_size=(WIDTH, HEIGHT))
    img = image.load_img('data/TransferTrainData/n04557648_waterbottle/n04557648_9553.JPEG', target_size=(WIDTH, HEIGHT))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images)
    print (classes)
