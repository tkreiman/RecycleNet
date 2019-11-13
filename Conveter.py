from keras.utils.generic_utils import CustomObjectScope
import keras

with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    from keras.models import load_model
    import coremltools


    output_labels = ['paperCup', 'waterbottle']
    your_model = coremltools.converters.keras.convert('transfer1.h5', input_names=['image'], output_names=['output'], class_labels=output_labels, image_input_names='image')

    # your_model.author = 'your name'
    # your_model.short_description = 'Digit Recognition with MNIST'
    # your_model.input_description['image'] = 'Takes as input an image'
    # your_model.output_description['output'] = 'Prediction of Digit

    your_model.save('transfer1.mlmodel')