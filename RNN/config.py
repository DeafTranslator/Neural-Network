import tensorflow as tf
"""Configuration and Hyperparameters"""

# Convolutional Layer 1.
filter_size1 = 3 
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 224

learning_rate = 0.0001

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info
# classes = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
classes = ['a', 'c', 'd', 'e', 'f', 'l']
num_classes = len(classes)

#batch size 
batch_size = 2

#iterations
iterations = 50

# validation split
validation_size = .2

# how long to wait after validation loss stops improving before terminating training
early_stopping = False  # use None if you dont want to implement early stoping

train_path = 'C:\\Users\\Juan Graciano\\Desktop\\Proyecto\\Dataset\\datasetTODO\\mujerePuca\\rnntest\\train'
test_path = 'C:\\Users\\Juan Graciano\\Desktop\\Proyecto\\Dataset\\datasetTODO\\mujerePuca\\rnntest\\test'
# checkpoint_dir = "C:/Projects/playground/tensorflow/tf_image_clf/models/"
# print train_path

model_path = 'C:\\Users\\Juan Graciano\\Desktop\\Proyecto\\Neural-network\\RNN\\temp'


