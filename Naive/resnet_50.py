# -*- coding: utf-8 -*-

!pip uninstall --yes h5py

# DEPENDENCIES ########################################################################################################################################

!pip install keras
!pip install keras_vggface
!pip install pandas
!pip install scikit_image
!pip install h5py

# IMPORTS #############################################################################################################################################

import numpy as np
import pandas as pd
from tensorflow.python.lib.io import file_io
from skimage.transform import resize
from keras import backend as K
from keras.utils import to_categorical
from keras_vggface.vggface import VGGFace
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, Callback
import h5py # For saving the model

# PARAMETERS ##########################################################################################################################################

# Folder where logs and models are stored
folder = 'logs/ResNet-50'

# Size of the images
img_height, img_width = 197, 197

# Parameters
num_classes         = 7     # ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
epochs_top_layers   = 5
epochs_all_layers   = 100
batch_size          = 128

# DATASETS ############################################################################################################################################

# Folder where logs and models are stored
folder = 'gs://emotion_recognition/logs/ResNet-50'

# Data paths
train_dataset	= 'gs://emotion_recognition/FER-2013/fer2013_train.csv'
eval_dataset 	= 'gs://emotion_recognition/FER-2013/fer2013_eval.csv'

# MODEL ###############################################################################################################################################

# Create the based on ResNet-50 architecture pre-trained model
    # model:        Selects one of the available architectures vgg16, resnet50 or senet50
    # include_top:  Whether to include the fully-connected layer at the top of the network
    # weights:      Pre-training on VGGFace
    # input_shape:  Optional shape tuple, only to be specified if include_top is False (otherwise the input
    #               shape has to be (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with
    #               'channels_first' data format). It should have exactly 3 inputs channels, and width and
    #               height should be no smaller than 197. E.g. (200, 200, 3) would be one valid value.
# Returns a keras Model instance
base_model = VGGFace(
    model       = 'resnet50',
    include_top = False,
    weights     = 'vggface',
    input_shape = (img_height, img_width, 3))

# Places x as the output of the pre-trained model
x = base_model.output

# Flattens the input. Does not affect the batch size
x = Flatten()(x)

# Add a fully-connected layer and a logistic layer
# Dense implements the operation: output = activation(dot(input, kernel) + bias(only applicable if use_bias is True))
    # units:        Positive integer, dimensionality of the output space
    # activation:   Activation function to use
    # input shape:  nD tensor with shape: (batch_size, ..., input_dim)
    # output shape: nD tensor with shape: (batch_size, ..., units)
x = Dense(1024, activation = 'relu')(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

# The model we will train
model = Model(inputs = base_model.input, outputs = predictions)
# model.summary()

# DATA PREPARATION ####################################################################################################################################

# Preprocesses a numpy array encoding a batch of images
    # x: Input array to preprocess
def preprocess_input(x):
    x -= 128.8006 # np.mean(train_dataset)
    return x

# Function that reads the data from the csv file, increases the size of the images and returns the images and their labels
    # dataset: Data path
def get_data(dataset):
    file_stream = file_io.FileIO(dataset, mode='r')
    data = pd.read_csv(file_stream)
    pixels = data['pixels'].tolist()
    images = np.empty((len(data), img_height, img_width, 3))
    i = 0

    for pixel_sequence in pixels:
        single_image = [float(pixel) for pixel in pixel_sequence.split(' ')]  # Extraction of each single
        single_image = np.asarray(single_image).reshape(48, 48) # Dimension: 48x48
        single_image = resize(single_image, (img_height, img_width), order = 3, mode = 'constant') # Dimension: 139x139x3 (Bicubic)
        ret = np.empty((img_height, img_width, 3))
        ret[:, :, 0] = single_image
        ret[:, :, 1] = single_image
        ret[:, :, 2] = single_image
        images[i, :, :, :] = ret
        i += 1

    images = preprocess_input(images)
    labels = to_categorical(data['emotion'])

    return images, labels

# Data preparation
train_data_x, train_data_y  = get_data(train_dataset)
val_data  = get_data(eval_dataset)

# Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches) indefinitely
# rescale:          Rescaling factor (defaults to None). Multiply the data by the value provided (before applying any other transformation)
# rotation_range:   Int. Degree range for random rotations
# shear_range:      Float. Shear Intensity (Shear angle in counter-clockwise direction as radians)
# zoom_range:       Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range]
# fill_mode :       Points outside the boundaries of the input are filled according to the given mode: {"constant", "nearest", "reflect" or "wrap"}
# horizontal_flip:  Boolean. Randomly flip inputs horizontally
train_datagen = ImageDataGenerator(
    rotation_range  = 10,
    shear_range     = 10, # 10 degrees
    zoom_range      = 0.1,
    fill_mode       = 'reflect',
    horizontal_flip = True)

# Takes numpy data & label arrays, and generates batches of augmented/normalized data. Yields batcfillhes indefinitely, in an infinite loop
    # x:            Data. Should have rank 4. In case of grayscale data, the channels axis should have value 1, and in case of RGB data,
    #               it should have value 3
    # y:            Labels
    # batch_size:   Int (default: 32)
train_generator = train_datagen.flow(
    train_data_x,
    train_data_y,
    batch_size  = batch_size)

# UPPER LAYERS TRAINING ###############################################################################################################################

# First: train only the top layers (which were randomly initialized) freezing all convolutional ResNet-50 layers
for layer in base_model.layers:
    layer.trainable = False

# Compile (configures the model for training) the model (should be done *AFTER* setting layers to non-trainable)
    # optimizer:    String (name of optimizer) or optimizer object
        # lr:       Float >= 0. Learning rate
        # beta_1:   Float, 0 < beta < 1. Generally close to 1
        # beta_2:   Float, 0 < beta < 1. Generally close to 1
        # epsilon:  Float >= 0. Fuzz factor
        # decay:    Float >= 0. Learning rate decay over each update
    # loss:     String (name of objective function) or objective function
    # metrics:  List of metrics to be evaluated by the model during training and testing
model.compile(
    optimizer   = Adam(lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0),
    loss        = 'categorical_crossentropy',
    metrics     = ['accuracy'])

# This callback writes a log for TensorBoard, which allows you to visualize dynamic graphs of your training and test metrics,
# as well as activation histograms for the different layers in your model
    # log_dir:          The path of the directory where to save the log files to be parsed by TensorBoard
    # histogram_freq:   Frequency (in epochs) at which to compute activation and weight histograms for the layers of the model
    #                   If set to 0, histograms won't be computed. Validation data (or split) must be specified for histogram visualizations
    # write_graph:      Whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True
    # write_grads:      Whether to visualize gradient histograms in TensorBoard. histogram_freq must be greater than 0
    # write_images:     Whether to write model weights to visualize as image in TensorBoard
# To visualize the files created during training, run in your terminal: tensorboard --logdir path_to_current_dir/Graph
tensorboard_top_layers = TensorBoard(
	log_dir         = folder + '/logs_top_layers',
	histogram_freq  = 0,
	write_graph     = True,
	write_grads     = False,
	write_images    = True)

# Train the model on the new data for a few epochs (Fits the model on data yielded batch-by-batch by a Python generator)
    # generator:        A generator or an instance of Sequence (keras.utils.Sequence) object in order to avoid duplicate data when using multiprocessing
    #                   The output of the generator must be either {a tuple (inputs, targets)} {a tuple (inputs, targets, sample_weights)}
    # steps_per_epoch:  Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch
    #                   It should typically be equal to the number of unique samples of your dataset divided by the batch size
    # epochs:           Integer, total number of iterations on the data
    # validation_data:  This can be either {a generator for the validation data } {a tuple (inputs, targets)} {a tuple (inputs, targets, sample_weights)}
    # callbacks:        List of callbacks to be called during training (to visualize the files created during training, run in your terminal:
    #                   tensorboard --logdir path_to_current_dir/Graph)
model.fit_generator(
    generator           = train_generator,
    steps_per_epoch     = len(train_data_x) // batch_size,  # samples_per_epoch / batch_size
    epochs              = epochs_top_layers,
    validation_data     = val_data,
    callbacks           = [tensorboard_top_layers])

# FULL NETWORK TRAINING ###############################################################################################################################

# At this point, the top layers are well trained and we can start fine-tuning convolutional layers from ResNet-50

# Fine-tuning of all the layers
for layer in model.layers:
    layer.trainable = True

# We need to recompile the model for these modifications to take effect (we use SGD with nesterov momentum and a low learning rate)
    # optimizer:    String (name of optimizer) or optimizer object
        # lr:       float >= 0. Learning rate
        # momentum: float >= 0. Parameter updates momentum
        # decay:    float >= 0. Learning rate decay over each update
        # nesterov: boolean. Whether to apply Nesterov momentum
    # loss:     String (name of objective function) or objective function
    # metrics:  List of metrics to be evaluated by the model during training and testing
model.compile(
    optimizer   = SGD(lr = 1e-4, momentum = 0.9, decay = 0.0, nesterov = True),
    loss        = 'categorical_crossentropy',
    metrics     = ['accuracy'])

# This callback writes a log for TensorBoard, which allows you to visualize dynamic graphs of your training and test metrics,
tensorboard_all_layers = TensorBoard(
    log_dir         = folder + '/logs_all_layers',
    histogram_freq  = 0,
    write_graph     = True,
    write_grads     = False,
    write_images    = True)

def scheduler(epoch):
    updated_lr = K.get_value(model.optimizer.lr) * 0.5
    if (epoch % 3 == 0) and (epoch != 0):
        K.set_value(model.optimizer.lr, updated_lr)
        print(K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)

# Learning rate scheduler
    # schedule: a function that takes an epoch index as input (integer, indexed from 0) and current learning
    #           rate and returns a new learning rate as output (float)
reduce_lr = LearningRateScheduler(scheduler)


# Reduce learning rate when a metric has stopped improving
	# monitor: 	Quantity to be monitored
	# factor: 	Factor by which the learning rate will be reduced. new_lr = lr * factor
	# patience:	Number of epochs with no improvement after which learning rate will be reduced
	# mode: 	One of {auto, min, max}
	# min_lr:	Lower bound on the learning rate
reduce_lr_plateau = ReduceLROnPlateau(
	monitor 	= 'val_loss',
	factor		= 0.5,
	patience	= 3,
	mode 		= 'auto',
	min_lr		= 1e-8)

# Stop training when a monitored quantity has stopped improving
	# monitor:		Quantity to be monitored
	# patience:		Number of epochs with no improvement after which training will be stopped
	# mode: 		One of {auto, min, max}
early_stop = EarlyStopping(
	monitor 	= 'val_loss',
	patience 	= 10,
	mode 		= 'auto')

class ModelCheckpoint(Callback):

	def __init__(self, filepath, folder, monitor = 'val_loss', verbose = 0, save_best_only = False, save_weights_only = False, mode = 'auto', period = 1):
		super(ModelCheckpoint, self).__init__()
		self.monitor 				= monitor
		self.verbose		 		= verbose
		self.filepath 				= filepath
		self.folder 				= folder
		self.save_best_only 		= save_best_only
		self.save_weights_only		= save_weights_only
		self.period 				= period
		self.epochs_since_last_save	= 0

		if mode not in ['auto', 'min', 'max']:
			warnings.warn('ModelCheckpoint mode %s is unknown, ' 'fallback to auto mode.' % (mode), RuntimeWarning)
			mode = 'auto'

		if mode == 'min':
			self.monitor_op = np.less
			self.best = np.Inf
		elif mode == 'max':
			self.monitor_op = np.greater
			self.best = -np.Inf
		else:
			if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
			    self.monitor_op = np.greater
			    self.best = -np.Inf
			else:
			    self.monitor_op = np.less
			    self.best = np.Inf

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}
		self.epochs_since_last_save += 1
		if self.epochs_since_last_save >= self.period:
			self.epochs_since_last_save = 0
			filepath = self.filepath.format(epoch = epoch + 1, **logs)
			if self.save_best_only:
				current = logs.get(self.monitor)
				if current is None:
				    warnings.warn('Can save best model only with %s available, ' 'skipping.' % (self.monitor), RuntimeWarning)
				else:
					if self.monitor_op(current, self.best):
					    if self.verbose > 0:
					        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,' ' saving model to %s' % (epoch + 1, self.monitor, self.best, current, filepath))
					    self.best = current
					    if self.save_weights_only:
					        self.model.save_weights(filepath, overwrite=True)
					    else:
							self.model.save(filepath, overwrite=True)
							# Save model.h5 on to google storage
							with file_io.FileIO(filepath, mode='r') as input_f:
								with file_io.FileIO(self.folder + '/checkpoints/' + filepath, mode='w+') as output_f:	# w+ : writing and reading
									output_f.write(input_f.read())
					else:
						if self.verbose > 0:
						    print('\nEpoch %05d: %s did not improve' %
						          (epoch + 1, self.monitor))
			else:
				if self.verbose > 0:
				    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
				if self.save_weights_only:
				    self.model.save_weights(filepath, overwrite=True)
				else:
					self.model.save(filepath, overwrite=True)
					# Save model.h5 on to google storage
					with file_io.FileIO(filepath, mode='r') as input_f:
						with file_io.FileIO(self.folder + '/checkpoints/' + filepath, mode='w+') as output_f:	# w+ : writing and reading
							output_f.write(input_f.read())

# Save the model after every epoch
	# filepath:       String, path to save the model file
	# monitor:        Quantity to monitor {val_loss, val_acc}
	# save_best_only: If save_best_only=True, the latest best model according to the quantity monitored will not be overwritten
	# mode:           One of {auto, min, max}. If save_best_only = True, the decision to overwrite the current save file is made based on either
	#			      the maximization or the minimization of the monitored quantity. For val_acc, this should be max, for val_loss this should
	#			      be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity
	# period:         Interval (number of epochs) between checkpoints
check_point = ModelCheckpoint(
	filepath		= 'ResNet-50_{epoch:02d}_{val_loss:.2f}.h5',
	folder 			= folder,
	monitor 		= 'val_loss', # Accuracy is not always a good indicator because of its yes or no nature
	save_best_only	= True,
	mode 			= 'auto',
	period			= 1)

# We train our model again (this time fine-tuning all the resnet blocks)
model.fit_generator(
    generator           = train_generator,
    steps_per_epoch     = len(train_data_x) // batch_size,  # samples_per_epoch / batch_size
    epochs              = epochs_all_layers,
    validation_data     = val_data,
    callbacks           = [tensorboard_all_layers, reduce_lr, reduce_lr_plateau, early_stop, check_point])

# SAVING ##############################################################################################################################################

# Saving the model in the workspace
model.save(folder + '/ResNet-50.h5')
# Save model.h5 on to google storage
with file_io.FileIO('ResNet-50.h5', mode='r') as input_f:
    with file_io.FileIO(folder + '/ResNet-50.h5', mode='w+') as output_f:  # w+ : writing and reading
        output_f.write(input_f.read())