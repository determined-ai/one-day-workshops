import keras
from determined.keras import TFKerasTrial, TFKerasTrialContext

import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

class MNISTTrial(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext):
        # Initialize the trial class.
        self.context = context

    def build_model(self):
        # Define and compile model graph.
        
        num_labels = self.context.get_hparam("num_labels")
        image_size = self.context.get_hparam("image_size")
        input_shape = (image_size, image_size, 1)
        
        filters = self.context.get_hparam("filters")
        kernel_size = self.context.get_hparam("kernel_size")
        dropout = self.context.get_hparam("dropout")
        
        
        # use functional API to build cnn layers
        inputs = Input(shape=input_shape)
        y = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu')(inputs)
        y = MaxPooling2D()(y)
        y = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu')(y)
        y = MaxPooling2D()(y)
        y = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu')(y)
        # image to vector before connecting to dense layer
        y = Flatten()(y)
        # dropout regularization
        y = Dropout(dropout)(y)
        outputs = Dense(num_labels, activation='softmax')(y)

        # build the model by supplying inputs/outputs
        model = Model(inputs=inputs, outputs=outputs)
        
        model = self.context.wrap_model(model)
        
        # Create and wrap optimizer.
        optimizer = Adam()
        optimizer = self.context.wrap_optimizer(optimizer)
        
        # classifier loss, Adam optimizer, classifier accuracy
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

    def build_training_data_loader(self):
        # Create the training data loader. This should return a keras.Sequence,
        # a tf.data.Dataset, or NumPy arrays.
        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # from sparse label to categorical
        num_labels = len(np.unique(y_train))
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # reshape and normalize input images
        image_size = x_train.shape[1]
        x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
        x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        
        return x_train, y_train

    def build_validation_data_loader(self):
        # Create the validation data loader. This should return a keras.Sequence,
        # a tf.data.Dataset, or NumPy arrays.
        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # from sparse label to categorical
        num_labels = len(np.unique(y_train))
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # reshape and normalize input images
        image_size = x_train.shape[1]
        x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
        x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        
        return x_test, y_test
        
        