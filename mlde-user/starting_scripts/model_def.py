import keras
from determined.keras import TFKerasTrial, TFKerasTrialContext


class MNISTTrial(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext):
        # Initialize the trial class.
        pass

    def build_model(self):
        # Define and compile model graph.
        pass

    def build_training_data_loader(self):
        # Create the training data loader. This should return a keras.Sequence,
        # a tf.data.Dataset, or NumPy arrays.
        pass

    def build_validation_data_loader(self):
        # Create the validation data loader. This should return a keras.Sequence,
        # a tf.data.Dataset, or NumPy arrays.
        pass