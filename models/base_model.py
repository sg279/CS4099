import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
    TensorBoard,
)
import numpy as np
from sklearn.utils import class_weight
import datetime
import random


"""
This class is used for instantiating a pre trained model, training the model, and storing 
predictions made on training data
"""


class BaseModel:
    def __init__(
        self,
        model_name=None,
        transformation_ratio=0.05,
        trainable_base_layers=0,
        resolution=400,
        seed=4099,
        model_dir="ensemble_members",
        base_model=Xception,
        base_layers=126,
    ):
        # hyper parameters for model
        self.nb_classes = 2  # number of classes
        # The number of pre trained layers in the base model
        self.based_model_last_block_layer_number = base_layers
        # Resolution of iamges for training
        self.img_width, self.img_height = resolution, resolution
        # Adjusted based on image resolution and model complexity
        self.batch_size = 16
        # Maximum number of epochs
        self.nb_epoch = 50
        # How much images are transformed before being fed to the model
        self.transformation_ratio = transformation_ratio
        # How many of the last pre trained layers can have weights adjusted
        self.trainable_base_layers = trainable_base_layers
        # Allows for repeatability of results
        self.seed = seed
        self.base_model = base_model
        # Used for saving the model and identifying later
        if model_name is None:
            self.name = datetime.datetime.now().strftime("%d-%m-%y")
        else:
            self.name = model_name
        self.model_path = os.path.join(os.getcwd(), ".", model_dir, self.name)
        os.makedirs(self.model_path, exist_ok=True)
        # Where the model weights that achieve highest val AUC are stored
        self.top_weights_path = os.path.join(
            os.path.abspath(self.model_path), "top_model_weights.h5"
        )
        self.tensorboard_path = os.path.join(
            os.path.abspath(self.model_path), "tensorboard"
        )
        os.makedirs(self.tensorboard_path, exist_ok=True)
        self.save_hyper_parameters()
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    """
    Store the model's hyper parameters
    """
    def save_hyper_parameters(self):
        f = open(os.path.join(self.model_path, "hyper_parameters.txt"), "w")
        f.write(
            "based_model_last_block_layer_number: "
            + str(self.based_model_last_block_layer_number)
            + "\n"
        )
        f.write("img_width: " + str(self.img_width) + "\n")
        f.write("img_height: " + str(self.img_height) + "\n")
        f.write("batch_size: " + str(self.batch_size) + "\n")
        f.write("nb_epoch: " + str(self.nb_epoch) + "\n")
        f.write("trainable base layers: " + str(self.trainable_base_layers) + "\n")
        f.write("random seed: " + str(self.seed) + "\n")
        f.write("transformation_ratio: " + str(self.transformation_ratio))
        f.close()

    """
    Create and compile the model to be used for training
    """
    def make_model(self, load=False):

        # Load the pre trained model from keras
        base_model = self.base_model(
            input_shape=(self.img_width, self.img_height, 3),
            weights="imagenet",
            include_top=False,
        )
        x = base_model.output
        # Create the final pooling and prediction layers
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.nb_classes, activation="softmax")(x)
        # Create the model object with the base model inputs and prediction layer outputs
        model = Model(base_model.input, predictions)
        # Set the required number of base layers to untrainable
        for layer in base_model.layers[
            : self.based_model_last_block_layer_number - self.trainable_base_layers
        ]:
            layer.trainable = False
        model.compile(
            optimizer="nadam",
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )
        # If using a trained model restore weights
        if load:
            model.load_weights(self.top_weights_path)
        self.model = model

    """
    Instantiate keras data generators to be used for training, validation, and prediction
    """
    def make_generators(self, train_data_dir, validation_data_dir, test_data_dir):
        self.train_data_dir, self.validation_data_dir, self.test_data_dir = (
            train_data_dir,
            validation_data_dir,
            test_data_dir,
        )
        # Train and validation generators use transformations
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=self.transformation_ratio,
            shear_range=self.transformation_ratio,
            zoom_range=self.transformation_ratio,
            cval=self.transformation_ratio,
            horizontal_flip=True,
            vertical_flip=True,
        )
        validation_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=self.transformation_ratio,
            shear_range=self.transformation_ratio,
            zoom_range=self.transformation_ratio,
            cval=self.transformation_ratio,
            horizontal_flip=True,
            vertical_flip=True,
        )
        # Test generator doesn't use any transformations
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        # Create flow from directory generators
        self.train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode="categorical",
        )
        self.validation_generator = validation_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode="categorical",
        )
        self.test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False,
        )
        self.test_classes = self.test_generator.classes
        # Used for storing classes to disk
        # self.train_generator.classes.dump("./classes/training_classes.npy")
        # self.validation_generator.classes.dump("./classes/val_classes.npy")
        # self.test_generator.classes.dump("./classes/test_classes.npy")

    """
    Train the class's model object using the train and validation generators
    """
    def train(self):
        # Create callbacks for use during training. Checkpoint on val AUC improvement and early stop after 5 epochs
        # of no improvement. Also defines saving tensorboards and logging training to CSV
        callbacks_list = [
            ModelCheckpoint(
                self.top_weights_path,
                monitor="val_auc",
                verbose=1,
                save_best_only=True,
                mode="max",
            ),
            EarlyStopping(
                monitor="val_auc",
                patience=5,
                verbose=0,
                restore_best_weights=True,
                mode="max",
            ),
            CSVLogger(self.model_path + "/log.csv", append=True, separator=";"),
            TensorBoard(self.tensorboard_path, update_freq=int(self.batch_size / 4)),
        ]
        # Calculate class weights for balancing
        class_weights = dict(
            enumerate(
                class_weight.compute_class_weight(
                    "balanced",
                    classes=np.unique(self.train_generator.classes),
                    y=self.train_generator.classes,
                )
            )
        )
        # Train the model and save
        self.model.fit(
            self.train_generator,
            validation_data=self.validation_generator,
            epochs=self.nb_epoch,
            callbacks=callbacks_list,
            class_weight=class_weights,
        )
        self.model.save(self.model_path + "/" + self.name)

    """
    Return the predictions made on the test generator
    """
    def test_predict(self):
        return self.model.predict(self.test_generator)

    """
    Save predictions on unshuffled and untransformed data to disk for use in ensembles
    """
    def save_preds(self):
        # The only processing done on images is to rescale pixel values
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)
        validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        # Generators used don't shuffle data in order to keep prediction orders consistent
        train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False,
        )
        validation_generator = validation_datagen.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False,
        )
        test_generator = test_datagen.flow_from_directory(
            self.test_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False,
        )
        os.makedirs(os.path.join(self.model_dir, "training_preds"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "validation_preds"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "test_preds"), exist_ok=True)
        pred_probas = self.model.predict(train_generator)
        np.save(os.path.join(self.model_dir, "training_preds", self.name), pred_probas)
        pred_probas = self.model.predict(validation_generator)
        np.save(
            os.path.join(self.model_dir, "validation_preds", self.name), pred_probas
        )
        pred_probas = self.model.predict(test_generator)
        np.save(os.path.join(self.model_dir, "test_preds", self.name), pred_probas)
