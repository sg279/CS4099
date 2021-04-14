import pandas as pd
from PIL import Image
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from tensorflow.keras import backend as k
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
from sklearn.utils import class_weight
import datetime
import random
import math
from tensorflow.keras.models import load_model

"""
This class is created to implement ensembles combined with a newly trained CNN
"""
class MixedData():

    def __init__(self, model_name = None, transformation_ratio = .05, trainable_base_layers = 0, resolution = 400, seed = 4099, members = None,
                 preds_dir = "ensemble_members", model_dir = "mixed_data_ensembles"):
        self.preds_dir = preds_dir
        # hyper parameters for model
        self.nb_classes = 2  # number of classes
        # The number of pre trained layers in the base model (always an xception model)
        self.based_model_last_block_layer_number = 126
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
        # Number of members in the ensemble
        self.members = members
        if model_name is None:
            self.name = datetime.datetime.now().strftime("%d-%m-%y")
        else:
            self.name = model_name
        self.model_path = os.path.join(os.getcwd(),".", model_dir, self.name)
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
        # Counter for loading predictions
        self.i = 0

    """
    Store the model's hyper parameters
    """
    def save_hyper_parameters(self):
        f = open(os.path.join(self.model_path, "hyper_parameters.txt"), 'w')
        f.write("based_model_last_block_layer_number: "+str(self.based_model_last_block_layer_number)+"\n")
        f.write("img_width: " + str(self.img_width)+"\n")
        f.write("img_height: " + str(self.img_height)+"\n")
        f.write("batch_size: "+str(self.batch_size)+"\n")
        f.write("nb_epoch: " + str(self.nb_epoch)+"\n")
        f.write("trainable base layers: " + str(self.trainable_base_layers)+"\n")
        f.write("random seed: " + str(self.seed)+"\n")
        f.write("transformation_ratio: " + str(self.transformation_ratio))
        f.close()

    def make_model(self, load=False):
        # Load the pre trained model from keras
        base_model = Xception(
            input_shape=(self.img_width, self.img_height, 3),
            weights="imagenet",
            include_top=False,
        )
        x = base_model.output
        # Create the final pooling and prediction layers
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.nb_classes, activation="softmax")(x)
        # Add pooling and predictions to base model
        x = Model(base_model.input, predictions)
        # Set the required number of base layers to untrainable
        for layer in base_model.layers[:self.based_model_last_block_layer_number-self.trainable_base_layers]:
            layer.trainable = False
        # Define the input for receiving member predictions
        preds = Input(shape=(self.members, ))
        # Concatenate the output of the new CNN with predictions from members
        combined = concatenate([x.output, preds])
        # Add dense layers to learn from the outputs of the new CNN and member models
        z = Dense(10, activation="relu")(combined)
        z = Dense(5, activation="relu")(z)
        z = Dense(self.nb_classes, activation="softmax")(z)
        # Create the model with the input to the new CNN and member predictions as inputs
        model = Model(inputs=[x.input, preds], outputs=z)
        model.compile(optimizer='nadam',
                      loss='binary_crossentropy',  # categorical_crossentropy if multi-class classifier
                      metrics=['accuracy', keras.metrics.AUC(name='auc')])
        # If using a trained model restore weights
        if load:
            model.load_weights(self.top_weights_path)
        self.model = model

    """
    A custom data generator to be used when training the model that generates batches of 
    images and member predictions made on the image
    """
    def custom_generator(self, images_list, preds, classes, batch_size, data_gen_args):
        # Reset the counter
        self.i=0
        datagen = ImageDataGenerator()
        # List of indexes that set the order of which images and predictions are included in batches
        indexes = list(range(0, len(images_list)))
        random.shuffle(indexes)
        # While true is used for data generators
        while True:
            batch = {'images': [], 'preds': [], 'labels': []}
            for b in range(batch_size):
                # If the end of the image list is reached reset counter and shuffle indexes
                if self.i == len(images_list):
                    self.i = 0
                    random.shuffle(indexes)
                # Load image at selected index and apply transformations
                image_path = images_list[indexes[self.i]]
                image = load_img(image_path, target_size=(self.img_height, self.img_width))
                image = datagen.apply_transform(image, data_gen_args)
                image = img_to_array(image)
                # Get the prediction and label at the selected index
                yield_preds = preds[indexes[self.i], :]
                label = classes[indexes[self.i]]
                batch['images'].append(image)
                batch['preds'].append(yield_preds)
                batch['labels'].append(label)
                self.i += 1
            # Format batch for being fed to model
            batch['images'] = np.array(batch['images'])
            batch['preds'] = np.array(batch['preds'])
            # Convert labels to categorical values
            batch['labels'] = np.eye(self.nb_classes)[batch['labels']]
            yield [batch['images'], batch['preds']], batch['labels']

    # Load either train, validation, or test predictions for each member
    def load_preds(self, models, mode):
        if self.members is None:
            self.members = len(models)
        labels = []
        for i in range(self.members):
            m = models[i]
            pred_probas = np.load(os.path.join(os.getcwd(), ".", self.preds_dir, mode + "_preds/", m))
            predicts = pred_probas[:, 1]
            labels.append(predicts)
            i += 1
        labels = np.array(labels)
        labels = np.transpose(labels, (1, 0))
        return labels

    # Get the absolute file paths of all files in a directory
    def absoluteFilePaths(self, directory):
        files = []
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                files.append(os.path.abspath(os.path.join(dirpath, f)))
        return files

    def make_generators(self, train_data_dir, validation_data_dir, test_data_dir, classes = ""):
        # Load member predictions and training, test, and validation data classes
        self.training_preds = self.load_preds(os.listdir(os.path.join(os.getcwd(), ".",self.preds_dir, "training_preds")), "training")
        self.training_classes = np.load(os.path.join(os.getcwd(), ".", "classes", "training_classes.npy"), allow_pickle=True)
        self.val_preds = self.load_preds(os.listdir(os.path.join(os.getcwd(), ".",self.preds_dir, "validation_preds")), "validation")
        self.val_classes = np.load(os.path.join(os.getcwd(), ".", "classes", classes+"val_classes.npy"), allow_pickle=True)
        self.test_preds = self.load_preds(os.listdir(os.path.join(os.getcwd(), ".",self.preds_dir, "test_preds")), "test")
        self.test_classes = np.load(os.path.join(os.getcwd(), ".", "classes", classes+"test_classes.npy"), allow_pickle=True)
        # Train and validation generators use transformations
        tv_args = dict(rescale=1. / 255,
                                           rotation_range=self.transformation_ratio,
                                           shear_range=self.transformation_ratio,
                                           zoom_range=self.transformation_ratio,
                                           cval=self.transformation_ratio,
                                           horizontal_flip=True,
                                           vertical_flip=True)
        self.train_generator = self.custom_generator(self.absoluteFilePaths(train_data_dir), self.training_preds, self.training_classes, self.batch_size, tv_args)
        self.validation_generator = self.custom_generator(self.absoluteFilePaths(validation_data_dir), self.val_preds, self.val_classes, self.batch_size, tv_args)
        self.test_images_list = self.absoluteFilePaths(test_data_dir)

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
                    classes=np.unique(self.training_classes),
                    y=self.training_classes,
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
            steps_per_epoch=math.ceil(len(self.training_classes)/self.batch_size),
            validation_steps=math.ceil(len(self.val_classes)/self.batch_size)
        )
        self.model.save(self.model_path + "/" + self.name)

    """
    Return the predictions made on the test data
    """
    def test_predict(self):
        datagen = ImageDataGenerator()
        preds = None
        data_gen_args = dict(rescale=1. / 255)
        # Iterate through test data
        for i in range(len(self.test_classes)):
            print(str(i)+"/"+str(len(self.test_classes)))
            # Load image
            image_path = self.test_images_list[i]
            image = load_img(image_path, target_size=(self.img_height, self.img_width))
            image = datagen.apply_transform(image, data_gen_args)
            image = img_to_array(image)
            # Add the model predictions on the image and test data predictions to an array
            if preds is None:
                preds = self.model.predict([np.array([image]), np.array([self.test_preds[i, :]])])
            else:
                preds = np.append(preds, self.model.predict([np.array([image]), np.array([self.test_preds[i, :]])]), axis=0)

        return preds

"""
This class trains a base model and layers on top of the base model output and metadata for the images being used
"""
class Metadata():

    def __init__(self, model_name = None, transformation_ratio = .05, trainable_base_layers = 0, resolution = 400, seed = 4099,
                 model_dir = "mixed_data_ensembles", same_level_metadata = False, metadata_prefix = ""):
        # hyper parameters for model
        self.nb_classes = 2  # number of classes
        # The number of pre trained layers in the base model (always an xception model)
        self.based_model_last_block_layer_number = 126
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
        # Used for fetching correct metadata
        self.metadata_prefix = metadata_prefix
        # Whether to combine all metadata features with the base model or a prediction made on metadata
        self.same_level_metadata = same_level_metadata
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
        # Counter for loading predictions
        self.i=0

    """
    Store the model's hyper parameters
    """
    def save_hyper_parameters(self):
        f = open(os.path.join(self.model_path, "hyper_parameters.txt"), 'w')
        f.write("based_model_last_block_layer_number: "+str(self.based_model_last_block_layer_number)+"\n")
        f.write("img_width: " + str(self.img_width)+"\n")
        f.write("img_height: " + str(self.img_height)+"\n")
        f.write("batch_size: "+str(self.batch_size)+"\n")
        f.write("nb_epoch: " + str(self.nb_epoch)+"\n")
        f.write("trainable base layers: " + str(self.trainable_base_layers)+"\n")
        f.write("random seed: " + str(self.seed)+"\n")
        f.write("transformation_ratio: " + str(self.transformation_ratio))
        f.close()

    def make_model(self, load=False, extra_block=False):
        # Load the pre trained xception model from keras
        base_model = Xception(input_shape=(self.img_width, self.img_height, 3), weights='imagenet', include_top=False)
        x = base_model.output
        # Create the final pooling and prediction layers
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.nb_classes, activation="softmax")(x)
        # Add pooling and predictions to base model
        x = Model(base_model.input, predictions)
        # Set the required number of base layers to untrainable
        for layer in base_model.layers[:self.based_model_last_block_layer_number - self.trainable_base_layers]:
            layer.trainable = False
        # If concatenating a single output with the base model prediction by adding layers
        # on top of metadata inputs
        if self.same_level_metadata:
            metadata = Sequential()
            # If there are more than 20 metadata features add extra layers on top of metadata before
            # combining with base model
            if self.training_metadata.shape[1] > 20:
                metadata.add(Dense(50, input_dim=self.training_metadata.shape[1]-1, activation='relu'))
                metadata.add(Dense(25, input_dim=10, activation='relu'))
                metadata.add(Dense(5, input_dim=10, activation='relu'))
            else:
                metadata.add(Dense(5, input_dim=self.training_metadata.shape[1]-1, activation='relu'))
            # Add the prediction layer to the layers on top of metadata
            metadata.add(Dense(self.nb_classes, activation='sigmoid'))
            # Concatenate with base model output and add final prediction layer
            combined = concatenate([x.output, metadata.output])
            z = Dense(self.nb_classes, activation="softmax")(combined)
            # Create model with image and metadata inputs
            model = Model(inputs=[x.input, metadata.input], outputs=z)
        # If all metadata features are being concatenated with the base model output
        if not self.same_level_metadata:
            # Create metadata input
            metadata = Input(shape=(self.training_metadata.shape[1] - 1,))
            # Concatenate metadata with base model output
            z = concatenate([x.output, metadata])
            # If there are more than 20 metadata features add extra layers on top of the metadata and base model
            if self.training_metadata.shape[1]>20:
                z = Dense(50, activation="relu")(z)
                z = Dense(25, activation="relu")(z)
            z = Dense(5, activation="relu")(z)
            # Create final prediction layer
            z = Dense(self.nb_classes, activation="softmax")(z)
            # Create model with image and metadata inputs
            model = Model(inputs=[x.input, metadata], outputs=z)
        model.compile(optimizer='nadam',
                      loss='binary_crossentropy',  # categorical_crossentropy if multi-class classifier
                      metrics=['accuracy', keras.metrics.AUC(name='auc')])
        # If using a trained model restore weights
        if load:
            model.load_weights(self.top_weights_path)
        self.model = model

    """
    A custom data generator to be used when training the model that generates batches of 
    images and metadata for the image
    """
    def custom_generator(self, images_list, metadata, classes, batch_size, data_gen_args):
        # Reset the counter
        self.i=0
        datagen = ImageDataGenerator()
        # List of indexes that set the order of which images and predictions are included in batches
        indexes = list(range(0, len(images_list)))
        random.shuffle(indexes)
        # While true is used for data generators
        while True:
            batch = {'images': [], 'metadata': [], 'labels': []}
            for b in range(batch_size):
                # If the end of the image list is reached reset counter and shuffle indexes
                if self.i == len(images_list):
                    self.i = 0
                    random.shuffle(indexes)
                # Load image at selected index and apply transformations
                image_path = images_list[indexes[self.i]]
                image = load_img(image_path, target_size=(self.img_height, self.img_width))
                image = datagen.apply_transform(image, data_gen_args)
                image = img_to_array(image)
                # Get the metadata and label at the selected index
                yield_metadata = np.array(metadata.loc[indexes[self.i]].drop('label'))
                label = classes[indexes[self.i]]
                batch['images'].append(image)
                batch['metadata'].append(yield_metadata)
                batch['labels'].append(label)
                self.i += 1
            # Format batch for being fed to model
            batch['images'] = np.array(batch['images'])
            batch['metadata'] = np.array(batch['metadata'])
            # Convert labels to categorical values
            batch['labels'] = np.eye(self.nb_classes)[batch['labels']]
            yield [batch['images'], batch['metadata']], batch['labels']

    # Get the absolute file paths of all files in a directory
    def absoluteFilePaths(self, directory):
        files = []
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                files.append(os.path.abspath(os.path.join(dirpath, f)))
        return files

    def make_generators(self, train_data_dir, validation_data_dir, test_data_dir, classes=""):
        # Load metadata and training, test, and validation data classes
        self.training_metadata = pd.read_csv(os.path.join(os.getcwd(), ".", "preprocessing", self.metadata_prefix+"training_metadata.csv"))
        self.training_classes = np.load(os.path.join(os.getcwd(), ".", "classes", "training_classes.npy"), allow_pickle=True)
        self.val_metadata = pd.read_csv(os.path.join(os.getcwd(), ".", "preprocessing", self.metadata_prefix+"val_metadata.csv"))
        self.val_classes = np.load(os.path.join(os.getcwd(), ".", classes+"classes", "val_classes.npy"), allow_pickle=True)
        self.test_metadata = pd.read_csv(os.path.join(os.getcwd(), ".", "preprocessing", self.metadata_prefix+"test_metadata.csv"))
        self.test_classes = np.load(os.path.join(os.getcwd(), ".", classes+"classes", "test_classes.npy"), allow_pickle=True)
        # Train and validation generators use transformations
        tv_args = dict(rescale=1. / 255,
                                           rotation_range=self.transformation_ratio,
                                           shear_range=self.transformation_ratio,
                                           zoom_range=self.transformation_ratio,
                                           cval=self.transformation_ratio,
                                           horizontal_flip=True,
                                           vertical_flip=True)
        self.train_generator = self.custom_generator(self.absoluteFilePaths(train_data_dir), self.training_metadata, self.training_classes, self.batch_size, tv_args)
        self.validation_generator = self.custom_generator(self.absoluteFilePaths(validation_data_dir), self.val_metadata, self.val_classes, self.batch_size, tv_args)
        self.test_images_list = self.absoluteFilePaths(test_data_dir)

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
                    classes=np.unique(self.training_classes),
                    y=self.training_classes,
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
            steps_per_epoch=math.ceil(len(self.training_classes)/self.batch_size),
            validation_steps=math.ceil(len(self.val_classes)/self.batch_size)
        )
        self.model.save(self.model_path + "/" + self.name)

    """
    Return the predictions made on the test data
    """
    def test_predict(self):
        datagen = ImageDataGenerator()
        preds = None
        data_gen_args = dict(rescale=1. / 255)
        # Iterate through test data
        for i in range(len(self.test_classes)):
            print(str(i)+"/"+str(len(self.test_classes)))
            # Load image
            image_path = self.test_images_list[i]
            image = load_img(image_path, target_size=(self.img_height, self.img_width))
            image = datagen.apply_transform(image, data_gen_args)
            image = img_to_array(image)
            # Add the model predictions on the image and metadata to an array
            if preds is None:
                preds = self.model.predict([np.array([image]), np.array(self.test_metadata.loc[i].drop('label')).astype('float32').reshape(1,-1)])
            else:
                preds = np.append(preds, self.model.predict([np.array([image]), np.array(self.test_metadata.loc[i].drop('label')).astype('float32').reshape(1,-1)]), axis=0)

        return preds


"""
This class is a combination of the previous two, combining model predictions and metadata with a base model
"""
class Metadata_ensemble():

    def __init__(self, model_name = None, transformation_ratio = .05, trainable_base_layers = 0, resolution = 400, seed = 4099, members = None,
                 preds_dir = "ensemble_members", model_dir = "mixed_data_ensembles", metadata_prefix = ""):
        self.preds_dir = preds_dir
        # hyper parameters for model
        self.nb_classes = 2  # number of classes
        # The number of pre trained layers in the base model (always an xception model)
        self.based_model_last_block_layer_number = 126
        # Resolution of iamges for training
        self.img_width, self.img_height = resolution, resolution
        # Adjusted based on image resolution and model complexity
        self.batch_size = 8
        # Maximum number of epochs
        self.nb_epoch = 50
        # How much images are transformed before being fed to the model
        self.transformation_ratio = transformation_ratio
        # How many of the last pre trained layers can have weights adjusted
        self.trainable_base_layers = trainable_base_layers
        # Allows for repeatability of results
        self.seed = seed
        # Used for fetching correct metadata
        self.metadata_prefix = metadata_prefix
        # Number of members in the ensemble
        self.members = members
        if model_name is None:
            self.name = datetime.datetime.now().strftime('%d-%m-%y')
        else:
            # self.name = model_name+"_"+datetime.datetime.now().strftime('%d-%m-%y')
            self.name = model_name
        # os.mkdir(".\\" + self.name)
        self.model_path = os.path.join(os.getcwd(),".", model_dir, self.name)

        os.makedirs(self.model_path, exist_ok=True)
        # Where the model weights that achieve highest val AUC are stored
        self.top_weights_path = os.path.join(os.path.abspath(self.model_path), 'top_model_weights.h5')
        self.final_weights_path = os.path.join(os.path.abspath(self.model_path), 'model_weights.h5')
        self.tensorboard_path = os.path.join(os.path.abspath(self.model_path), 'tensorboard')
        os.makedirs(self.tensorboard_path, exist_ok=True)
        self.save_hyper_parameters()
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        # Counter for loading predictions
        self.i=0

    """
    Store the model's hyper parameters
    """
    def save_hyper_parameters(self):
        f = open(os.path.join(self.model_path, "hyper_parameters.txt"), 'w')
        f.write("based_model_last_block_layer_number: "+str(self.based_model_last_block_layer_number)+"\n")
        f.write("img_width: " + str(self.img_width)+"\n")
        f.write("img_height: " + str(self.img_height)+"\n")
        f.write("batch_size: "+str(self.batch_size)+"\n")
        f.write("nb_epoch: " + str(self.nb_epoch)+"\n")
        f.write("trainable base layers: " + str(self.trainable_base_layers)+"\n")
        f.write("random seed: " + str(self.seed)+"\n")
        f.write("transformation_ratio: " + str(self.transformation_ratio))
        f.close()

    def make_model(self, load=False):
        # Load the pre trained xception model from keras
        base_model = Xception(input_shape=(self.img_width, self.img_height, 3), weights='imagenet', include_top=False)
        x = base_model.output
        # Create the final pooling and prediction layers
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.nb_classes, activation="softmax")(x)
        # Add pooling and predictions to base model
        x = Model(base_model.input, predictions)
        # Set the required number of base layers to untrainable
        for layer in base_model.layers[:self.based_model_last_block_layer_number - self.trainable_base_layers]:
            layer.trainable = False
        metadata = Sequential()
        # If there are more than 20 metadata features add extra layers on top of metadata before
        # combining with base model
        if self.training_metadata.shape[1] > 20:
            metadata.add(Dense(50, input_dim=self.training_metadata.shape[1] - 1, activation='relu'))
            metadata.add(Dense(25, input_dim=10, activation='relu'))
            metadata.add(Dense(5, input_dim=10, activation='relu'))
        else:
            metadata.add(Dense(5, input_dim=self.training_metadata.shape[1] - 1, activation='relu'))
        # Add the prediction layer to the layers on top of metadata
        metadata.add(Dense(self.nb_classes, activation='sigmoid'))
        # Define the input for receiving member predictions
        preds = Input(shape=(self.members, ))
        # Concatenate base model output with feed forward layer on top of metadata output and member predictions
        combined = concatenate([x.output, preds,metadata.output])
        # Add dense layers to learn from the outputs of the new CNN and member models
        z = Dense(10, activation="relu")(combined)
        z = Dense(5, activation="relu")(z)
        z = Dense(self.nb_classes, activation="softmax")(z)
        # Create the model with the input to the new CNN, member predictions, and metadata as inputs
        model = Model(inputs=[x.input, preds, metadata.input], outputs=z)
        model.compile(optimizer='nadam',
                      loss='binary_crossentropy',  # categorical_crossentropy if multi-class classifier
                      metrics=['accuracy', keras.metrics.AUC(name='auc')])
        # If using a trained model restore weights
        if load:
            model.load_weights(self.top_weights_path)
        self.model = model

    """
    A custom data generator to be used when training the model that generates batches of 
    images, metadata, and member predictions made on the image
    """
    def custom_generator(self, images_list, metadata, preds, classes, batch_size, data_gen_args):
        # Reset the counter
        self.i=0
        datagen = ImageDataGenerator()
        indexes = list(range(0, len(images_list)))
        random.shuffle(indexes)
        # While true is used for data generators
        while True:
            batch = {'images': [], 'preds': [], 'metadata': [],  'labels': []}
            for b in range(batch_size):
                # If the end of the image list is reached reset counter and shuffle indexes
                if self.i == len(images_list):
                    self.i = 0
                    random.shuffle(indexes)
                # Load image at selected index and apply transformations
                image_path = images_list[indexes[self.i]]
                image = load_img(image_path, target_size=(self.img_height, self.img_width))
                image = datagen.apply_transform(image, data_gen_args)
                image = img_to_array(image)
                # Get the prediction and label at the selected index
                yield_preds = preds[indexes[self.i], :]
                # Get the metadata and label at the selected index
                yield_metadata = np.array(metadata.loc[indexes[self.i]].drop('label'))
                label = classes[indexes[self.i]]
                batch['images'].append(image)
                batch['preds'].append(yield_preds)
                batch['metadata'].append(yield_metadata)
                batch['labels'].append(label)
                self.i += 1
            # Format batch for being fed to model
            batch['images'] = np.array(batch['images'])
            batch['preds'] = np.array(batch['preds'])
            batch['metadata'] = np.array(batch['metadata'])
            # Convert labels to categorical values
            batch['labels'] = np.eye(self.nb_classes)[batch['labels']]
            yield [batch['images'], batch['preds'], batch['metadata']], batch['labels']

    # Load either train, validation, or test predictions for each member
    def load_preds(self, models, mode):
        if self.members is None:
            self.members = len(models)
        labels = []
        for i in range(self.members):
            m = models[i]
            pred_probas = np.load(os.path.join(os.getcwd(), ".", self.preds_dir, mode + "_preds/", m))
            predicts = pred_probas[:, 1]
            labels.append(predicts)
            i += 1
        labels = np.array(labels)
        labels = np.transpose(labels, (1, 0))
        return labels

    # Get the absolute file paths of all files in a directory
    def absoluteFilePaths(self, directory):
        files = []
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                files.append(os.path.abspath(os.path.join(dirpath, f)))
        return files

    def make_generators(self, train_data_dir, validation_data_dir, test_data_dir, classes=""):
        # Load member predictions, metadata, and training, test, and validation data classes
        self.training_preds = self.load_preds(os.listdir(os.path.join(os.getcwd(), ".",self.preds_dir, "training_preds")), "training")
        self.training_metadata = pd.read_csv(os.path.join(os.getcwd(), ".", "preprocessing", self.metadata_prefix+"training_metadata.csv"))
        self.training_classes = np.load(os.path.join(os.getcwd(), ".", "classes", "training_classes.npy"), allow_pickle=True)
        self.val_preds = self.load_preds(os.listdir(os.path.join(os.getcwd(), ".",self.preds_dir, "validation_preds")), "validation")
        self.val_metadata = pd.read_csv(os.path.join(os.getcwd(), ".", "preprocessing", self.metadata_prefix+"val_metadata.csv"))
        self.val_classes = np.load(os.path.join(os.getcwd(), ".", "classes.npy", classes+"val_classes.npy"), allow_pickle=True)
        self.test_preds = self.load_preds(os.listdir(os.path.join(os.getcwd(), ".",self.preds_dir, "test_preds")), "test")
        self.test_metadata = pd.read_csv(os.path.join(os.getcwd(), ".", "preprocessing", self.metadata_prefix+"test_metadata.csv"))
        self.test_classes = np.load(os.path.join(os.getcwd(), ".", "classes.npy", classes+"test_classes.npy"), allow_pickle=True)
        # Train and validation generators use transformations
        tv_args = dict(rescale=1. / 255,
                                           rotation_range=self.transformation_ratio,
                                           shear_range=self.transformation_ratio,
                                           zoom_range=self.transformation_ratio,
                                           cval=self.transformation_ratio,
                                           horizontal_flip=True,
                                           vertical_flip=True)
        self.train_generator = self.custom_generator(self.absoluteFilePaths(train_data_dir),self.training_metadata, self.training_preds, self.training_classes, self.batch_size, tv_args)
        self.validation_generator = self.custom_generator(self.absoluteFilePaths(validation_data_dir), self.val_metadata, self.val_preds, self.val_classes, self.batch_size, tv_args)
        self.test_images_list = self.absoluteFilePaths(test_data_dir)

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
                    classes=np.unique(self.training_classes),
                    y=self.training_classes,
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
            steps_per_epoch=math.ceil(len(self.training_classes)/self.batch_size),
            validation_steps=math.ceil(len(self.val_classes)/self.batch_size)
        )
        self.model.save(self.model_path + "/" + self.name)

    """
    Return the predictions made on the test data
    """
    def test_predict(self):
        datagen = ImageDataGenerator()
        preds = None
        data_gen_args = dict(rescale=1. / 255)
        # Iterate through test data
        for i in range(len(self.test_classes)):
            print(str(i)+"/"+str(len(self.test_classes)))
            # Load image
            image_path = self.test_images_list[i]
            image = load_img(image_path, target_size=(self.img_height, self.img_width))
            image = datagen.apply_transform(image, data_gen_args)
            image = img_to_array(image)
            # Add the model predictions on the image, metadata and test data predictions to an array
            if preds is None:
                preds = self.model.predict([np.array([image]), np.array([self.test_preds[i, :]]), np.array(self.test_metadata.loc[i].drop('label')).astype('float32').reshape(1,-1)])
            else:
                preds = np.append(preds, self.model.predict([np.array([image]), np.array([self.test_preds[i, :]]),
                                                            np.array(self.test_metadata.loc[i].drop('label')).astype('float32').reshape(1,-1)]), axis=0)
        return preds
