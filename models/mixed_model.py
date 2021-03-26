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


class MixedModel():

    def __init__(self, model_name=None, transformation_ratio=.05, trainable_base_layers=0, resolution=400, seed=4099,
                 members=None,
                 preds_dir="ensemble_members", model_dir="mixed_model_ensembles"):
        # hyper parameters for model
        self.nb_classes = 2  # number of classes
        self.xception_model_last_block_layer_number = 126  # value is based on based model selected.
        self.mobilenet_model_last_block_layer_number = 88
        self.inception_model_last_block_layer_number = 159  # value is based on based model selected.
        self.inceptionresnet_model_last_block_layer_number = 572
        self.resnet_model_last_block_layer_number = 564
        self.img_width, self.img_height = resolution, resolution  # change based on the shape/structure of your images
        self.batch_size = 4  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
        self.nb_epoch = 50  # number of iteration the algorithm gets trained.
        self.learn_rate = 1e-4  # sgd learning rate
        self.momentum = .9  # sgd momentum to avoid local minimum
        self.transformation_ratio = transformation_ratio
        self.trainable_base_layers = trainable_base_layers
        self.seed = seed
        self.members = members
        self.preds_dir = preds_dir
        if model_name is None:
            self.name = datetime.datetime.now().strftime('%d-%m-%y')
        else:
            # self.name = model_name+"_"+datetime.datetime.now().strftime('%d-%m-%y')
            self.name = model_name
        # os.mkdir(".\\" + self.name)
        self.model_path = os.path.join(os.getcwd(), ".", model_dir, self.name)

        os.makedirs(self.model_path, exist_ok=True)
        # self.top_weights_path = os.path.join(os.path.abspath(self.model_path), 'top_model_weights.h5')
        self.top_weights_path = os.path.join(os.path.abspath(self.model_path), 'top_model_weights.h5')
        self.final_weights_path = os.path.join(os.path.abspath(self.model_path), 'model_weights.h5')
        self.tensorboard_path = os.path.join(os.path.abspath(self.model_path), 'tensorboard')
        os.makedirs(self.tensorboard_path, exist_ok=True)
        self.save_hyper_parameters()
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.i = 0

    def save_hyper_parameters(self):
        # f = open(self.model_path+"/hyper_parameters.txt", 'w')
        f = open(os.path.join(self.model_path, "hyper_parameters.txt"), 'w')
        f.write("img_width: " + str(self.img_width) + "\n")
        f.write("img_height: " + str(self.img_height) + "\n")
        f.write("batch_size: " + str(self.batch_size) + "\n")
        f.write("nb_epoch: " + str(self.nb_epoch) + "\n")
        f.write("trainable base layers: " + str(self.trainable_base_layers) + "\n")
        f.write("random seed: " + str(self.seed) + "\n")
        f.write("transformation_ratio: " + str(self.transformation_ratio))
        f.close()

    def make_model(self, load=False):
        # if load:
        #     self.model = load_model(os.path.join(os.path.abspath(self.model_path), self.name))
        #     return
        # Pre-Trained CNN Model using imagenet dataset for pre-trained weights
        input = Input(shape=(self.img_width, self.img_height, 3))
        xception = self.make_member(Xception(input_tensor=input, weights='imagenet', include_top=False),
                                    self.xception_model_last_block_layer_number)
        # mobilenet = self.make_member(MobileNetV2(input_tensor=input, weights='imagenet', include_top=False), self.mobilenet_model_last_block_layer_number)
        resnet = self.make_member(ResNet152V2(input_tensor=input, weights='imagenet', include_top=False),
                                  self.resnet_model_last_block_layer_number)
        # inceptionresnet = self.make_member(InceptionResNetV2(input_tensor=input, weights='imagenet', include_top=False), self.inceptionresnet_model_last_block_layer_number)
        inception = self.make_member(InceptionV3(input_tensor=input, weights='imagenet', include_top=False),
                                     self.inception_model_last_block_layer_number)
        # combined = concatenate([x.output, y.output])
        # combined = concatenate([xception, mobilenet, resnet, inceptionresnet, inception])
        combined = concatenate([xception, resnet, inception])
        z = Dense(10, activation="relu")(combined)
        z = Dense(5, activation="relu")(z)
        z = Dense(self.nb_classes, activation="softmax")(z)
        model = Model(inputs=input, outputs=z)
        model.compile(optimizer='nadam',
                      loss='binary_crossentropy',  # categorical_crossentropy if multi-class classifier
                      metrics=['accuracy', keras.metrics.AUC(name='auc')])
        if load:
            model.load_weights(self.top_weights_path)
            # model = load_model(os.path.join(self.model_path, self.name))
        self.model = model

    def make_member(self, base_model, base_layers):
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.nb_classes, activation='softmax')(x)
        # add your top layer block to your base model
        # x = Model(base_model.input, predictions)
        for layer in base_model.layers[:base_layers - self.trainable_base_layers]:
            layer.trainable = False
        return x

    def absoluteFilePaths(self, directory):
        files = []
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                files.append(os.path.abspath(os.path.join(dirpath, f)))
        return files

    def make_generators(self, train_data_dir, validation_data_dir, test_data_dir, make=True):
        self.training_classes = np.load(os.path.join(os.getcwd(), ".", "classes", "training_classes.npy"))
        self.val_classes = np.load(os.path.join(os.getcwd(), ".", "classes", "val_classes.npy"))
        self.test_classes = np.load(os.path.join(os.getcwd(), ".", "classes", "test_classes.npy"))
        self.train_data_dir, self.validation_data_dir, self.test_data_dir = train_data_dir, validation_data_dir, test_data_dir
        if make:
            train_datagen = ImageDataGenerator(rescale=1. / 255,
                                               rotation_range=self.transformation_ratio,
                                               shear_range=self.transformation_ratio,
                                               zoom_range=self.transformation_ratio,
                                               cval=self.transformation_ratio,
                                               horizontal_flip=True,
                                               vertical_flip=True
                                               )

            validation_datagen = ImageDataGenerator(rescale=1. / 255,
                                                    rotation_range=self.transformation_ratio,
                                                    shear_range=self.transformation_ratio,
                                                    zoom_range=self.transformation_ratio,
                                                    cval=self.transformation_ratio,
                                                    horizontal_flip=True,
                                                    vertical_flip=True
                                                    )

            test_datagen = ImageDataGenerator(rescale=1. / 255)
            self.train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                                     target_size=(self.img_width, self.img_height),
                                                                     batch_size=self.batch_size,
                                                                     class_mode='categorical'
                                                                     # ,shuffle=False
                                                                     )

            self.validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                               target_size=(
                                                                               self.img_width, self.img_height),
                                                                               batch_size=self.batch_size,
                                                                               class_mode='categorical'
                                                                               #   ,shuffle=False
                                                                               )

            self.test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                                   target_size=(self.img_width, self.img_height),
                                                                   batch_size=self.batch_size,
                                                                   class_mode='categorical', shuffle=False)

            self.class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                                  classes=np.unique(
                                                                                      self.train_generator.classes),
                                                                                  y=self.train_generator.classes)))
            self.test_classes = self.test_generator.classes

    def train(self):

        # top_weights_path = os.path.join(os.path.abspath(self.model_path), 'top_model_weights.h5')
        callbacks_list = [
            ModelCheckpoint(self.top_weights_path, monitor='val_auc', verbose=1, save_best_only=True, mode="max"),
            EarlyStopping(monitor='val_auc', patience=5, verbose=0, restore_best_weights=True, mode="max"),
            CSVLogger(self.model_path + '/log.csv', append=True, separator=';'),
            TensorBoard(self.tensorboard_path, update_freq=int(self.batch_size / 4))
        ]
        class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                         classes=np.unique(
                                                                             self.training_classes),
                                                                         y=self.training_classes)))
        # Train Simple CNN
        self.model.fit_generator(self.train_generator, validation_data=self.validation_generator, epochs=self.nb_epoch,
                                 callbacks=callbacks_list,
                                 class_weight=class_weights,
                                 steps_per_epoch=math.ceil(len(self.training_classes) / self.batch_size),
                                 validation_steps=math.ceil(len(self.val_classes) / self.batch_size))
        self.model.save(self.model_path + "/" + self.name)

    def test_predict(self):
        return self.model.predict(self.test_generator)

    def save_preds(self, tuning):

        train_datagen = ImageDataGenerator(rescale=1. / 255
                                           )

        validation_datagen = ImageDataGenerator(rescale=1. / 255
                                                )

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(self.train_data_dir,
                                                            target_size=(self.img_width, self.img_height),
                                                            batch_size=self.batch_size,
                                                            class_mode='categorical'
                                                            , shuffle=False
                                                            )

        validation_generator = validation_datagen.flow_from_directory(self.validation_data_dir,
                                                                      target_size=(self.img_width, self.img_height),
                                                                      batch_size=self.batch_size,
                                                                      class_mode='categorical'
                                                                      , shuffle=False
                                                                      )

        test_generator = test_datagen.flow_from_directory(self.test_data_dir,
                                                          target_size=(self.img_width, self.img_height),
                                                          batch_size=self.batch_size,
                                                          class_mode='categorical', shuffle=False)
        os.makedirs(os.path.join(self.model_path, "training_preds"), exist_ok=True)
        os.makedirs(os.path.join(self.model_path, "validation_preds"), exist_ok=True)
        os.makedirs(os.path.join(self.model_path, "test_preds"), exist_ok=True)
        pred_probas = self.model.predict(train_generator)
        predicts = np.argmax(pred_probas, axis=1)
        np.save(os.path.join(self.model_path, "training_preds", self.name + tuning), pred_probas)
        pred_probas = self.model.predict(validation_generator)
        predicts = np.argmax(pred_probas, axis=1)
        np.save(os.path.join(self.model_path, "validation_preds", self.name + tuning), pred_probas)
        pred_probas = self.model.predict(test_generator)
        predicts = np.argmax(pred_probas, axis=1)
        np.save(os.path.join(self.model_path, "test_preds", self.name + tuning), pred_probas)


class MixedModelPretrained():

    def __init__(self, model_name=None, transformation_ratio=.05, trainable_base_layers=0, resolution=400, seed=4099,
                 members=None,
                 preds_dir="ensemble_members", model_dir="mixed_model_ensembles"):
        # hyper parameters for model
        self.nb_classes = 2  # number of classes
        self.xception_model_last_block_layer_number = 126  # value is based on based model selected.
        self.mobilenet_model_last_block_layer_number = 88
        self.inception_model_last_block_layer_number = 159  # value is based on based model selected.
        self.inceptionresnet_model_last_block_layer_number = 572
        self.resnet_model_last_block_layer_number = 564
        self.img_width, self.img_height = resolution, resolution  # change based on the shape/structure of your images
        self.batch_size = 4  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
        self.nb_epoch = 50  # number of iteration the algorithm gets trained.
        self.learn_rate = 1e-4  # sgd learning rate
        self.momentum = .9  # sgd momentum to avoid local minimum
        self.transformation_ratio = transformation_ratio
        self.trainable_base_layers = trainable_base_layers
        self.seed = seed
        self.members = members
        self.preds_dir = preds_dir
        if model_name is None:
            self.name = datetime.datetime.now().strftime('%d-%m-%y')
        else:
            # self.name = model_name+"_"+datetime.datetime.now().strftime('%d-%m-%y')
            self.name = model_name
        # os.mkdir(".\\" + self.name)
        self.model_path = os.path.join(os.getcwd(), ".", model_dir, self.name)

        os.makedirs(self.model_path, exist_ok=True)
        # self.top_weights_path = os.path.join(os.path.abspath(self.model_path), 'top_model_weights.h5')
        self.top_weights_path = os.path.join(os.path.abspath(self.model_path), 'top_model_weights.h5')
        self.final_weights_path = os.path.join(os.path.abspath(self.model_path), 'model_weights.h5')
        self.tensorboard_path = os.path.join(os.path.abspath(self.model_path), 'tensorboard')
        os.makedirs(self.tensorboard_path, exist_ok=True)
        self.save_hyper_parameters()
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.i = 0

    def save_hyper_parameters(self):
        # f = open(self.model_path+"/hyper_parameters.txt", 'w')
        f = open(os.path.join(self.model_path, "hyper_parameters.txt"), 'w')
        f.write("img_width: " + str(self.img_width) + "\n")
        f.write("img_height: " + str(self.img_height) + "\n")
        f.write("batch_size: " + str(self.batch_size) + "\n")
        f.write("nb_epoch: " + str(self.nb_epoch) + "\n")
        f.write("trainable base layers: " + str(self.trainable_base_layers) + "\n")
        f.write("random seed: " + str(self.seed) + "\n")
        f.write("transformation_ratio: " + str(self.transformation_ratio))
        f.close()

    def make_model(self, load=False):
        # if load:
        #     self.model = load_model(os.path.join(os.path.abspath(self.model_path), self.name))
        #     return
        # Pre-Trained CNN Model using imagenet dataset for pre-trained weights
        input = Input(shape=(self.img_width, self.img_height, 3), name="i")
        base_model = Xception(input_shape=(self.img_width, self.img_height, 3), weights='imagenet', include_top=False)
        for layer in base_model.layers[:self.xception_model_last_block_layer_number - self.trainable_base_layers]:
            layer.trainable = False
        base_model = base_model(input)
        # x = base_model.output
        base_model = GlobalAveragePooling2D()(base_model)
        base_model = Dense(self.nb_classes, activation='softmax')(base_model)
        # add your top layer block to your base model
        # x = Model(base_model.input, predictions)(input)

        ensemble_member_0 = keras.models.load_model(
            os.path.join(os.getcwd(), ".", self.preds_dir, "mobilenetV2", "mobilenetV2"))
        print(ensemble_member_0.name)
        ensemble_member_0._name = "ensemble_member_0"
        self.fix_model(ensemble_member_0, self.xception_model_last_block_layer_number)
        ensemble_member_0 = ensemble_member_0(input)
        ensemble_member_1 = keras.models.load_model(
            os.path.join(os.getcwd(), ".", self.preds_dir, "resnet152V2", "resnet152V2"))
        print(ensemble_member_1.name)
        ensemble_member_1._name = "ensemble_member_1"
        self.fix_model(ensemble_member_1, self.xception_model_last_block_layer_number)
        ensemble_member_1 = ensemble_member_1(input)
        ensemble_member_2 = keras.models.load_model(
            os.path.join(os.getcwd(), ".", self.preds_dir, "inceptionresnetV2", "inceptionresnetV2"))
        ensemble_member_2._name = "ensemble_member_2"
        print(ensemble_member_2.name)
        self.fix_model(ensemble_member_2, self.xception_model_last_block_layer_number)
        ensemble_member_2 = ensemble_member_2(input)
        ensemble_member_3 = keras.models.load_model(
            os.path.join(os.getcwd(), ".", self.preds_dir, "inceptionV3", "inceptionV3"))
        print(ensemble_member_3.name)
        ensemble_member_3._name = "ensemble_member_3"
        self.fix_model(ensemble_member_3, self.xception_model_last_block_layer_number)
        ensemble_member_3 = ensemble_member_3(input)
        ensemble_member_4 = keras.models.load_model(
            os.path.join(os.getcwd(), ".", self.preds_dir, "xception", "xception"))
        print(ensemble_member_4.name)
        ensemble_member_4._name = "ensemble_member_4"
        self.fix_model(ensemble_member_4, self.xception_model_last_block_layer_number)
        ensemble_member_4 = ensemble_member_4(input)

        base_model = concatenate(
            [ensemble_member_0, ensemble_member_1, ensemble_member_2, ensemble_member_3, ensemble_member_4, base_model],
            name="concat")
        base_model = Dense(10, activation="relu", name="desnse1")(base_model)
        base_model = Dense(5, activation="relu", name="dense2")(base_model)
        base_model = Dense(self.nb_classes, activation="softmax", name="dense3")(base_model)

        model = Model(inputs=input, outputs=base_model, name="test")
        model.compile(optimizer='nadam',
                      loss='binary_crossentropy',  # categorical_crossentropy if multi-class classifier
                      metrics=['accuracy', keras.metrics.AUC(name='auc')])
        if load:
            model.load_weights(self.top_weights_path)
            # model = load_model(os.path.join(self.model_path, self.name))
        self.model = model

    def fix_model(self, base_model, base_layers):
        for layer in base_model.layers:
            layer.trainable = False

    def make_member(self, base_model, base_layers, path, input):
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.nb_classes, activation='softmax')(x)
        # add your top layer block to your base model
        model = Model(base_model.input, x)
        if path is not None:
            model.load_weights(path)
        model._layers.pop(0)
        model = model(input)
        for layer in base_model.layers[:base_layers]:
            layer.trainable = False
        return model

    def absoluteFilePaths(self, directory):
        files = []
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                files.append(os.path.abspath(os.path.join(dirpath, f)))
        return files

    def make_generators(self, train_data_dir, validation_data_dir, test_data_dir, make=True):
        self.train_data_dir, self.validation_data_dir, self.test_data_dir = train_data_dir, validation_data_dir, test_data_dir
        if make:
            train_datagen = ImageDataGenerator(rescale=1. / 255,
                                               rotation_range=self.transformation_ratio,
                                               shear_range=self.transformation_ratio,
                                               zoom_range=self.transformation_ratio,
                                               cval=self.transformation_ratio,
                                               horizontal_flip=True,
                                               vertical_flip=True
                                               )

            validation_datagen = ImageDataGenerator(rescale=1. / 255,
                                                    rotation_range=self.transformation_ratio,
                                                    shear_range=self.transformation_ratio,
                                                    zoom_range=self.transformation_ratio,
                                                    cval=self.transformation_ratio,
                                                    horizontal_flip=True,
                                                    vertical_flip=True
                                                    )

            test_datagen = ImageDataGenerator(rescale=1. / 255)
            self.train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                                     target_size=(self.img_width, self.img_height),
                                                                     batch_size=self.batch_size,
                                                                     class_mode='categorical'
                                                                     # ,shuffle=False
                                                                     )

            self.validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                               target_size=(
                                                                               self.img_width, self.img_height),
                                                                               batch_size=self.batch_size,
                                                                               class_mode='categorical'
                                                                               #   ,shuffle=False
                                                                               )

            self.test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                                   target_size=(self.img_width, self.img_height),
                                                                   batch_size=self.batch_size,
                                                                   class_mode='categorical', shuffle=False)

            self.class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                                  classes=np.unique(
                                                                                      self.train_generator.classes),
                                                                                  y=self.train_generator.classes)))
            self.test_classes = self.test_generator.classes

    def train(self):

        # top_weights_path = os.path.join(os.path.abspath(self.model_path), 'top_model_weights.h5')
        callbacks_list = [
            ModelCheckpoint(self.top_weights_path, monitor='val_auc', verbose=1, save_best_only=True, mode="max"),
            EarlyStopping(monitor='val_auc', patience=5, verbose=0, restore_best_weights=True, mode="max"),
            CSVLogger(self.model_path + '/log.csv', append=True, separator=';'),
            TensorBoard(self.tensorboard_path, update_freq=int(self.batch_size / 4))
        ]
        class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                         classes=np.unique(
                                                                             self.train_generator.classes),
                                                                         y=self.train_generator.classes)))
        # Train Simple CNN
        self.model.fit_generator(self.train_generator, validation_data=self.validation_generator, epochs=self.nb_epoch,
                                 callbacks=callbacks_list,
                                 class_weight=class_weights,
                                 steps_per_epoch=math.ceil(len(self.train_generator.classes) / self.batch_size),
                                 validation_steps=math.ceil(len(self.validation_generator.classes) / self.batch_size))
        self.model.save(self.model_path + "/" + self.name)

    def test_predict(self):
        return self.model.predict(self.test_generator)

    def save_preds(self, tuning):

        train_datagen = ImageDataGenerator(rescale=1. / 255
                                           )

        validation_datagen = ImageDataGenerator(rescale=1. / 255
                                                )

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(self.train_data_dir,
                                                            target_size=(self.img_width, self.img_height),
                                                            batch_size=self.batch_size,
                                                            class_mode='categorical'
                                                            , shuffle=False
                                                            )

        validation_generator = validation_datagen.flow_from_directory(self.validation_data_dir,
                                                                      target_size=(self.img_width, self.img_height),
                                                                      batch_size=self.batch_size,
                                                                      class_mode='categorical'
                                                                      , shuffle=False
                                                                      )

        test_generator = test_datagen.flow_from_directory(self.test_data_dir,
                                                          target_size=(self.img_width, self.img_height),
                                                          batch_size=self.batch_size,
                                                          class_mode='categorical', shuffle=False)
        os.makedirs(os.path.join(self.model_path, "training_preds"), exist_ok=True)
        os.makedirs(os.path.join(self.model_path, "validation_preds"), exist_ok=True)
        os.makedirs(os.path.join(self.model_path, "test_preds"), exist_ok=True)
        pred_probas = self.model.predict(train_generator)
        predicts = np.argmax(pred_probas, axis=1)
        np.save(os.path.join(self.model_path, "training_preds", self.name + tuning), pred_probas)
        pred_probas = self.model.predict(validation_generator)
        predicts = np.argmax(pred_probas, axis=1)
        np.save(os.path.join(self.model_path, "validation_preds", self.name + tuning), pred_probas)
        pred_probas = self.model.predict(test_generator)
        predicts = np.argmax(pred_probas, axis=1)
        np.save(os.path.join(self.model_path, "test_preds", self.name + tuning), pred_probas)
