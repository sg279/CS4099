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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from tensorflow.keras import backend as k
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
from sklearn.utils import class_weight
import datetime
import random



class BaseModel():

    def __init__(self, model_name = None, transformation_ratio = .05, trainable_base_layers = 0, resolution = 400,
                 seed = 4099, model_dir="ensemble_members", base_model = Xception, base_layers=126):
        # hyper parameters for model
        self.nb_classes = 2  # number of classes
        self.based_model_last_block_layer_number = base_layers  # value is based on based model selected.
        self.img_width, self.img_height = resolution, resolution  # change based on the shape/structure of your images
        self.batch_size = 16  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
        self.nb_epoch = 50  # number of iteration the algorithm gets trained.
        self.transformation_ratio = transformation_ratio
        self.trainable_base_layers = trainable_base_layers
        self.seed = seed
        self.base_model=base_model
        if model_name is None:
            self.name = datetime.datetime.now().strftime('%d-%m-%y')
        else:
            self.name = model_name
        self.model_dir = model_dir
        self.model_path = os.path.join(os.getcwd(),".", model_dir, self.name)
        os.makedirs(self.model_path, exist_ok=True)
        self.top_weights_path = os.path.join(os.path.abspath(self.model_path), 'top_model_weights.h5')
        self.final_weights_path = os.path.join(os.path.abspath(self.model_path), 'model_weights.h5')
        self.tensorboard_path = os.path.join(os.path.abspath(self.model_path), 'tensorboard')
        os.makedirs(self.tensorboard_path, exist_ok=True)
        self.save_hyper_parameters()
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def save_hyper_parameters(self):
        # f = open(self.model_path+"/hyper_parameters.txt", 'w')
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

    def make_model(self, load=False, extra_block=False, pre_trained_model = None):
        # Pre-Trained CNN Model using imagenet dataset for pre-trained weights
        if pre_trained_model is None:
            base_model = self.base_model(input_shape=(self.img_width, self.img_height, 3), weights='imagenet', include_top=False)
        else:
            base_model = keras.models.load_model(pre_trained_model)
        if extra_block:
            input = Input(shape=(self.img_width, self.img_height, 3))
            x = Conv2D(16, (7, 7),
                                  activation='relu',
                                  padding='same')(input)

            x =Conv2D(16, (7, 7),
                                  activation='relu',
                                  padding='same')(x)

            x = MaxPooling2D((2, 2), strides=(2, 2))(x)

            x = Conv2D(32, (5, 5),
                                  activation='relu',
                                  padding='same')(x)

            x = Conv2D(32, (5, 5),
                                  activation='relu',
                                  padding='same')(x)

            x = MaxPooling2D((2, 2), strides=(2, 2))(x)
            base_model = base_model(x)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        predictions = Dense(self.nb_classes, activation='softmax')(x)

        # add your top layer block to your base model
        model = Model(base_model.input, predictions)
        for layer in base_model.layers[:self.based_model_last_block_layer_number-self.trainable_base_layers]:
            layer.trainable = False

        model.compile(optimizer='nadam',
                      loss='binary_crossentropy',  # categorical_crossentropy if multi-class classifier
                      metrics=['accuracy', keras.metrics.AUC(name='auc')])
        if load:
            model.load_weights(self.top_weights_path)
        self.model = model

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
                                                                          target_size=(self.img_width, self.img_height),
                                                                          batch_size=self.batch_size,
                                                                          class_mode='categorical'
                                                                        #   ,shuffle=False
                                                                          )

            self.test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                              target_size=(self.img_width, self.img_height),
                                                              batch_size=self.batch_size,
                                                              class_mode='categorical',shuffle=False)

            self.class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                             classes=np.unique(
                                                                                 self.train_generator.classes),
                                                                             y=self.train_generator.classes)))
            self.test_classes = self.test_generator.classes
            # self.test_images_list = self.absoluteFilePaths(test_data_dir)
            self.train_generator.classes.dump("./classes/training_classes.npy")
            self.validation_generator.classes.dump("./classes/val_classes.npy")
            self.test_generator.classes.dump("./classes/test_classes.npy")


    def train(self):

        # top_weights_path = os.path.join(os.path.abspath(self.model_path), 'top_model_weights.h5')
        callbacks_list = [
            ModelCheckpoint(self.top_weights_path, monitor='val_auc', verbose=1, save_best_only=True, mode="max"),
            EarlyStopping(monitor='val_auc', patience=5, verbose=0, restore_best_weights=True, mode="max"),
            CSVLogger(self.model_path+'/log.csv', append=True, separator=';'),
            TensorBoard(self.tensorboard_path, update_freq=int(self.batch_size/4))
        ]
        class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                         classes=np.unique(
                                                                             self.train_generator.classes),
                                                                         y=self.train_generator.classes)))
        # Train Simple CNN
        self.model.fit(self.train_generator, validation_data=self.validation_generator, epochs=self.nb_epoch,
                  callbacks=callbacks_list,
                  class_weight=class_weights)
        self.model.save(self.model_path+"/"+self.name)

    def test_predict(self):
        return self.model.predict(self.test_generator)

    def save_preds(self):

        train_datagen = ImageDataGenerator(rescale=1. / 255
                                           )

        validation_datagen = ImageDataGenerator(rescale=1. / 255
                                                )

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(self.train_data_dir,
                                                            target_size=(self.img_width, self.img_height),
                                                            batch_size=self.batch_size,
                                                            class_mode='categorical'
                                                            ,shuffle=False
                                                            )

        validation_generator = validation_datagen.flow_from_directory(self.validation_data_dir,
                                                                      target_size=(self.img_width, self.img_height),
                                                                      batch_size=self.batch_size,
                                                                      class_mode='categorical'
                                                                      ,shuffle=False
                                                                      )

        test_generator = test_datagen.flow_from_directory(self.test_data_dir,
                                                          target_size=(self.img_width, self.img_height),
                                                          batch_size=self.batch_size,
                                                          class_mode='categorical',shuffle=False)
        os.makedirs(os.path.join(self.model_dir,"training_preds"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir,"validation_preds"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir,"test_preds"), exist_ok=True)
        pred_probas = self.model.predict(train_generator)
        predicts = np.argmax(pred_probas, axis=1)
        np.save(os.path.join(self.model_dir,"training_preds",self.name), pred_probas)
        pred_probas = self.model.predict(validation_generator)
        predicts = np.argmax(pred_probas, axis=1)
        np.save(os.path.join(self.model_dir,"validation_preds",self.name), pred_probas)
        pred_probas = self.model.predict(test_generator)
        predicts = np.argmax(pred_probas, axis=1)
        np.save(os.path.join(self.model_dir,"test_preds",self.name), pred_probas)
