import pandas as pd
from PIL import Image
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras import backend as k
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
from sklearn.utils import class_weight
import datetime

class xception():

    def __init__(self, model_name = None):
        # hyper parameters for model
        self.nb_classes = 2  # number of classes
        self.based_model_last_block_layer_number = 126  # value is based on based model selected.
        self.img_width, self.img_height = 750, 750  # change based on the shape/structure of your images
        self.batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
        self.nb_epoch = 50  # number of iteration the algorithm gets trained.
        self.learn_rate = 1e-4  # sgd learning rate
        self.momentum = .9  # sgd momentum to avoid local minimum
        self.transformation_ratio = .05
        if model_name is None:
            self.name = datetime.datetime.now().strftime('%d-%m-%y')
        else:
            self.name = model_name+"_"+datetime.datetime.now().strftime('%d-%m-%y')
        # os.mkdir(".\\" + self.name)
        self.model_path = "./trained_models/" + self.name
        self.model_path = os.path.join(os.getcwd(),".", "trained_models", self.name)

        os.makedirs(self.model_path, exist_ok=True)
        # self.top_weights_path = os.path.join(os.path.abspath(self.model_path), 'top_model_weights.h5')
        self.top_weights_path = os.path.join(os.path.abspath(self.model_path), 'top_model_weights.h5')
        self.final_weights_path = os.path.join(os.path.abspath(self.model_path), 'model_weights.h5')
        self.save_hyper_parameters()
        np.random.seed(10)
        tf.random.set_seed(7)

    def save_hyper_parameters(self):
        # f = open(self.model_path+"/hyper_parameters.txt", 'w')
        f = open(os.path.join(self.model_path, "hyper_parameters.txt"), 'w')
        f.write("based_model_last_block_layer_number: "+str(self.based_model_last_block_layer_number)+"\n")
        f.write("img_width: " + str(self.img_width)+"\n")
        f.write("img_height: " + str(self.img_height)+"\n")
        f.write("batch_size: "+str(self.batch_size)+"\n")
        f.write("nb_epoch: " + str(self.nb_epoch)+"\n")
        f.write("learn_rate: " + str(self.learn_rate)+"\n")
        f.write("momentum: " + str(self.momentum)+"\n")
        f.write("transformation_ratio: " + str(self.transformation_ratio))
        f.close()

    def make_model(self, load=False):
        # Pre-Trained CNN Model using imagenet dataset for pre-trained weights
        base_model = Xception(input_shape=(self.img_width, self.img_height, 3), weights='imagenet', include_top=False)

        # Top Model Block
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.nb_classes, activation='softmax')(x)

        # add your top layer block to your base model
        model = Model(base_model.input, predictions)
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='nadam',
                      loss='binary_crossentropy',  # categorical_crossentropy if multi-class classifier
                      metrics=['accuracy', keras.metrics.AUC()])
        if load:
            model.load_weights(self.top_weights_path)
        self.model = model

    def make_generators(self, train_data_dir, validation_data_dir, test_data_dir):
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

    def train(self):

        # top_weights_path = os.path.join(os.path.abspath(self.model_path), 'top_model_weights.h5')
        callbacks_list = [
            ModelCheckpoint(self.top_weights_path, monitor='val_auc', verbose=1, save_best_only=True, mode="max"),
            EarlyStopping(monitor='val_auc', patience=5, verbose=0, restore_best_weights=True, mode="max"),
            CSVLogger(self.model_path+'/log.csv', append=True, separator=';')
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

    def tune(self):
        self.model.load_weights(self.top_weights_path)

        # based_model_last_block_layer_number points to the layer in your model you want to train.
        # For example if you want to train the last block of a 19 layer VGG16 model this should be 15
        # If you want to train the last Two blocks of an Inception model it should be 172
        # layers before this number will used the pre-trained weights, layers above and including this number
        # will be re-trained based on the new data.
        for layer in self.model.layers[:self.based_model_last_block_layer_number]:
            layer.trainable = False
        for layer in self.model.layers[self.based_model_last_block_layer_number:]:
            layer.trainable = True

        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        # self.model.compile(optimizer='nadam',
        #               loss='categorical_crossentropy',
        #               metrics=['accuracy'])

        callbacks_list = [
            ModelCheckpoint(self.final_weights_path, monitor='val_auc', verbose=1, save_best_only=True, mode="max"),
            EarlyStopping(monitor='val_auc', patience=5, verbose=0, restore_best_weights=True, mode="max"),
            CSVLogger(self.model_path + '/tuning_log.csv', append=True, separator=';')
        ]

        # fine-tune the model
        self.model.fit(self.train_generator, validation_data=self.validation_generator, epochs=self.nb_epoch, callbacks=callbacks_list,
                  class_weight=self.class_weights)

    def test(self):
        Y_pred = self.model.predict(self.test_generator, self.test_generator.samples // self.batch_size + 1)
        y_pred = np.argmax(Y_pred, axis=1)
        print('Confusion Matrix')
        print(confusion_matrix(self.test_generator.classes, y_pred))
        print('Classification Report')
        target_names = ['B', 'M', 'N']
        print(classification_report(self.test_generator.classes, y_pred, target_names=target_names))



    def doeverything(self, model_path):
        # hyper parameters for model
        nb_classes = 2  # number of classes
        based_model_last_block_layer_number = 126  # value is based on based model selected.
        img_width, img_height = 350, 350 # change based on the shape/structure of your images
        batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
        nb_epoch = 50  # number of iteration the algorithm gets trained.
        learn_rate = 1e-4  # sgd learning rate
        momentum = .9  # sgd momentum to avoid local minimum
        transformation_ratio = .05  # how aggressive will be the data augmentation/transformation
        # Pre-Trained CNN Model using imagenet dataset for pre-trained weights
        base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)

        # Top Model Block
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(nb_classes, activation='softmax')(x)

        # add your top layer block to your base model
        model = Model(base_model.input, predictions)
        # print(model.summary())

        # # let's visualize layer names and layer indices to see how many layers/blocks to re-train
        # # uncomment when choosing based_model_last_block_layer
        # for i, layer in enumerate(model.layers):
        #     print(i, layer.name)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all layers of the based model that is already pre-trained.
        for layer in base_model.layers:
            layer.trainable = False


        model.compile(optimizer='nadam',
                      loss='binary_crossentropy',  # categorical_crossentropy if multi-class classifier
                      metrics=['accuracy', keras.metrics.AUC()])

        # save weights of best training epoch: monitor either val_loss or val_acc

        top_weights_path = os.path.join(os.path.abspath(model_path), 'top_model_weights.h5')
        callbacks_list = [
            ModelCheckpoint(top_weights_path, monitor='val_auc', verbose=1, save_best_only=True, mode="max"),
            EarlyStopping(monitor='val_auc', patience=5, verbose=0, restore_best_weights=True, mode="max")
        ]
        class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                         classes=np.unique(self.train_generator.classes),
                                                                         y=self.train_generator.classes)))
        # Train Simple CNN
        model.fit(self.train_generator, validation_data=self.validation_generator, epochs=nb_epoch, callbacks=callbacks_list,
                  class_weight=class_weights)

        Y_pred = model.predict(self.test_generator, self.test_generator.samples // batch_size + 1)
        y_pred = np.argmax(Y_pred, axis=1)
        print('Confusion Matrix')
        print(confusion_matrix(self.test_generator.classes, y_pred))
        print('Classification Report')
        target_names = ['B', 'M']
        print(classification_report(self.test_generator.classes, y_pred, target_names=target_names))

        # model.fit(train_generator, validation_split=0.2, epochs=nb_epoch, callbacks=callbacks_list)
        # add the best weights from the train top model
        # at this point we have the pre-train weights of the base model and the trained weight of the new/added top model
        # we re-load model weights to ensure the best epoch is selected and not the last one.
        # model.load_weights(top_weights_path)
        #
        # # based_model_last_block_layer_number points to the layer in your model you want to train.
        # # For example if you want to train the last block of a 19 layer VGG16 model this should be 15
        # # If you want to train the last Two blocks of an Inception model it should be 172
        # # layers before this number will used the pre-trained weights, layers above and including this number
        # # will be re-trained based on the new data.
        # for layer in model.layers[:based_model_last_block_layer_number]:
        #     layer.trainable = False
        # for layer in model.layers[based_model_last_block_layer_number:]:
        #     layer.trainable = True
        #
        # # compile the model with a SGD/momentum optimizer
        # # and a very slow learning rate.
        # model.compile(optimizer='nadam',
        #               loss='binary_crossentropy',
        #               metrics=['accuracy'])
        #
        # # save weights of best training epoch: monitor either val_loss or val_acc
        # final_weights_path = os.path.join(os.path.abspath(model_path), 'model_weights.h5')
        # callbacks_list = [
        #     ModelCheckpoint(final_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True),
        #     EarlyStopping(monitor='val_loss', patience=5, verbose=0)
        # ]
        #
        # # fine-tune the model
        # model.fit(self.train_generator, validation_data=self.validation_generator, epochs=nb_epoch, callbacks=callbacks_list,
        #           class_weight=class_weights)
        # # Confution Matrix and Classification Report
        # Y_pred = model.predict(self.test_generator, self.test_generator.samples // batch_size + 1)
        # y_pred = np.argmax(Y_pred, axis=1)
        # print('Confusion Matrix')
        # print(confusion_matrix(self.test_generator.classes, y_pred))
        # print('Classification Report')
        # target_names = ['B', 'M']
        # print(classification_report(self.test_generator.classes, y_pred, target_names=target_names))

    def test_predict(self):
        return self.model.predict(self.test_generator)