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


class MixedData():

    def __init__(self, model_name = None):
        # hyper parameters for model
        self.nb_classes = 2  # number of classes
        self.based_model_last_block_layer_number = 126  # value is based on based model selected.
        self.img_width, self.img_height = 400, 400  # change based on the shape/structure of your images
        self.batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
        self.nb_epoch = 50  # number of iteration the algorithm gets trained.
        self.learn_rate = 1e-4  # sgd learning rate
        self.momentum = .9  # sgd momentum to avoid local minimum
        self.transformation_ratio = .05
        if model_name is None:
            self.name = datetime.datetime.now().strftime('%d-%m-%y')
        else:
            # self.name = model_name+"_"+datetime.datetime.now().strftime('%d-%m-%y')
            self.name = model_name
        # os.mkdir(".\\" + self.name)
        self.model_path = "./resolution_gridsearch/" + self.name
        self.model_path = os.path.join(os.getcwd(),".", "trained_models", self.name)

        os.makedirs(self.model_path, exist_ok=True)
        # self.top_weights_path = os.path.join(os.path.abspath(self.model_path), 'top_model_weights.h5')
        self.top_weights_path = os.path.join(os.path.abspath(self.model_path), 'top_model_weights.h5')
        self.final_weights_path = os.path.join(os.path.abspath(self.model_path), 'model_weights.h5')
        self.tensorboard_path = os.path.join(os.path.abspath(self.model_path), 'tensorboard')
        os.makedirs(self.tensorboard_path, exist_ok=True)
        self.save_hyper_parameters()
        self.i=0

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

    def make_model(self, load=False, extra_block=False, trainable_base_layers = 0):
        # if load:
        #     self.model = load_model(os.path.join(os.path.abspath(self.model_path), self.name))
        #     return
        # Pre-Trained CNN Model using imagenet dataset for pre-trained weights
        base_model = Xception(input_shape=(self.img_width, self.img_height, 3), weights='imagenet', include_top=False)

        if extra_block:
            # Top Model Block
            x = base_model.output
            x = Conv2D(32, (3, 3))(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(32, (3, 3))(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            # x = Conv2D(64, (3, 3))(x)
            # x = Activation('relu')(x)
            # x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Flatten()(x)
            x = Dense(64)(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)
            x = GlobalAveragePooling2D()(x)
        else:
            x = base_model.output
            x = GlobalAveragePooling2D()(x)

        predictions = Dense(self.nb_classes, activation='softmax')(x)

        # add your top layer block to your base model
        x = Model(base_model.input, predictions)
        for layer in base_model.layers[:self.based_model_last_block_layer_number-trainable_base_layers]:
            layer.trainable = False

        # y = Sequential()
        # y.add(Dense(10, input_dim=self.training_preds.shape[1], activation='relu'))
        # y.add(Dense(1, activation='sigmoid'))
        preds = Input(shape=(9, ))
        # combined = concatenate([x.output, y.output])
        combined = concatenate([x.output, preds])
        # z = Dense(15, activation="relu")(combined)
        # z = Dense(10, activation="relu")(z)
        z = Dense(self.nb_classes, activation="softmax")(combined)
        model = Model(inputs=[x.input, preds], outputs=z)
        model.compile(optimizer='nadam',
                      loss='binary_crossentropy',  # categorical_crossentropy if multi-class classifier
                      metrics=['accuracy', keras.metrics.AUC()])
        if load:
            model.load_weights(self.top_weights_path)
            # model = load_model(os.path.join(self.model_path, self.name))
        self.model = model


    def custom_generator(self, images_list, preds, classes, batch_size, data_gen_args, test):
        # i = 0
        self.i=0
        datagen = ImageDataGenerator()
        indexes = list(range(0, len(images_list)-1))
        if not test:
            random.shuffle(indexes)
        while True:
            batch = {'images': [], 'preds': [], 'labels': []}
            for b in range(batch_size):
                try:
                    if self.i == len(images_list)-1:
                        self.i = 0
                        if not test:
                            random.shuffle(indexes)
                    # Read image from list and convert to array
                    image_path = images_list[indexes[self.i]]
                    image = load_img(image_path, target_size=(self.img_height, self.img_width))
                    image = datagen.apply_transform(image, data_gen_args)
                    image = img_to_array(image)

                    # Read data from csv using the name of current image
                    yield_preds = preds[indexes[self.i], :]
                    label = classes[indexes[self.i]]
                    batch['images'].append(image)
                    batch['preds'].append(yield_preds)
                    batch['labels'].append(label)
                    self.i += 1
                except:
                    print("i: "+ str(self.i)+" image list length: "+ str(len(images_list))+" index "+str(indexes[self.i]))
                    random.shuffle(indexes)
                    self.i = 0
                    b=b-1

            batch['images'] = np.array(batch['images'])
            batch['preds'] = np.array(batch['preds'])
            # Convert labels to categorical values
            batch['labels'] = np.eye(self.nb_classes)[batch['labels']]
            yield [batch['images'], batch['preds']], batch['labels']

    def load_preds(self, models, mode):
        labels = []
        i = 0
        for m in models:
            pred_probas = np.load("./models/" + mode + "_preds/" + m)
            predicts = pred_probas[:, 1]
            # np.save("./preds/model_"+str(i+1)+"_preds", pred_probas)
            labels.append(predicts)
            i += 1

        # Ensemble with voting
        labels = np.array(labels)
        labels = np.transpose(labels, (1, 0))
        return labels

    def absoluteFilePaths(self, directory):
        files = []
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                files.append(os.path.abspath(os.path.join(dirpath, f)))
        return files

    def make_generators(self, train_data_dir, validation_data_dir, test_data_dir):
        self.training_preds = self.load_preds(os.listdir("./models/training_preds"), "training")
        self.training_classes = np.load("./models/classes/training_classes.npy")
        self.val_preds = self.load_preds(os.listdir("./models/validation_preds"), "validation")
        self.val_classes = np.load("./models/classes/val_classes.npy")
        self.test_preds = self.load_preds(os.listdir("./models/test_preds"), "test")
        self.test_classes = np.load("./models/classes/test_classes.npy")

        tv_args = dict(rescale=1. / 255,
                                           rotation_range=self.transformation_ratio,
                                           shear_range=self.transformation_ratio,
                                           zoom_range=self.transformation_ratio,
                                           cval=self.transformation_ratio,
                                           horizontal_flip=True,
                                           vertical_flip=True)
        test_args = dict(rescale=1. / 255)
        self.train_generator = self.custom_generator(self.absoluteFilePaths(train_data_dir), self.training_preds, self.training_classes, self.batch_size, tv_args, False)
        print("image list length: "+str(len(self.absoluteFilePaths(train_data_dir)))+"preds shape: "+str(self.training_preds.shape[0])+" classes: "+str(len(self.training_classes)))
        self.validation_generator = self.custom_generator(self.absoluteFilePaths(validation_data_dir), self.val_preds, self.val_classes, self.batch_size, tv_args, False)
        print("image list length: "+str(len(self.absoluteFilePaths(validation_data_dir)))+"preds shape: "+str(self.val_preds.shape[0])+" classes: "+str(len(self.val_classes)))
        self.test_generator = self.custom_generator(self.absoluteFilePaths(test_data_dir), self.test_preds, self.test_classes, self.batch_size, test_args, True)
        print("image list length: "+str(len(self.absoluteFilePaths(test_data_dir)))+"preds shape: "+str(self.test_preds.shape[0])+" classes: "+str(len(self.test_classes)))
        self.class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                         classes=np.unique(
                                                                             self.training_classes),
                                                                         y=self.training_classes)))
        self.test_images_list = self.absoluteFilePaths(test_data_dir)

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
                                                                             self.training_classes),
                                                                         y=self.training_classes)))
        # Train Simple CNN
        self.model.fit_generator(self.train_generator, validation_data=self.validation_generator, epochs=self.nb_epoch,
                  callbacks=callbacks_list,
                  class_weight=class_weights, steps_per_epoch=math.ceil(len(self.training_classes)/self.batch_size), validation_steps=math.ceil(len(self.val_classes)/self.batch_size))
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
        self.model.save(self.model_path+"/"+self.name+"_tuned")

    def test(self):
        Y_pred = self.model.predict(self.test_generator, self.test_generator.samples // self.batch_size + 1)
        y_pred = np.argmax(Y_pred, axis=1)
        print('Confusion Matrix')
        print(confusion_matrix(self.test_generator.classes, y_pred))
        print('Classification Report')
        target_names = ['B', 'M', 'N']
        print(classification_report(self.test_generator.classes, y_pred, target_names=target_names))

    def test_predict(self):
        datagen = ImageDataGenerator()
        preds = None
        data_gen_args = dict(rescale=1. / 255)
        for i in range(len(self.test_classes)):
            print(str(i)+"/"+str(len(self.test_classes)))
            image_path = self.test_images_list[i]
            image = load_img(image_path, target_size=(self.img_height, self.img_width))
            image = datagen.apply_transform(image, data_gen_args)
            image = img_to_array(image)
            if preds is None:
                preds = self.model.predict([np.array([image]), np.array([self.test_preds[i, :]])])
            else:
                preds = np.append(preds, self.model.predict([np.array([image]), np.array([self.test_preds[i, :]])]), axis=0)

        return preds
        # return self.model.predict(self.train_generator, steps=math.ceil(len(self.test_classes)/self.batch_size), verbose=1).round()[:len(self.test_classes)]

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
        os.makedirs(os.path.join(self.model_path,"training_preds"), exist_ok=True)
        os.makedirs(os.path.join(self.model_path,"validation_preds"), exist_ok=True)
        os.makedirs(os.path.join(self.model_path,"test_preds"), exist_ok=True)
        pred_probas = self.model.predict(train_generator)
        predicts = np.argmax(pred_probas, axis=1)
        np.save(os.path.join(self.model_path,"training_preds",self.name+tuning), pred_probas)
        pred_probas = self.model.predict(validation_generator)
        predicts = np.argmax(pred_probas, axis=1)
        np.save(os.path.join(self.model_path,"validation_preds",self.name+tuning), pred_probas)
        pred_probas = self.model.predict(test_generator)
        predicts = np.argmax(pred_probas, axis=1)
        np.save(os.path.join(self.model_path,"test_preds",self.name+tuning), pred_probas)
