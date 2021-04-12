# stacked generalization with linear meta model on blobs dataset
import scipy.stats as stats
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.metrics import AUC
import datetime
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import random
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import load_model

"""
This class is used for constructing a feedforward network that makes predictions on only metadata
"""
class MetadataNN:

    def __init__(self, model_name=None, seed = 4099, model_dir = "metadata_nn"):
        self.model_dir = model_dir
        # Load metadata and training, validation, and test classes
        self.training_metadata = pd.read_csv(os.path.join(os.getcwd(), ".", "preprocessing", "all_training_metadata.csv")).drop(['label'], axis=1).astype('float32')
        self.training_classes = np.load(os.path.join(os.getcwd(), ".", "classes", "training_classes.npy"), allow_pickle=True)
        self.val_metadata = pd.read_csv(os.path.join(os.getcwd(), ".", "preprocessing", "val_metadata.csv")).drop('label', axis=1).astype('float32')
        self.val_classes = np.load(os.path.join(os.getcwd(), ".", "classes", "val_classes.npy"), allow_pickle=True)
        self.test_metadata = pd.read_csv(os.path.join(os.getcwd(), ".", "preprocessing", "all_test_metadata.csv")).drop('label', axis=1).astype('float32')
        self.test_classes = np.load(os.path.join(os.getcwd(), ".", "classes", "test_classes.npy"), allow_pickle=True)
        if model_name is None:
            self.name = datetime.datetime.now().strftime('%d-%m-%y')
        else:
            self.name = model_name
        self.model_path = os.path.join(os.getcwd(), ".", self.model_dir, self.name)
        os.makedirs(self.model_path, exist_ok=True)
        self.top_weights_path = os.path.join(os.path.abspath(self.model_path), 'top_model_weights.h5')
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)


    def make_model(self, load=False):
        if load:
            self.model = load_model(os.path.join(os.path.abspath(self.model_path), self.name))
            return
        model = Sequential()
        # If there are more than 20 metadata features add extra layers on top of metadata before
        # combining with base model
        if self.training_metadata.shape[1]>20:
            model.add(Dense(50, input_dim=self.training_metadata.shape[1], activation='relu'))
            model.add(Dense(25, input_dim=10, activation='relu'))
        model.add(Dense(5, input_dim=10, activation='relu'))
        model.add(Dense(1, input_dim=self.training_metadata.shape[1], activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', keras.metrics.AUC(name='auc')])
        if load:
            model.load_weights(self.top_weights_path)
        self.model = model

    def train(self):
        class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                         classes=np.unique(
                                                                             self.training_classes),
                                                                         y=self.training_classes)))
        callbacks_list = [
            ModelCheckpoint(self.top_weights_path, monitor='val_auc', verbose=1, save_best_only=True, mode="max"),
            EarlyStopping(monitor='val_auc', patience=5, verbose=0, restore_best_weights=True, mode="max"),
            CSVLogger(self.model_path + '/log.csv', append=True, separator=';'),
        ]
        self.model.fit(self.training_metadata, self.training_classes, epochs=200, validation_data=(self.val_metadata, self.val_classes), verbose=1,
                  class_weight=class_weights,
                       callbacks=callbacks_list)
        self.model.save(self.model_path + "/" + self.name)

    """
    Return the predictions made on the test generator
    """
    def test_predict(self):
        return to_categorical(self.model.predict(self.test_metadata).round())
