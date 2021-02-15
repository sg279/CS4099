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


class FfEnsemble:

    def __init__(self, model_name=None, members=None, seed = 4099, preds_dir = "ensemble_members", model_dir = "nn_ensembles", mode = "lr"):
        self.preds_dir = preds_dir
        self.model_dir = model_dir
        self.members = members
        self.mode=mode
        self.training_preds = self.load_preds(os.listdir(os.path.join(os.getcwd(), ".",self.preds_dir, "training_preds")), "training")
        self.training_classes = np.load(os.path.join(os.getcwd(), ".",self.preds_dir, "classes", "training_classes.npy"))
        self.val_preds = self.load_preds(os.listdir(os.path.join(os.getcwd(), ".",self.preds_dir, "validation_preds")), "validation")
        self.val_classes = np.load(os.path.join(os.getcwd(), ".",self.preds_dir, "classes", "val_classes.npy"))
        self.test_preds = self.load_preds(os.listdir(os.path.join(os.getcwd(), ".",self.preds_dir, "test_preds")), "test")
        self.test_classes = np.load(os.path.join(os.getcwd(), ".",self.preds_dir, "classes", "test_classes.npy"))
        if model_name is None:
            self.name = datetime.datetime.now().strftime('%d-%m-%y')
        else:
            self.name = model_name+"_"+datetime.datetime.now().strftime('%d-%m-%y')
        self.model_path = os.path.join(os.getcwd(), ".", self.model_dir, self.name)

        os.makedirs(self.model_path, exist_ok=True)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def load_preds(self, models, mode):
        if self.members is None:
            self.members = len(models)
        else:
            self.members = self.members
        labels = []
        for i in range(self.members):
            m = models[i]
            pred_probas = np.load(os.path.join(os.getcwd(), ".", self.preds_dir, mode + "_preds/", m))
            predicts = pred_probas[:, 1]
            # np.save("./preds/model_"+str(i+1)+"_preds", pred_probas)
            labels.append(predicts)
            i += 1

        # Ensemble with voting
        labels = np.array(labels)
        labels = np.transpose(labels, (1, 0))
        return labels

    def make_model(self, load=False):
        model = Sequential()
        if self.mode=="nn":
            model.add(Dense(10, input_dim=self.training_preds.shape[1], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', AUC(name="auc")])
        if load:
            model.load_weights(self.top_weights_path)
        self.model = model

    def train(self):
        class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                         classes=np.unique(
                                                                             self.training_classes),
                                                                         y=self.training_classes)))
        callbacks_list = [
            EarlyStopping(monitor='val_auc', patience=30, verbose=0, restore_best_weights=True, mode="max")
        ]
        self.model.fit(self.training_preds, self.training_classes, epochs=200, validation_data=(self.val_preds, self.val_classes), verbose=0,
                  class_weight=class_weights,
                       callbacks=callbacks_list)
        self.model.save(self.model_path + "/" + self.name)

    def test_predict(self):
        return to_categorical(self.model.predict(self.test_preds).round())
