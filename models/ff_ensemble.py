import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.metrics import AUC
import datetime
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import random


"""
This class is used for creating ensembles that combine member predictions using feed forward networks, either logistic
regression or ANN
"""
class FfEnsemble:

    def __init__(self, model_name=None, members=None, seed = 4099, preds_dir = "ensemble_members", model_dir = "nn_ensembles", mode = "lr"):
        self.preds_dir = preds_dir
        self.model_dir = model_dir
        self.members = members
        self.mode=mode
        # Load the predictions made my members on training, validation, and test data
        self.training_preds = self.load_preds(os.listdir(os.path.join(os.getcwd(), ".",self.preds_dir, "training_preds")), "training")
        self.training_classes = np.load(os.path.join(os.getcwd(), ".", "classes", "training_classes.npy"), allow_pickle=True)
        self.val_preds = self.load_preds(os.listdir(os.path.join(os.getcwd(), ".",self.preds_dir, "validation_preds")), "validation")
        self.val_classes = np.load(os.path.join(os.getcwd(), ".", "classes", "val_classes.npy"), allow_pickle=True)
        self.test_preds = self.load_preds(os.listdir(os.path.join(os.getcwd(), ".",self.preds_dir, "test_preds")), "test")
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

    # Load either train, validation, or test predictions for each member
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
            labels.append(predicts)
            i += 1
        labels = np.array(labels)
        labels = np.transpose(labels, (1, 0))
        return labels

    """
    Create and train the class's model object using the loaded member predictions
    """
    def make_model(self, load=False):
        model = Sequential()
        # If the feed forward method is ANN, add a hidden layer on top of the input, otherwise only use the
        # final prediction layer
        if self.mode=="nn":
            nodes = self.members
            # With 9 members use 10 nodes in the hidden layer. A sharp decrease in performance was noticed with 9.
            # Otherwise, use nodes equal to the number of members
            if self.members == 9:
                nodes= 10
            model.add(Dense(nodes, input_dim=self.training_preds.shape[1], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', AUC(name="auc")])
        if load:
            model.load_weights(self.top_weights_path)
        self.model = model

    """
    Train the class's model object using the train and validation predictions
    """
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
        self.model.fit(self.training_preds, self.training_classes, epochs=200, validation_data=(self.val_preds, self.val_classes), verbose=0,
                  class_weight=class_weights,
                       callbacks=callbacks_list)
        self.model.save(self.model_path + "/" + self.name)

    """
    Return the predictions made on test data by the ensemble
    """
    def test_predict(self):
        return to_categorical(self.model.predict(self.test_preds).round())
