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


class NnEnsemble():

    def __init__(self, model_name=None, nodes = None):
        self.training_preds = self.load_preds(os.listdir("./models/training_preds"), "training")
        self.training_classes = np.load("./models/classes/training_classes.npy")
        self.val_preds = self.load_preds(os.listdir("./models/validation_preds"), "validation")
        self.val_classes = np.load("./models/classes/val_classes.npy")
        self.test_preds = self.load_preds(os.listdir("./models/test_preds"), "test")
        if model_name is None:
            self.name = datetime.datetime.now().strftime('%d-%m-%y')
        else:
            self.name = model_name+"_"+datetime.datetime.now().strftime('%d-%m-%y')
        self.model_path = "./lr_gridsearch/" + self.name
        self.model_path = os.path.join(os.getcwd(), ".", "lr_gridsearch", self.name)
        self.test_classes = np.load("./models/classes/test_classes.npy")
        os.makedirs(self.model_path, exist_ok=True)
        if nodes is None:
            self.nodes = 60
        else:
            self.nodes = nodes

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

    def make_model(self, load=False):
        model = Sequential()
        model.add(Dense(self.nodes, input_dim=self.training_preds.shape[1], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', AUC()])
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


def lr_ensemble():
    training_preds = load_preds(os.listdir("../models/training_preds"), "training")
    training_classes = np.load("../models/classes/training_classes.npy")
    val_preds = load_preds(os.listdir("../models/validation_preds"), "validation")
    val_classes = np.load("../models/classes/val_classes.npy")
    test_preds = load_preds(os.listdir("../models/test_preds"), "test")
    model = Sequential()
    model.add(Dense(10, input_dim=training_preds.shape[1], activation='relu'))
    model.add(Dense(20, input_dim=training_preds.shape[1], activation='relu'))
    model.add(Dense(10, input_dim=training_preds.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                     classes=np.unique(
                                                                         training_classes),
                                                                     y=training_classes)))
    callbacks_list = [
        EarlyStopping(monitor='val_auc', patience=20, verbose=0, restore_best_weights=True, mode="max")
    ]
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', AUC()])
    model.fit(training_preds, training_classes, epochs=200, validation_data=(val_preds, val_classes), verbose=1, class_weight=class_weights, callbacks=callbacks_list)
    return np.squeeze(model.predict_proba(test_preds).round())


def load_preds(models, mode):
    labels = []
    i = 0
    for m in models:
        pred_probas = np.load("../models/"+mode+"_preds/" + m)
        predicts = pred_probas[:, 1]
        # np.save("./preds/model_"+str(i+1)+"_preds", pred_probas)
        labels.append(predicts)
        i += 1
    labels = np.array(labels)
    labels = np.transpose(labels, (1, 0))
    return labels


def evaluate_error(pred, y_test):
    # pred = np.argmax(pred, axis=1)
    # pred = np.expand_dims(pred, axis=1)  # make same shape as y_test
    error = np.sum(np.equal(pred, y_test)) / y_test.shape[0]
    return error

def main_lr():
    test_preds = os.listdir("../models/test_preds")
    test_classes = np.load("../models/classes/test_classes.npy")
    for model in test_preds:
        print(model+" "+str(evaluate_error(np.argmax(np.load("../models/test_preds/"+model), axis=1), test_classes)))
    lr_preds = lr_ensemble()
    print("voting " + str(evaluate_error(lr_preds,test_classes)))
    fpr, tpr, _ = roc_curve(test_classes, lr_preds)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    pass
    # vote= voting_ensemble(members, test_generator)
    # print(np.sum(np.equal(vote, test_generator.classes)) / test_generator.classes.shape[0])

if __name__ == '__main__':
    main_lr()