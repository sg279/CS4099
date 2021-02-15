# stacked generalization with linear meta model on blobs dataset
import scipy.stats as stats
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import datetime
from tensorflow.keras.utils import to_categorical

class ExpertEnsemble:

    def __init__(self, model_name=None, members=None, preds_dir = "ensemble_members", model_dir = "voting_ensembles", mode="voting"):
        self.preds_dir = preds_dir
        self.model_dir = model_dir
        self.mode=mode
        models = os.listdir(os.path.join(os.getcwd(), ".",self.preds_dir, "test_preds"))
        if members is None:
            self.members = len(models)
        else:
            self.members = members
        labels = []
        for i in range(self.members):
            m = models[i]
            pred_probas = np.load(os.path.join(os.getcwd(), ".",self.preds_dir, "test_preds", m))
            if mode=="average":
                predicts = pred_probas[:, 1]
            else:
                predicts = np.argmax(pred_probas, axis=1)
            # np.save("./preds/model_"+str(i+1)+"_preds", pred_probas)
            labels.append(predicts)
            i += 1
        self.labels = labels
        if model_name is None:
            self.name = datetime.datetime.now().strftime('%d-%m-%y')
        else:
            self.name = model_name + "_" + datetime.datetime.now().strftime('%d-%m-%y')
        self.model_path = os.path.join(os.getcwd(), ".", self.model_dir, self.name)

        os.makedirs(self.model_path, exist_ok=True)
        self.test_classes = np.load(os.path.join(os.getcwd(),self.preds_dir, "classes", "test_classes.npy"))

    def test_predict(self):
        labels = np.array(self.labels)
        labels = np.transpose(labels, (1, 0))
        # labels = stats.mode(labels, axis=1)[0]
        if self.mode=="average":
            labels = labels.mean(axis=1).round()
        else:
            labels = np.max(labels, axis=1)
        # labels = np.squeeze(labels)
        return to_categorical(labels)
