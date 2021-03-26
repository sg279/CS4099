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
            self.name = model_name
        self.model_path = os.path.join(os.getcwd(), ".", self.model_dir, self.name)

        os.makedirs(self.model_path, exist_ok=True)
        self.test_classes = np.load(os.path.join(os.getcwd(), "classes", "test_classes.npy"), allow_pickle=True)

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

    def val_predict(self):
        models = os.listdir(os.path.join(os.getcwd(), ".", self.preds_dir, "validation_preds"))
        labels = []
        for i in range(self.members):
            m = models[i]
            pred_probas = np.load(os.path.join(os.getcwd(), ".", self.preds_dir, "validation_preds", m))
            if self.mode == "average":
                predicts = pred_probas[:, 1]
            else:
                predicts = np.argmax(pred_probas, axis=1)
            # np.save("./preds/model_"+str(i+1)+"_preds", pred_probas)
            labels.append(predicts)
            i += 1
        val_classes = np.load(os.path.join(os.getcwd(), "classes", "val_classes.npy"), allow_pickle=True)

        labels = np.array(labels)
        labels = np.transpose(labels, (1, 0))
        # labels = stats.mode(labels, axis=1)[0]
        if self.mode == "average":
            labels = labels.mean(axis=1).round()
        else:
            labels = np.max(labels, axis=1)
        # labels = np.squeeze(labels)
        y_pred = np.argmax(to_categorical(labels), axis=1)
        f = open(os.path.join(self.model_path,"log.csv"), 'w')
        f.write('val_auc;\n')
        fpr, tpr, _ = roc_curve(val_classes, y_pred)
        roc_auc = auc(fpr, tpr)
        f.write(str(roc_auc) + ";\n")