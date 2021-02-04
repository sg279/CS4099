# stacked generalization with linear meta model on blobs dataset
import scipy.stats as stats
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import datetime
from tensorflow.keras.utils import to_categorical


class AverageEnsemble:

    def __init__(self, model_name=None, members = None, preds_dir = "ensemble_members", model_dir = "average_ensembles"):
        self.preds_dir = preds_dir
        self.model_dir = model_dir
        models = os.listdir(os.path.join(os.getcwd(), ".",self.preds_dir, "test_preds"))

        if members is None:
            self.members = len(models)
        else:
            self.members = members
        labels = []
        for i in range(self.members):
            m = models[i]
            pred_probas = np.load(os.path.join(os.getcwd(), ".",self.preds_dir, "test_preds", m))
            predicts = pred_probas[:,1]
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
        self.test_classes = np.load(os.path.join(os.getcwd(), ".",self.preds_dir, "classes", "test_classes.npy"))

    def test_predict(self):
        labels = np.array(self.labels)
        labels = np.transpose(labels, (1, 0))
        labels = labels.mean(axis=1).round()
        # labels = np.squeeze(labels)
        return to_categorical(labels)

def average_ensemble(models):
    labels = []
    i=0
    for m in models:
        pred_probas = np.load("../ensemble_members/test_preds/"+m)
        predicts = pred_probas[:,1]
        # np.save("./preds/model_"+str(i+1)+"_preds", pred_probas)
        labels.append(predicts)
        i+=1

    # Ensemble with voting
    labels = np.array(labels)
    labels = np.transpose(labels, (1, 0))
    labels = labels.mean(axis=1).round()
    labels = np.squeeze(labels)
    return labels

def evaluate_error(pred, y_test):
    # pred = np.argmax(pred, axis=1)
    # pred = np.expand_dims(pred, axis=1)  # make same shape as y_test
    error = np.sum(np.equal(pred, y_test)) / y_test.shape[0]
    return error

def main_average():
    test_preds = os.listdir("../models/test_preds")
    test_classes = np.load("../models/classes/test_classes.npy")
    for model in test_preds:
        print(model+" "+str(evaluate_error(np.argmax(np.load("../models/test_preds/"+model), axis=1), test_classes)))
    average_preds = average_ensemble(test_preds)
    print("voting " + str(evaluate_error(average_preds,test_classes)))
    fpr, tpr, _ = roc_curve(test_classes, average_preds)
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
    main_average()