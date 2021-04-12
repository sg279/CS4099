import pandas as pd
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.applications import *
from matplotlib import pyplot as plt
import numpy as np
from models.base_model import BaseModel
from models.mixed_data import MixedData, Metadata, Metadata_ensemble
from models.expert_ensemble import ExpertEnsemble
from models.ff_ensemble import FfEnsemble
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import random
from models.metadata_nn import MetadataNN

'''
This method takes a model as a parameter and calls its test predict method to get the test data results, and 
saves various metrics and the ROC AUC curve
'''
def evaluate(model, modifier=""):
    Y_pred = model.test_predict()
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(model.test_classes, y_pred))
    print('Classification Report')
    f = open(os.path.join(model.model_path,modifier+"results.txt"), 'w')
    f.write('Confusion Matrix\n')
    f.write(str(confusion_matrix(model.test_classes, y_pred))+"\n")
    f.write('Classification Report\n')

    target_names = ['B', 'M']
    print(classification_report(model.test_classes, y_pred, target_names=target_names))
    f.write(classification_report(model.test_classes, y_pred, target_names=target_names)+"\n")

    fpr, tpr, _ = roc_curve(model.test_classes, y_pred)
    roc_auc = auc(fpr, tpr)

    f.write("FPR: "+str(fpr)+ " TPR: "+str(tpr)+ " AUC: "+str(roc_auc)+"\n")
    fn, tn = 0,0
    for i in range(len(y_pred)):
        if y_pred[i]==0 and model.test_classes[i] ==0:
            tn = tn+1
        if y_pred[i]==0 and model.test_classes[i] ==1:
            fn = fn+1
    fnr = fn/len(y_pred)
    tnr=tn/len(y_pred)
    f.write("FNR: " + str(fnr) + " TNR: " + str(tnr)+"\n")

    f.close()
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
    # plt.savefig(model.model_path + "/AUC.png")
    # plt.show()

'''
These scripts are used for training collections of models with a single method call and have no bearing on functionality
'''

def train_members():
    for i in range(1, 8):
        seed = int("4" + str(i + 1) + "99")
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        name = "ensemble_member_" + str(i + 1)
        xm = BaseModel(model_name=name, trainable_base_layers=5, resolution=600, transformation_ratio=0.15, seed=4099,
                       model_dir="ensemble_members")
        data_dir = os.path.join("..", "..", "..", "..", "..", "data", "sg279", "DDSM data", "pngs_corrected")
        # data_dir = 'F:\\DDSM data\\pngs'
        xm.make_generators(os.path.join(data_dir, "train"), os.path.join(data_dir, "val"),
                           os.path.join(data_dir, "test"))
        xm.make_model(False, False)
        xm.train()
        evaluate(xm)
        xm.save_preds()

def train_metadata_members():
    resolutions = [450,500,550,600,650]
    for i in resolutions:
        random.seed(4099)
        np.random.seed(4099)
        tf.random.set_seed(4099)
        name = "ensemble_member_"+str(i)
        xm = Metadata(model_name=name, trainable_base_layers=5, resolution=i, transformation_ratio=0.15, seed=4099, model_dir="xception_metadata_members")
        # data_dir = os.path.join("..","..","..","..","..","data","sg279", "DDSM data", "pngs_corrected")
        data_dir = 'F:\\DDSM data\\pngs'
        xm.make_generators(os.path.join(data_dir, "train"), os.path.join(data_dir, "val"), os.path.join(data_dir, "test"))
        xm.make_model(False)
        xm.train()
        evaluate(xm)
        xm.save_preds()

def create_mixed_data_ensembles():
    members = [3, 5, 7, 9]
    for m in members:
        name = "mixed_data_" + str(m) + "_members"
        md = MixedData(model_name=name, trainable_base_layers=5, resolution=600, transformation_ratio=0.15, seed=4099,
                       members=m,
                       preds_dir="ensemble_members", model_dir="mixed_data_ensembles")
        data_dir = os.path.join("..", "..", "..", "..", "..", "data", "sg279", "DDSM data", "pngs_corrected")
        # data_dir = 'F:\\DDSM data\\pngs'
        md.make_generators(os.path.join(data_dir, "train"), os.path.join(data_dir, "val"),
                           os.path.join(data_dir, "test"))
        md.make_model(False, False)
        md.train()
        evaluate(md)
        # xm.save_preds("")

def create_voting_ensembles():
    members = [3, 5, 7, 9]
    for m in members:
        name = "voting_" + str(m) + "_members"
        voting = ExpertEnsemble(model_name=name, members=m, model_dir="parameter_gridsearch_voting_ensemble",
                                preds_dir="parameter_gridsearch_members", mode="voting")
        voting.val_predict()
        evaluate(voting)

def create_average_ensembles():
    members = [3, 5, 7, 9]
    for m in members:
        name = "average_" + str(m) + "_members"
        avg = ExpertEnsemble(model_name=name, members=m, model_dir="parameter_gridsearch_average_ensemble",
                             preds_dir="parameter_gridsearch_members", mode="average")
        avg.val_predict()
        evaluate(avg)

def create_lr_ensembles():
    members = [3, 5, 7, 9]
    for m in members:
        name = "lr_" + str(m) + "_members"
        # lr = LrEnsemble(model_name=name, members=m, model_dir="lr_gridsearch_ensemble", preds_dir="parameter_gridsearch")
        lr = FfEnsemble(model_name=name, members=m, mode="lr", model_dir="starting_weights_lr_ensembles", preds_dir="ensemble_members")
        lr.make_model()
        lr.train()
        evaluate(lr)

def create_nn_ensembles():
    members = [3, 5, 7, 9]
    for m in members:
        name = "nn_" + str(m) + "_members"
        # nn = NnEnsemble(model_name=name, members=m, nodes=10, model_dir="nn_gridsearch_ensemble", preds_dir="parameter_gridsearch")
        nn = FfEnsemble(model_name=name, members=m, mode="nn", model_dir="starting_weights_nn_ensembles", preds_dir="ensemble_members")
        nn.make_model()
        nn.train()
        evaluate(nn)

def parameter_gridsearch():
    resolutions = [800]
    trainable_layers = [5, 10]
    transformation_ratios = [0.05, 0.1, 0.15]

    for r in resolutions:
        for tl in trainable_layers:
            for tr in transformation_ratios:
                name = str(tl) + "trainable_layers_" + str(r) + "_resolution_" + str(tr) + "_transformation_ration"
                model = BaseModel(model_name=name, transformation_ratio=tr, trainable_base_layers=tl, resolution=r,
                                  model_dir="parameter_gridsearch")
                data_dir = os.path.join("..", "..", "..", "..", "..", "data", "sg279", "DDSM data", "pngs_corrected")
                # data_dir = 'F:\\DDSM data\\pngs_corrected'
                model.make_generators(os.path.join(data_dir, "train"), os.path.join(data_dir, "val"),
                                      os.path.join(data_dir, "test"))
                model.make_model(False)
                model.train()
                evaluate(model)
                model.save_preds()

def create_diverse_models():
    models = [
        [Xception, "xception", 126, 600],
        [MobileNetV2, "mobilenetV2", 88, 224],
        [InceptionV3, "inceptionV3", 159, 400],
        [InceptionResNetV2, "inceptionresnetV2", 572, 300],
        [ResNet152V2, "resnet152V2", 564, 500]
    ]
    for m in models:
        name = m[1]
        model = BaseModel(model_name=name, base_model=m[0], base_layers=m[2],
                          model_dir="diverse_model_tests_different_resolutions",
                          trainable_base_layers=5, transformation_ratio=0.15, resolution=m[3])
        # data_dir = os.path.join("..", "..", "..", "..", "..", "data", "sg279", "DDSM data", "pngs_corrected")
        data_dir = 'F:\\DDSM data\\pngs_corrected'
        model.make_generators(os.path.join(data_dir, "train"), os.path.join(data_dir, "val"),
                              os.path.join(data_dir, "test"))
        model.make_model(False)
        model.train()
        evaluate(model)
        model.save_preds()


def main():
    random.seed(4099)
    np.random.seed(4099)
    tf.random.set_seed(4099)


if __name__ == '__main__':
    main()
