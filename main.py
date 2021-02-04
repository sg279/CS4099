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
from tensorflow.keras.preprocessing.image import ImageDataGenerator,  array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as k
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import class_weight
from models.xception import xception
from models.lr_ensemble import LrEnsemble
from models.average_ensemble import AverageEnsemble
from models.voting_ensemble import VotingEnsemble
from models.nn_ensemble import NnEnsemble
from models.mixed_data import MixedData
from models.vgg16 import vgg
from models.mobilenetv2 import mobilenet
from models.inceptionv3 import inception
from models.inception_resnet_v2 import inceptionResnet
from models.mixed_model import MixedModel
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import random

def evaluate(model, tuning=False):
    Y_pred = model.test_predict()
    y_pred = np.argmax(Y_pred, axis=1)
    tuning_string = ""
    if tuning:
        tuning_string = "tuning_"
    print('Confusion Matrix')
    print(confusion_matrix(model.test_classes, y_pred))
    print('Classification Report')
    f = open(os.path.join(model.model_path,tuning_string+"results.txt"), 'w')
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
    plt.savefig(model.model_path + "/"+tuning_string+"AUC.png")
    # plt.show()


def train_members():
    for i in range(7):
        seed = int("4"+str(i+1)+"99")
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        name = "ensemble_member_"+str(i+1)
        xm = xception(model_name=name, trainable_base_layers=10, resolution=600, transformation_ratio=0.1, seed=4099)
        data_dir = os.path.join("..","..","..","..","..","data","sg279", "DDSM data", "pngs")
        # data_dir = 'F:\\DDSM data\\pngs'
        xm.make_generators(os.path.join(data_dir, "train"), os.path.join(data_dir, "val"), os.path.join(data_dir, "test"))
        xm.make_model(False, False)
        xm.train()
        evaluate(xm)
        xm.save_preds("")


def create_mixed_data_ensembles():
    members = [5,7,9]
    for m in members:
        name = "mixed_data_"+str(m)+"_members"
        md = MixedData(model_name=name, trainable_base_layers=10, resolution=600, transformation_ratio=0.1, seed=4099, members=m,
                       preds_dir="parameter_gridsearch", model_dir="mixed_data_gridsearch_ensembles")
        # data_dir = os.path.join("..","..","..","..","..","data","sg279", "DDSM data", "pngs")
        data_dir = 'F:\\DDSM data\\pngs'
        md.make_generators(os.path.join(data_dir, "train"), os.path.join(data_dir, "val"), os.path.join(data_dir, "test"))
        md.make_model(False, False)
        md.train()
        evaluate(md)
        # xm.save_preds("")

def create_voting_ensembles():
    members = [3,5,7,9]
    for m in members:
        name = "voting_"+str(m)+"_members"
        voting = VotingEnsemble(model_name=name, members=m, model_dir="voting_gridsearch_ensemble", preds_dir="parameter_gridsearch")
        evaluate(voting)

def create_average_ensembles():
    members = [3,5,7,9]
    for m in members:
        name = "average_"+str(m)+"_members"
        avg = AverageEnsemble(model_name=name, members=m, model_dir="average_gridsearch_ensemble", preds_dir="parameter_gridsearch")
        evaluate(avg)

def create_lr_ensembles():
    members = [3, 5, 7, 9]
    for m in members:
        name = "lr_" + str(m) + "_members"
        # lr = LrEnsemble(model_name=name, members=m, model_dir="lr_gridsearch_ensemble", preds_dir="parameter_gridsearch")
        lr = LrEnsemble(model_name=name, members=m)
        lr.make_model()
        lr.train()
        evaluate(lr)

def create_nn_ensembles():
    members = [3, 5, 7, 9]
    for m in members:
        name = "nn_" + str(m) + "_members"
        # nn = NnEnsemble(model_name=name, members=m, nodes=10, model_dir="nn_gridsearch_ensemble", preds_dir="parameter_gridsearch")
        nn = NnEnsemble(model_name=name, members=m, nodes=10)
        nn.make_model()
        nn.train()
        evaluate(nn)


def main():
    random.seed(4099)
    np.random.seed(4099)
    tf.random.set_seed(4099)

    name = "mixed_data_diverse_members"
    mm = MixedModel(model_name=name, trainable_base_layers=10, resolution=600, transformation_ratio=0.1, seed=4099,preds_dir="diverse_ensemble_members",
                    model_dir="diverse_ensembles")
    # data_dir = os.path.join("..","..","..","..","..","data","sg279", "DDSM data", "pngs")
    data_dir = 'F:\\DDSM data\\pngs'
    mm.make_generators(os.path.join(data_dir, "small_test"), os.path.join(data_dir, "small_test"), os.path.join(data_dir, "small_test"))
    mm.make_model(False, False)
    mm.train()
    evaluate(mm)


    # name = "nn_diverse_members"
    # nn = NnEnsemble(model_name=name, members=5, nodes=10, model_dir="diverse_ensembles", preds_dir="diverse_ensemble_members")
    # nn.make_model()
    # nn.train()
    # evaluate(nn)
    # name = "lr_diverse_members"
    # lr = LrEnsemble(model_name=name, members=5, model_dir="diverse_ensembles", preds_dir="diverse_ensemble_members")
    # lr.make_model()
    # lr.train()
    # evaluate(lr)
    # name = "average_diverse_members"
    # avg = AverageEnsemble(model_name=name, members=5, model_dir="diverse_ensembles", preds_dir="diverse_ensemble_members")
    # evaluate(avg)
    # name = "voting_diverse_members"
    # voting = VotingEnsemble(model_name=name, members=5, model_dir="diverse_ensembles", preds_dir="diverse_ensemble_members")
    # evaluate(voting)
    # name = "mixed_data_diverse_members"
    # md = MixedData(model_name=name, trainable_base_layers=10, resolution=600, transformation_ratio=0.1, seed=4099,
    #                members=5,
    #                preds_dir="diverse_ensemble_members", model_dir="diverse_ensembles")
    # # data_dir = os.path.join("..","..","..","..","..","data","sg279", "DDSM data", "pngs")
    # data_dir = 'F:\\DDSM data\\pngs'
    # md.make_generators(os.path.join(data_dir, "train"), os.path.join(data_dir, "val"), os.path.join(data_dir, "test"))
    # md.make_model(False, False)
    # md.train()
    # evaluate(md)

    # create_mixed_data_ensembles()
    # create_average_ensembles()
    # create_voting_ensembles()
    # create_lr_ensembles()
    # create_nn_ensembles()

    # xm = xception(model_name=name, trainable_base_layers=10, resolution=600, transformation_ratio=0.1, seed=4099)
    # data_dir = os.path.join("..","..","..","..","..","data","sg279", "DDSM data", "pngs")
    # data_dir = 'F:\\DDSM data\\pngs'
    # xm.make_generators(os.path.join(data_dir, "train"), os.path.join(data_dir, "val"), os.path.join(data_dir, "test"))
    # xm.make_model(True, False)
    # xm.train()
    # evaluate(xm)
    # xm.save_preds("")



    # xm = MixedData(name)
    # data_dir = os.path.join("..","..","..","..","..","data","sg279", "DDSM data", "pngs")
    # data_dir = 'F:\\DDSM data\\pngs'
    # xm.make_generators(os.path.join(data_dir, "train"), os.path.join(data_dir, "val"), os.path.join(data_dir, "test"))
    # xm.make_model(False, False, 0)
    # xm.train()
    # evaluate(xm)
    # xm.save_preds()

    # name = "average"
    # avg = AverageEnsemble(name, members=3)
    # evaluate(avg)
    # voting = VotingEnsemble(model_name="new_member_voting_test", model_dir="trained_models", members=3)
    # evaluate(voting)

    # lr = LrEnsemble(model_name="new_member_lr_test", model_dir="trained_models", members=3)
    # lr.make_model()
    # lr.train()
    # evaluate(lr)
    #
    # nn = NnEnsemble(model_name="new_member_nn_test", model_dir="trained_models", members=7, nodes=3)
    # nn.make_model()
    # nn.train()
    # evaluate(nn)

if __name__ == '__main__':
    # mias_conversion("val")
    # mias_conversion("train")
    # mias_conversion("test")
    # print(os.getcwd())
    # np.seterr(all='raise')
    main()


    # train(".\mias\\train", ".\mias\\val",".\mias\\test", ".\\")