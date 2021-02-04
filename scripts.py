import pandas as pd
from PIL import Image
import os
from tensorflow.keras.applications import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator,  array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as k
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import class_weight
from models.xception import xception


def distort_images():
    datagen = ImageDataGenerator(zoom_range=0.7
                                            )
    img = load_img('F:\\DDSM data\\pngs\\val\\B\\Calc-Test_P_00077_RIGHT_MLO.png')
    i = 0
    for batch in datagen.flow_from_directory("F:\\DDSM data\\pngs\\preview2", batch_size=1, color_mode='grayscale',
                              save_to_dir='F:\\DDSM data\\pngs\\preview', save_prefix='zoom', save_format='jpg', target_size=(800,800)):
        i += 1
        if i > 0:
            break  #


def get_gridsearch_aucs(dir):
    models = os.listdir("./"+dir)
    df = pd.DataFrame(columns=['transformation ratio', 'resolution', 'trainable layers', 'max_val_auc', 'test_auc'])
    for m in models:
        if 'preds' not in m and m!='classes':
            aucs = pd.read_csv("./"+dir+"/"+m+"/log.csv", sep=";")["val_auc"]
            test_auc = round(float(open("./"+dir+"/"+m+"/results.txt").read().split("AUC: ")[1].split("\n")[0]),4)
            # np.save("./preds/model_"+str(i+1)+"_preds", pred_probas)
            tl = int(m.split("_")[0])
            r = int(m.split("_")[3])
            tr = float(m.split("_")[5])
            df = df.append(pd.Series({'transformation ratio':tr, 'resolution':r, 'trainable layers':tl, 'max_val_auc':round(max(aucs),4), 'test_auc': test_auc}), ignore_index=True)
    # df = df.sort_values("max_val_auc", ascending=False)
    df.to_csv("./parameter_gridsearch.csv", index=False)
    return df

def get_aucs(dir):
    models = os.listdir("./"+dir)
    df = pd.DataFrame(columns=['model', 'max_val_auc', 'test_auc'])
    for m in models:
        if m.startswith("ensemble"):
        # if 'preds' not in m and m != 'classes':
            aucs = pd.read_csv("./"+dir+"/"+m+"/log.csv", sep=";")["val_auc"]
            test_auc = round(float(open("./"+dir+"/"+m+"/results.txt").read().split("AUC: ")[1].split("\n")[0]),4)
            # np.save("./preds/model_"+str(i+1)+"_preds", pred_probas)
            df = df.append(pd.Series({'model':m, 'max_val_auc':round(max(aucs),4), 'test_auc': test_auc}), ignore_index=True)
    df=df.head(9)
    df = df.sort_values("max_val_auc", ascending=False)
    df.to_csv("./gridsearch_ensemble_members.csv", index=False)
    return df


def eval_ensembles():
    dirs = ["voting_ensembles", "average_ensembles", "lr_ensembles", "nn_ensembles", "mixed_data_ensemble"]
    dirs = ["voting_gridsearch_ensemble", "average_gridsearch_ensemble", "lr_gridsearch_ensemble", "nn_gridsearch_ensemble", "mixed_data_gridsearch_ensembles"]
    members = [3,5,7,9]
    df = pd.DataFrame(columns=['ensemble_method', '3', '5', '7', '9'])
    for dir in dirs:
        test_aucs = []
        models = os.listdir("./"+dir)
        for i in range(len(models)):
            m = models[i]
            test_aucs.append(round(float(open("./" + dir + "/" + m + "/results.txt").read().split("AUC: ")[1].split("\n")[0]), 4))
            # np.save("./preds/model_"+str(i+1)+"_preds", pred_probas)
        df = df.append(pd.Series({'ensemble_method': dir.split("_")[0], '3': test_aucs[0], '5': test_aucs[1], '7': test_aucs[2], '9': test_aucs[3]}), ignore_index=True)
    df = df.append(pd.Series({'ensemble_method': "best_individual", '3': 0.6902, '5': 0.6902, '7': 0.6902, '9': 0.6902}), ignore_index=True)
    df = df.append(pd.Series({'ensemble_method': "best_gridsearch_individual", '3': 0.6984, '5': 0.6984, '7': 0.6984, '9': 0.6984}), ignore_index=True)
    df = df.append(pd.Series({'ensemble_method': "individual with extra capacity", '3': 0.6146, '5': 0.6146, '7': 0.6146, '9': 0.6146}), ignore_index=True)
    df.to_csv("./gridsearch_ensemble_results.csv", index=False)
    return df

def eval_ensembles_fnr():
    dirs = ["voting_ensembles", "average_ensembles", "lr_ensembles", "nn_ensembles", 'mixed_data_ensemble']
    members = [3,5,7,9]
    df = pd.DataFrame(columns=['ensemble_method', '3', '5', '7', '9'])
    for dir in dirs:
        test_aucs = []
        models = os.listdir("./"+dir)
        for i in range(len(models)):
            m = models[i]
            test_aucs.append(round(float(open("./" + dir + "/" + m + "/results.txt").read().split("FNR: ")[1].split(" ")[0]),4))
            # np.save("./preds/model_"+str(i+1)+"_preds", pred_probas)
        df = df.append(pd.Series({'ensemble_method': dir.split("_")[0], '3': test_aucs[0], '5': test_aucs[1], '7': test_aucs[2], '9': test_aucs[3]}), ignore_index=True)
    df = df.append(pd.Series({'ensemble_method': "best_individual", '3': 0.6902, '5': 0.6902, '7': 0.69015, '9': 0.69015}), ignore_index=True)
    df.to_csv("./ensemble_fnr_results.csv", index=False)
    return df

def get_preds():
    models = os.listdir("./parameter_gridsearch")
    for i in range(9):
        m=models[i]
        m="0_trainable_layers_600_resolution_0.05_transformation_ratio"
        xm = xception(model_name=m, trainable_base_layers=10, resolution=int(m.split("_")[3]), transformation_ratio=0.1, seed=4099, dir="parameter_gridsearch")
        xm.make_model(True, False)
        data_dir = 'F:\\DDSM data\\pngs'
        xm.make_generators(os.path.join(data_dir, "train"), os.path.join(data_dir, "val"), os.path.join(data_dir, "test"), make=False)
        xm.save_preds("")



if __name__ == '__main__':
    # get_aucs("parameter_gridsearch")
    # get_aucs("ensemble_members")
    eval_ensembles()
    # get_preds()
    # get_gridsearch_aucs("parameter_gridsearch")