import pandas as pd
from PIL import Image
import os
# from tensorflow.keras.applications import *
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator,  array_to_img, img_to_array, load_img
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.keras import backend as k
# from matplotlib import pyplot as plt
# import numpy as np
# from sklearn.utils import class_weight

'''
These scripts were used for compiling results and have no impact on functionality
'''


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
            tl = int(m.split("train")[0])
            r = int(m.split("_")[2])
            tr = float(m.split("_")[4])
            df = df.append(pd.Series({'transformation ratio':tr, 'resolution':r, 'trainable layers':tl, 'max_val_auc':round(max(aucs),4), 'test_auc': test_auc}), ignore_index=True)
    # df = df.sort_values("max_val_auc", ascending=False)
    df.to_csv("./diverse_models.csv", index=False)
    return df

def get_aucs(dir):
    models = os.listdir("./"+dir)
    df = pd.DataFrame(columns=['model', 'max_val_auc', 'test_auc'])
    for m in models:
        # if m.startswith("ensemble"):
        if 'preds' not in m and m != 'classes':
            aucs = pd.read_csv("./"+dir+"/"+m+"/log.csv", sep=";")["val_auc"]
            test_auc = round(float(open("./"+dir+"/"+m+"/results.txt").read().split("AUC: ")[1].split("\n")[0]),4)
            # np.save("./preds/model_"+str(i+1)+"_preds", pred_probas)
            df = df.append(pd.Series({'model':m, 'max_val_auc':round(max(aucs),4), 'test_auc': test_auc}), ignore_index=True)
    # df=df.head(9)
    # df = df.sort_values("max_val_auc", ascending=False)
    df.to_csv("./mass_ensemble_members.csv", index=False)
    return df

def eval_diverse_ensembles(dir):
    models = os.listdir("./"+dir)
    df = pd.DataFrame(columns=['model', 'test_auc'])
    for m in models:
        # if m.startswith("ensemble"):
        if 'preds' not in m and m != 'classes':
            # aucs = pd.read_csv("./"+dir+"/"+m+"/log.csv", sep=";")["val_auc"]
            test_auc = round(float(open("./"+dir+"/"+m+"/results.txt").read().split("AUC: ")[1].split("\n")[0]),4)
            # np.save("./preds/model_"+str(i+1)+"_preds", pred_probas)
            df = df.append(pd.Series({'model':m, 'test_auc': test_auc}), ignore_index=True)
    # df=df.head(9)
    df = df.sort_values("test_auc", ascending=False)
    df.to_csv("./diverse_ensembles.csv", index=False)
    return df

def eval_ensembles():
    dirs = ["starting_weights_voting_ensemble", "starting_weights_average_ensemble", "starting_weights_lr_ensembles", "starting_weights_nn_ensembles",
            "starting_weights_mixed_data_ensembles"
            ]
    # dirs = ["parameter_gridsearch_voting_ensemble", "parameter_gridsearch_average_ensemble", "parameter_gridsearch_lr_ensembles", "parameter_gridsearch_nn_ensembles",
    #         "parameter_gridsearch_mixed_data_ensembles"
    #         ]
    members = [3,5,7,9]
    df = pd.DataFrame(columns=['ensemble_method', 'v3', 'v5', 'v7', 'v9','t3', 't5', 't7', 't9'])
    for dir in dirs:
        test_aucs = []
        val_aucs = []
        models = os.listdir("./"+dir)
        for i in range(len(models)):
            m = models[i]
            test_aucs.append(round(float(open("./" + dir + "/" + m + "/results.txt").read().split("AUC: ")[1].split("\n")[0]), 4))
            val_aucs.append(round(max(pd.read_csv("./" + dir + "/" + m + "/log.csv", sep=";")["val_auc"]), 4))
            # np.save("./preds/model_"+str(i+1)+"_preds", pred_probas)
        df = df.append(pd.Series({'ensemble_method': dir.split("_")[-2], 't3': test_aucs[0], 't5': test_aucs[1], 't7': test_aucs[2], 't9': test_aucs[3],
                                  'v3': val_aucs[0], 'v5': val_aucs[1], 'v7': val_aucs[2], 'v9': val_aucs[3]}), ignore_index=True)
    df.to_csv("./starting_weights_ensemble_results.csv", index=False)
    return df


def get_preds():
    models = os.listdir("./parameter_gridsearch")
    for i in range(9):
        m=models[i]
        m="0_trainable_layers_600_resolution_0.05_transformation_ratio"
        xm = Xc(model_name=m, trainable_base_layers=10, resolution=int(m.split("_")[3]), transformation_ratio=0.1, seed=4099, dir="parameter_gridsearch")
        xm.make_model()
        data_dir = 'F:\\DDSM data\\pngs'
        xm.make_generators(os.path.join(data_dir, "train"), os.path.join(data_dir, "val"), os.path.join(data_dir, "test"), make=False)
        os.mkdir("./classes/")
        xm.train_generator.classes.dump("./classes/training_classes.npy")
        xm.validation_generator.classes.dump("./classes/val_classes.npy")
        xm.test_generator.classes.dump("./classes/test_classes.npy")



if __name__ == '__main__':
    # get_aucs("trained_models_and_ensembles/calc_ensemble_members")
    get_aucs("trained_models_and_ensembles/mass_ensemble_members")
    # get_aucs("metadata_search")
    # get_aucs("diverse_model_ensembles")
    # eval_ensembles()
    # eval_val_ensembles()
    # get_preds()
    # get_gridsearch_aucs("parameter_gridsearch")