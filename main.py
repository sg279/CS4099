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


def main():
    random.seed(4099)
    np.random.seed(4099)
    tf.random.set_seed(4099)
    name = "3_trainable_base_redo"
    # print(name)
    xm = xception(name)
    # data_dir = os.path.join("..","..","..","..","..","data","sg279", "DDSM data", "pngs")
    data_dir = 'F:\\DDSM data\\pngs'
    xm.make_generators(os.path.join(data_dir, "train"), os.path.join(data_dir, "val"), os.path.join(data_dir, "test"))
    xm.make_model(False, False, 3)
    xm.train()
    evaluate(xm)
    xm.save_preds()

    # name = "average"
    # avg = AverageEnsemble(name)
    # evaluate(avg)
    # name = "voting"
    # voting = VotingEnsemble(name)
    # evaluate(voting)
    # name = "LR"
    # lr = LrEnsemble(name, nodes=0)
    # lr.make_model()
    # lr.train()
    # evaluate(lr)
    # name = "NN"
    # nn = NnEnsemble(name, 10)
    # nn.make_model()
    # nn.train()
    # evaluate(nn)
    # name = "best_performing"
    # voting = LrEnsemble(name)
    # evaluate(voting)

    # for i in range(10, 100, 10):
    #     name = "LR_"+str(i)
    #     lr = LrEnsemble(name, nodes=i)
    #     lr.make_model()
    #     lr.train()
    #     evaluate(lr)

    # datagen = ImageDataGenerator(shear_range=20
    #                                         )
    #
    # img = load_img('F:\\DDSM data\\pngs\\val\\B\\Calc-Test_P_00077_RIGHT_MLO.png')  # this is a PIL image
    # # x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    # # x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    #
    # # the .flow() command below generates batches of randomly transformed images
    # # and saves the results to the `preview/` directory
    # i = 0
    # for batch in datagen.flow_from_directory("F:\\DDSM data\\pngs\\preview2", batch_size=1, color_mode='grayscale',
    #                           save_to_dir='F:\\DDSM data\\pngs\\preview', save_prefix='shear', save_format='jpg', target_size=(800,800)):
    #     i += 1
    #     if i > 0:
    #         break  #


if __name__ == '__main__':
    # mias_conversion("val")
    # mias_conversion("train")
    # mias_conversion("test")
    # print(os.getcwd())
    # np.seterr(all='raise')
    main()


    # train(".\mias\\train", ".\mias\\val",".\mias\\test", ".\\")