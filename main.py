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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as k
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import class_weight
from models.xception import xception
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import random






# hyper parameters for model
nb_classes = 3  # number of classes
based_model_last_block_layer_number = 126  # value is based on based model selected.
img_width, img_height = 300, 300  # change based on the shape/structure of your images
batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 50  # number of iteration the algorithm gets trained.
learn_rate = 1e-4  # sgd learning rate
momentum = .9  # sgd momentum to avoid local minimum
transformation_ratio = .05  # how aggressive will be the data augmentation/transformation
def train(train_data_dir, validation_data_dir, test_data_dir, model_path):
    # Pre-Trained CNN Model using imagenet dataset for pre-trained weights
    base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)

    # Top Model Block
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)

    # add your top layer block to your base model
    model = Model(base_model.input, predictions)
    print(model.summary())

    # # let's visualize layer names and layer indices to see how many layers/blocks to re-train
    # # uncomment when choosing based_model_last_block_layer
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all layers of the based model that is already pre-trained.
    for layer in base_model.layers:
        layer.trainable = False

    # Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.
    # To save augmentations un-comment save lines and add to your flow parameters.
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=transformation_ratio,
                                       shear_range=transformation_ratio,
                                       zoom_range=transformation_ratio,
                                       cval=transformation_ratio,
                                       horizontal_flip=True,
                                       vertical_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255,
                                            rotation_range=transformation_ratio,
                                            shear_range=transformation_ratio,
                                            zoom_range=transformation_ratio,
                                            cval=transformation_ratio,
                                            horizontal_flip=True,
                                            vertical_flip=True
                                            )

    test_datagen = ImageDataGenerator(rescale=1. / 255,
                                            rotation_range=transformation_ratio,
                                            shear_range=transformation_ratio,
                                            zoom_range=transformation_ratio,
                                            cval=transformation_ratio,
                                            horizontal_flip=True,
                                            vertical_flip=True
                                            )

    os.makedirs(os.path.join(os.path.abspath(train_data_dir), '../preview'), exist_ok=True)
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
    # save_to_dir=os.path.join(os.path.abspath(train_data_dir), '../preview')
    # save_prefix='aug',
    # save_format='jpeg')
    # use the above 3 commented lines if you want to save and look at how the data augmentations look like

    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=(img_width, img_height),
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                                  target_size=(img_width, img_height),
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical')

    model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',  # categorical_crossentropy if multi-class classifier
                  metrics=['accuracy', keras.metrics.AUC()])

    # save weights of best training epoch: monitor either val_loss or val_acc

    top_weights_path = os.path.join(os.path.abspath(model_path), 'top_model_weights.h5')
    callbacks_list = [
        ModelCheckpoint(top_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    ]
    class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                 classes = np.unique(train_generator.classes),
                                                 y = train_generator.classes)))
    # Train Simple CNN
    model.fit(train_generator, validation_data= validation_generator, epochs=nb_epoch, callbacks=callbacks_list, class_weight=class_weights)
    Y_pred = model.predict(test_generator, test_generator.samples // batch_size + 1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_generator.classes, y_pred))
    print('Classification Report')
    target_names = ['B', 'M', 'N']
    print(classification_report(test_generator.classes, y_pred, target_names=target_names))
    # model.fit(train_generator, validation_split=0.2, epochs=nb_epoch, callbacks=callbacks_list)
    # add the best weights from the train top model
    # at this point we have the pre-train weights of the base model and the trained weight of the new/added top model
    # we re-load model weights to ensure the best epoch is selected and not the last one.
    model.load_weights(top_weights_path)

    # based_model_last_block_layer_number points to the layer in your model you want to train.
    # For example if you want to train the last block of a 19 layer VGG16 model this should be 15
    # If you want to train the last Two blocks of an Inception model it should be 172
    # layers before this number will used the pre-trained weights, layers above and including this number
    # will be re-trained based on the new data.
    for layer in model.layers[:based_model_last_block_layer_number]:
        layer.trainable = False
    for layer in model.layers[based_model_last_block_layer_number:]:
        layer.trainable = True

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # save weights of best training epoch: monitor either val_loss or val_acc
    final_weights_path = os.path.join(os.path.abspath(model_path), 'model_weights.h5')
    callbacks_list = [
        ModelCheckpoint(final_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_accuracy', patience=5, verbose=0, mode="max")
    ]

    # fine-tune the model
    model.fit(train_generator, validation_data= validation_generator, epochs=nb_epoch, callbacks=callbacks_list, class_weight=class_weights)
    # Confution Matrix and Classification Report
    Y_pred = model.predict(test_generator, test_generator.samples // batch_size + 1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_generator.classes, y_pred))
    print('Classification Report')
    target_names = ['B', 'M', 'N']
    print(classification_report(test_generator.classes, y_pred, target_names=target_names))

def evaluate(model, tuning=False):
    Y_pred = model.test_predict()
    y_pred = np.argmax(Y_pred, axis=1)
    tuning_string = ""
    if tuning:
        tuning_string = "tuning_"
    print('Confusion Matrix')
    print(confusion_matrix(model.test_generator.classes, y_pred))
    print('Classification Report')
    f = open(model.model_path + "/"+tuning_string+"results.txt", 'w')
    f.write('Confusion Matrix\n')
    f.write(str(confusion_matrix(model.test_generator.classes, y_pred))+"\n")
    f.write('Classification Report\n')

    target_names = ['B', 'M']
    print(classification_report(model.test_generator.classes, y_pred, target_names=target_names))
    f.write(classification_report(model.test_generator.classes, y_pred, target_names=target_names)+"\n")
    
    fpr, tpr, _ = roc_curve(model.test_generator.classes, y_pred)
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
    name = "400"
    # print(name)
    xm = xception(name)
    data_dir = os.path.join("..","..","..","..","..","data","sg279", "DDSM data", "pngs")
    xm.make_generators(os.path.join(data_dir, "train"), os.path.join(data_dir, "val"), os.path.join(data_dir, "test"))
    # xm.make_generators("F:\\DDSM data\\pngs\\train", "F:\\DDSM data\\pngs\\val", "F:\\DDSM data\\pngs\\test")
    # xm.predict(".\\", "F:\\DDSM data\\pngs\\test")
    # xm.doeverything(".\\")
    # xm.make_generators(".\mias\\train", ".\mias\\val", ".\mias\\test")
    # xm.doeverything(".\mias\\train", ".\mias\\val",".\mias\\test", ".\\")
    xm.make_model(False)
    xm.train()
    evaluate(xm)
    xm.save_preds("")
    # xm.tune()
    # evaluate(xm, True)
    # xm.save_preds("_tuning")
    # xm.make_generators(".\mias\\train", ".\mias\\val",".\mias\\test")
    # xm.train()
    # xm.test()
    # xm.tune()
    # xm.test()


if __name__ == '__main__':
    # mias_conversion("val")
    # mias_conversion("train")
    # mias_conversion("test")
    # print(os.getcwd())
    # np.seterr(all='raise')
    main()


    # train(".\mias\\train", ".\mias\\val",".\mias\\test", ".\\")