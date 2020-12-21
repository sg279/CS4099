
# stacked generalization with linear meta model on blobs dataset
import scipy
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from numpy import dstack
from sklearn.datasets import make_blobs
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Average, Input
from matplotlib import pyplot
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.regularizers import L1L2
from tensorflow import one_hot

# load models from file
def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = 'trained/model_' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
        # make prediction
        yhat = model.predict(inputX, verbose=1)[:,1]
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = np.column_stack((stackX,yhat))
    # flatten predictions to [rows, members x probabilities]
    # stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
    return stackX


# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, class_weights):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    y = one_hot(inputX.classes,2)
    # fit standalone model
    model = Sequential()

    model.add(Dense(1,  # output dim is 2, one score per each class
                    activation='softmax',
                    # kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                    input_dim=len(members))) # input dimension = number of features your data has
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(stackedX, inputX.classes, epochs=100, class_weight=class_weights)
    return model


# fit model on dataset
def fit_model(trainX, trainy):
	# define model
	model = Sequential()
	model.add(Dense(25, input_dim=2, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
	model.fit(trainX, trainy, epochs=100, verbose=1)
	return model

# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat

def make_generators(train_data_dir, validation_data_dir, test_data_dir):
    transformation_ratio = .05
    img_width, img_height = 400, 400 # change based on the shape/structure of your images
    batch_size = 32
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

    test_datagen = ImageDataGenerator(rescale=1. / 255
                                      )
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
    validation_generator = None
    # validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
    #                                                               target_size=(img_width, img_height),
    #                                                               batch_size=batch_size,
    #                                                               class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                      target_size=(img_width, img_height),
                                                      batch_size=batch_size,
                                                      class_mode='categorical', shuffle=False)

    class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                     classes=np.unique(
                                                                         train_generator.classes),
                                                                     y=train_generator.classes)))
    return train_generator, validation_generator, test_generator, class_weights

def ddsm_ensemble():
    n_members = 2
    members = load_all_models(n_members)
    train_generator, validation_generator, test_generator, class_weights = make_generators("F:\\DDSM data\\pngs\\small_test", "F:\\DDSM data\\pngs\\val", "F:\\DDSM data\\pngs\\small_test")
    for model in members:
        acc = model.evaluate(test_generator, verbose=0)
        print(dict(zip(model.metrics_names, acc)))
        # acc = evaluate_error(model, test_generator, test_generator.classes)
        # print(acc)
    model = fit_stacked_model(members, train_generator, class_weights)
    # evaluate model on test set
    yhat = stacked_prediction(members, model, test_generator)
    acc = accuracy_score(test_generator.classes, yhat)
    print('Stacked Test Accuracy: %.3f' % acc)


def avg_ensemble(models, model_input):
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(Input(model_input), y, name='ensemble')

    return model

def voting_ensemble(models, test):
    labels = []
    i=0
    for m in models:
        pred_probas = m.predict(test)
        predicts = np.argmax(pred_probas, axis=1)
        np.save("./preds/model_"+str(i+1)+"_preds", pred_probas)
        labels.append(predicts)
        i+=1

    # Ensemble with voting
    labels = np.array(labels)
    labels = np.transpose(labels, (1, 0))
    labels = scipy.stats.mode(labels, axis=1)[0]
    labels = np.squeeze(labels)
    return labels

def evaluate_error(model, x_test, y_test):
    pred = model.predict(x_test, verbose=1)
    pred = np.argmax(pred, axis=1)
    # pred = np.expand_dims(pred, axis=1)  # make same shape as y_test
    error = np.sum(np.equal(pred, y_test)) / y_test.shape[0]
    return error

def main_avg():
    n_members = 2
    members = load_all_models(n_members)
    train_generator, validation_generator, test_generator, class_weights = make_generators(
        "F:\\DDSM data\\pngs\\small_test", "F:\\DDSM data\\pngs\\val", "F:\\DDSM data\\pngs\\small_test")
    # for model in members:
    #     acc = evaluate_error(model, test_generator, test_generator.classes)
    #     print(acc)
    model = avg_ensemble(members, train_generator)
    acc = evaluate_error(model, test_generator, test_generator.classes)
    print(acc)

def main_voting():
    os.mkdir("./preds/")
    n_members = 5
    members = load_all_models(n_members)
    train_generator, validation_generator, test_generator, class_weights = make_generators(
        "F:\\DDSM data\\pngs\\small_test", "F:\\DDSM data\\pngs\\val", "F:\\DDSM data\\pngs\\test")
    for model in members:
        acc = evaluate_error(model, test_generator, test_generator.classes)
        print(acc)
    vote= voting_ensemble(members, test_generator)
    print(np.sum(np.equal(vote, test_generator.classes)) / test_generator.classes.shape[0])

def main():
    # generate 2d classification dataset
    X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
    # y = to_categorical(y)
    # split into train and test
    n_train = 100
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    print(trainX.shape, testX.shape)
    # load all models
    n_members = 3
    # os.makedirs('ensemble_models', exist_ok=True)
    # for i in range(n_members):
    #     # fit model
    #     model = fit_model(trainX, trainy)
    #     # save model
    #     filename = 'ensemble_models/model_' + str(i + 1) + '.h5'
    #     model.save(filename)
    #     print('>Saved %s' % filename)
    # exit(0)
    members = load_all_models(n_members)
    print('Loaded %d models' % len(members))
    # evaluate standalone models on test dataset
    for model in members:
        testy_enc = to_categorical(testy)
        _, acc = model.evaluate(testX, testy_enc, verbose=0)
        print('Model Accuracy: %.3f' % acc)
    # fit stacked model using the ensemble
    model = fit_stacked_model(members, testX, testy)
    # evaluate model on test set
    yhat = stacked_prediction(members, model, testX)
    acc = accuracy_score(testy, yhat)
    print('Stacked Test Accuracy: %.3f' % acc)

if __name__ == '__main__':
    # ddsm_ensemble()
    # main_avg()
    transformation_ratio = .05
    img_width, img_height = 650, 650  # change based on the shape/structure of your images
    batch_size = 32
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory("F:\\DDSM data\\pngs\\train",
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical', shuffle=False)
    # main_voting()
    filename = '../trained_models/650_12-12-20/650_12-12-20/'
    # load model from file
    model = load_model(filename)
    pred_probas = model.predict(train_generator)
    predicts = np.argmax(pred_probas, axis=1)
    np.save("./training_preds/model_650_preds", pred_probas)