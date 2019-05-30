
# coding: utf-8

# To specify the site-packages location:
import sys
sys.path.insert(0, '/home/.conda/envs/tensorflow/lib/python3.6/site-packages')


import glob
import os
import itertools
import re
from collections import Counter

import numpy
import tensorflow
import keras
import sklearn.metrics

import deepometry.model

import csv


def _shape(pathname):
    """
    Infer the shape of the sample data from a single sample.

    :param pathname: Path to a sample.
    :return: Sample dimensions.
    """
    return numpy.load(pathname).shape


def load(pathnames, labels, patient_to_exclude):
    """
    Load training and target data.

    Assumes data is stored in a directory corresponding to some class label.

    :param pathnames: List of image pathnames.
    :param labels: List of class labels.
    :return: Tuple (training, target) data, as NumPy arrays.
    """
    print('Before exclusion: ',len(pathnames))
    pathnames = [x for x in pathnames if patient_to_exclude not in x]
    print('After exclusion: ',len(pathnames))

    x = numpy.empty((len(pathnames),) + _shape(pathnames[0]), dtype=numpy.uint8)

    y = numpy.empty((len(pathnames),), dtype=numpy.uint8)

    label_to_index = {label: index for index, label in enumerate(sorted(labels))}

    for index, pathname in enumerate(pathnames):
        if (os.path.isfile(pathname) == True):

            label = os.path.split(os.path.dirname(pathname))[-1]

            x[index] = numpy.load(pathname)

            y[index] = label_to_index[label]

    return x, y


def sample(directories):
    """
    Sample pathnames from directories.

    For each directory, samples are randomly selected equally across subdirectories.

    :param directories: List of directories to select samples from. Assumes subdirectories of each directory
                        correspond to class labels. Contents of subdirectories are NPY files containing data
                        of that label.
    :return: List of sampled pathnames.
    """
    pathnames = []

    for directory in directories:
        subdirectories = sorted(glob.glob(os.path.join(directory, "*")))

        subdirectory_pathnames = [glob.glob(os.path.join(subdirectory, "*")) for subdirectory in subdirectories]

        nsamples = max([len(pathnames) for pathnames in subdirectory_pathnames])
        #nsamples = 70000

        pathnames += [list(numpy.random.permutation(pathnames)[:nsamples]) for pathnames in subdirectory_pathnames]

    pathnames = sum(pathnames, [])

    return pathnames


def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: float(majority/count) for cls, count in counter.items()}


def collect_pathnames(directories, labels):
    """
    :param directories: List of directories to select samples from. Assumes subdirectories of each directory
                        correspond to class labels. Contents of subdirectories are NPY files containing data
                        of that label.
    :return: List of pathnames.
    """
    pathnames = []

    for directory in directories:
        subdirectories = sorted(glob.glob(os.path.join(directory, "*")))

        # transform the files of the same label into directory
        subdirectory_pathnames = [glob.glob("{}/*.npy".format(subdirectory)) for subdirectory in subdirectories ]

        nsamples = max([len(pathnames) for pathnames in subdirectory_pathnames])

        pathnames += [list(numpy.random.permutation(pathnames)[:nsamples]) for pathnames in subdirectory_pathnames]

    pathnames = sum(pathnames, [])

    return pathnames


def load_include(pathnames, labels, patient_to_include):
    """
    Load training and target data.

    Assumes data is stored in a directory corresponding to some class label.

    :param pathnames: List of image pathnames.
    :param labels: List of class labels.
    :return: Tuple (training, target) data, as NumPy arrays.
    """
    #print('All cells in treated patients: ',len(pathnames))
    pathnames = [x for x in pathnames if patient_to_include in x]
    print('Cells in this patient: ',len(pathnames))

    x = numpy.empty((len(pathnames),) + _shape(pathnames[0]), dtype=numpy.uint8)

    y = numpy.empty((len(pathnames),), dtype=numpy.uint8)

    label_to_index = {label: index for index, label in enumerate(sorted(labels))}

    for index, pathname in enumerate(pathnames):
        if (os.path.isfile(pathname) == True):

            label = os.path.split(os.path.dirname(pathname))[-1]

            x[index] = numpy.load(pathname)

            y[index] = label_to_index[label]

    return x, y


labels = ["Leukemic", "Normal", "Others"]


directories = ["/parsed_data/"]


samples = sample(directories)


pathnames_to_test = collect_pathnames(directories, labels)


patients_to_test = [
'LK155_pres',
'LK155_day11',
'LK155_day29',

'LK157_pres',
'LK157_day8',
'LK157_day15',

'LK167_pres',
'LK167_day12',

'LK171_pres',
'LK171_day11',

'LK172_pres',
'LK172_day29',

'LK174_pres',
'LK174_day29',

'LK175_pres',
'LK175_day8',

'LK177_pres',
'LK177_day8',

'LK181_pres',
'LK181_day8'
]


# build session running on GPU 1
configuration = tensorflow.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.visible_device_list = "1"
session = tensorflow.Session(config = configuration)

# apply session
keras.backend.set_session(session)

drop_how_many = 6

model = deepometry.model.Model(shape=(48,48,(8-drop_how_many)), units=len(labels))

model.compile()

model.model.load_weights('/models/resnet_drop_' + str(drop_how_many) + '_channels/mode


for patient_to_test in patients_to_test:

    for i in [drop_how_many]:#range(8):

        print("Testing: ", patient_to_test,", dropped ",i, " channels")


        model_directory = str('/models/resnet_to_test_' + patient_to_test + '_drop_' + str(i) + '_channels')

        if not os.path.exists(model_directory):
             os.makedirs(model_directory)


        xx_test, y_test = load_include(pathnames_to_test, labels, patient_to_test)

        x_test = xx_test[:,:,:,i:]

        #print("Testing set: ", x_test.shape)

        predictions = model.predict(x=x_test, batch_size=256, verbose=0)
        y_pred = numpy.argmax(predictions, -1)
        cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
        numpy.save(os.path.join(model_directory, str('confusion_matrix_'+patient_to_test + '_drop_' + str(i) + '_channels'+'.npy') ), cm)

        del(xx_test,x_test,y_test)

keras.backend.clear_session()
