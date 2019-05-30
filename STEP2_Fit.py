
# coding: utf-8

import sys
sys.path.insert(0, '/home/.conda/envs/tensorflow/lib/python3.6/site-packages')

import glob
import os

import keras
import matplotlib.pyplot
import numpy
import pandas
import pkg_resources
import tensorflow

import deepometry.model
import deepometry.utils


#--------#
drop = 6
model_directory = '/models/drop'+str(drop)

# build session running on GPU 1:
configuration = tensorflow.ConfigProto()
configuration.gpu_options.visible_device_list = "3"
#--------#


patients_to_test = ['157pres','157day8','157day15','171pres','171day11','172pres','172day29','175pres','175day8','177pres','177day8']

directories = ["/parsed_data/"]

xx, y, units = deepometry.utils.load(directories, patients_to_test, sample=True)


# Drop channels

x = xx[:,:,:,drop:]
del(xx)
print('Shape of x to be trained on: ', x.shape)


# apply session
configuration.gpu_options.allow_growth = True
session = tensorflow.Session(config = configuration)
keras.backend.set_session(session)


model = deepometry.model.Model(shape=x.shape[1:], units=units)

model.compile()

model.fit(
    x,
    y,
    batch_size=32,
    class_weight="auto",
    epochs=512,
    validation_split=0.3,
    verbose=1
)


if not os.path.exists(model_directory):
    os.makedirs(model_directory)

model.model.save( os.path.join(model_directory, 'resnet50_drop'+str(drop)+'.h5') )
del(x)

keras.backend.clear_session()
