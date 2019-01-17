from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os
from PIL import Image
import pickle
import keras
from keras.layers import *
from keras.applications.vgg16 import VGG16
from keras import Model
from keras.utils import multi_gpu_model
from keras import backend as K
from keras.models import load_model
import os
import sys
from keras.losses import categorical_crossentropy
import random

gpus = "2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
data_path = "../anime_i2v_data"
gpu_num = len(gpus.split(","))
bs = 96 * gpu_num
epochs = 300
single_classes = 512
classes = single_classes * 3
input_tensor = Input(shape=(256,256,3))
model_name = sys.argv[0].replace(".py","")
now_e = 0
def get_model():
    for i in range(epochs):
        e = epochs - i
        if(os.path.exists("%s_at_epoch_%d.h5"%(model_name,e))):
            now_e = e
            existing_model = load_model ("%s_at_epoch_%d.h5"%(model_name,e))
            #for layer in existing_model.layers:
            #    layer.trainable = True
            return existing_model
    vgg16_model = VGG16(include_top=False, weights="imagenet", input_shape=(256,256,3), input_tensor=input_tensor)
    for layer in vgg16_model.layers[0:12]:
        layer.trainable = False
    x = vgg16_model.layers[-1].output
    x = BatchNormalization()(x)
    low_f = Dropout(0.3,name="low_features")(x)

    a = AveragePooling2D(7)(low_f)
    a = Flatten()(a)
    a = Dense(single_classes)(a)
    a = Dense(single_classes)(a)
    a = Dropout(0.3)(a)
    a = Activation("sigmoid",name="attr_label")(a)

    high_f = Conv2D(512,3,activation='relu',name="high_features")(low_f)
    high_f = BatchNormalization()(high_f)
    high_f = Dropout(0.3)(high_f)
    high_f = AveragePooling2D(5)(high_f)

    y0 = Flatten()(high_f)
    y0 = Dense(single_classes)(y0)
    y0 = Dropout(0.3)(y0)
    y0 = Dense(single_classes)(y0)
    y0 = Lambda(lambda x: K.tf.nn.softmax(x),name="chara_label")(y0)

    y1 = Flatten()(high_f)
    y1 = Dense(single_classes)(y1)
    y1 = Dropout(0.3)(y1)
    y1 = Dense(single_classes)(y1)
    y1 = Lambda(lambda x: K.tf.nn.softmax(x),name="cpy_label")(y1)

    return Model(inputs=vgg16_model.input, outputs=[y0, y1, a])

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def load_data(path,index=None,train_set=True):
    if(train_set):
        x_npz = path+"/train_x_"+str(index)+".npz"
        y0_npy = path + "/train_y0_" + str(index) + ".npy"
        y1_npy = path + "/train_y1_" + str(index) + ".npy"
        y2_npy = path + "/train_y2_" + str(index) + ".npy"
    else:
        x_npz = path+"/val_x.npz"
        y0_npy = path + "/val_y0.npy"
        y1_npy = path + "/val_y1.npy"
        y2_npy = path + "/val_y2.npy"
    x = np.load(x_npz)
    x = x['arr_0']
    y0 = np.load(y0_npy)
    y1 = np.load(y1_npy)
    y2 = np.load(y2_npy)
    return x,y0,y1,y2


def my_generator(train_set=True,path="../anime_i2v_data",batch_size=96):

    img_gen = ImageDataGenerator()
    if(train_set):
        index = 0
        while(True):
            x,y0,y1,y2=load_data(path,index,train_set=True)
            seed = np.random.randint(100)
            random.Random(seed).shuffle(x)
            random.Random(seed).shuffle(y0)
            random.Random(seed).shuffle(y1)
            random.Random(seed).shuffle(y2)

            for i in range(len(x)//batch_size):
                b_x = x[i*batch_size:(i+1)*batch_size]
                b_y0 = y0[i*batch_size:(i+1)*batch_size]
                b_y1 = y1[i*batch_size:(i+1)*batch_size]
                b_y2 = y2[i*batch_size:(i+1)*batch_size]
                b_x = (b_x-127.5)/127.5
                yield b_x,[b_y0,b_y1,b_y2]
            index = (index+1) % 10
    else:
        x,y0,y1,y2=load_data(path,train_set=False)
        seed = np.random.randint(100)
        while(True):
            random.Random(seed).shuffle(x)
            random.Random(seed).shuffle(y0)
            random.Random(seed).shuffle(y1)
            random.Random(seed).shuffle(y2)
            for i in range(len(x)//batch_size):
                b_x = x[i*batch_size:(i+1)*batch_size]
                b_y0 = y0[i*batch_size:(i+1)*batch_size]
                b_y1 = y1[i*batch_size:(i+1)*batch_size]
                b_y2 = y2[i*batch_size:(i+1)*batch_size]
                b_x = (b_x-127.5)/127.5
                yield b_x,[b_y0,b_y1,b_y2]


# ref:https://github.com/keras-team/keras/issues/8649
class MyCbk(keras.callbacks.Callback):

    def __init__(self, model):
        self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save("%s_at_epoch_%d.h5"%(model_name,epoch+now_e))
        # keras.callbacks.ModelCheckpoint(filepath="i2v.h5",monitor="val_loss",save_best_only=True)]


model = get_model()
model.summary()
my_train_generator = my_generator(train_set=True, batch_size=bs)
my_valid_generator = my_generator(train_set=False, batch_size=bs)
try:
    parallel_model = multi_gpu_model(model,gpus=gpu_num, cpu_merge=False)
    print("Training using multiple GPUs..")
except Exception as e:
    parallel_model = model
    print(e)
    print("Training using single GPU or CPU..")

parallel_model.compile(loss=['categorical_crossentropy','categorical_crossentropy','binary_crossentropy'],
                       optimizer='adam',
                       metrics=["accuracy",recall,precision])
saving_model_cbk = MyCbk(model)
callbacks = [keras.callbacks.TensorBoard(log_dir=model_name),
             saving_model_cbk]
TEST = False
if(TEST):
    STEP_SIZE_TRAIN = 1
    STEP_SIZE_VALID = 1
    parallel_model.fit_generator(generator=my_train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                                 validation_data=my_valid_generator, validation_steps=STEP_SIZE_VALID,
                                 #use_multiprocessing=True, workers=20,
                                 callbacks=callbacks,
                                 epochs=300)
else:
    STEP_SIZE_TRAIN = 2000000//bs
    STEP_SIZE_VALID = 200000//bs
    parallel_model.fit_generator(generator=my_train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                                 validation_data=my_valid_generator, validation_steps=STEP_SIZE_VALID,
                                 #use_multiprocessing=True, workers=120,
                                 callbacks=callbacks,
                                 epochs=300)
