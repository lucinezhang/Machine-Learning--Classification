# -*- coding: utf-8 -*-

# 好好学习队

import os
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adadelta, Adagrad

from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# 配置utf-8输出环境
try:
	reload(sys)
	sys.setdefaultencoding('utf-8')
except:
	pass

# 读入数据
train = datasets.load_files("data/train")
# 返回值：Bunch Dictionary-like object.主要属性有
#   data:原始数据；
#   filenames:每个文件的名字；
#   target:类别标签（从0开始的整数索引）；
#   target_names:类别标签的具体含义（由子文件夹的名字`category_1_folder`等决定）

# 验证数据是否加载成功
print len(train.data),len(train.filenames),len(train.target),len(train.target_names)
for i in range(len(train.target_names)):
	print i, ' ', train.target_names[i]
# print train.filenames[0], '\n'
# print train.target[0], ' ', train.target_names[train.target[0]]

# 读入图片
# dimensions of our images.
img_width, img_height = 128, 128
input_shape = (3, img_width, img_height)
print input_shape

l = len(train.filenames)
X = np.empty((l, 3, img_width, img_height), dtype="float32")  
for i in range(l):
	print 'reading image: ', i, train.filenames[i]
	img = Image.open(train.filenames[i])
	img = img.resize((img_width, img_height))
	arr = np.asarray(img, dtype="float32")
	if img.mode == 'L':
		X[i,0,:,:] = arr
		X[i,0,:,:] = arr
		X[i,0,:,:] = arr
	else:
		X[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]  

# convert integers to dummy variables (one hot encoding)
dummy_y = np_utils.to_categorical(train.target, 19)
print dummy_y

# define model structure
model = Sequential()
  
# 第一个卷积层，4个卷积核，每个卷积核大小3*3。 
# 激活函数用tanh  
# 还可以在model.add(Activation('tanh'))后加上dropout的技巧: model.add(Dropout(0.5))  
model.add(Conv2D(4, (3, 3), padding='valid', input_shape=input_shape))
model.add(Activation('tanh'))   
model.add(MaxPooling2D(pool_size=(2, 2)))

# 第二个卷积层，8个卷积核，每个卷积核大小3*3。
# 激活函数用tanh  
# 采用maxpooling，poolsize为(2,2)  
model.add(Conv2D(8, (3, 3), padding='valid'))  
model.add(Activation('tanh'))  
model.add(MaxPooling2D(pool_size=(2, 2)))  
  
# 第三个卷积层，16个卷积核，每个卷积核大小3*3  
# 激活函数用tanh  
# 采用maxpooling，poolsize为(2,2)  
model.add(Conv2D(16, (3, 3), padding='valid'))  
model.add(Activation('tanh'))  
model.add(MaxPooling2D(pool_size=(2, 2)))  
  
# 全连接层，先将前一层输出的二维特征图flatten为一维的。  
model.add(Flatten())  
model.add(Dense(128, kernel_initializer='normal'))  
model.add(Activation('tanh'))  
model.add(Dropout(0.25))

# Softmax分类，输出是19类别  
model.add(Dense(19, kernel_initializer='normal'))  
model.add(Activation('softmax'))

sgd = SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)  
model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode="categorical", metrics=['accuracy']) 

# splitting data into training set and test set. If random_state is set to an integer, the split datasets are fixed.
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.1, random_state=0)
print len(X_train),len(X_test)
nb_train_samples = len(X_train)
nb_validation_samples = len(X_test)

# 数据提升
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

batch_size = 120
train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size)

validation_generator = test_datagen.flow(X_test, Y_test, batch_size=batch_size)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=1000,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')

