# -*- coding: utf-8 -*-

# 好好学习队

import glob
import numpy as np
import mahotas as mh
from sklearn import metrics
from sklearn import datasets
from mahotas.features import surf
from sklearn.svm import SVC 		# SVM（损失函数hinge LOSS）
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

# 配置utf-8输出环境
try:
	reload(sys)
	sys.setdefaultencoding('utf-8')
except:
	pass

# 读入图片
train = datasets.load_files("image_datas")
# 返回值：Bunch Dictionary-like object.主要属性有
#   data:原始数据；
#   filenames:每个文件的名字；
#   target:类别标签（从0开始的整数索引）；
#   target_names:类别标签的具体含义（由子文件夹的名字`category_1_folder`等决定）

#验证数据是否加载成功
print len(train.data),len(train.filenames),len(train.target),len(train.target_names)
for i in range(len(train.target_names)):
	print i, ' ', train.target_names[i]
#print train.filenames[0], '\n'
#print train.target[0], ' ', train.target_names[train.target[0]]

# ----------提取特征----------
# 从每张图片中提取SURF特征
surf_features = []
counter = 0
for f in train.filenames:
	print('Reading image:', f)
	image = mh.imread(f, as_grey=True)
	surf_features.append(surf.surf(image)[:, 6:])

train_len = len(train.filenames)
train_surf_features = np.concatenate(surf_features[:train_len])

# 所有特征用KMeans聚类
n_clusters=1000
print('Clustering', len(train_surf_features), 'features')
estimator = MiniBatchKMeans(init_size=3000, n_clusters=n_clusters)
estimator.fit_transform(train_surf_features)

# BOW方法生成每个图片的特征
X_train = []
for instance in surf_features[:train_len]:
	if len(instance) == 0:
		features = []
	else:
		clusters = estimator.predict(instance)
		features = np.bincount(clusters)

	if len(features) < n_clusters:
		features = np.append(features, np.zeros((1, n_clusters - len(features))))

	X_train.append(features)

# ----------训练分类器----------
# LR
print "----------LR----------"
model = LogisticRegression(C=0.001, penalty='l2')
model.fit_transform(X_train, train.target)
# 测试
expected = train.target
predicted = model.predict(X_train)
# 结果
print "训练集上的准确率", np.mean(predicted == expected)
print metrics.classification_report(expected, predicted, target_names = train.target_names)
print metrics.confusion_matrix(expected, predicted)
# 交叉检验
scores = cross_val_score(model, X_train, train.target, cv = 10)
print "交叉检验", scores
print "交叉检验均值", scores.mean()
print "交叉检验方差", scores.var()
print "交叉检验标准差", scores.std()

# SVM(default with kernel 'rbf' )
print "----------SVM----------"
model = SVC(decision_function_shape='ovo',kernel='linear')
model.fit(X_train, train.target)
# 预测
expected = train.target
predicted = model.predict(X_train)
# 结果
print "训练集上的准确率", np.mean(predicted == expected)
print metrics.classification_report(expected, predicted, target_names = train.target_names)
print metrics.confusion_matrix(expected, predicted)
# 交叉检验
scores = cross_val_score(model, X_train, train.target, cv = 10)
print "交叉检验", scores
print "交叉检验均值", scores.mean()
print "交叉检验方差", scores.var()
print "交叉检验标准差", scores.std()



