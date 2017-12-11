# -*- coding: utf-8 -*-

# 好好学习队

import os
import sys
import numpy as np
from sklearn import metrics 
from sklearn import datasets
from sklearn.svm import SVC                         # SVM（损失函数hinge LOSS）
from sklearn.cluster import KMeans                  # Kmeans
from sklearn.naive_bayes import GaussianNB          # Naive Bayes
from sklearn.naive_bayes import MultinomialNB       # Naive Bayes
from sklearn.linear_model import SGDClassifier      # 随机梯度下降(快速版SVM、LR?抽出部分样本加快计算,LOSS为hinge时是SVM，LOSS为log时是LR)
from sklearn.tree import DecisionTreeClassifier     # 决策树
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.linear_model import LogisticRegression # LR（损失函数log loss）
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer  
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#配置utf-8输出环境
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

#加载数据
train = datasets.load_files("text_datas", None, None, True, True, 'utf-8', 'strict', 0)
#返回值：Bunch Dictionary-like object.主要属性有
#   data:原始数据;
#   filenames:每个文件的名字；
#   target:类别标签（从0开始的整数索引）；
#   target_names:类别标签的具体含义（由子文件夹的名字`category_1_folder`等决定）

#验证数据是否加载成功
print len(train.data),len(train.filenames),len(train.target),len(train.target_names)
for i in range(len(train.target_names)):
	print i, ' ', train.target_names[i]
#print "\n".join(train.data[0].split("\n")[:3])
#print train.filenames[0], '\n'
#print train.target[0], ' ', train.target_names[train.target[0]]

X_train, X_test, y_train, y_test = train_test_split(train.data, train.target, test_size=0.1, random_state=512)

#提取特征
count_vect = CountVectorizer(decode_error='ignore', max_features=9000)
X_train_counts = count_vect.fit_transform(X_train)
print X_train_counts.shape

tf_transformer = TfidfTransformer(use_idf = False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print X_train_tf.shape

X_test_counts = count_vect.transform(X_test)
X_test_tf = tf_transformer.transform(X_test_counts)

# 训练分类器,预测结果分析
# MultinomialNB
print "----------MultinomialNB----------"
model = MultinomialNB()
model.fit(X_train_tf, y_train)

acc_MNB = model.score(X_train_tf, y_train)
print acc_MNB
# 预测
expected = y_test
predicted_MNB = model.predict(X_test_tf)
# 结果
print np.mean(predicted_MNB == expected)
print metrics.classification_report(expected, predicted_MNB)
print metrics.confusion_matrix(expected, predicted_MNB)

# SGD
print "----------SGD----------"
model = SGDClassifier(loss = 'hinge', penalty = 'l2', alpha = 1e-3, n_iter = 5, random_state = 512)
model.fit(X_train_tf, y_train)

acc_SGD = model.score(X_train_tf, y_train)
print acc_SGD
# 预测
expected = y_test
predicted_SGD = model.predict(X_test_tf)
# 结果
print np.mean(predicted_SGD == expected)
print metrics.classification_report(expected, predicted_SGD)
print metrics.confusion_matrix(expected, predicted_SGD)

# SVM(default with kernel 'rbf' )
print "----------SVM----------"
model = SVC(decision_function_shape='ovo',kernel='linear')
model.fit(X_train_tf, y_train)

acc_SVM  = model.score(X_train_tf, y_train)
print acc_SVM
# 预测
expected = y_test
predicted_SVM = model.predict(X_test_tf)
# 结果
print np.mean(predicted_SVM == expected)
print metrics.classification_report(expected, predicted_SVM)
print metrics.confusion_matrix(expected, predicted_SVM)

# LR
print "----------Logistic Regression----------"
model = LogisticRegression()
model.fit(X_train_tf, y_train)

acc_LR = model.score(X_train_tf, y_train)
print acc_LR
# 预测
expected = y_test
predicted_LR = model.predict(X_test_tf)
# 结果
print np.mean(predicted_LR == expected)
print metrics.classification_report(expected, predicted_LR)
print metrics.confusion_matrix(expected, predicted_LR)

# GaussianNB
print "----------GaussianNB----------"
model = GaussianNB()
model.fit(X_train_tf.toarray(), y_train)
# 训练集上的准确率
acc_GNB = model.score(X_train_tf.toarray(), y_train)
print acc_GNB
# 预测
expected = y_test
predicted_GNB = model.predict(X_test_tf.toarray())
# 结果
print np.mean(predicted_GNB == expected)
print metrics.classification_report(expected, predicted_GNB)
print metrics.confusion_matrix(expected, predicted_GNB)

print "----------Decision Tree----------"
model = DecisionTreeClassifier()
model.fit(X_train_tf, y_train)
# 训练集上的准确率
acc_DT = model.score(X_train_tf, y_train)
print acc_DT
# 预测
expected = y_test
predicted_DT = model.predict(X_test_tf)
# 结果
print np.mean(predicted_DT == expected)
print metrics.classification_report(expected, predicted_DT)
print metrics.confusion_matrix(expected, predicted_DT)

# votes
print "----------Votes----------"
predicted = np.zeros(len(y_test))
for i in range(len(y_test)):
	votes = np.zeros(9)
	votes[predicted_MNB[i]] += acc_MNB;
	votes[predicted_SGD[i]] += acc_SGD;
	votes[predicted_SVM[i]] += acc_SVM;
	votes[predicted_LR[i]] += acc_LR;
	votes[predicted_GNB[i]] += acc_GNB;
	votes[predicted_DT[i]] += acc_DT;
	max_num = 0
	for j in range(9):
		if(votes[j] > max_num):
			max_num = votes[j]
			predicted[i] = j

print np.mean(predicted == expected)
print metrics.classification_report(expected, predicted)
print metrics.confusion_matrix(expected, predicted)


