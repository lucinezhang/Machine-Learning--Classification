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

# 配置utf-8输出环境
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

# 加载数据
train = datasets.load_files("text_datas", None, None, True, True, 'utf-8', 'strict', 0)
# 返回值：Bunch Dictionary-like object.主要属性有
#   data:原始数据;
#   filenames:每个文件的名字；
#   target:类别标签（从0开始的整数索引）；
#   target_names:类别标签的具体含义（由子文件夹的名字`category_1_folder`等决定）

# 验证数据是否加载成功
print len(train.data),len(train.filenames),len(train.target),len(train.target_names)
for i in range(len(train.target_names)):
	print i, ' ', train.target_names[i]
# print "\n".join(train.data[0].split("\n")[:3])
# print train.filenames[0], '\n'
# print train.target[0], ' ', train.target_names[train.target[0]]

# 分割训练集和测试集，random_state取值可使分割随机
X_train, X_test, y_train, y_test = train_test_split(train.data, train.target, test_size=0.1, random_state=512)

# 提取特征
count_vect = CountVectorizer(decode_error='ignore', max_features=9000)
X_train_counts = count_vect.fit_transform(X_train)
# print X_train_counts.shape

# 使用tf还是tf_idf，use_idf = False or True
tf_transformer = TfidfTransformer(use_idf = False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print "训练样本特征大小：", X_train_tf.shape

X_test_counts = count_vect.transform(X_test)
# print X_test_counts.shape
X_test_tf = tf_transformer.transform(X_test_counts)
print "测试样本特征大小：", X_test_tf.shape

X_counts = count_vect.fit_transform(train.data)
# print X_counts.shape
tfidf_transformer = TfidfTransformer(use_idf = False).fit(X_counts)
X_tf = tfidf_transformer.transform(X_counts)
print "所有样本特征大小：", X_tf.shape

# 训练分类器,预测结果分析

# MultinomialNB
print "----------MultinomialNB----------"
model = MultinomialNB()
model.fit(X_train_tf, y_train)
# 训练集上的准确率
predicted_train = model.predict(X_train_tf)
print "训练集上的准确率：", np.mean(predicted_train == y_train)
# 预测
expected = y_test
predicted = model.predict(X_test_tf)
# 结果
print "测试集上的准确率：", np.mean(predicted == expected)
print metrics.classification_report(expected, predicted)
print metrics.confusion_matrix(expected, predicted)

# 交叉检验
scores = cross_val_score(model, X_tf, train.target, cv = 10)
print "交叉检验", scores
print "交叉检验均值", scores.mean()
print "交叉检验方差", scores.var()
print "交叉检验标准差", scores.std()

# GaussianNB
print "----------GaussianNB----------"
model = GaussianNB()
model.fit(X_train_tf.toarray(), y_train)
# 训练集上的准确率
print "训练集上的准确率：", model.score(X_train_tf.toarray(), y_train)
# 预测
expected = y_test
predicted = model.predict(X_test_tf.toarray())
# 结果
print "测试集上的准确率：", np.mean(predicted == expected)
print metrics.classification_report(expected, predicted)
print metrics.confusion_matrix(expected, predicted)
# 交叉检验
scores = cross_val_score(model, X_tf.toarray(), train.target, cv = 10)
print "交叉检验", scores
print "交叉检验均值", scores.mean()
print "交叉检验方差", scores.var()
print "交叉检验标准差", scores.std()

# KMeans
print "----------KMeans----------"
model = KMeans(n_clusters = 9)
model.fit(X_train_tf, y_train)
# 训练集上的准确率
print "训练集上的准确率：", model.score(X_train_tf, y_train)
# 预测
expected = y_test
predicted = model.predict(X_test_tf)
# 结果
print "测试集上的准确率：", np.mean(predicted == expected)
print metrics.classification_report(expected, predicted)
print metrics.confusion_matrix(expected, predicted)

# KNN(default with k=5)
print "----------KNN----------"
model = KNeighborsClassifier()
model.fit(X_train_tf, y_train)
# 训练集上的准确率
print "训练集上的准确率：", model.score(X_train_tf, y_train)
# 预测
expected = y_test
predicted = model.predict(X_test_tf)
# 结果
print "测试集上的准确率：", np.mean(predicted == expected)
print metrics.classification_report(expected, predicted)
print metrics.confusion_matrix(expected, predicted)
scores = cross_val_score(model, X_tf, train.target, cv = 10)
# 交叉检验
print "交叉检验", scores
print "交叉检验均值", scores.mean()
print "交叉检验方差", scores.var()
print "交叉检验标准差", scores.std()

# SGD
print "----------SGD----------"
model = SGDClassifier(loss = 'hinge', penalty = 'l2', alpha = 1e-3, n_iter = 5, random_state = 42)
model.fit(X_train_tf, y_train)
# 训练集上的准确率
print "训练集上的准确率：", model.score(X_train_tf, y_train)
# 预测
expected = y_test
predicted = model.predict(X_test_tf)
# 结果
print "测试集上的准确率：", np.mean(predicted == expected)
print metrics.classification_report(expected, predicted)
print metrics.confusion_matrix(expected, predicted)
# 交叉检验
scores = cross_val_score(model, X_tf, train.target, cv = 10)
print "交叉检验", scores
print "交叉检验均值", scores.mean()
print "交叉检验方差", scores.var()
print "交叉检验标准差", scores.std()

# SVM(default with kernel 'rbf' )
print "----------SVM----------"
model = SVC(decision_function_shape='ovo',kernel='linear')
model.fit(X_train_tf, y_train)
# 训练集上的准确率
predicted_train = model.predict(X_train_tf)
print "训练集上的准确率：", np.mean(predicted_train == y_train)
# 预测
expected = y_test
predicted = model.predict(X_test_tf)
# 结果
print "测试集上的准确率：", np.mean(predicted == expected)
# print accuracy_score(predicted1, expected)
print metrics.classification_report(expected, predicted)
print metrics.confusion_matrix(expected, predicted)

# 交叉检验
scores = cross_val_score(model, X_tf, train.target, cv = 10)
print "交叉检验", scores
print "交叉检验均值", scores.mean()
print "交叉检验方差", scores.var()
print "交叉检验标准差", scores.std()

print "-----SVM--AddWeights-----"
model = SVC(decision_function_shape='ovo',kernel='linear')
weights = np.ones(len(X_train))
# 修改训错了的样本的权重
for i in range(len(X_train)):
	if predicted_train[i] != y_train[i]:
		weights[i] += 0.5
# 用新的权重进行训练
model.fit(X_train_tf, y_train, sample_weight = weights)
predicted_train = model.predict(X_train_tf)
print "训练集上的准确率：", np.mean(predicted_train == y_train)
# 预测
expected = y_test
predicted = model.predict(X_test_tf)
# 结果
print "测试集上的准确率：", np.mean(predicted == expected)
print metrics.classification_report(expected, predicted)
print metrics.confusion_matrix(expected, predicted)

print "-----SVM--AddNewWeights-----"
model = SVC(decision_function_shape='ovo',kernel='linear')
# 修改训错了的样本的权重
for i in range(len(X_train)):
	if predicted_train[i] != y_train[i]:
		weights[i] += 0.25
# 用新的权重进行训练
model.fit(X_train_tf, y_train, sample_weight = weights)
predicted_train = model.predict(X_train_tf)
print "训练集上的准确率：", np.mean(predicted_train == y_train)
# 预测
expected = y_test
predicted3 = model.predict(X_test_tf)
# 结果
print "测试集上的准确率：", np.mean(predicted == expected)
print metrics.classification_report(expected, predicted)
print metrics.confusion_matrix(expected, predicted)

#decision tree（在训练数据上跑准确率很高）
print "----------Decision Tree----------"
model = DecisionTreeClassifier()
model.fit(X_train_tf, y_train)
# 训练集上的准确率
print "训练集上的准确率：", model.score(X_train_tf, y_train)
# 预测
expected = y_test
predicted = model.predict(X_test_tf)
# 结果
print "测试集上的准确率：", np.mean(predicted == expected)
print metrics.classification_report(expected, predicted)
print metrics.confusion_matrix(expected, predicted)
# 交叉检验
scores = cross_val_score(model, X_tf, train.target, cv = 10)
print "交叉检验", scores
print "交叉检验均值", scores.mean()
print "交叉检验方差", scores.var()
print "交叉检验标准差", scores.std()

#LR
print "----------Logistic Regression----------"
model = LogisticRegression()
model.fit(X_train_tf, y_train)
# 训练集上的准确率
print "训练集上的准确率：", model.score(X_train_tf, y_train)
# 预测
expected = y_test
predicted = model.predict(X_test_tf)
# 结果
print "测试集上的准确率：", np.mean(predicted == expected)
print metrics.classification_report(expected, predicted)
print metrics.confusion_matrix(expected, predicted)
# 交叉检验
scores = cross_val_score(model, X_tf, train.target, cv = 10)
print "交叉检验", scores
print "交叉检验均值", scores.mean()
print "交叉检验方差", scores.var()
print "交叉检验标准差", scores.std()



