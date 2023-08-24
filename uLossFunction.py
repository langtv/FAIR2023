#!/usr/bin/env python
# coding: utf-8

# In[1]:


#######################################################
# Activity Prediction by loss function regularization #
# @author: A.Prof. Tran Van Lang, PhD                 #
# File: usingLossFunction.py                          #
#######################################################

import pandas as pd
import csv
from time import time

import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from visualLang import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# In[2]:


path = 'data/BioassayDatasets/AID456red'
df_train = pd.read_csv( path + '_train.csv')
df_test  = pd.read_csv( path + '_test.csv' )

# Delete the missing values datapoint
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

# Split the dataset into features and labels
X_tr = df_train.drop('Outcome', axis=1)
X_te = df_test.drop('Outcome', axis=1)

y_tr = df_train['Outcome']
y_te = df_test['Outcome']

# Convert the features and labels to NumPy arrays
X_train = np.array(X_tr)
y_train = np.array(y_tr)

X_test = np.array(X_te)
y_test = np.array(y_te)

y_train = np.where(y_train == 'Active', 0, 1)
y_test = np.where(y_test == 'Active', 0, 1)

results = []

print("Số lượng mẫu dùng huấn luyện:", len(X_train))
num_minority_samples = np.sum(y_train==0)
num_majority_samples = np.sum(y_train==1)
print( 'Số mẫu của 2 nhãn là %d và %d' % (num_minority_samples,num_majority_samples) )


# In[3]:


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

# Compile mô hình với hàm mất mát tùy chỉnh (Focal Loss)
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.cast(y_true, dtype=K.floatx())   # Chuyển đổi kiểu dữ liệu của nhãn thành float32
        bce = K.binary_crossentropy(y_true, y_pred) # Tính binary cross-entropy    
        focal_weight = alpha * K.pow(1 - y_pred, gamma) # Tính focal weight dựa trên dự đoán và tham số alpha
        focal_loss = focal_weight * bce # Tính focal loss
        return K.mean(focal_loss)
    return focal_loss_fixed

model.compile(optimizer='adam', loss=focal_loss(gamma=2, alpha=0.25), metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

estimator,metric,npred = visualization_proba('FocalLoss_'+str(len(X_train)),model,X_test,y_test)
row = [estimator,metric[0],metric[1],metric[2],metric[3],metric[4],metric[5],npred[0],npred[1],npred[2],npred[3]]
results.append(row)


# In[4]:


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

# Compile mô hình với hàm mất mát tùy chỉnh (Balanced Cross Entropy)
def balanced_crossentropy(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # Tránh trường hợp log(0)
    y_true = K.cast(y_true, dtype=K.floatx())  # Chuyển đổi kiểu dữ liệu của nhãn thành float32
    loss = -K.mean(K.sum(y_true * K.log(y_pred), axis=-1))
    return loss

class_weight = {0: 10.0, 1: 1.0}
model.compile(optimizer='adam', loss=balanced_crossentropy, metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2, class_weight=class_weight)

estimator,metric,npred = visualization_proba('BalancedCrossEntropy_'+str(len(X_train)),model,X_test,y_test)
row = [estimator,metric[0],metric[1],metric[2],metric[3],metric[4],metric[5],npred[0],npred[1],npred[2],npred[3]]
results.append(row)


# In[5]:


from keras.losses import BinaryCrossentropy
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

# Define hàm mất mát sử dụng Weighted Cross-Entropy Loss
# Tăng trọng số của lớp positive để cân bằng dữ liệu mất cân bằng
def weighted_binary_crossentropy(y_true, y_pred):
    # Tăng trọng số của lớp positive (nhãn 0) so với lớp negative (nhãn 1)
    weight_positive = num_majority_samples / num_minority_samples
    weight_negative = 1.0
    bce = BinaryCrossentropy()
    y_true = K.cast(y_true, dtype=K.floatx())  # Chuyển đổi kiểu dữ liệu của nhãn thành float32
    return (weight_positive * y_true * bce(y_true, y_pred)) + (weight_negative * (1 - y_true) * bce(y_true, y_pred))

# Compile model với hàm mất mát là weighted_binary_crossentropy
model.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

estimator,metric,npred = visualization_proba('WeightedCrossEntropy_'+str(len(X_train)),model,X_test,y_test)
row = [estimator,metric[0],metric[1],metric[2],metric[3],metric[4],metric[5],npred[0],npred[1],npred[2],npred[3]]
results.append(row)


# In[6]:


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

def dice_loss(y_true, y_pred, smooth=1e-7):
    y_true = K.cast(y_true, dtype=K.floatx())  # Chuyển đổi kiểu dữ liệu của nhãn thành float32   
    intersection = K.sum(y_true * y_pred, axis=-1)
    dice = (2. * intersection + smooth) / (K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1) + smooth)
    dice_loss = 1.0 - dice
    return dice_loss
    
model.compile(optimizer='adam', loss=focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

estimator,metric,npred = visualization_proba('DiceLoss_'+str(len(X_train)),model,X_test,y_test)
row = [estimator,metric[0],metric[1],metric[2],metric[3],metric[4],metric[5],npred[0],npred[1],npred[2],npred[3]]
results.append(row)


# In[7]:


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=1.0):
    y_true = K.cast(y_true, dtype=K.floatx())  # Chuyển đổi kiểu dữ liệu của nhãn thành float32   
    # Tính các thành phần của Focal Tversky Loss
    tp = K.sum(y_true * y_pred, axis=-1)
    fp = K.sum((1 - y_true) * y_pred, axis=-1)
    fn = K.sum(y_true * (1 - y_pred), axis=-1)
    
    tversky = (tp + K.epsilon()) / (tp + alpha * fp + beta * fn + K.epsilon())
    focal_tversky = K.pow((1 - tversky), gamma)
    return -K.mean(focal_tversky)

model.compile(optimizer='adam', loss=focal_tversky_loss, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

estimator,metric,npred = visualization_proba('FocalTverskyLoss_'+str(len(X_train)),model,X_test,y_test)
row = [estimator,metric[0],metric[1],metric[2],metric[3],metric[4],metric[5],npred[0],npred[1],npred[2],npred[3]]
results.append(row)


# In[8]:


import tensorflow as tf

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

# Định nghĩa hàm mất mát Class-Balanced Loss
def class_balanced_loss(y_true, y_pred):
    y_true = K.cast(y_true, dtype=K.floatx())  # Chuyển đổi kiểu dữ liệu của nhãn thành float32   
    # Tính số lượng mẫu của từng lớp
    num_samples_per_class = tf.reduce_sum(y_true, axis=0)
    num_total_samples = tf.reduce_sum(num_samples_per_class)
    
    # Tính trọng số của từng lớp
    beta = num_total_samples / (num_samples_per_class * len(num_samples_per_class))
    weights = tf.reduce_sum(beta * y_true, axis=1)
    
    # Tính hàm mất mát
    bce = BinaryCrossentropy()
    loss = bce(y_true, y_pred)
    balanced_loss = tf.reduce_mean(weights * loss)
    return balanced_loss

model.compile(optimizer='adam', loss=class_balanced_loss, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

estimator,metric,npred = visualization_proba('ClassBalancedLoss_'+str(len(X_train)),model,X_test,y_test)
row = [estimator,metric[0],metric[1],metric[2],metric[3],metric[4],metric[5],npred[0],npred[1],npred[2],npred[3]]
results.append(row)


# In[9]:


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

# Định nghĩa hàm mất mát Focal Cosine Loss
def focal_cosine_loss(y_true, y_pred):
    y_true = K.cast(y_true, dtype=K.floatx())  # Chuyển đổi kiểu dữ liệu của nhãn thành float32   
    alpha = 0.25
    gamma = 2.0
    bce = BinaryCrossentropy() # Tính hàm mất mát Focal Loss
    focal_loss = alpha * tf.pow(1 - y_pred, gamma) * bce(y_true, y_pred)
    cosine_loss = tf.losses.cosine_similarity(y_true, y_pred) # Tính hàm mất mát Cosine Loss
    return tf.reduce_mean(focal_loss + cosine_loss) # Tổng hợp Focal Cosine Loss

# Compile mô hình với Focal Cosine Loss
model.compile(optimizer='adam', loss=focal_cosine_loss, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

estimator,metric,npred = visualization_proba('ClassBalancedLoss_'+str(len(X_train)),model,X_test,y_test)
row = [estimator,metric[0],metric[1],metric[2],metric[3],metric[4],metric[5],npred[0],npred[1],npred[2],npred[3]]
results.append(row)


# In[10]:


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

# Định nghĩa hàm mất mát Online Hard Example Mining (OHEM) Loss
def ohem_loss(y_true, y_pred):
    y_true = K.cast(y_true, dtype=K.floatx())  # Chuyển đổi kiểu dữ liệu của nhãn thành float32   
    n_pos = tf.reduce_sum(y_true)
    n_neg = tf.reduce_sum(1 - y_true)

    pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=1 - y_true, logits=y_pred)

    pos_loss_sorted, _ = tf.nn.top_k(pos_loss, k=tf.cast(n_pos, dtype=tf.int32))
    neg_loss_sorted, _ = tf.nn.top_k(neg_loss, k=tf.cast(n_neg, dtype=tf.int32))

    hard_pos_loss = tf.reduce_mean(pos_loss_sorted)
    hard_neg_loss = tf.reduce_mean(neg_loss_sorted)

    return hard_pos_loss + hard_neg_loss

# Compile mô hình với OHEM Loss
model.compile(optimizer='adam', loss=ohem_loss, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

estimator,metric,npred = visualization_proba('ClassBalancedLoss_'+str(len(X_train)),model,X_test,y_test)
row = [estimator,metric[0],metric[1],metric[2],metric[3],metric[4],metric[5],npred[0],npred[1],npred[2],npred[3]]
results.append(row)


# In[11]:


import csv
import os

filename = os.path.basename(path) + '_Loss'
with open(filename + '.csv', "w") as file:
    writer = csv.writer(file, delimiter=",")
    writer.writerow(['Phương pháp tiếp cận','Precision','G-mean','AUC','Accuracy','Recall','F1-score','Active: Đoán đúng','Active: Đoán nhầm','Inactive: Đoán đúng','Inactive: Đoán nhầm'])
    for row in results:
        writer.writerow(row)
file.close()

df = pd.read_csv(filename + '.csv')
df.to_excel(filename + '.xlsx', index=False)



