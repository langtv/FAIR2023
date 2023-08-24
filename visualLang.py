#######################################
# Data Visualization and et cetera    #
# @author: A.Prof. Tran Van Lang, PhD #
# File: visualLang.py                 #
#######################################

from sklearn.metrics import *
from joblib import dump, load
import numpy as np
import matplotlib.pyplot as plt

import warnings
from sklearn.exceptions import UndefinedMetricWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UndefinedMetricWarning)

# Trực quan hoá dữ liệu
def visualization(estimator, model, X_test, y_test ):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    report = classification_report(y_test, y_pred)
        
    print(report)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('AUC:',roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    
    #dump( model, 'Bioassay'+estimator+'.joblib' )
    cm = confusion_matrix(y_test, y_pred)

    accuracy  = (cm[1,1] + cm[0,0])/(cm[1,0] + cm[1,1] + cm[0,0] + cm[0,1])
    precision = cm[0,0] / (cm[0,0] + cm[0,1])
    recall    = cm[0,0] / (cm[0,0] + cm[1,0])
    g_mean    = np.sqrt(recall * cm[1,1] / (cm[1,1] + cm[0,1]))
    f1_score  = 2 * (precision * recall) / (precision + recall)
    
    TActive   = cm[0,0]
    FActive   = cm[0,1]
    FInactive = cm[1,0]
    TInactive = cm[1,1]
    
    metric = precision, g_mean, auc, accuracy, recall, f1_score
    npred = TActive, FActive, TInactive, FInactive 

    ConfusionMatrixDisplay( confusion_matrix=cm,display_labels=["Active","Inactive"] ).plot()
    plt.show()
    return estimator,metric,npred
    
# Trực quan hoá dữ liệu
def visualization_proba(estimator, model, X_test, y_test ):
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    loss, accuracy = model.evaluate(X_test, y_test)
    auc = roc_auc_score(y_test,model.predict(X_test))
    report = classification_report(y_test, y_pred)

    print('\n',report)
    print('Loss:',loss)
    print('Accuracy:', accuracy)
    print('AUC:',auc)
    
    #dump( model, 'Bioassay'+estimator+'.joblib' )
    cm = confusion_matrix(y_test, y_pred)
    
    accuracy  = (cm[1,1] + cm[0,0])/(cm[1,0] + cm[1,1] + cm[0,0] + cm[0,1])
    precision = cm[0,0] / (cm[0,0] + cm[0,1])
    recall    = cm[0,0] / (cm[0,0] + cm[1,0])
    g_mean    = np.sqrt(recall * cm[1,1] / (cm[1,1] + cm[0,1]))
    f1_score  = 2 * (precision * recall) / (precision + recall)
    
    TActive   = cm[0,0]
    FActive   = cm[0,1]
    TInactive = cm[1,1]
    FInactive = cm[1,0]
    
    metric = precision, g_mean, auc, accuracy, recall, f1_score
    npred = TActive, FActive, TInactive, FInactive 

    ConfusionMatrixDisplay( confusion_matrix=cm,display_labels=["Active","Inactive"] ).plot()
    plt.show()
    return estimator,metric,npred
