import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
import warnings
import os
import sys
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


#Using GPUs
os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
else:
  print("No GPU")

# Remove FutureWarning that appears during training
warnings.simplefilter(action='ignore', category=FutureWarning)

#Load dataset
train = pd.read_csv('suicidal_ideation_data1.csv')

# Split trainig data, test data
x = train.drop(columns=['Suicide thinking'], axis=1)
y = train['Suicide thinking']

# Know the number of data classes
from collections import Counter
counter1 = Counter(x)
counter2 = Counter(y)
print(counter1)
print(counter2)

# Proceed with raw data without scaling or pca
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=777)

#Undersampling is performed to resolve imbalanced data
X_train, y_train = RandomUnderSampler(random_state=0).fit_resample(X_train, y_train)
# Know the number of data classes after undersampling.
from collections import Counter
counter3 = Counter(X_train)
print(counter3)

# Training performed using logistic regression methodology
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)

#Evaluate generated model performance
cfx = confusion_matrix(y_test, lr_y_pred)
sensitivity = cfx[0,0]/(cfx[0,0]+cfx[0,1])
specificity = cfx[1,1]/(cfx[1,0]+cfx[1,1])
print('Logistic Regression Sensitivity = \n', sensitivity)
print('Logistic Regression Specificity = \n', specificity)
print('Logistic Regression Accuracy = ', accuracy_score(y_true=y_test, y_pred=lr_y_pred))
print('Logistic Regression AUC = ', roc_auc_score(y_true=y_test, y_score= lr_y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test, lr_y_pred))
print("Logistic Regression Finish""\n")

# Training performed using gradient boosting methodology
from sklearn.ensemble import GradientBoostingClassifier
gbc_clf = GradientBoostingClassifier(random_state=51, n_estimators=1000, learning_rate=0.001)
gbc_clf.fit(X_train, y_train)

gradientboosting_pred = gbc_clf.predict(X_test)

# Feature importance during gradient boosting training
ser = pd.Series(gbc_clf.feature_importances_, index= X_train.columns)
# Sort in descending order
top10 = ser.sort_values(ascending=False)[:10]
print(top10)
# Display Feature Importance as an image
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2.0) #Change the picture font size
plt.figure(figsize=(8,10))
plt.title('Feature Importances Top 10')
sns.barplot(x=top10, y=top10.index)
plt.show()
print("GradientBoosting Feature Importance Finish""\n")

#Evaluate generated model performance
cfx = confusion_matrix(y_test, gradientboosting_pred)
sensitivity = cfx[0,0]/(cfx[0,0]+cfx[0,1])
specificity = cfx[1,1]/(cfx[1,0]+cfx[1,1])
print('GradientBoosting Sensitivity = \n', sensitivity)
print('GradientBoosting Specificity = \n', specificity)
print('GradientBoosting Accuracy = ', accuracy_score(y_true=y_test, y_pred=gradientboosting_pred))
print('GradientBoosting AUC = ', roc_auc_score(y_true=y_test, y_score= gradientboosting_pred))

print(classification_report(y_test, gradientboosting_pred))
print("GradientBoosting Finish""\n")


# Training performed using Random forest methodology
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
forest_params = {'max_depth': range(1, 20), 'max_features': ['auto', 'sqrt', 'log2']}

gridsearch_rf = GridSearchCV(RandomForestClassifier(n_estimators=1000,
                                                    random_state=17, n_jobs=-1), forest_params, cv=5, verbose=1)
gridsearch_rf.fit(X_train, y_train)
gridsearch_rf_pred = gridsearch_rf.predict(X_test)

cfx = confusion_matrix(y_test, gridsearch_rf_pred)
sensitivity = cfx[0,0]/(cfx[0,0]+cfx[0,1])
specificity = cfx[1,1]/(cfx[1,0]+cfx[1,1])
print('GridSearch Random Forest Sensitivity = \n', sensitivity)
print('GridSearch Random Forest Specificity = \n', specificity)
print('GridSearch Random Forest Accuracy = ', accuracy_score(y_true=y_test, y_pred=gridsearch_rf_pred))
print('GridSearch Random Forest AUC = ', roc_auc_score(y_true=y_test, y_score= gridsearch_rf_pred))

print(classification_report(y_test, gridsearch_rf_pred))
print("GridSearch Random Forest Finish""\n")


# Training performed using Naive Bayes methodology
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
naivebayse_pred = gnb.predict(X_test)

cfx = confusion_matrix(y_test, naivebayse_pred)
sensitivity = cfx[0,0]/(cfx[0,0]+cfx[0,1])
specificity = cfx[1,1]/(cfx[1,0]+cfx[1,1])
print('Naive Bayse Sensitivity = \n', sensitivity)
print('Naive Bayse Specificity = \n', specificity)
print('Naive Bayse Accuracy = ', accuracy_score(y_true=y_test, y_pred=naivebayse_pred))
print('Naive Bayse AUC = ', roc_auc_score(y_true=y_test, y_score= naivebayse_pred))

print(classification_report(y_test, naivebayse_pred))
print("Naive Bayse Finish""\n")

#Training performed using k-NN methodology
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=1, n_neighbors=2, p=2,
                     weights='uniform')
knn_pred = knn.predict(X_test)

cfx = confusion_matrix(y_test, knn_pred)
sensitivity = cfx[0,0]/(cfx[0,0]+cfx[0,1])
specificity = cfx[1,1]/(cfx[1,0]+cfx[1,1])
print('KNN Sensitivity = \n', sensitivity)
print('KNN Specificity = \n', specificity)
print('KNN Accuracy = ', accuracy_score(y_true=y_test, y_pred=knn_pred))
print('KNN AUC = ', roc_auc_score(y_true=y_test, y_score= knn_pred))

print(classification_report(y_test, knn_pred))
print("KNN Finish""\n")


#Training performed using SVM methodology
from sklearn.svm import SVC
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

cfx = confusion_matrix(y_test, svm_pred)
sensitivity = cfx[0,0]/(cfx[0,0]+cfx[0,1])
specificity = cfx[1,1]/(cfx[1,0]+cfx[1,1])
print('SVM Sensitivity = \n', sensitivity)
print('SVM Specificity = \n', specificity)
print('SVM Accuracy = ', accuracy_score(y_true=y_test, y_pred=svm_pred))
print('SVM AUC = ', roc_auc_score(y_true=y_test, y_score= svm_pred))

print(classification_report(y_test, svm_pred))
print("SVM Finish""\n")


# Training performed using DNN methodology
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model = Sequential()
model.add(Dense(128, input_dim=len(x.columns), activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.1, batch_size=16, epochs=200)

dnn_model_pred = model.predict(X_test)

preds_1d = dnn_model_pred.flatten()  # dimensional spread
pred_class = np.where(preds_1d > 0.5, 1 , 0)

cfx = confusion_matrix(y_test, pred_class)
sensitivity = cfx[0, 0] / (cfx[0, 0] + cfx[0, 1])
specificity = cfx[1, 1] / (cfx[1, 0] + cfx[1, 1])
print('DNN Sensitivity = \n', sensitivity)
print('DNN Specificity = \n', specificity)
print('DNN Accuracy = ', accuracy_score(y_true=y_test, y_pred=pred_class))

print('DNN AUC = ', roc_auc_score(y_true=y_test, y_score=pred_class))

print(classification_report(y_test, pred_class))
print("DNN Finish""\n")