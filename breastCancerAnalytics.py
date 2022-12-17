# Databricks notebook source
# MAGIC %md
# MAGIC # Wisconsin Breast Cancer: Case Study 
# MAGIC 
# MAGIC ### Case Study Tasks
# MAGIC 
# MAGIC * Prep data and build an analytical model/solution
# MAGIC * Share code and results 
# MAGIC * Present findings
# MAGIC 
# MAGIC ### Data Overview
# MAGIC 
# MAGIC 1. **How and when were the data collected?**<br/>
# MAGIC Samples arrive periodically as Dr. Wolberg reports his clinical cases. The database therefore reflects this chronological grouping of the data. This grouping information appears immediately below, having been removed from the data itself:<br/>
# MAGIC Group 1: 367 instances (January 1989)<br/>
# MAGIC Group 2:  70 instances (October 1989)<br/>
# MAGIC Group 3:  31 instances (February 1990)<br/>
# MAGIC Group 4:  17 instances (April 1990)<br/>
# MAGIC Group 5:  48 instances (August 1990)<br/>
# MAGIC Group 6:  49 instances (Updated January 1991)<br/>
# MAGIC Group 7:  31 instances (June 1991)<br/>
# MAGIC Group 8:  86 instances (November 1991)<br/>
# MAGIC 
# MAGIC 2. **What is the intended purpose of the data?**<br/>
# MAGIC Breast Cancer is one of the most common types of cancers in women which is affecting approximately 12.5% of women all around the world. Moreover, developing countries have a growing breast cancer epidemic with an increasing number of younger women who are susceptible to cancer. Since early detection of this cancer can help treatment to be duly started, intented purpose of this data is to build predictive models for identifying a malignant condition.<br/>
# MAGIC 
# MAGIC 3. **Identify all the issues with the dataset; if possible, show it in visual form**<br/>
# MAGIC There are 16 instances in Groups 1 to 6 that contain a single missing (i.e., unavailable) attribute value, now denoted by "?". 
# MAGIC 
# MAGIC 4. **Are there pre-processing needs on data? What are they and why?**<br/>
# MAGIC Two pre-processing techniques are used:<br/>
# MAGIC ***Missing data***: Since there's a small number of missing data, rows containing missing data are removed from the dataset to develop the model. This is to reduce error while traning the model.<br/>
# MAGIC ***Feature scaling***: Feature scaling (standardization) is used to scale the features. This is to improve efficiency (gradient descent) and classification accuracy of the model.<br/>
# MAGIC 
# MAGIC ### Analytical areas
# MAGIC 
# MAGIC For this case study, two classificatoin models are developed and compared, one machine learning and the other one is deep learning:
# MAGIC 
# MAGIC 1.	**Machine learning**. A logistic regression model is developed using python sklearn library.<br/>
# MAGIC Assumptions are binary outcome, features should not be too highly correlated with each other and linearity of independent variables and log odds.<br/>
# MAGIC Model predicts an outcome of benign or malignant based on features.<br/>
# MAGIC Model quality is determined by using 4 metrics i.e. accuracy, precision, recall and f-1 score.<br/>
# MAGIC Data play nice with the suggested method<br/>
# MAGIC 
# MAGIC 2.	**Deep learning**. An Artifical Neural Network (ANN) model is developed using python tensorflow library.<br/>
# MAGIC No specific assumptions are made on data and their distrbutions.<br/>
# MAGIC Model predicts an outcome of benign or malignant based on features.<br/>
# MAGIC Model quality is determined by using 4 metrics i.e. accuracy, precision, recall and f-1 score.<br/>
# MAGIC Data play nice with the suggested method<br/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate model performance
# MAGIC * In this case study, malignant is termed as positive (1) and benign is considered as negative (0)
# MAGIC * The terms are as follows:
# MAGIC TP = true positive, TN = true negative, FP = false positive, FN = false negative
# MAGIC * **Accuracy** is the overall prediction accuracy of the model (TN+TP/TN+TP+FN+FP)
# MAGIC * **Precision** is class based prediction accuracy of the model (TP/TP+FP)
# MAGIC * **Recall/sensitivity** is the class based evaluation against truth (TP/TP+FN)
# MAGIC * **f1-score** is used to represent both precision and recall (harmonic mean)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Importing the libraries

# COMMAND ----------

# MAGIC %pip install --upgrade tensorflow

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap
import tensorflow as tf
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# COMMAND ----------

tf.__version__

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Pre-processing
# MAGIC * Bare_nuclei column contains 16 null values
# MAGIC * These rows are removed from the analysis 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bcw_data

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bcw_data WHERE bare_nuclei IS NULL 

# COMMAND ----------

df =  spark.sql('''
                SELECT *,
                CASE 
                  WHEN class = 2 THEN 'benign'
                  WHEN class = 4 THEN 'malignant'
                END AS outcome,
                CASE 
                  WHEN class = 2 THEN 0
                  WHEN class = 4 THEN 1
                END AS class_model
                FROM bcw_data WHERE bare_nuclei IS NOT NULL
                ''')
display(df)

# COMMAND ----------

pdf = df.select("*").toPandas()
dataset = pdf.drop(['sample_code_number', 'class', 'outcome'], axis = 1)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# COMMAND ----------

fig = plt.figure(figsize=(8,6))
pdf.groupby('outcome').outcome.count().plot.bar(ylim=0)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Splitting the dataset into the Training set and Test set

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# COMMAND ----------

print(X_train)

# COMMAND ----------

print(y_train)

# COMMAND ----------

print(X_test)

# COMMAND ----------

print(y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature Scaling

# COMMAND ----------

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# COMMAND ----------

print(X_train)

# COMMAND ----------

print(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Building the Logistic Regression

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1. Training the Logistic Regression model on the Training set

# COMMAND ----------

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2. Predicting the Test set results

# COMMAND ----------

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3. Making the Confusion Matrix

# COMMAND ----------

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4. Create classification report

# COMMAND ----------

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred, output_dict=True)
df = pd.DataFrame(report).transpose()
df_reg = df.iloc[:3 , :3].rename(columns={"precision": "precision_reg", "recall": "recall_reg", "f1-score": "f1_score_reg"})
df_reg

# COMMAND ----------

# MAGIC %md
# MAGIC #### Building the ANN

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1. Initializing the ANN

# COMMAND ----------

ann = tf.keras.models.Sequential()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2. Adding the input layer and the first hidden layer

# COMMAND ----------

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3. Adding the second hidden layer

# COMMAND ----------

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4. Adding the output layer

# COMMAND ----------

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5. Compiling the ANN

# COMMAND ----------

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 6. Training the ANN on the Training set

# COMMAND ----------

ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 7. Predicting the Test set results

# COMMAND ----------

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 8. Making the Confusion Matrix

# COMMAND ----------

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 9. Create classification report

# COMMAND ----------

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred, output_dict=True)
df = pd.DataFrame(report).transpose()
df_ann = df.iloc[:3 , :3].rename(columns={"precision": "precision_ann", "recall": "recall_ann", "f1-score": "f1_score_ann"})
df_ann

# COMMAND ----------

# MAGIC %md
# MAGIC #### Compare model performance

# COMMAND ----------

pd.merge(df_reg, df_ann, left_index=True, right_index=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Model performance highlights
# MAGIC * Both machine learning (logistic regression) and deep learning (ann) provides similar overall prediction accuracy. DL slightly improves the overall accuracy over ML. 
# MAGIC * Biggest difference is observed in recall and f1-score metrics, where DL performed better than ML. As recall is calculated by (TP/TP+FN), any false negative (malignant is determined as benign) prediction can reduce this value. This may signify DL's better performance by reducing false negative predictions. 
