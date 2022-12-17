### breast-cancer-classification-model
Early detection of Breast Cancer can help treatment to be duly started, intended purpose of this project is to build predictive models for identifying a malignant condition.

### Analytical areas

For this case study, two classificatoin models are developed and compared, one machine learning and the other one is deep learning:

**Machine learning**. A logistic regression model is developed using python sklearn library.<br/>
Assumptions are binary outcome, features should not be too highly correlated with each other and linearity of independent variables and log odds.<br/>
Model predicts an outcome of benign or malignant based on features.<br/>
Model quality is determined by using 4 metrics i.e. accuracy, precision, recall and f-1 score.<br/>
Data play nice with the suggested method<br/>

**Deep learning**. An Artifical Neural Network (ANN) model is developed using python tensorflow library.<br/>
No specific assumptions are made on data and their distrbutions.<br/>
Model predicts an outcome of benign or malignant based on features.<br/>
Model quality is determined by using 4 metrics i.e. accuracy, precision, recall and f-1 score.<br/>
Data play nice with the suggested method<br/>

### Data Overview

**How and when were the data collected?**<br/>
Samples arrive periodically as Dr. Wolberg reports his clinical cases. The database therefore reflects this chronological grouping of the data. This grouping information appears immediately below, having been removed from the data itself:<br/>
Group 1: 367 instances (January 1989)<br/>
Group 2:  70 instances (October 1989)<br/>
Group 3:  31 instances (February 1990)<br/>
Group 4:  17 instances (April 1990)<br/>
Group 5:  48 instances (August 1990)<br/>
Group 6:  49 instances (Updated January 1991)<br/>
Group 7:  31 instances (June 1991)<br/>
Group 8:  86 instances (November 1991)<br/>

![image](https://user-images.githubusercontent.com/37245809/208222130-3ba24779-45fe-463e-a309-b4eca1eaf4f3.png)

Class distribution is shown below:

![image](https://user-images.githubusercontent.com/37245809/208222167-b1b72f7b-e957-436d-947c-4b2e95cbc449.png)

### Evaluate model performance
* In this case study, malignant is termed as positive (1) and benign is considered as negative (0)
* The terms are as follows:
TP = true positive, TN = true negative, FP = false positive, FN = false negative
* **Accuracy** is the overall prediction accuracy of the model (TN+TP/TN+TP+FN+FP)
* **Precision** is class based prediction accuracy of the model (TP/TP+FP)
* **Recall/sensitivity** is the class based evaluation against truth (TP/TP+FN)
* **f1-score** is used to represent both precision and recall (harmonic mean)

### Evaluate model performance

![image](https://user-images.githubusercontent.com/37245809/208226656-8234ef6c-4e88-4a0c-bb6c-744ffbf91516.png)

![image](https://user-images.githubusercontent.com/37245809/208226693-83834301-65a4-4d26-bc8d-5345e09d1a15.png)

