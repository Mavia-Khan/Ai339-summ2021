CODE 

import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm

setwd("D:/Kaggle Datasets/titanic")
training_set = read.csv("Train.csv")
test_set = read.csv("test.csv")
 
test_setSurvived = NA
 
complete_data = rbind(training_set, test_set)
 
library(Amelia)
missmap(complete_data, main = "Missing value map")
 
complete_dataAge[is.na(complete_dataAge)] <- median(complete_dataAge, na.rm=T)
complete_dataEmbarked[complete_dataEmbarked==""] <- "S"
 
drop <- c("Ticket","Name","Cabin")
df = complete_data[,!(names(complete_data) %in% drop)]
 
for (i in c("Survived","Pclass","Sex","Embarked")){
  df[,i]=as.factor(df[,i])
} 
# Create dummy variables for categorical variables
# install.packages("dummies")
library(dummies)
df <- dummy.data.frame(df, names=c("Pclass","Sex","Embarked"), sep="_")
training_set = df[1:669, ]
test_set = df[670:891, ] 
# install.packages("tidyverse")
library(tidyverse)
training_set_sub = training_set %>% select(Pclass_1, Pclass_2, Age, Sex_female, SibSp)
test_set_sub = test_set %>% select(Pclass_1, Pclass_2, Age, Sex_female, SibSp)
library(e1071)
classifier = svm(x = training_set_sub
                 , y = training_setSurvived
                 , kernel = 'linear'
                 , type = 'C-classification')
 
y_pred = predict(classifier, newdata = test_set_sub)
library(caret)
confusionMatrix(table(y_pred, test_set[, 2])) 
output = pd.DataFrame({'PassengerId': titanic_test.index,'Survived': y_test_pred})
output.to_csv('SVM-TitanicSurvival.csv',index=False)

"Support Vector Machine (SVM) is a supervised machine learning algorithm capable of performing classification, regression and even outlier detection. The linear SVM classifier works by drawing a straight line between two classes."


![knn](https://user-images.githubusercontent.com/53654229/126382347-774fa614-c135-406d-bafd-f10e49479e0c.PNG)

" It is the learning where the value or result that we want to predict is within the training data (labeled data) and the value which is in data that we want to study is known as Target or Dependent Variable or Response Variable."

![linear_classifeir](https://user-images.githubusercontent.com/53654229/126382379-8f1653a8-6e6c-4018-b923-5f3df4506d75.PNG)

"Linear classifiers classify data into labels based on a linear combination of input features. Therefore, these classifiers separate data using a line or plane or a hyperplane (a plane in more than 2 dimensions). They can only be used to classify data that is linearly separable."
