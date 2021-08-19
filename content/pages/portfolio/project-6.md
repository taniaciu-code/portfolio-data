---
title: Logistic Regression Prediction Model for Cardiovascular Disease
subtitle: Aug 2019 - Feb 2020
date: '2018-12-18'
thumb_image: /images/cardio.jpg
thumb_image_alt: A handheld game console on a yellow background
image: /images/cardio-ff9fcd88.jpg
image_alt: A handheld game console on a yellow background
seo:
  title: Project Title 6
  description: This is the project 6 description
  extra:
    - name: 'og:type'
      value: website
      keyName: property
    - name: 'og:title'
      value: Project Title 6
      keyName: property
    - name: 'og:description'
      value: This is the project 6 description
      keyName: property
    - name: 'og:image'
      value: images/6.jpg
      keyName: property
      relativeUrl: true
    - name: 'twitter:card'
      value: summary_large_image
    - name: 'twitter:title'
      value: Project Title 6
    - name: 'twitter:description'
      value: This is the project 6 description
    - name: 'twitter:image'
      value: images/6.jpg
      relativeUrl: true
layout: project
---
<div align="justify">
The number of cardiovascular disease sufferes is also increasing yearly. This disease occurs due to several factors, such as age, blood pressure, cholesterol levels, diabetes, hypertension, genes, obesity, and unhealthy lifestyles. Various symptoms can be identified through physical signs such as chest pain, shortness of breath, dizziness, and easy feeling of fatigue.

Cardiovascular disease identification techniques are complicated to do. It is essential to know the the complication of cardiovascular disease can give a impact on one's life as a whole. The diagnosis and treatment of cardiovascular disease are very complex. While still using invasive-based techniques through analysis of the patients medical history, report of physical examinations performed by the medical tend to be less accurate and require a relatively long time. For this reason, a support system is implemented to predict cardiovascular disease through a machine learning model.

</div>

## Mengimport library

```{r, echo = TRUE, message = FALSE, warning = FALSE}
install.packages("DataExplorer")
library(DataExplorer)
library(data.table)
library(dplyr)
library(car)
install.packages("psych")
library(psych)
library(caret)
```

## Retrieve Data

<div align="justify">
The first process is Data Retrieval. In this process, 
Heart Disease UCI Dataset -published by Ronit in 
Kaggle website (https://www.kaggle.com/ronitf/heartdisease-uci)- will be used. It will be imported into the 
Rstudio software. 
</div>

```{r, echo = TRUE, message = FALSE, warning = FALSE}
library(readr)
heart <- read_csv("C:/Users/Tania Ciu/Downloads/DataAnalysis/heart.csv")
View(heart)
Data<-heart

```
![png](/images/data1.JPG)


## Variable as factor

```{r, echo = TRUE, message = FALSE, warning = FALSE}
Data1 <- copy(Data)
Data1$sex <- as.factor(Data1$sex)
Data1$cp <- as.factor(Data1$cp)
Data1$fbs <- as.factor(Data1$fbs)
Data1$restecg <- as.factor(Data1$restecg)
Data1$exang <- as.factor(Data1$exang)
Data1$ca <- as.factor(Data1$ca)
Data1$thal <- as.factor(Data1$thal)
Data1$target <- as.factor(Data1$target)
describe(Data1)
str(Data1)

```
![png](/images/data2.JPG)


## Plot histogram
<div align="justify">
The dataset obtained by the researcher as a basis for analysis is imported into RStudio.The data retrieval process is also performed in the data visualization to see the value of each variable involved in the overall research analysis.
</div>

```{r, echo = TRUE, message = FALSE, warning = FALSE}
library(ggplot2)
plot_histogram(Data,
               ggtheme = theme_bw(), 
               title="Variables in Data")

```
![png](/images/data3.JPG)


## Plot Correlation

```{r, echo = TRUE, message = FALSE, warning = FALSE}
install.packages("GGally")
library(GGally)
ggcorr(Data, nbreaks=8, 
       palette='RdGy', 
       label=TRUE, 
       label_size=5, 
       label_color='black')

```
![png](/images/data4.JPG)

<div align="justify">
In this process, the correlation between variables will be examined, which will be used as a basis for analysis to predict cardiovascular disease. Based on 
the matrix, it was found that the variables 
induced angia (exang), chest pain type (cp), ST depression induced by exercise relative to rest (oldpeak), maximal heart rate (thalac) had a strong correlation with the target variable. Meanwhile, blood sugar (fbs) and cholesterol (chol) levels do not 
correlate with the target variable. Meanwhile, among the independent variables, there is a strong correlation between the slope and oldpeak variables. Besides, 
thalac, exhang, oldpeak, and slope variables are also strongly correlated. Strong correlation also applies to variables Exang, cp, and thalac. It proves that there is 
no multicollinearity in the relationship between variables where each independent variable does not correlate with each other.
</div>

## Split data

<div align="justify">
Data that has been imported will be taken as many as 293 random data as a basis for analysis. The data is divided into train data and test data. 
</div>

```{r, echo = TRUE, message = FALSE, warning = FALSE}
set.seed(293)
trainIndex<-createDataPartition(y=Data1$target
                                , p=0.7, list=FALSE)
train_data<-Data1[trainIndex,]
train_data
describe(train_data)
test_data<-Data1[trainIndex,]
test_data
describe(test_data)

```
![png](/images/hasil1.JPG)

## Modeling Logistic Regression

```{r, echo = TRUE, message = FALSE, warning = FALSE}
LogisticMod <- glm(target ~ age+sex+trestbps+chol+fbs+restecg+thalach+exang+oldpeak+slope+ca+thal, data=train_data, family="binomial"(link="logit"))
LogisticPred <- predict(LogisticMod, test_data, 
                        type='response')
LogisticPred <- ifelse(LogisticPred > 0.5, 1, 0)
LogisticPredCorrect <- data.frame(target=test_data$target, 
                                  predicted=LogisticPred, 
                                  match=(test_data$target == LogisticPred))
summary(LogisticMod)
LogisticPrediction <- predict(LogisticMod, 
                        test_data, 
                        type='response')
LogisticPrediction
summary(LogisticPrediction)

```
![png](/images/hasil2.JPG)

<div align="justify">
The training data is used to build a logistic regression model using the glm () function because logistic regression is 
included in the generalized linear model with binomial type families. Based on the results of using the logistic regression method, it is predicted that the sex, cp, 
trestbps, restecg, ca and that variables influence the target variable at an alpha value of 5% significantly. 
The selected variables are the variables that significantly affect the target variable. In logistic regression, the effect of each variable on the target variable can be seen from the odds ratio value. For
example, for the sex variable having a coefficient value of -1.547601 with a reference category with a male value, the odds ratio value is 4.2655 which means that for male patients, the odds of getting heart 
disease are 4.2655 times the female odds or it can be said the tendency of men to heart disease is higher than women. 
<br>
For the trestbps variable with a coefficient value of -0.029713, it is found that the odds ratio value is 0.0822 which means that for the trestbps variable there will be a significant increase when trestbps enters the value 0.0822 mmHg. On the other 
hand, the thalach variable with a coefficient of 0.032028 will have an odds of 0.08856 which means that at that value there will be a significant change in the performance of the heart rate or cardiovascular rate. The exang1 variable is exercise-induced angina with an estimated coefficient of -1.05855 so that the exang variable with a reference value of 1 will have an odds of 2.92710 which means that if the value is achieved then cardiovascular performance will decrease. 
<br>
Next is the variable ca with reference ca values 1, 2, and 3. Ca1 with an estimated coefficient of -1.430110 will have odds of 3.955, while ca2 with an estimated ratio of -3.329874 will have odds of 9.1777 and ca3 with an estimated factor of -0.553711 will have odds in the amount of 1.5261. It proves that when the number of fluoroscopy vessels reaches its value odds, this will have an impact on decreasing cardiac performance which will affect the increased potential 
for cardiovascular disease.
</div>

## Validation data with Confusion Matrix

```{r, echo = TRUE, message = FALSE, warning = FALSE}
library(tools)
conf<-confusionMatrix(table(LogisticPred, 
                            test_data$target))
conf

```
![png](/images/hasil3.JPG)

## Validation data with k-fold cross validation

```{r, echo = TRUE, message = FALSE, warning = FALSE}
library(boot)
set.seed(293)
glm.fit <- glm(target ~ age+sex+trestbps
               +chol+fbs+restecg
               +thalach+exang+oldpeak
               +slope+ca+thal, 
               family = quasibinomial, 
               data = Data)
cv.err.10 <- cv.glm(data = Data, 
                    glmfit = glm.fit,
                    K = 10)
cv.err.10$delta

```

```
>cv.err.10$delta
[1] 0.1406565 0.1397245
```
<div align="justify">
The method used to validate the logistic regression model used in this study is the k-fold cross-validation method with k-fold value of 10. Following is the syntax of the k-fold cross-validation method. Based on 
the k-fold cross-validation data method, it was found that the prediction data using the logistic regression method had an error rate that tended to be lower at 0.1406565. It proves that referring to the two validation methods that have been done, and it can be 
concluded that the logistic regression model is an appropriate and effective model for this research.
</div>

## Conclusions
By using the Heart Disease UCI dataset consisting of fourteen variables, including age, sex, cp, fbs, restecg, thalac, exang, oldpeak, slope, ca, thal, and target, it was found that the use of the logistic regression algorithm is effective and efficient in predicting cardiovascular disease where based on the results of data validation it is found that the accuracy 
of the prediction results with the algorithm reaches 85% with an error rate that tends to be small at 0.1406565. It proves that this algorithm is suitable for use as a prediction algorithm in this study.
Based on the results of cardiovascular disease predictions, it can be concluded that cardiovascular disease is significantly affected by gender, trestbps - blood pressure level, thalach - heart rate, and canumber of vessels affected by fluoroscopy. An increase in the value of these variables will have an impact on overall cardiovascular performance where 
the cardiovascular performance will decrease, while the potential for cardiovascular disease is predicted to 
increase. The use of the logistic regression algorithm is successful in predicting the main factors causing cardiovascular disease where the main elements of the disease are gender factors, blood pressure level factors, heart rate level factors, and blood vessel colour factors (vessels).