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

```{r, echo = TRUE, message = FALSE, warning = FALSE}
library(readr)
heart <- read_csv("C:/Users/Tania Ciu/Downloads/DataAnalysis/heart.csv")
View(heart)
Data<-heart
```
![png](/images/data1.png)

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
![png](/images/data2.png)

## Plot histogram

```{r, echo = TRUE, message = FALSE, warning = FALSE}
library(ggplot2)
plot_histogram(Data,
               ggtheme = theme_bw(), 
               title="Variables in Data")

```
![png](/images/data3.png)

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

## Split data

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

## Validation data with Confusion Matrix

```{r, echo = TRUE, message = FALSE, warning = FALSE}
library(tools)
conf<-confusionMatrix(table(LogisticPred, 
                            test_data$target))
conf

```

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
