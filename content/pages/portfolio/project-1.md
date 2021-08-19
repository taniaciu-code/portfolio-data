---
title: >-
  Implementation of Decision Tree, K-Nearest Neighbors, and Support Vector
  Machine Algorithm to Predict Potential Client in Banking Sector
subtitle: Jan 2021 - Jun 2021
date: '2019-05-10'
thumb_image: /images/client.jpg
thumb_image_alt: 'White, black, and red shoe sole'
image: /images/client-bb6a2564.jpg
image_alt: 'White, black, and red shoe sole'
seo:
  title: Project Title 1
  description: This is the project 1 description
  extra:
    - name: 'og:type'
      value: website
      keyName: property
    - name: 'og:title'
      value: Project Title 1
      keyName: property
    - name: 'og:description'
      value: This is the project 1 description
      keyName: property
    - name: 'og:image'
      value: images/1.jpg
      keyName: property
      relativeUrl: true
    - name: 'twitter:card'
      value: summary_large_image
    - name: 'twitter:title'
      value: Project Title 1
    - name: 'twitter:description'
      value: This is the project 1 description
    - name: 'twitter:image'
      value: images/1.jpg
      relativeUrl: true
layout: project
---
<div align="justify">
The high internal competition and the financial crisis in Portuguese bank institutions prompted the company to apply the telemarketing method. It is because telemarketing usage minimizes the resources spent on the marketing process so it can reduce financial assets pressure. However, the telemarketing implementation is constrained by the predictions of potential clients who are interested in the long-term deposits offered. Moreover, the large amount of client data causes the traditional data processing unable to accommodate the data processing. This study aims to build a classification model that will assist Portuguese banks in predicting potential clients as part of the client targeting process in telemarketing. </div>

## Data Understanding

### Import Dataset

```python
import pandas as pd
data = pd.read_csv('bank-full.csv', sep=';')
data.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>2143</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>261</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>technician</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>29</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>151</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33</td>
      <td>entrepreneur</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2</td>
      <td>yes</td>
      <td>yes</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>76</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>unknown</td>
      <td>no</td>
      <td>1506</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>92</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>unknown</td>
      <td>single</td>
      <td>unknown</td>
      <td>no</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>198</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>

### Data Exploration

```python
#Check data dimension
print("Ukuran data: {}".format(data.shape))
```

    Ukuran data: (45211, 17)

```python
#Check missing value
data.isnull().sum()
```

    age          0
    job          0
    marital      0
    education    0
    default      0
    balance      0
    housing      0
    loan         0
    contact      0
    day          0
    month        0
    duration     0
    campaign     0
    pdays        0
    previous     0
    poutcome     0
    y            0
    dtype: int64

```python
#Data Information
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 45211 entries, 0 to 45210
    Data columns (total 17 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   age        45211 non-null  int64 
     1   job        45211 non-null  object
     2   marital    45211 non-null  object
     3   education  45211 non-null  object
     4   default    45211 non-null  object
     5   balance    45211 non-null  int64 
     6   housing    45211 non-null  object
     7   loan       45211 non-null  object
     8   contact    45211 non-null  object
     9   day        45211 non-null  int64 
     10  month      45211 non-null  object
     11  duration   45211 non-null  int64 
     12  campaign   45211 non-null  int64 
     13  pdays      45211 non-null  int64 
     14  previous   45211 non-null  int64 
     15  poutcome   45211 non-null  object
     16  y          45211 non-null  object
    dtypes: int64(7), object(10)
    memory usage: 5.9+ MB

```python
# Statistic Descriptive Analysis of Data
data.describe().T
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>45211.0</td>
      <td>40.936210</td>
      <td>10.618762</td>
      <td>18.0</td>
      <td>33.0</td>
      <td>39.0</td>
      <td>48.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>balance</th>
      <td>45211.0</td>
      <td>1362.272058</td>
      <td>3044.765829</td>
      <td>-8019.0</td>
      <td>72.0</td>
      <td>448.0</td>
      <td>1428.0</td>
      <td>102127.0</td>
    </tr>
    <tr>
      <th>day</th>
      <td>45211.0</td>
      <td>15.806419</td>
      <td>8.322476</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>16.0</td>
      <td>21.0</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>duration</th>
      <td>45211.0</td>
      <td>258.163080</td>
      <td>257.527812</td>
      <td>0.0</td>
      <td>103.0</td>
      <td>180.0</td>
      <td>319.0</td>
      <td>4918.0</td>
    </tr>
    <tr>
      <th>campaign</th>
      <td>45211.0</td>
      <td>2.763841</td>
      <td>3.098021</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>pdays</th>
      <td>45211.0</td>
      <td>40.197828</td>
      <td>100.128746</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>871.0</td>
    </tr>
    <tr>
      <th>previous</th>
      <td>45211.0</td>
      <td>0.580323</td>
      <td>2.303441</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>275.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
#Import Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
data.hist(figsize=(15,15), density=True, color="#330033")
plt.show()
```

![](/images/hist.jpg)

```python
fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'job', data=data, palette="rocket")
ax.set_xlabel('Job', fontsize = 15)
ax.set_ylabel('Count', fontsize = 15)
ax.set_title('Job Count Distribution', fontsize = 15)
ax.tick_params
for rect in ax.patches:
    height = rect.get_height()
    ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
                xytext=(0, 5), textcoords='offset points', ha='center', va='bottom') 
```

![png](/images/output\_10\_0.png)

```python
fig, axs = plt.subplots(ncols=3)
fig.set_size_inches(20, 8)
sns.countplot(x = 'marital', data=data, palette="rocket", ax=axs[0])
sns.countplot(x = 'education', data=data, palette="rocket", ax=axs[1])
sns.countplot(x = 'default', data=data, palette="rocket", ax=axs[2])
ax.set_ylabel('Count', fontsize = 15)
```

    Text(22.200000000000017, 0.5, 'Count')




![png](/images/output_111.png)



```python
fig, axs = plt.subplots(ncols=3)
fig.set_size_inches(20, 8)
sns.countplot(x = 'housing', data=data, palette="rocket", ax=axs[0])
sns.countplot(x = 'loan', data=data, palette="rocket", ax=axs[1])
sns.countplot(x = 'contact', data=data, palette="rocket", ax=axs[2])
ax.set_ylabel('Count', fontsize = 15)
```

    Text(22.200000000000017, 0.5, 'Count')

![png](/images/output\_12\_1.png)

```python
fig, axs = plt.subplots(ncols=3)
fig.set_size_inches(20, 8)
sns.countplot(x = 'month', data=data, palette="rocket", ax=axs[0])
sns.countplot(x = 'poutcome', data=data, palette="rocket", ax=axs[1])
sns.countplot(x = 'y', data=data, palette="rocket", ax=axs[2])
ax.set_ylabel('Count', fontsize = 15)
```

    Text(22.200000000000017, 0.5, 'Count')

![png](/images/output\_13\_1.png)

## Data Preparation

### Label Encoding

```python
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)
data.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3036</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>261</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26</td>
      <td>9</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>945</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>151</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>918</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>76</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>2420</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>92</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15</td>
      <td>11</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>917</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>198</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

## Modeling

### Data Splitting

```python
from sklearn.model_selection import train_test_split
X = data.drop(['y'], axis=1)
y = data['y']

X_train, X_test, y_train, y_test =\
    train_test_split (X, y,
                     test_size = 0.3,
                     random_state = 0,
                     stratify=y)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))
```

    X_train shape: (31647, 16)
    y_train shape: (31647,)
    X_test shape: (13564, 16)
    y_test shape: (13564,)

### Decision Tree Model

```python
from sklearn.tree import DecisionTreeClassifier
clf_DT = DecisionTreeClassifier(criterion="gini", max_depth=5,
                                min_samples_split=4, min_samples_leaf=2)
model_DT = clf_DT.fit(X_train, y_train)
predict_DT = model_DT.predict(X_test)
print("Test prediction: {}".format(predict_DT))
```

    Test prediction: [0 0 0 ... 0 0 0]

```python
df_DT = pd.concat([y_test, pd.Series(predict_DT, name='Predicted_y')], axis=1)
df_DT.dropna(axis=0, inplace=True)
df_DT.tail()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>Predicted_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13550</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13553</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13559</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13560</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13563</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
#Feature Selection
feature_names = data[['age','job','marital','education','default','balance',
                             'housing','loan','contact','day',
                             'month','duration','campaign','previous',
                             'poutcome','pdays']]
```

```python
#Decision tree building
from sklearn.tree import export_graphviz
export_graphviz(model_DT, out_file="tree.dot", class_names=["yes", "no"],
               feature_names=feature_names.columns.values, impurity=False, filled=True)

import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
```

```python
from sklearn import tree
import pydotplus
import matplotlib.image as pltimg
import matplotlib.pyplot as plt

data_tree = tree.export_graphviz(model_DT, out_file=None, feature_names=feature_names.columns.values)
graph = pydotplus.graph_from_dot_data(data_tree)
graph.write_png('00000027615-TaniaCiu_DecisionTree.png')

img=pltimg.imread('00000027615-TaniaCiu_DecisionTree.png')
imgplot = plt.imshow(img)
plt.show() 
```

![png](/images/output\_25\_0.png)

### K-Nearest Neighbors Model

```python
from sklearn.neighbors import KNeighborsClassifier
clf_KNN = KNeighborsClassifier(n_neighbors=3, weights='distance')
model_KNN = clf_KNN.fit(X_train, y_train)
predict_KNN = model_KNN.predict(X_test)
print("Test prediction: {}".format(predict_KNN))
```

    Test prediction: [0 0 0 ... 0 0 0]

```python
#Prediction probability
print("Prediction probability: {}".format(model_KNN.predict_proba(X_test)))
```

    Prediction probability: [[1. 0.]
     [1. 0.]
     [1. 0.]
     ...
     [1. 0.]
     [1. 0.]
     [1. 0.]]

```python
df_KNN = pd.concat([y_test, pd.Series(predict_KNN, name='Predicted_y')], axis=1)
df_KNN.dropna(axis=0, inplace=True)
df_KNN.tail()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>Predicted_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13550</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13553</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13559</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13560</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13563</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

### Support Vector Machine Model

```python
# Supress warnings
import warnings
warnings.filterwarnings('ignore')
```

```python
from sklearn.svm import LinearSVC
clf_SVM = LinearSVC(max_iter=100000, class_weight='balanced', random_state=0)
model_SVM = clf_SVM.fit(X_train, y_train)
predict_SVM = model_SVM.predict(X_test)
print("Test prediction: {}".format(predict_SVM))
```

    Test prediction: [0 0 0 ... 0 0 0]

```python
df_SVM = pd.concat([y_test, pd.Series(predict_SVM, name='Predicted_y')], axis=1)
df_SVM.dropna(axis=0, inplace=True)
df_SVM.tail()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>Predicted_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13550</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13553</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13559</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13560</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13563</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

## Evaluation

### Evaluation Model Function

```python
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def metrics(y_true, y_pred):
    print('Accuracy: %.4f' % accuracy_score(y_true, y_pred))
    print('Precision: %.4f' % precision_score(y_true, y_pred))
    print('Recall: %.4f' % recall_score(y_true, y_pred))
    print('F1 Score: %.4f' % f1_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    cm = pd.DataFrame(cm, [1,2], [1,2])

    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()
```

### Evaluation of Decision Tree Model

```python
metrics(y_test, predict_DT)
```

    Accuracy: 0.8945
    Precision: 0.5880
    Recall: 0.3283
    F1 Score: 0.4214
                  precision    recall  f1-score   support

               0       0.92      0.97      0.94     11977
               1       0.59      0.33      0.42      1587

        accuracy                           0.89     13564
       macro avg       0.75      0.65      0.68     13564
    weighted avg       0.88      0.89      0.88     13564
    
    
![png test](/images/output_38_1.png)


### Evaluation of K-Nearest Neighbor Model

```python
metrics(y_test, predict_KNN)
```

    Accuracy: 0.8795
    Precision: 0.4789
    Recall: 0.3359
    F1 Score: 0.3948
                  precision    recall  f1-score   support

               0       0.92      0.95      0.93     11977
               1       0.48      0.34      0.39      1587

        accuracy                           0.88     13564
       macro avg       0.70      0.64      0.66     13564
    weighted avg       0.86      0.88      0.87     13564

![](/images/output\_40\_1.png)

### Evaluation of Support Vector Machine Model

```python
metrics(y_test, predict_SVM)
```

    Accuracy: 0.8882
    Precision: 0.6127
    Recall: 0.1216
    F1 Score: 0.2029
                  precision    recall  f1-score   support

               0       0.89      0.99      0.94     11977
               1       0.61      0.12      0.20      1587

        accuracy                           0.89     13564
       macro avg       0.75      0.56      0.57     13564
    weighted avg       0.86      0.89      0.85     13564

![](/images/output\_42\_1.png)

### Model Comparison

```python
import numpy as np
def perf_measure(y_actual, y_hat):
    y_actual=np.array(y_actual)
    y_hat=np.array(y_hat)
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i] and y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)
```

```python
classifiers=[]
classifiers.append(('DT',clf_DT))
classifiers.append(('KNN',clf_KNN))
classifiers.append(('SVM',clf_SVM))

result=[]
cnf_matric_parameter=[]
for i,v in classifiers:
    pred=v.predict(X_test)
    acc=accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall=recall_score(y_test, pred)
    f_measure=f1_score(y_test,pred)
    result.append((i,acc,precision,recall,f_measure))

    TP,FP,TN,FN=perf_measure(y_test,pred)
    cnf_matric_parameter.append((i,TP,FP,TN,FN))
```

```python
column_names=['Algorithm','Accuracy','Precision','Recall','F1-score']
df1=pd.DataFrame(result,columns=column_names)
df1
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algorithm</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DT</td>
      <td>0.894500</td>
      <td>0.588036</td>
      <td>0.328292</td>
      <td>0.421351</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNN</td>
      <td>0.879534</td>
      <td>0.478886</td>
      <td>0.335854</td>
      <td>0.394815</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SVM</td>
      <td>0.888234</td>
      <td>0.612698</td>
      <td>0.121613</td>
      <td>0.202944</td>
    </tr>
  </tbody>
</table>
</div>

```python
df1.plot(kind='bar', ylim=(0.2,1.0), align='center', colormap="RdBu")
plt.xticks(np.arange(3), df1['Algorithm'],fontsize=12)
plt.ylabel('Score',fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=10)
```

![png](/images/output_47_1.png)


