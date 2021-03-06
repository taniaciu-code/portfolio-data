---
title: Support Vector Machine & Naive Bayes Based Spam SMS Detection
subtitle: Aug 2020 - Dec 2020
date: '2019-04-08'
thumb_image: /images/sms.jpg
thumb_image_alt: A yellow retro telephone on a yellow background
image: /images/sms-b4dff986.jpg
image_alt: A yellow retro telephone on a yellow background
seo:
  title: Project Title 3
  description: This is the project 3 description
  extra:
    - name: 'og:type'
      value: website
      keyName: property
    - name: 'og:title'
      value: Project Title 3
      keyName: property
    - name: 'og:description'
      value: This is the project 3 description
      keyName: property
    - name: 'og:image'
      value: images/3.jpg
      keyName: property
      relativeUrl: true
    - name: 'twitter:card'
      value: summary_large_image
    - name: 'twitter:title'
      value: Project Title 3
    - name: 'twitter:description'
      value: This is the project 3 description
    - name: 'twitter:image'
      value: images/3.jpg
      relativeUrl: true
layout: project
---

<div align="justify">
The high number of SMS spamming cases causes this case to become one of unsettling cases in society. Therefore, it is necessary to have a spam filter with the concept of SMS spam detection. This study compares the performance of Support Vector Machine (SVM) and Naïve Bayes algorithm to find the appropriate algorithm for SMS spam detection. The SMS spam data collected from the UCI dataset repository will be used in 
data pre-processing which includes the process of labeling and dropping column, removing stop words, stemming, and feature extraction. Data that has been previously 
processed will be divided into training data for data modeling with Support Vector Machine and Naïve Bayes algorithms, as well as data testing for data validation process using confusion matrix. </div>

## Retrieve Data

<div align="justify">
In the data retrieving process, the SMS spam data obtained from the UCI dataset repository in http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection will be imported into Jupyter Notebook. The data consists of 5 variables with 5572 rows of data as the basis for analysis. The variables contained in the data consist of the variables “v1”, “v2”, “Unamed: 2”, “Unamed:3”, “Unamed:4”. The data used as the basis for the analysis is imported into Jupyter Notebook using the pandas library. The data import process also involves encoding parameters to be able to convert the data obtained into latin-1 format because the data is obtained in UTF-8 format so it is difficult to process.
</div>

```python
import pandas as pd
data = pd.read_csv('SMSSpamCollection.csv', encoding='latin-1')
data
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
      <th>v1</th>
      <th>v2</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5567</th>
      <td>spam</td>
      <td>This is the 2nd time we have tried 2 contact u...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5568</th>
      <td>ham</td>
      <td>Will Ì_ b going to esplanade fr home?</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5569</th>
      <td>ham</td>
      <td>Pity, * was in mood for that. So...any other s...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5570</th>
      <td>ham</td>
      <td>The guy did some bitching but I acted like i'd...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5571</th>
      <td>ham</td>
      <td>Rofl. Its true to its name</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5572 rows × 5 columns</p>
</div>



## Data Pre-Processing
<div align="justify">
In this process, the imported data will be prepared before it can be used in the next process. Variables from imported data will be labeled “label” and “text”. Then, from the 5 data variables obtained, only the first 2 variables will be used because the other three variables do not provide any information.
</div>

```python
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data.columns = ["label", "text"]
data
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
      <th>label</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5567</th>
      <td>spam</td>
      <td>This is the 2nd time we have tried 2 contact u...</td>
    </tr>
    <tr>
      <th>5568</th>
      <td>ham</td>
      <td>Will Ì_ b going to esplanade fr home?</td>
    </tr>
    <tr>
      <th>5569</th>
      <td>ham</td>
      <td>Pity, * was in mood for that. So...any other s...</td>
    </tr>
    <tr>
      <th>5570</th>
      <td>ham</td>
      <td>The guy did some bitching but I acted like i'd...</td>
    </tr>
    <tr>
      <th>5571</th>
      <td>ham</td>
      <td>Rofl. Its true to its name</td>
    </tr>
  </tbody>
</table>
<p>5572 rows × 2 columns</p>
</div>

```python
data['length'] = data['text'].apply(len)
data
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
      <th>label</th>
      <th>text</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>49</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>61</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5567</th>
      <td>spam</td>
      <td>This is the 2nd time we have tried 2 contact u...</td>
      <td>161</td>
    </tr>
    <tr>
      <th>5568</th>
      <td>ham</td>
      <td>Will Ì_ b going to esplanade fr home?</td>
      <td>37</td>
    </tr>
    <tr>
      <th>5569</th>
      <td>ham</td>
      <td>Pity, * was in mood for that. So...any other s...</td>
      <td>57</td>
    </tr>
    <tr>
      <th>5570</th>
      <td>ham</td>
      <td>The guy did some bitching but I acted like i'd...</td>
      <td>125</td>
    </tr>
    <tr>
      <th>5571</th>
      <td>ham</td>
      <td>Rofl. Its true to its name</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
<p>5572 rows × 3 columns</p>
</div>

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['patch.force_edgecolor'] = True
plt.style.use('seaborn-bright')
data.hist(column='length', by='label', bins=50,figsize=(11,5), color = "darkturquoise")
```

    array([<matplotlib.axes._subplots.AxesSubplot object at 0x000002630923E190>,
           <matplotlib.axes._subplots.AxesSubplot object at 0x0000026309231910>],
          dtype=object)

![png](/images/output\_5\_1.png)

<div align="justify">
Based on the graph above, it is found that the amount of spam SMS data is more than the number of “Ham” data. The words that are often used in the data identified as SMS "Ham" data.
</div>

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
def generate_wordcloud(all_words):
    wordcloud = WordCloud(width=800, height=500, random_state=21, 
                          max_font_size=100, relative_scaling=0.5, 
                          colormap='RdBu').generate(all_words)

    plt.figure(figsize=(6, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()
```

```python
ham = ' '.join([text for text in data['text'][data.label == "ham"]])
generate_wordcloud(ham)
```

![png](/images/output\_7\_0.png)

```python
spam = ' '.join([text for text in data['text'][data.label == "spam"]])
generate_wordcloud(spam)
```

![png](/images/output\_8\_0.png)

```python
from nltk.stem import SnowballStemmer
def pre_process(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() 
            if word.lower() not in stopwords.words('english')]
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words

```

```python
import nltk
import string
#nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

textFeatures = data['text'].copy()
textFeatures = textFeatures.apply(pre_process)
vectorizer = TfidfVectorizer("english")
features = vectorizer.fit_transform(textFeatures)
```

    C:\Users\Tania Ciu\anaconda3\lib\site-packages\sklearn\utils\validation.py:68: FutureWarning: Pass input=english as keyword args. From version 0.25 passing these as positional arguments will result in an error
      warnings.warn("Pass {} as keyword args. From version 0.25 "
    

## Data Splitting
<div align="justify">
Data that has undergone data pre-processing in the previous process will be mapped into a numerical value where the "Ham" data will be assigned a value of 0 and the "Spam" data will be assigned a value of 1.
</div>

```python
# Map dataframe to encode values and put values into a numpy array
encoded_labels = data['label'].map(lambda x: 1 if x == 'spam' else 0).values # ham will be 0 and spam will be 1
```

<div align="justify">
Furthermore, the data will be divided into training data and testing data where 70% of the data will be used as training data and 30% of the data will be used as testing data.
</div>

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, 
                                                    test_size=0.3, random_state=272)
```

## Data Modeling and Data Validation

<div align="justify">
Training data is used for data modeling using the Support Vector Machine (SVM) and Naive Bayes algorithm.
</div>

```python
import numpy as np 

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score

# Calculate accuracy by using confusion matrix
def plot_confusion_matrix(matrix):
    plt.clf()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.GnBu)
    classNames = ['Positive', 'Negative']
    plt.title('Confusion Matrix')
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    s = [['TP','FP'], ['FN', 'TN']]

    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(matrix[i][j]))
    plt.show()
```

### Modeling with Support Vector Machine Algorithm


```python
from sklearn import svm

svm_clf = svm.SVC(kernel='sigmoid', gamma=1.0,probability=True)
svm_clf.fit(X_train,y_train)
pred_svm = svm_clf.predict(X_test.toarray())
accuracy_svm = accuracy_score(y_test, pred_svm)
print(f'Accuracy Score = {accuracy_svm}')
conf_matrix_svm = confusion_matrix(y_test, pred_svm)
plot_confusion_matrix(conf_matrix_svm)
```

    Accuracy Score = 0.9706937799043063

![png](/images/output\_17\_1.png)

### Modeling with Naive Bayes Algorithm


```python
from sklearn.naive_bayes import GaussianNB

nb_clf = GaussianNB()
nb_clf.fit(X_train.toarray(), y_train)
pred_nb = nb_clf.predict(X_test.toarray())
accuracy_nb = accuracy_score(y_test, pred_nb)
print(f'Accuracy Score = {accuracy_nb}')
conf_matrix_nb = confusion_matrix(y_test, pred_nb)
plot_confusion_matrix(conf_matrix_nb)
```

    Accuracy Score = 0.8738038277511961


![png](/images/output_191.png)


## Comparation of Support Vector Machine and Naive Bayes Algorithm Model

<div align="justify">
Based on the modeling results, it is found that the SMS spam data modeling using the Support Vector Machine algorithm is more accurate than using the Naïve Bayes algorithm where the accuracy of the Support Vector Machine modeling is 97% while the accuracy of the Naïve Bayes modeling is 87%. To compare the two algorithms specifically, further data validation was carried out using additional parameters in the form of precision, recall, and f-measure parameters.
</div>

```python
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
classifiers.append(('SVM',svm_clf))
classifiers.append(('NB',nb_clf))

result=[]
cnf_matric_parameter=[]
for i,v in classifiers:
    pred=v.predict(X_test.todense())
    acc=accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall=recall_score(y_test, pred)
    #print(precision)
    f_measure=f1_score(y_test,pred)
    result.append((i,acc,precision,recall,f_measure))
    
    TP,FP,TN,FN=perf_measure(y_test,pred)
    cnf_matric_parameter.append((i,TP,FP,TN,FN))
```

```python
column_names=['Algorithm','Accuracy','Precision','Recall','F-measure']
df1=pd.DataFrame(result,columns=column_names)
print(df1)
```

      Algorithm  Accuracy  Precision    Recall  F-measure
    0       SVM  0.970694   0.975490  0.818930   0.890380
    1        NB  0.873804   0.540404  0.880658   0.669797

```python
df1.plot(kind='bar', ylim=(0.2,1.0), align='center', colormap="RdBu")
plt.xticks(np.arange(2), df1['Algorithm'],fontsize=12)
plt.ylabel('Score',fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=10)
```

    <matplotlib.legend.Legend at 0x2630c58e3d0>




![png](/images/output_compare.png)

<div align="justify">
By using SMS spam data and modeling using the Support Vector Machine and Naïve Bayes algorithms, it is found that the implementation of the Support Vector Machine algorithm is better than the implementation of the Naïve Bayes algorithm. This is based on the accuracy results where the accuracy of modeling with the Support Vector Machine is 97%, while the modeling with Naïve Bayes only produces an accuracy of 87%.
The data validation process in this study is not only limited to accuracy, but can also use a number of additional parameters. The first parameter used is the precision parameter. Based on these parameters, it is found that modeling with Support Vector Machine produces a precision value of 97% and modeling with Naïve Bayes produces a precision value of 54%. Thus, it can be said that modeling using the Support Vector Machine is more precise than modeling using Naïve Bayes.
<br>
The recall parameter is also used as a parameter to measure the suitability of the algorithm. Based on the recall parameter, the use of the Support Vector Machine algorithm produces a smaller recall value of 82% compared to the use of the Naïve Bayes algorithm which is able to produce a recall value of 88%.
<br>
In addition, in order to determine a more suitable algorithm for SMS spam data, the f-measure parameter can be used. If viewed from the f-measure value, it is found that the modeling using the Support Vector Machine algorithm has a higher f-measure than the modeling using the Naïve Bayes algorithm where the Support Vector Machine f-measure modeling is 89%, while the Naïve Bayes modeling has f-measure. -measure of 66%. Thus, it is found that the Support Vector Machine algorithm is more suitable for use on SMS spam data than the Naïve Bayes algorithm.
</div>