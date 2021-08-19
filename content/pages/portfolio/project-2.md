---
title: >-
  Multiclass Text Classification using Long Short Term Memory: Product Review &
  Rating Prediction
subtitle: Jan 2021 - Jun 2021
date: '2019-04-30'
thumb_image: /images/productreview1.jpg
thumb_image_alt: An orange on a blue background
image: /images/productreview1-49285502.jpg
image_alt: An orange on a blue background
seo:
  title: Project Title 2
  description: This is the project 2 description
  extra:
    - name: 'og:type'
      value: website
      keyName: property
    - name: 'og:title'
      value: Project Title 2
      keyName: property
    - name: 'og:description'
      value: This is the project 2 description
      keyName: property
    - name: 'og:image'
      value: images/2.jpg
      keyName: property
      relativeUrl: true
    - name: 'twitter:card'
      value: summary_large_image
    - name: 'twitter:title'
      value: Project Title 2
    - name: 'twitter:description'
      value: This is the project 2 description
    - name: 'twitter:image'
      value: images/2.jpg
      relativeUrl: true
layout: project
---

<div align="justify">
With the easiness of expressing one’s expression through digital advancements in technologies, social media has become a medium for people to express their feelings. This phenomena is also happening in the e-commerce industry, where a user’s sentiment towards a product can be seen through the 
rating and review given. In this project, we will build a multiclass text classification model using a dataset that is used at Shopee Code League 2020 based on various algorithms, which include Long-Short Term Memory, Support Vector Machine, Naïve Bayes and Random Forest. Final results and observation of this research is expected to be in the form of a model that is able to predict a product’s rating out of customer’s review with high accuracy
</div>

## Data Preparation

<div align="justify">
Data preparation is a data preparation stage that can include the process of retrieving or retrieval of raw data for use in the machine learning model development process. The training data that will be used in the study are as follows.
</div>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('train.csv')
df.head()
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
      <th>review_id</th>
      <th>review</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Ga disappointed neat products .. Meletot Hilsn...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Rdtanya replace broken glass, broken chargernya</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Nyesel bngt dsni shopping antecedent photo mes...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Sent a light blue suit goods ga want a refund</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Pendants came with dents and scratches on its ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

<div align="justify">
In this study, an exploration was carried out on two data provided by the Shopee Code League, namely train.csv data and test.csv data. Based on the exploration results, it was found that the training data contains 146,811 rows of data with 3 attributes including “review_id”, “review”, and "rating” and does not contain null values.
</div>


```python
df.isnull().sum()
```




    review_id    0
    review       0
    rating       0
    dtype: int64




```python
df.shape
```




    (146811, 3)



```python
df['rating'].value_counts()
```




    4    41865
    5    41515
    3    35941
    1    14785
    2    12705
    Name: rating, dtype: int64

<div align="justify">
The training data has an uneven rating distribution so that it can be judged to have an imbalanced class. Exploration results show that there are 14,785 data with a rating of 1, 12,705 data with a rating of 2, 35,941 with a rating of 3, 41,865 with a rating of 4, and 41,515 with a rating of 5. The visualization of the distribution of "rating" data by class is as follows.
</div>

<br>

```python
plt.hist(df['rating'],density=1, bins=20)
plt.show()
```


![png](/images/output_70.png)


## Data Cleansing & Text Preprocessing


```python
import string
import re
```


```python
df = df[['review', 'rating']]
df.head()
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
      <th>review</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ga disappointed neat products .. Meletot Hilsn...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Rdtanya replace broken glass, broken chargernya</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nyesel bngt dsni shopping antecedent photo mes...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sent a light blue suit goods ga want a refund</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pendants came with dents and scratches on its ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# separate majority and minority classes
df_1 = df[df.rating == 1] 
df_3 = df[df.rating == 3] 
df_4 = df[df.rating == 4] 
df_5 = df[df.rating == 5] 
df_minority = df[df.rating == 2] 
print(df_minority['rating'].value_counts())
```

    2    12705
    Name: rating, dtype: int64
    


```python
from sklearn.utils import resample
df1 = resample(df_1, replace=False, n_samples=12705, random_state=42)
df3 = resample(df_3, replace=False, n_samples=12705, random_state=42)
df4 = resample(df_4, replace=False, n_samples=12705, random_state=42)
df5 = resample(df_5, replace=False, n_samples=12705, random_state=42)
data = pd.concat([df1, df3, df4, df5, df_minority])
data.rating.value_counts()
```




    5    12705
    4    12705
    3    12705
    2    12705
    1    12705
    Name: rating, dtype: int64




```python
len(data)
```




    63525




```python
def remove_punctuation(text):
    no_punct=[words for words in text if words not in string.punctuation]
    words_wo_punct= ''.join(no_punct)
    return words_wo_punct
data['clean_text']=data['review'].apply(lambda x: remove_punctuation(x))
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
      <th>review</th>
      <th>rating</th>
      <th>clean_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10390</th>
      <td>Item different from picture</td>
      <td>1</td>
      <td>Item different from picture</td>
    </tr>
    <tr>
      <th>5549</th>
      <td>Dapet apes who like this again</td>
      <td>1</td>
      <td>Dapet apes who like this again</td>
    </tr>
    <tr>
      <th>3774</th>
      <td>Items like antiques through it. Motor force Bl...</td>
      <td>1</td>
      <td>Items like antiques through it Motor force Bla...</td>
    </tr>
    <tr>
      <th>8409</th>
      <td>I did not receive the item.</td>
      <td>1</td>
      <td>I did not receive the item</td>
    </tr>
    <tr>
      <th>6018</th>
      <td>The product quality is not good</td>
      <td>1</td>
      <td>The product quality is not good</td>
    </tr>
  </tbody>
</table>
</div>




```python
#remove hyperlink
data['clean_text'] = data['clean_text'].str.replace(r"http\S+", "") 
#remove emoji
data['clean_text'] = data['clean_text'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
#convert all words to lowercase
data['clean_text'] = data['clean_text'].str.lower()
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
      <th>review</th>
      <th>rating</th>
      <th>clean_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10390</th>
      <td>Item different from picture</td>
      <td>1</td>
      <td>item different from picture</td>
    </tr>
    <tr>
      <th>5549</th>
      <td>Dapet apes who like this again</td>
      <td>1</td>
      <td>dapet apes who like this again</td>
    </tr>
    <tr>
      <th>3774</th>
      <td>Items like antiques through it. Motor force Bl...</td>
      <td>1</td>
      <td>items like antiques through it motor force bla...</td>
    </tr>
    <tr>
      <th>8409</th>
      <td>I did not receive the item.</td>
      <td>1</td>
      <td>i did not receive the item</td>
    </tr>
    <tr>
      <th>6018</th>
      <td>The product quality is not good</td>
      <td>1</td>
      <td>the product quality is not good</td>
    </tr>
  </tbody>
</table>
</div>




```python
# not my original code (spanyol code lol)
def recover_shortened_words(text):
  
    text = re.sub(r'\bapaa\b', 'apa', text)
    
    text = re.sub(r'\bbsk\b', 'besok', text)
    text = re.sub(r'\bbrngnya\b', 'barangnya', text)
    text = re.sub(r'\bbrp\b', 'berapa', text)
    text = re.sub(r'\bbgt\b', 'banget', text)
    text = re.sub(r'\bbngt\b', 'banget', text)
    text = re.sub(r'\bgini\b', 'begini', text)
    text = re.sub(r'\bbrg\b', 'barang', text)
    
    text = re.sub(r'\bdtg\b', 'datang', text)
    text = re.sub(r'\bd\b', 'di', text)
    text = re.sub(r'\bsdh\b', 'sudah', text)
    text = re.sub(r'\bdri\b', 'dari', text)
    text = re.sub(r'\bdsni\b', 'disini', text)
    
    text = re.sub(r'\bgk\b', 'gak', text)
    
    text = re.sub(r'\bhrs\b', 'harus', text)
    
    text = re.sub(r'\bjd\b', 'jadi', text)
    text = re.sub(r'\bjg\b', 'juga', text)
    text = re.sub(r'\bjgn\b', 'jangan', text)
    
    text = re.sub(r'\blg\b', 'lagi', text)
    text = re.sub(r'\blgi\b', 'lagi', text)
    text = re.sub(r'\blbh\b', 'lebih', text)
    text = re.sub(r'\blbih\b', 'lebih', text)
    
    text = re.sub(r'\bmksh\b', 'makasih', text)
    text = re.sub(r'\bmna\b', 'mana', text)
    
    text = re.sub(r'\borg\b', 'orang', text)
    
    text = re.sub(r'\bpjg\b', 'panjang', text)
    
    text = re.sub(r'\bka\b', 'kakak', text)
    text = re.sub(r'\bkk\b', 'kakak', text)
    text = re.sub(r'\bklo\b', 'kalau', text)
    text = re.sub(r'\bkmrn\b', 'kemarin', text)
    text = re.sub(r'\bkmrin\b', 'kemarin', text)
    text = re.sub(r'\bknp\b', 'kenapa', text)
    text = re.sub(r'\bkcil\b', 'kecil', text)
    
    text = re.sub(r'\bgmn\b', 'gimana', text)
    text = re.sub(r'\bgmna\b', 'gimana', text)
    
    text = re.sub(r'\btp\b', 'tapi', text)
    text = re.sub(r'\btq\b', 'thanks', text)
    text = re.sub(r'\btks\b', 'thanks', text)
    text = re.sub(r'\btlg\b', 'tolong', text)
    text = re.sub(r'\bgk\b', 'tidak', text)
    text = re.sub(r'\bgak\b', 'tidak', text)
    text = re.sub(r'\bgpp\b', 'tidak apa apa', text)
    text = re.sub(r'\bgapapa\b', 'tidak apa apa', text)
    text = re.sub(r'\bga\b', 'tidak', text)
    text = re.sub(r'\btgl\b', 'tanggal', text)
    text = re.sub(r'\btggl\b', 'tanggal', text)
    text = re.sub(r'\bgamau\b', 'tidak mau', text)
    
    text = re.sub(r'\bsy\b', 'saya', text)
    text = re.sub(r'\bsis\b', 'sister', text)
    text = re.sub(r'\bsdgkan\b', 'sedangkan', text)
    text = re.sub(r'\bmdh2n\b', 'semoga', text)
    text = re.sub(r'\bsmoga\b', 'semoga', text)
    text = re.sub(r'\bsmpai\b', 'sampai', text)
    text = re.sub(r'\bnympe\b', 'sampai', text)
    text = re.sub(r'\bdah\b', 'sudah', text)
    
    text = re.sub(r'\bberkali2\b', 'repeated', text)
  
    text = re.sub(r'\byg\b', 'yang', text)
    
    return text
```


```python
data['clean_text'] = data['clean_text'].apply(recover_shortened_words)
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
      <th>review</th>
      <th>rating</th>
      <th>clean_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10390</th>
      <td>Item different from picture</td>
      <td>1</td>
      <td>item different from picture</td>
    </tr>
    <tr>
      <th>5549</th>
      <td>Dapet apes who like this again</td>
      <td>1</td>
      <td>dapet apes who like this again</td>
    </tr>
    <tr>
      <th>3774</th>
      <td>Items like antiques through it. Motor force Bl...</td>
      <td>1</td>
      <td>items like antiques through it motor force bla...</td>
    </tr>
    <tr>
      <th>8409</th>
      <td>I did not receive the item.</td>
      <td>1</td>
      <td>i did not receive the item</td>
    </tr>
    <tr>
      <th>6018</th>
      <td>The product quality is not good</td>
      <td>1</td>
      <td>the product quality is not good</td>
    </tr>
  </tbody>
</table>
</div>




```python
un_words_count = pd.Series(' '.join(data.clean_text).split()).value_counts()
len(un_words_count)
```




    43146




```python
un_words_count.head(10)
```




    the         38726
    good        29923
    is          22511
    product     22055
    quality     19670
    very        15685
    not         14241
    delivery    14126
    to          13508
    of          11519
    dtype: int64



## Modeling with Long Short Term Memory


```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional, SpatialDropout1D
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
```


```python
print(tf.__version__)
```

    2.6.0
    


```python
# set hyperparameter
vocab_size = 7000 # make the top list of words (common words)
embedding_dim = 32
max_length = 354
oov_tok = '<OOV>' # OOV = Out of Vocabulary
```


```python
# splitting dataset
X = data['clean_text']
Y = data['rating']
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
```

    (57172,) (57172,)
    (6353,) (6353,)
    


```python
# set tokenizer
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
```


```python
 # training set
train_sequences = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(train_sequences, maxlen=max_length)
```


```python
# validation set
validation_sequences = tokenizer.texts_to_sequences(X_test)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length)
```


```python
from tensorflow.keras.utils import to_categorical
print(set(data['rating']))
#training_label_seq = np.asarray(train_labels).astype(np.float32)
#validation_label_seq = np.asarray(validation_labels).astype(np.float32)
training_label_seq = to_categorical(y_train)
validation_label_seq = to_categorical(y_test)
```

    {1, 2, 3, 4, 5}
    


```python
# dropout to avoid overfitting
# softmax as multiclass clf
model = Sequential()

model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=354))
model.add(Dropout(0.5))
model.add(Bidirectional(CuDNNLSTM(100)))
model.add(Dense(6, activation='softmax'))

model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 354, 32)           224000    
    _________________________________________________________________
    dropout (Dropout)            (None, 354, 32)           0         
    _________________________________________________________________
    bidirectional (Bidirectional (None, 200)               107200    
    _________________________________________________________________
    dense (Dense)                (None, 6)                 1206      
    =================================================================
    Total params: 332,406
    Trainable params: 332,406
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# pakai sparse_categorical_crossentropy kalau ada 1D integer hot encode
# categorical_crossentropy for 2D w/o hot encode
model.compile(
    loss='categorical_crossentropy',
    optimizer='Adam',
    metrics=['accuracy']
)
```

## Training process


```python
phrase_len = X_train.apply(lambda p: len(p.split(' ')))
phrase_len.max()
```




    354




```python
print(train_padded.shape)
print(training_label_seq.shape)
print(validation_padded.shape)
print(validation_label_seq.shape)
```

    (57172, 354)
    (57172, 6)
    (6353, 354)
    (6353, 6)
    


```python
training_label_seq
```




    array([[0., 0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 0., 1.],
           ...,
           [0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0.],
           [0., 0., 1., 0., 0., 0.]], dtype=float32)




```python
num_epochs = 10
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, batch_size=10, validation_data=(validation_padded, validation_label_seq), verbose=1)
```

    Epoch 1/10
    5718/5718 [==============================] - 165s 28ms/step - loss: 1.1802 - accuracy: 0.4595 - val_loss: 1.0914 - val_accuracy: 0.5013
    Epoch 2/10
    5718/5718 [==============================] - 156s 27ms/step - loss: 1.0511 - accuracy: 0.5190 - val_loss: 1.0646 - val_accuracy: 0.5024
    Epoch 3/10
    5718/5718 [==============================] - 157s 27ms/step - loss: 1.0104 - accuracy: 0.5312 - val_loss: 1.0727 - val_accuracy: 0.4993
    Epoch 4/10
    5718/5718 [==============================] - 157s 27ms/step - loss: 0.9842 - accuracy: 0.5451 - val_loss: 1.0609 - val_accuracy: 0.5119
    Epoch 5/10
    5718/5718 [==============================] - 156s 27ms/step - loss: 0.9637 - accuracy: 0.5549 - val_loss: 1.0684 - val_accuracy: 0.5127
    Epoch 6/10
    5718/5718 [==============================] - 156s 27ms/step - loss: 0.9461 - accuracy: 0.5643 - val_loss: 1.0785 - val_accuracy: 0.5083
    Epoch 7/10
    5718/5718 [==============================] - 157s 27ms/step - loss: 0.9312 - accuracy: 0.5702 - val_loss: 1.0800 - val_accuracy: 0.5062
    Epoch 8/10
    5718/5718 [==============================] - 157s 27ms/step - loss: 0.9173 - accuracy: 0.5786 - val_loss: 1.0878 - val_accuracy: 0.5067
    Epoch 9/10
    5718/5718 [==============================] - 158s 28ms/step - loss: 0.9010 - accuracy: 0.5860 - val_loss: 1.0933 - val_accuracy: 0.5045
    Epoch 10/10
    5718/5718 [==============================] - 158s 28ms/step - loss: 0.8900 - accuracy: 0.5910 - val_loss: 1.1165 - val_accuracy: 0.5046
    


```python
# plot history accuracy
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
```


![png](/images/output_graph.png)



![png](/images/output_graph1.png)


## Modeling with Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
```


```python
Encoder = LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_test = Encoder.fit_transform(y_test)
```


```python
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(data['clean_text'])
X_train_Tfidf = Tfidf_vect.transform(X_train)
X_test_Tfidf = Tfidf_vect.transform(X_test)
```


```python
rf_model = RandomForestClassifier(random_state=3)
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
```


```python
from sklearn.model_selection import GridSearchCV
CV_rfc = GridSearchCV(estimator = rf_model, param_grid = param_grid, cv = 5)
CV_rfc.fit(X_train_Tfidf, y_train)
CV_rfc.best_params_
```




    {'criterion': 'entropy',
     'max_depth': 8,
     'max_features': 'auto',
     'n_estimators': 500}




```python
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
rf_best_model =RandomForestClassifier(random_state=42, max_features='log2', n_estimators= 500, max_depth=8, criterion='entropy')
rf_best_model.fit(X_train_Tfidf, y_train)

y_pred = rf_best_model.predict(X_train_Tfidf)

print('Model accuracy: %s' % accuracy_score(y_train, y_pred))
print('F1 Score :', f1_score(y_train, y_pred, average='macro'))
print('Precision :', precision_score(y_train, y_pred, average='macro'))
print('Recall :', recall_score(y_train, y_pred, average='macro'))
print(classification_report(y_train, y_pred))

cm = confusion_matrix(y_train, y_pred)
cm = pd.DataFrame(cm, [1,2,3,4,5], [1,2,3,4,5])

sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt="d")
plt.show()
```

    Model accuracy: 0.49387812215769955
    F1 Score : 0.4780980042560764
    Precision : 0.5158184429643616
    Recall : 0.4936791974983524
                  precision    recall  f1-score   support
    
               0       0.49      0.76      0.60     11435
               1       0.46      0.47      0.47     11406
               2       0.57      0.40      0.47     11441
               3       0.44      0.59      0.51     11482
               4       0.61      0.24      0.35     11408
    
        accuracy                           0.49     57172
       macro avg       0.52      0.49      0.48     57172
    weighted avg       0.52      0.49      0.48     57172
    
    


![png](/images/output_43_1.png)



```python
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
rf_best_model =RandomForestClassifier(random_state=42, max_features='log2', n_estimators= 500, max_depth=8, criterion='entropy')
rf_best_model.fit(X_train_Tfidf, y_train)


ytest = np.array(y_test)
y_pred = rf_best_model.predict(X_test_Tfidf)

print('Model accuracy: %s' % accuracy_score(y_test, y_pred))
print('F1 Score :', f1_score(y_test, y_pred, average='macro'))
print('Precision :', precision_score(y_test, y_pred, average='macro'))
print('Recall :', recall_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cm = pd.DataFrame(cm, [1,2,3,4,5], [1,2,3,4,5])

sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt="d")
plt.show()
```

    Model accuracy: 0.44215331339524633
    F1 Score : 0.41938213306233785
    Precision : 0.44761147216141295
    Recall : 0.44429202087808745
                  precision    recall  f1-score   support
    
               0       0.47      0.75      0.58      1270
               1       0.41      0.41      0.41      1299
               2       0.53      0.37      0.43      1264
               3       0.39      0.55      0.46      1223
               4       0.44      0.15      0.22      1297
    
        accuracy                           0.44      6353
       macro avg       0.45      0.44      0.42      6353
    weighted avg       0.45      0.44      0.42      6353
    
    


![png](/images/output_44_1.png)


## Modeling with Multinomial Naive Bayes


```python
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
clf_NB = MultinomialNB()
model_NB = clf_NB.fit(X_train_Tfidf, y_train)

y_train_predict_NB = model_NB.predict(X_train_Tfidf)

print('Model accuracy: %s' % accuracy_score(y_train, y_train_predict_NB))
print('F1 Score :', f1_score(y_train, y_train_predict_NB, average='macro'))
print('Precision :', precision_score(y_train, y_train_predict_NB, average='macro'))
print('Recall :', recall_score(y_train, y_train_predict_NB, average='macro'))
print(classification_report(y_train, y_train_predict_NB))

cm = confusion_matrix(y_train, y_train_predict_NB)
cm = pd.DataFrame(cm, [1,2,3,4,5], [1,2,3,4,5])

sns.heatmap(cm, annot=True, cmap="RdBu", fmt="d")
plt.show()
```

    Model accuracy: 0.5501119429091164
    F1 Score : 0.5453538238538348
    Precision : 0.5461088626154376
    Recall : 0.5502059171489326
                  precision    recall  f1-score   support
    
               0       0.61      0.76      0.68     11435
               1       0.55      0.56      0.55     11406
               2       0.55      0.49      0.52     11441
               3       0.52      0.43      0.47     11482
               4       0.51      0.51      0.51     11408
    
        accuracy                           0.55     57172
       macro avg       0.55      0.55      0.55     57172
    weighted avg       0.55      0.55      0.55     57172
    
    


![png](/images/output_46_1.png)



```python
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
clf_NB = MultinomialNB()
model_NB = clf_NB.fit(X_train_Tfidf, y_train)

y_test_predict_NB = model_NB.predict(X_test_Tfidf)

print('Model accuracy: %s' % accuracy_score(y_test, y_test_predict_NB))
print('F1 Score :', f1_score(y_test, y_test_predict_NB, average='macro'))
print('Precision :', precision_score(y_test, y_test_predict_NB, average='macro'))
print('Recall :', recall_score(y_test, y_test_predict_NB, average='macro'))
print(classification_report(y_test, y_test_predict_NB))

cm = confusion_matrix(y_test, y_test_predict_NB)
cm = pd.DataFrame(cm, [1,2,3,4,5], [1,2,3,4,5])

sns.heatmap(cm, annot=True, cmap="RdBu", fmt="d")
plt.show()
```

    Model accuracy: 0.4753659688336219
    F1 Score : 0.4687774011246607
    Precision : 0.46857469990087675
    Recall : 0.4746133278536215
                  precision    recall  f1-score   support
    
               0       0.55      0.70      0.62      1270
               1       0.47      0.48      0.47      1299
               2       0.49      0.44      0.47      1264
               3       0.41      0.34      0.37      1223
               4       0.43      0.41      0.42      1297
    
        accuracy                           0.48      6353
       macro avg       0.47      0.47      0.47      6353
    weighted avg       0.47      0.48      0.47      6353
    
    


![png](/images/output_47_1.png)


## Modeling with Support Vector Machine


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
```


```python
#Train
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

clf = LinearSVC() #Kalau ingin regularisasi data C = 10 atau lebih dan class_weight = 'balanced'
clf.fit(X_train_Tfidf, y_train)

pred_SVM = clf.predict(X_train_Tfidf)

print('Model accuracy: %s' % accuracy_score(y_train, pred_SVM))
print('F1 Score :', f1_score(y_train, pred_SVM, average='macro'))
print('Precision :', precision_score(y_train, pred_SVM, average='macro'))
print('Recall :', recall_score(y_train, pred_SVM, average='macro'))
print(classification_report(y_train, pred_SVM))

cm = confusion_matrix(y_train, pred_SVM)
cm = pd.DataFrame(cm, [1,2,3,4,5], [1,2,3,4,5])

sns.heatmap(cm, annot=True, cmap="Pastel1", fmt="d")
plt.show()
```

    Model accuracy: 0.6071503533198069
    F1 Score : 0.6036313175586325
    Precision : 0.6027360016658762
    Recall : 0.6071948620082958
                  precision    recall  f1-score   support
    
               0       0.69      0.82      0.75     11435
               1       0.62      0.60      0.61     11406
               2       0.61      0.59      0.60     11441
               3       0.56      0.50      0.53     11482
               4       0.54      0.52      0.53     11408
    
        accuracy                           0.61     57172
       macro avg       0.60      0.61      0.60     57172
    weighted avg       0.60      0.61      0.60     57172
    
    


![png](/images/output_50_1.png)



```python
#Test
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

clf = LinearSVC()
clf.fit(X_train_Tfidf, y_train)

y_pred1 = clf.predict(X_test_Tfidf)

print('Model accuracy: %s' % accuracy_score(y_test, y_pred1))
print('F1 Score :', f1_score(y_test, y_pred1, average='macro'))
print('Precision :', precision_score(y_test, y_pred1, average='macro'))
print('Recall :', recall_score(y_test, y_pred1, average='macro'))
print(classification_report(y_test, y_pred1))

cm = confusion_matrix(y_test, y_pred1)
cm = pd.DataFrame(cm, [1,2,3,4,5], [1,2,3,4,5])

sns.heatmap(cm, annot=True, cmap="Pastel1", fmt="d")
plt.show()
```

    Model accuracy: 0.47993074138202424
    F1 Score : 0.4744259165642724
    Precision : 0.4725665355846921
    Recall : 0.4797036838202741
                  precision    recall  f1-score   support
    
               0       0.59      0.72      0.65      1270
               1       0.46      0.43      0.45      1299
               2       0.49      0.49      0.49      1264
               3       0.40      0.37      0.38      1223
               4       0.43      0.39      0.41      1297
    
        accuracy                           0.48      6353
       macro avg       0.47      0.48      0.47      6353
    weighted avg       0.47      0.48      0.47      6353
    
    


![png](/images/output_511.png)


## Predict with Long Short Term Memory


```python
# predict 1 - fake generated review
txt = ["barangnya bagus, seller oke dan terpercaya, mantaplah pokoknya."]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)
print(np.argmax(pred))
```

    5
    


```python
# predict 2 - fake generated review
txt = ["dissapointed, seller galak dan barang telat nyampe."]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)
print(np.argmax(pred))
```

    1
    


```python
# predict 3 - taken from shopee real customer review with 3 stars rating
txt = ["Kl aku LBh suka yg real gini tmn tmn,kadang ad jg fg yg suka konyol kadang bajunya  wrna merah di edit look nya ala tone kuning gt pas baju merah nya berubah jd kuning, lahhh percuma donk manten Ny pilih merah bajunya di edit kuning 😤😤😤"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)
print(np.argmax(pred))
```

    4
    


```python
# predict 4 - taken from shopee real customer review with 2 stars rating
txt = ["Alhamdulillah paket sudah saya terima. Tpi saya kecewa, Krn pesen warna mustard koq datangnya coklat? Pdhl bwt dress code acara"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)
print(np.argmax(pred))
```

    2
    


```python
# predict 5 - taken from shopee real customer review with 4 stars rating
txt = ["Harga sama kualitas sesuailah yaaaa....suka sekali sama modelnya warna navy nya juga bagus... ga bladus....meskipun ada sobekannya dikit....tolong diperhatikan lagi yaaaa...."]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)
print(np.argmax(pred))
```

    3
    
