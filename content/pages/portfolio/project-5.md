---
title: Twitter Sentiment Analysis on Bank Nasional Indonesia Stock
subtitle: Optional Project Subtitle
date: '2019-01-05'
thumb_image: /images/stock.jpg
thumb_image_alt: A white gamepad on a green background
image: images/5.jpg
image_alt: A white gamepad on a green background
seo:
  title: Project Title 5
  description: This is the project 5 description
  extra:
    - name: 'og:type'
      value: website
      keyName: property
    - name: 'og:title'
      value: Project Title 5
      keyName: property
    - name: 'og:description'
      value: This is the project 5 description
      keyName: property
    - name: 'og:image'
      value: images/5.jpg
      keyName: property
      relativeUrl: true
    - name: 'twitter:card'
      value: summary_large_image
    - name: 'twitter:title'
      value: Project Title 5
    - name: 'twitter:description'
      value: This is the project 5 description
    - name: 'twitter:image'
      value: images/5.jpg
      relativeUrl: true
layout: project
---

**Web Scrapping**

Authentication


```python
import tweepy
import pandas as pd

# Variables that contains the credentials to access Twitter API
ACCESS_TOKEN = '1090888242335932421-S4eIEaJeEU62Jt8hW1VsqEuVVqsvp3'
ACCESS_SECRET = '4mCzr8Dq9QhCftWJFY87o1PFcgxwDlhyUAH58loK2ThxW'
CONSUMER_KEY = 'bHJY3dVFriRHJ8AizVnBywzGf'
CONSUMER_SECRET = 'jOs9kXL3cX6f43IvCXq6b28cL2AG8bNYEJEYySSRkeLYgYqnQn'


# Setup access to API
def connect_to_twitter_OAuth():
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    api = tweepy.API(auth)
    return api

# Create API object
api = connect_to_twitter_OAuth()
```

Get Tweets according to keyword "BNI"


```python
text_query = 'BNI'
max_tweets = 1000
 
# Creation of query method using parameters
tweets = tweepy.Cursor(api.search,q=text_query,since="2020-11-01", until="2020-11-29").items(max_tweets)
 
# Pulling information from tweets iterable object
tweets_list = [[tweet.id_str,
                tweet.created_at, 
                tweet.text, 
                tweet.user.name, 
                tweet.user.screen_name, 
                tweet.user.id_str, 
                tweet.user.location, 
                tweet.user.followers_count] for tweet in tweets]
 
# Creation of dataframe from tweets_list
tweets_df = pd.DataFrame(tweets_list, columns=['id_str',
                                               'created_at',
                                               'tweets',
                                               'user_name',
                                               'user_screenname',
                                               'user_id',
                                               'user_location',
                                               'user_followerscount'])
tweets_df
```

Mount to Google Drive


```python
from google.colab import drive
drive.mount('/gdrive')
%cd /gdrive
```

Saved to CSV


```python
result = pd.DataFrame(tweets_df)
result.to_csv('/gdrive/My Drive/Project_DE/data.csv', index = False)
print("Data has been saved successfully")
```

**Retrieve Data**


```python
from google.colab import drive
drive.mount('/gdrive')
%cd /gdrive
```

    Mounted at /gdrive
    /gdrive
    


```python
import pandas as pd
data21 = pd.read_csv('/gdrive/My Drive/Project_DE/data_21.csv')
data21
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
      <th>time</th>
      <th>tweets_id</th>
      <th>tweets</th>
      <th>retweets</th>
      <th>count_likes</th>
      <th>source</th>
      <th>source_url</th>
      <th>Unnamed: 7</th>
      <th>Unnamed: 8</th>
      <th>Unnamed: 9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21-11-20 11:59</td>
      <td>1.330300e+18</td>
      <td>@masmasjawabgt Saldonya buat nabung boleh ga m...</td>
      <td>0</td>
      <td>0</td>
      <td>Twitter for Android</td>
      <td>http://twitter.com/download/android</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21-11-20 11:56</td>
      <td>1.330300e+18</td>
      <td>RT @hhjdinn: [Help RT]\n\nBismillah🙏🏻\nYukk ak...</td>
      <td>27</td>
      <td>0</td>
      <td>Twitter for Android</td>
      <td>http://twitter.com/download/android</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21-11-20 11:54</td>
      <td>1.330300e+18</td>
      <td>@banjarbase Kalo aku kmrian BNI mbanking tingg...</td>
      <td>0</td>
      <td>0</td>
      <td>Twitter for Android</td>
      <td>http://twitter.com/download/android</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21-11-20 23:51</td>
      <td>1.330300e+18</td>
      <td>@DORbertanya @BNI @BPJSTKinfo @KemnakerRI Saya...</td>
      <td>0</td>
      <td>2</td>
      <td>Twitter for Android</td>
      <td>http://twitter.com/download/android</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21-11-20 11:48</td>
      <td>1.330300e+18</td>
      <td>Transfer k No rekening\nBRI 411401008723530\nB...</td>
      <td>0</td>
      <td>0</td>
      <td>twittbot.net</td>
      <td>http://twittbot.net/</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>376</th>
      <td>30-11-20 15:01</td>
      <td>1.330000e+18</td>
      <td>@niagarapokpou @JNECare @jntexpressid Halo @BN...</td>
      <td>0</td>
      <td>0</td>
      <td>Twitter for Android</td>
      <td>http://twitter.com/download/android</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>377</th>
      <td>30-11-20 15:01</td>
      <td>1.330000e+18</td>
      <td>@restuachill_ @astro_rizki @Iqbalfas @hrdbacot...</td>
      <td>0</td>
      <td>0</td>
      <td>Twitter for iPhone</td>
      <td>http://twitter.com/download/iphone</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>378</th>
      <td>30-11-20 15:00</td>
      <td>1.330000e+18</td>
      <td>RT @johfamfess: Cara Berdonasi\n1. Klik Tombol...</td>
      <td>4</td>
      <td>0</td>
      <td>Twitter for Android</td>
      <td>http://twitter.com/download/android</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>379</th>
      <td>30-11-20 14:59</td>
      <td>1.330000e+18</td>
      <td>@QnA18MENFESS BNI TAPLUS MUDA nder, admin nya ...</td>
      <td>0</td>
      <td>0</td>
      <td>Twitter for Android</td>
      <td>http://twitter.com/download/android</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>380</th>
      <td>30-11-20 14:58</td>
      <td>1.330000e+18</td>
      <td>@sellkpopfess 414,250 dana/BNI/ovo</td>
      <td>0</td>
      <td>0</td>
      <td>Twitter for Android</td>
      <td>http://twitter.com/download/android</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>381 rows × 10 columns</p>
</div>



**Text Pre-Processing**


```python
data21 = pd.DataFrame(data21,columns=['time','tweets'])
data21.head()
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
      <th>time</th>
      <th>tweets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21-11-20 11:59</td>
      <td>@masmasjawabgt Saldonya buat nabung boleh ga m...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21-11-20 11:56</td>
      <td>RT @hhjdinn: [Help RT]\n\nBismillah🙏🏻\nYukk ak...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21-11-20 11:54</td>
      <td>@banjarbase Kalo aku kmrian BNI mbanking tingg...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21-11-20 23:51</td>
      <td>@DORbertanya @BNI @BPJSTKinfo @KemnakerRI Saya...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21-11-20 11:48</td>
      <td>Transfer k No rekening\nBRI 411401008723530\nB...</td>
    </tr>
  </tbody>
</table>
</div>



Removing '@names'


```python
import numpy as np
import re
def remove_pattern(text, pattern_regex):
    r = re.findall(pattern_regex, text)
    for i in r:
        text = re.sub(i, '', text)
    
    return text 
```


```python
data21['tweets'] = np.vectorize(remove_pattern)(data21['tweets'], "@[\w]*")
data21.head()
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
      <th>time</th>
      <th>tweets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21-11-20 11:59</td>
      <td>Saldonya buat nabung boleh ga mas? Kalo boleh...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21-11-20 11:56</td>
      <td>RT : [Help RT]\n\nBismillah🙏🏻\nYukk aku jual j...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21-11-20 11:54</td>
      <td>Kalo aku kmrian BNI mbanking tinggal unintall...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21-11-20 23:51</td>
      <td>Saya BNI jg blm cair sampe skrg, padahal p...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21-11-20 11:48</td>
      <td>Transfer k No rekening\nBRI 411401008723530\nB...</td>
    </tr>
  </tbody>
</table>
</div>



Removing links, such as http, https


```python
cleaned_tweets = []
for index, row in data21.iterrows():
    # Here we are filtering out all the words that contains link
    words_without_links = [word for word in row.tweets.split() if 'http' not in word]
    cleaned_tweets.append(' '.join(words_without_links))
data21['tweets'] = cleaned_tweets
data21.head()
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
      <th>time</th>
      <th>tweets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21-11-20 11:59</td>
      <td>Saldonya buat nabung boleh ga mas? Kalo boleh ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21-11-20 11:56</td>
      <td>RT : [Help RT] Bismillah🙏🏻 Yukk aku jual jasa ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21-11-20 11:54</td>
      <td>Kalo aku kmrian BNI mbanking tinggal unintall ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21-11-20 23:51</td>
      <td>Saya BNI jg blm cair sampe skrg, padahal pas b...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21-11-20 11:48</td>
      <td>Transfer k No rekening BRI 411401008723530 BCA...</td>
    </tr>
  </tbody>
</table>
</div>



Removing tweets with empty text


```python
data21 = data21[data21['tweets']!='']
data21.head()
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
      <th>time</th>
      <th>tweets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21-11-20 11:59</td>
      <td>Saldonya buat nabung boleh ga mas? Kalo boleh ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21-11-20 11:56</td>
      <td>RT : [Help RT] Bismillah🙏🏻 Yukk aku jual jasa ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21-11-20 11:54</td>
      <td>Kalo aku kmrian BNI mbanking tinggal unintall ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21-11-20 23:51</td>
      <td>Saya BNI jg blm cair sampe skrg, padahal pas b...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21-11-20 11:48</td>
      <td>Transfer k No rekening BRI 411401008723530 BCA...</td>
    </tr>
  </tbody>
</table>
</div>



Removing Punctuations, Special characters and Emoticon


```python
data21['tweets'] = data21['tweets'].str.replace("[^a-zA-Z0-9]#?", " ")
data21['tweets'] = data21['tweets'].str.replace(r'[^\x00-\x7F]+', '', regex=True)
data21.head()
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
      <th>time</th>
      <th>tweets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21-11-20 11:59</td>
      <td>Saldonya buat nabung boleh ga mas  Kalo boleh ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21-11-20 11:56</td>
      <td>RT    Help RT  Bismillah   Yukk aku jual jasa ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21-11-20 11:54</td>
      <td>Kalo aku kmrian BNI mbanking tinggal unintall ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21-11-20 23:51</td>
      <td>Saya BNI jg blm cair sampe skrg  padahal pas b...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21-11-20 11:48</td>
      <td>Transfer k No rekening BRI 411401008723530 BCA...</td>
    </tr>
  </tbody>
</table>
</div>



Removing Stop words


```python
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = set(stopwords.words("indonesian"))
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    


```python
stopwords_set = set(stopwords)
cleaned_tweets = []

for index, row in data21.iterrows():
    # filerting out all the stopwords 
    words_without_stopwords = [word for word in row.tweets.split() if not word in stopwords_set and '#' not in word.lower()]  
    # finally creating tweets list of tuples containing stopwords(list) and sentimentType 
    cleaned_tweets.append(' '.join(words_without_stopwords))
    
data21['tweets'] = cleaned_tweets
data21.head()
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
      <th>time</th>
      <th>tweets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21-11-20 11:59</td>
      <td>Saldonya nabung ga mas Kalo ikutan kirim BNI s...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21-11-20 11:56</td>
      <td>RT Help RT Bismillah Yukk jual jasa premium ap...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21-11-20 11:54</td>
      <td>Kalo kmrian BNI mbanking tinggal unintall inst...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21-11-20 23:51</td>
      <td>Saya BNI jg blm cair sampe skrg pas blt termin...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21-11-20 11:48</td>
      <td>Transfer k No rekening BRI 411401008723530 BCA...</td>
    </tr>
  </tbody>
</table>
</div>



Tokenization


```python
from nltk.stem import WordNetLemmatizer# Tokenization
nltk.download('wordnet')
```

    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    




    True




```python
tokenized_tweet = data21['tweets'].apply(lambda x: x.split())

# Finding Lemma for each word
word_lemmatizer = WordNetLemmatizer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [word_lemmatizer.lemmatize(i) for i in x])

#joining words into sentences (from where they came from)
for i, tokens in enumerate(tokenized_tweet):
    tokenized_tweet[i] = ' '.join(tokens)

data21['tweets'] = tokenized_tweet
tokenized_tweet.head()
```




    0    Saldonya nabung ga ma Kalo ikutan kirim BNI se...
    1    RT Help RT Bismillah Yukk jual jasa premium ap...
    2    Kalo kmrian BNI mbanking tinggal unintall inst...
    3    Saya BNI jg blm cair sampe skrg pa blt termin ...
    4    Transfer k No rekening BRI 411401008723530 BCA...
    Name: tweets, dtype: object



**Visualization**

Fetch Sentiments using Sentiment Analyzer TextBlob


```python
# To identify the sentiment of text
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.np_extractors import ConllExtractor
```


```python
def fetch_sentiment_using_textblob(text):
    analysis = TextBlob(text)
    return 'positive' if analysis.sentiment.polarity >= 0 else 'negative'
sentiments_using_textblob21 = data21.tweets.apply(lambda tweet: fetch_sentiment_using_textblob(tweet))
pd.DataFrame(sentiments_using_textblob21.value_counts())
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
      <th>tweets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>positive</th>
      <td>368</td>
    </tr>
    <tr>
      <th>negative</th>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
data21['sentiment'] = sentiments_using_textblob21
data21.head()
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
      <th>time</th>
      <th>tweets</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21-11-20 11:59</td>
      <td>Saldonya nabung ga ma Kalo ikutan kirim BNI se...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21-11-20 11:56</td>
      <td>RT Help RT Bismillah Yukk jual jasa premium ap...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21-11-20 11:54</td>
      <td>Kalo kmrian BNI mbanking tinggal unintall inst...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21-11-20 23:51</td>
      <td>Saya BNI jg blm cair sampe skrg pa blt termin ...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21-11-20 11:48</td>
      <td>Transfer k No rekening BRI 411401008723530 BCA...</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>




```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
```


```python
def generate_wordcloud(all_words):
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=100, relative_scaling=0.5, colormap='RdBu').generate(all_words)

    plt.figure(figsize=(14, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()
```


```python
all_words21 = ' '.join([text for text in data21['tweets'][data21.sentiment == 'positive']])
generate_wordcloud(all_words21)
```


![png](/images/output_36_0.png)



```python
all_words21 = ' '.join([text for text in data21['tweets'][data21.sentiment == 'negative']])
generate_wordcloud(all_words21)
```


![png](/images/output_37_0.png)


**Feature Extraction Using CountVectorizer**


```python
from sklearn.feature_extraction.text import CountVectorizer
```


```python
# Count Vectorization features
count_word_vectorizer = CountVectorizer(max_df=0.90, min_df=2, stop_words='english')
# Count Vectorization feature matrix
count_word_feature = count_word_vectorizer.fit_transform(data21['tweets'])
```

**Model Building: Sentiment Analysis**


```python
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
```


```python
target_variable = data21['sentiment'].apply(lambda x: 0 if x=='negative' else 1)
```


```python
# Calculate accuracy by using confusion matrix
def plot_confusion_matrix(matrix):
    plt.clf()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Set2_r)
    classNames = ['Positive', 'Negative']
    plt.title('Confusion Matrix')
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    s = [['TP','FP'], ['FN', 'TN']]
    #TP = True Positive
    #FP = False Positive
    #FN = False Negative
    #TN = True Negative --> data negative terdeteksi benar

    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(matrix[i][j]))
    plt.show()
```

Split to Training Data and Testing Data


```python
X_train, X_test, y_train, y_test = train_test_split(count_word_feature, target_variable, test_size=0.3, random_state=272)
```

Support Vector Machine Model


```python
clf_svm = svm.SVC(kernel='sigmoid', gamma=1.0,probability=True)
clf_svm.fit(X_train,y_train)
pred_svm = clf_svm.predict(X_test.toarray())
print(f'Accuracy Score = {accuracy_score(y_test, pred_svm)}')
conf_matrix = confusion_matrix(y_test, pred_svm, labels=[True, False])
plot_confusion_matrix(conf_matrix)
```

    Accuracy Score = 0.991304347826087
    


![png](output_48_1.png)


Count Sentiment Positive and Negative Per Day


```python
# convert time_date col to datetime64 dtype
data21['time'] = pd.to_datetime(data21['time'], utc=True) 
data21.set_index('time', inplace=True)
print(data21.index.date)
```

    [datetime.date(2020, 11, 21) datetime.date(2020, 11, 21)
     datetime.date(2020, 11, 21) datetime.date(2020, 11, 21)
     datetime.date(2020, 11, 21) datetime.date(2020, 11, 21)
     datetime.date(2020, 11, 21) datetime.date(2020, 11, 21)
     datetime.date(2020, 11, 21) datetime.date(2020, 11, 21)
     datetime.date(2020, 11, 21) datetime.date(2020, 11, 21)
     datetime.date(2020, 11, 21) datetime.date(2020, 11, 21)
     datetime.date(2020, 11, 21) datetime.date(2020, 11, 21)
     datetime.date(2020, 11, 21) datetime.date(2020, 11, 21)
     datetime.date(2020, 11, 21) datetime.date(2020, 11, 21)
     datetime.date(2020, 11, 21) datetime.date(2020, 11, 21)
     datetime.date(2020, 11, 21) datetime.date(2020, 11, 21)
     datetime.date(2020, 11, 21) datetime.date(2020, 11, 21)
     datetime.date(2020, 11, 21) datetime.date(2020, 11, 21)
     datetime.date(2020, 11, 21) datetime.date(2020, 11, 21)
     datetime.date(2020, 11, 22) datetime.date(2020, 11, 22)
     datetime.date(2020, 11, 22) datetime.date(2020, 11, 22)
     datetime.date(2020, 11, 22) datetime.date(2020, 11, 22)
     datetime.date(2020, 11, 22) datetime.date(2020, 11, 22)
     datetime.date(2020, 11, 22) datetime.date(2020, 11, 22)
     datetime.date(2020, 11, 22) datetime.date(2020, 11, 22)
     datetime.date(2020, 11, 22) datetime.date(2020, 11, 22)
     datetime.date(2020, 11, 22) datetime.date(2020, 11, 22)
     datetime.date(2020, 11, 22) datetime.date(2020, 11, 22)
     datetime.date(2020, 11, 22) datetime.date(2020, 11, 22)
     datetime.date(2020, 11, 22) datetime.date(2020, 11, 22)
     datetime.date(2020, 11, 22) datetime.date(2020, 11, 22)
     datetime.date(2020, 11, 22) datetime.date(2020, 11, 22)
     datetime.date(2020, 11, 22) datetime.date(2020, 11, 22)
     datetime.date(2020, 11, 22) datetime.date(2020, 11, 22)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 23)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 24) datetime.date(2020, 11, 25)
     datetime.date(2020, 11, 25) datetime.date(2020, 11, 25)
     datetime.date(2020, 11, 25) datetime.date(2020, 11, 25)
     datetime.date(2020, 11, 25) datetime.date(2020, 11, 25)
     datetime.date(2020, 11, 25) datetime.date(2020, 11, 25)
     datetime.date(2020, 11, 25) datetime.date(2020, 11, 25)
     datetime.date(2020, 11, 25) datetime.date(2020, 11, 25)
     datetime.date(2020, 11, 25) datetime.date(2020, 11, 25)
     datetime.date(2020, 11, 25) datetime.date(2020, 11, 25)
     datetime.date(2020, 11, 25) datetime.date(2020, 11, 25)
     datetime.date(2020, 11, 25) datetime.date(2020, 11, 25)
     datetime.date(2020, 11, 25) datetime.date(2020, 11, 25)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 26) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 27) datetime.date(2020, 11, 27)
     datetime.date(2020, 11, 27) datetime.date(2020, 11, 27)
     datetime.date(2020, 11, 27) datetime.date(2020, 11, 27)
     datetime.date(2020, 11, 27) datetime.date(2020, 11, 27)
     datetime.date(2020, 11, 27) datetime.date(2020, 11, 27)
     datetime.date(2020, 11, 27) datetime.date(2020, 11, 27)
     datetime.date(2020, 11, 27) datetime.date(2020, 11, 27)
     datetime.date(2020, 11, 27) datetime.date(2020, 11, 27)
     datetime.date(2020, 11, 27) datetime.date(2020, 11, 27)
     datetime.date(2020, 11, 27) datetime.date(2020, 11, 27)
     datetime.date(2020, 11, 27) datetime.date(2020, 11, 27)
     datetime.date(2020, 11, 27) datetime.date(2020, 11, 27)
     datetime.date(2020, 11, 27) datetime.date(2020, 11, 27)
     datetime.date(2020, 11, 27) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 28) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 28) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 28) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 28) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 28) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 28) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 28) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 28) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 28) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 28) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 28) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 28) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 28) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 28) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 28) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 28) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 28) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 28) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 29)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30) datetime.date(2020, 11, 30)
     datetime.date(2020, 11, 30)]
    


```python
print(data21.groupby(data21.index.date).count())
```

                tweets  sentiment
    2020-11-21      30         30
    2020-11-22      30         30
    2020-11-23      72         72
    2020-11-24      37         37
    2020-11-25      23         23
    2020-11-26      38         38
    2020-11-27      27         27
    2020-11-28      36         36
    2020-11-29      46         46
    2020-11-30      42         42
    


```python
data21["_dummy"]=1
df1=data21.pivot_table(index=["time"], columns="sentiment", values="_dummy", aggfunc="sum").fillna(0)
df1.replace(0, np.nan, inplace=True) # change value 0 to NaN
df1

result = pd.DataFrame(df1)
result
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
      <th>sentiment</th>
      <th>negative</th>
      <th>positive</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-11-21 11:19:00+00:00</th>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2020-11-21 11:22:00+00:00</th>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2020-11-21 11:24:00+00:00</th>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2020-11-21 11:30:00+00:00</th>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2020-11-21 11:38:00+00:00</th>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-11-30 15:28:00+00:00</th>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2020-11-30 15:29:00+00:00</th>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2020-11-30 15:30:00+00:00</th>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2020-11-30 15:32:00+00:00</th>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2020-11-30 15:33:00+00:00</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>278 rows × 2 columns</p>
</div>




```python
output = result.groupby(result.index.date).count()
output
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
      <th>sentiment</th>
      <th>negative</th>
      <th>positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-11-21</th>
      <td>0</td>
      <td>26</td>
    </tr>
    <tr>
      <th>2020-11-22</th>
      <td>1</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2020-11-23</th>
      <td>1</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2020-11-24</th>
      <td>3</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2020-11-25</th>
      <td>0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2020-11-26</th>
      <td>1</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2020-11-27</th>
      <td>2</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2020-11-28</th>
      <td>2</td>
      <td>32</td>
    </tr>
    <tr>
      <th>2020-11-29</th>
      <td>0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2020-11-30</th>
      <td>3</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
</div>



Plot with Time series


```python
import matplotlib
import matplotlib.pyplot as plt

output.plot()
plt.xlabel("Date",size=12)
plt.show()
```


![png](output_55_0.png)



```python
stock = pd.read_csv('/gdrive/My Drive/Project_DE/data_union/BBNI.JK.csv')
stock.tail(12)
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
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>235</th>
      <td>2020-11-21</td>
      <td>5700</td>
      <td>5800</td>
      <td>5650</td>
      <td>5675</td>
      <td>5675.0</td>
      <td>53141400</td>
    </tr>
    <tr>
      <th>236</th>
      <td>2020-11-22</td>
      <td>5688</td>
      <td>5888</td>
      <td>5638</td>
      <td>5788</td>
      <td>5788.0</td>
      <td>67838750</td>
    </tr>
    <tr>
      <th>237</th>
      <td>2020-11-23</td>
      <td>5675</td>
      <td>5975</td>
      <td>5625</td>
      <td>5900</td>
      <td>5900.0</td>
      <td>82536100</td>
    </tr>
    <tr>
      <th>238</th>
      <td>2020-11-24</td>
      <td>5975</td>
      <td>5975</td>
      <td>5900</td>
      <td>5950</td>
      <td>5950.0</td>
      <td>61987700</td>
    </tr>
    <tr>
      <th>239</th>
      <td>2020-11-25</td>
      <td>6000</td>
      <td>6175</td>
      <td>6000</td>
      <td>6050</td>
      <td>6050.0</td>
      <td>101206400</td>
    </tr>
    <tr>
      <th>240</th>
      <td>2020-11-26</td>
      <td>6050</td>
      <td>6350</td>
      <td>6025</td>
      <td>6300</td>
      <td>6300.0</td>
      <td>66773100</td>
    </tr>
    <tr>
      <th>241</th>
      <td>2020-11-27</td>
      <td>6325</td>
      <td>6375</td>
      <td>6200</td>
      <td>6350</td>
      <td>6350.0</td>
      <td>64918200</td>
    </tr>
    <tr>
      <th>242</th>
      <td>2020-11-28</td>
      <td>6338</td>
      <td>6363</td>
      <td>6075</td>
      <td>6175</td>
      <td>6175.0</td>
      <td>112662650</td>
    </tr>
    <tr>
      <th>243</th>
      <td>2020-11-29</td>
      <td>6338</td>
      <td>6363</td>
      <td>6075</td>
      <td>6175</td>
      <td>6175.0</td>
      <td>112662650</td>
    </tr>
    <tr>
      <th>244</th>
      <td>2020-11-30</td>
      <td>6350</td>
      <td>6350</td>
      <td>5950</td>
      <td>6000</td>
      <td>6000.0</td>
      <td>160407100</td>
    </tr>
    <tr>
      <th>245</th>
      <td>2020-12-01</td>
      <td>6075</td>
      <td>6250</td>
      <td>5925</td>
      <td>6250</td>
      <td>6250.0</td>
      <td>99785300</td>
    </tr>
    <tr>
      <th>246</th>
      <td>2020-12-02</td>
      <td>6300</td>
      <td>6375</td>
      <td>6200</td>
      <td>6350</td>
      <td>6350.0</td>
      <td>51993700</td>
    </tr>
  </tbody>
</table>
</div>




```python
#stock_10 = pd.concat([stock.loc[235:244,['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close']]], axis=1)
stock_10 = pd.concat([stock.loc[235:244,['Date','Close']]], axis=1)
#stock_10.columns = ['time', 'Open', 'High', 'Low', 'Close', 'Adj Close']
stock_10.columns = ['time','Close']
stock_10
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
      <th>time</th>
      <th>Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>235</th>
      <td>2020-11-21</td>
      <td>5675</td>
    </tr>
    <tr>
      <th>236</th>
      <td>2020-11-22</td>
      <td>5788</td>
    </tr>
    <tr>
      <th>237</th>
      <td>2020-11-23</td>
      <td>5900</td>
    </tr>
    <tr>
      <th>238</th>
      <td>2020-11-24</td>
      <td>5950</td>
    </tr>
    <tr>
      <th>239</th>
      <td>2020-11-25</td>
      <td>6050</td>
    </tr>
    <tr>
      <th>240</th>
      <td>2020-11-26</td>
      <td>6300</td>
    </tr>
    <tr>
      <th>241</th>
      <td>2020-11-27</td>
      <td>6350</td>
    </tr>
    <tr>
      <th>242</th>
      <td>2020-11-28</td>
      <td>6175</td>
    </tr>
    <tr>
      <th>243</th>
      <td>2020-11-29</td>
      <td>6175</td>
    </tr>
    <tr>
      <th>244</th>
      <td>2020-11-30</td>
      <td>6000</td>
    </tr>
  </tbody>
</table>
</div>




```python
stock_10['time'] = pd.to_datetime(stock_10['time'], errors='coerce')
stock_10
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
      <th>time</th>
      <th>Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>235</th>
      <td>2020-11-21</td>
      <td>5675</td>
    </tr>
    <tr>
      <th>236</th>
      <td>2020-11-22</td>
      <td>5788</td>
    </tr>
    <tr>
      <th>237</th>
      <td>2020-11-23</td>
      <td>5900</td>
    </tr>
    <tr>
      <th>238</th>
      <td>2020-11-24</td>
      <td>5950</td>
    </tr>
    <tr>
      <th>239</th>
      <td>2020-11-25</td>
      <td>6050</td>
    </tr>
    <tr>
      <th>240</th>
      <td>2020-11-26</td>
      <td>6300</td>
    </tr>
    <tr>
      <th>241</th>
      <td>2020-11-27</td>
      <td>6350</td>
    </tr>
    <tr>
      <th>242</th>
      <td>2020-11-28</td>
      <td>6175</td>
    </tr>
    <tr>
      <th>243</th>
      <td>2020-11-29</td>
      <td>6175</td>
    </tr>
    <tr>
      <th>244</th>
      <td>2020-11-30</td>
      <td>6000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# convert time_date col to datetime64 dtype
#stock_10['time'] = pd.to_datetime(stock_10['time'], utc=True) 
stock_10.set_index('time', inplace=True)
print(stock_10.index.date)
```

    [datetime.date(2020, 11, 21) datetime.date(2020, 11, 22)
     datetime.date(2020, 11, 23) datetime.date(2020, 11, 24)
     datetime.date(2020, 11, 25) datetime.date(2020, 11, 26)
     datetime.date(2020, 11, 27) datetime.date(2020, 11, 28)
     datetime.date(2020, 11, 29) datetime.date(2020, 11, 30)]
    


```python
stock_10
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
      <th>Close</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-11-21</th>
      <td>5675</td>
    </tr>
    <tr>
      <th>2020-11-22</th>
      <td>5788</td>
    </tr>
    <tr>
      <th>2020-11-23</th>
      <td>5900</td>
    </tr>
    <tr>
      <th>2020-11-24</th>
      <td>5950</td>
    </tr>
    <tr>
      <th>2020-11-25</th>
      <td>6050</td>
    </tr>
    <tr>
      <th>2020-11-26</th>
      <td>6300</td>
    </tr>
    <tr>
      <th>2020-11-27</th>
      <td>6350</td>
    </tr>
    <tr>
      <th>2020-11-28</th>
      <td>6175</td>
    </tr>
    <tr>
      <th>2020-11-29</th>
      <td>6175</td>
    </tr>
    <tr>
      <th>2020-11-30</th>
      <td>6000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calculate the difference between rows - By default, periods = 1
difference = stock_10.diff(axis=0);
print("Difference between rows(Period=1):");
print(difference);
```

    Difference between rows(Period=1):
                Close
    time             
    2020-11-21    NaN
    2020-11-22  113.0
    2020-11-23  112.0
    2020-11-24   50.0
    2020-11-25  100.0
    2020-11-26  250.0
    2020-11-27   50.0
    2020-11-28 -175.0
    2020-11-29    0.0
    2020-11-30 -175.0
    


```python
end_result = pd.concat([difference, output], axis=1)
end_result
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
      <th>Close</th>
      <th>negative</th>
      <th>positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-11-21</th>
      <td>NaN</td>
      <td>0</td>
      <td>26</td>
    </tr>
    <tr>
      <th>2020-11-22</th>
      <td>113.0</td>
      <td>1</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2020-11-23</th>
      <td>112.0</td>
      <td>1</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2020-11-24</th>
      <td>50.0</td>
      <td>3</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2020-11-25</th>
      <td>100.0</td>
      <td>0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2020-11-26</th>
      <td>250.0</td>
      <td>1</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2020-11-27</th>
      <td>50.0</td>
      <td>2</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2020-11-28</th>
      <td>-175.0</td>
      <td>2</td>
      <td>32</td>
    </tr>
    <tr>
      <th>2020-11-29</th>
      <td>0.0</td>
      <td>0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2020-11-30</th>
      <td>-175.0</td>
      <td>3</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
</div>




```python
end_result1 = end_result.fillna(0)
end_result1
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
      <th>Close</th>
      <th>negative</th>
      <th>positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-11-21</th>
      <td>0.0</td>
      <td>0</td>
      <td>26</td>
    </tr>
    <tr>
      <th>2020-11-22</th>
      <td>113.0</td>
      <td>1</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2020-11-23</th>
      <td>112.0</td>
      <td>1</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2020-11-24</th>
      <td>50.0</td>
      <td>3</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2020-11-25</th>
      <td>100.0</td>
      <td>0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2020-11-26</th>
      <td>250.0</td>
      <td>1</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2020-11-27</th>
      <td>50.0</td>
      <td>2</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2020-11-28</th>
      <td>-175.0</td>
      <td>2</td>
      <td>32</td>
    </tr>
    <tr>
      <th>2020-11-29</th>
      <td>0.0</td>
      <td>0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2020-11-30</th>
      <td>-175.0</td>
      <td>3</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
</div>



Correlation Test Close ~ Sentiment



```python
x = end_result1['Close']
y = end_result1['positive']
z = end_result1['negative']
```


```python
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr

corr, _ = pearsonr(x, y)
print('Pearsons correlation Close ~ Positive: %.3f' % corr)
corr1,_ = pearsonr(x, z)
print('Pearsons correlation Close ~ Negative: %.3f' % corr1)
```

    Pearsons correlation Close ~ Positive: -0.104
    Pearsons correlation Close ~ Negative: -0.420
    

Plot Sentimen dan Saham


```python
import matplotlib
import matplotlib.pyplot as plt

end_result1.plot()
plt.xlabel("Date",size=16)
plt.ylim(-300,300)

plt.show()
```


![png](output_68_0.png)


Linear Regression


```python
x = end_result1['Close'].values.reshape(-1,1)
y = end_result1['positive'].values.reshape(-1,1)
z = end_result1['negative'].values.reshape(-1,1)
```


```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x, y)
print(model.coef_)
print(model.intercept_)
#Y = 0.183x — 27.9178022
```

    [[-0.00261888]]
    [27.08511373]
    


```python
#Plotting
plt.scatter(x, y)
plt.xlabel('Close')
plt.ylabel('Positive')
plt.title('Plot Close dengan Positive Sentiment')
```




    Text(0.5, 1.0, 'Plot Close dengan Positive Sentiment')




![png](output_72_1.png)



```python
#Plotting
plt.scatter(x, z)
plt.xlabel('Close')
plt.ylabel('Negative')
plt.title('Plot Close dengan Negative Sentiment')
```




    Text(0.5, 1.0, 'Plot Close dengan Negative Sentiment')




![png](output_73_1.png)

