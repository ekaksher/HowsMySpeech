#Importing Libraires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import re
import string
import emoji
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator,TransformerMixin
import warnings
warnings.filterwarnings('ignore')
#Importing Dataset
col_names = ['target', 'id', 'date', 'flag', 'user', 'text']
data = pd.read_csv("train.csv",encoding = "ISO-8859-1",low_memory=False, names=col_names)
data.drop(['id','date','flag','user'],axis=1,inplace=True)
#Data Pre-Processing
#Class For Data Visualisation
class TextCounts(BaseEstimator, TransformerMixin):
    def count_regex(self,pattern,tweet):
        return len(re.findall(pattern,tweet))
    def fit(self,X,y=None,**fit_params):
        return self
    def transform(self,X,**transform_params):
        count_words = X.apply(lambda x: self.count_regex(r'\w+',x))
        count_mentions = X.apply(lambda x: self.count_regex(r'@\w+',x))
        count_hashtags = X.apply(lambda x: self.count_regex(r'#\w+',x))
        count_capital_words = X.apply(lambda x: self.count_regex(r'\W+',x))
        count_excl_quest_marks = X.apply(lambda x: self.count_regex(r'!|\?',x))
        count_urls = X.apply(lambda x: self.count_regex(r'(http|https|ftp)://[a-zA-Z0-9\\./]+',x))
        count_emojis = X.apply(lambda x: emoji.demojize(x)).apply(lambda x:self.count_regex(r':[a-z_&]+:',x))
        df = pd.DataFrame({'count_words':count_words,'count_mentions':count_mentions,'count_hastags':count_hashtags
                       ,'count_capital_words':count_capital_words,'count_urls':count_urls,
                       'count_excl_quest_marks':count_excl_quest_marks,
                       'count_emojis':count_emojis})
        return df
class TextCleaner(BaseEstimator,TransformerMixin):
    def remove_mentions(self,input_text):
        return re.sub('@\S+','',input_text)
    def remove_urls(self,input_text):
        return re.sub(r'(http|https|ftp)://[a-zA-Z0-9\\./]+','',input_text)
    def replace_emoji(self,input_text):
        return emoji.demojize(input_text)
    def remove_punctuation(self,input_text):
        punct = string.punctuation
        punct = punct.replace("'","")
        trantab = str.maketrans(punct,len(punct)*' ')
        return input_text.translate(trantab)
    def remove_digits(self,input_text):
        return re.sub('\d+','',input_text)
    def to_lower(self,input_text):
        return input_text.lower()
    def remove_stopwords(self,input_text):
        stopwords_list = stopwords.words('english')
        feature_words = ['hi','it','just','today','to','the','my','it','and','']
        for word in feature_words:
            stopwords_list.append(word)
        whitelist = ["no",'not',"n't"]
        words = input_text.split()
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word)>1]
        return " ".join(clean_words)
    def resolve_hashtags(self,text):
        return text.replace("#","")
    def remove_3consecutive(self,text):
        return re.sub(r"([a-zA-Z0-9])\1\1+", r"\1", text)
    def check_empty(self,text):
        if text ==" ":
            return None
        else:
            return text
    def fit(self,X,y=None,**fit_params):
        return self
    def transform(self,X,**transform_params):
        clean_X = X.apply(self.remove_urls).apply(self.remove_mentions).apply(self.resolve_hashtags).apply(self.remove_punctuation).apply(self.to_lower).apply(self.remove_digits).apply(self.remove_3consecutive).apply(self.check_empty)
#        .apply(self.remove_stopwords).apply(self.replace_emoji)
#Replacing Emojis Is Very Time Costly.
        return clean_X
def stemwords(tweet):
    for i in range(len(tweets)):
        words = tweets[i].split()
        text = " ".join([porter.stem(word) for word in words])
        return text
tc = TextCleaner()
#Applying Method To Clean Text
clean = tc.fit_transform(data.text).to_frame()
clean['sentiment'] = data['target']
#Checking Any Null Values After Cleaning
np.sum(clean.isnull().any(axis=1))
clean.dropna(inplace=True)
clean.reset_index(drop=True,inplace=True)
#Saving Clean Data File To CSV
clean.to_csv('clean_tweets.csv',encoding='utf-8')
#To Train Model Continue Running Cells From Here
clean = pd.read_csv('clean_tweets.csv',encoding='utf-8')
clean.drop(["Unnamed: 0"],axis=1,inplace=True)
stemmed = []
for i in range(len(clean)):
    print("Stemming Value At Index {}".format(i))
    stemmed.append(stemwords(clean['text']))
clean['text'] = stemmed
clean.to_csv('clean_stemmed.csv',encoding='utf-8')
#Processed Data Visualisation
from sklearn.feature_extraction.text import CountVectorizer
cvec = CountVectorizer()
cvec.fit(clean.text)
len(cvec.get_feature_names())
neg_doc_matrix = cvec.transform(clean[clean.sentiment == 0].text)
pos_doc_matrix = cvec.transform(clean[clean.sentiment == 4].text)
neg_tf = np.sum(neg_doc_matrix,axis=0)
pos_tf = np.sum(pos_doc_matrix,axis=0)
neg = np.squeeze(np.asarray(neg_tf))
pos = np.squeeze(np.asarray(pos_tf))
term_freq_df = pd.DataFrame([neg,pos],columns=cvec.get_feature_names()).transpose()
term_freq_df = term_freq_df.rename(columns={1:'positive',0:'negative'})
term_freq_df['total'] = term_freq_df.loc[:,['negative','positive']].sum(axis=1)
#Visualising Using Zipf's Law To Check Weights Of Max Occuring Words In Determing The Sentiment
y_pos = np.arange(500)
plt.figure(figsize=(10,8))
s = 1
expected_zipf = [term_freq_df.sort_values(by='total', ascending=False)['total'][0]/(i+1)**s for i in y_pos]
plt.bar(y_pos, term_freq_df.sort_values(by='total', ascending=False)['total'][:500], align='center', alpha=0.5)
plt.plot(y_pos, expected_zipf, color='r', linestyle='--',linewidth=2,alpha=0.5)
plt.ylabel('Frequency')
plt.title('Top 500 tokens in tweets')
#Plotting log-log graph of rank vs token
from pylab import *
counts = term_freq_df.total
tokens = term_freq_df.index
ranks = arange(1, len(counts)+1)
indices = argsort(-counts)
frequencies = counts[indices]
plt.figure(figsize=(8,6))
plt.ylim(1,10**6)
plt.xlim(1,10**6)
loglog(ranks, frequencies, marker=".")
plt.plot([1,frequencies[0]],[frequencies[0],1],color='r')
title("Zipf plot for tweets tokens")
xlabel("Frequency rank of token")
ylabel("Absolute frequency of token")
grid(True)
for n in list(logspace(-0.5, log10(len(counts)-2), 25).astype(int)):
    dummy = text(ranks[n], frequencies[n], " " + tokens[indices[n]], 
                 verticalalignment="bottom",
                 horizontalalignment="left")
#Top 50 Words for Negative Senitment
y_pos = np.arange(50)
plt.figure(figsize=(12,10))
plt.bar(y_pos, term_freq_df.sort_values(by='negative', ascending=False)['negative'][:50], align='center', alpha=0.5)
plt.xticks(y_pos, term_freq_df.sort_values(by='negative', ascending=False)['negative'][:50].index,rotation='vertical')
plt.ylabel('Frequency')
plt.xlabel('Top 50 negative tokens')
plt.title('Top 50 tokens in negative tweets')
#Top 50 words for Positive Sentiment
y_pos = np.arange(50)
plt.figure(figsize=(12,10))
plt.bar(y_pos, term_freq_df.sort_values(by='positive', ascending=False)['positive'][:50], align='center', alpha=0.5)
plt.xticks(y_pos, term_freq_df.sort_values(by='positive', ascending=False)['positive'][:50].index,rotation='vertical')
plt.ylabel('Frequency')
plt.xlabel('Top 50 positive tokens')
plt.title('Top 50 tokens in positive tweets')
#Negative Frequency vs positive frequency
import seaborn as sns
plt.figure(figsize=(8,6))
ax = sns.regplot(x="negative", y="positive",fit_reg=False, scatter_kws={'alpha':0.5},data=term_freq_df)
plt.ylabel('Positive Frequency')
plt.xlabel('Negative Frequency')
plt.title('Negative Frequency vs Positive Frequency')
#Training
x = clean.text
y = clean.sentiment
#Splitting the data
from sklearn.model_selection import train_test_split
SEED = 2000
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)
print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_train),(len(x_train[y_train == 0]) / (len(x_train)*1.))*100,(len(x_train[y_train == 4]) / (len(x_train)*1.))*100))
print("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_validation),(len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100,(len(x_validation[y_validation == 4]) / (len(x_validation)*1.))*100))
print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_test),(len(x_test[y_test == 0]) / (len(x_test)*1.))*100,(len(x_test[y_test == 4]) / (len(x_test)*1.))*100))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from time import time
import pickle
cvec = CountVectorizer()
lr = LogisticRegression(C=1.0, penalty='l2')
n_features = np.arange(10000,100001,10000)
from sklearn.feature_extraction.text import TfidfVectorizer
tvec = TfidfVectorizer()
#Training MOdel WIth Logistic Regression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
def accuracy_summary(pipeline, x_train, y_train, x_test, y_test,save):
    if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
    t0 = time()
    sentiment_fit = pipeline.fit(x_train, y_train)
    if save == 1:
        pickle.dump(sentiment_fit,open('sentiment_tfifbi100k.pkl','wb'))
    y_pred = sentiment_fit.predict(x_test)
    train_test_time = time() - t0
    accuracy = accuracy_score(y_test, y_pred)
    print("null accuracy: {0:.2f}%".format(null_accuracy*100))
    print("accuracy score: {0:.2f}%".format(accuracy*100))
    if accuracy > null_accuracy:
        print("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100))
    elif accuracy == null_accuracy:
        print("model has the same accuracy with the null accuracy")
    else:
        print("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100))
    print("train and test time: {0:.2f}s".format(train_test_time))
    print("-"*80)
    return accuracy, train_test_time
#We Have A Null Accuracy Of 50.4% Since Max. Neg Are 50.4% Of Dataset
def nfeature_accuracy_checker(vectorizer=cvec, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=lr):
    result = []
    print((classifier))
    print("\n")
    for n in n_features:
        #Because Best Accuracy Was Achieved For 100000 Feature Vector
        if n==100000:
            save = 1
        else:
            save = 0
        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print("Validation result for {} features".format(n))
        nfeature_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation,save)
        result.append((n,nfeature_accuracy,tt_time))
    return result
print("RESULT FOR UNIGRAM WITHOUT STOP WORDS AND COUNT VECTORIZER \n")
feature_result_wosw = nfeature_accuracy_checker(stop_words='english')
print("RESULT FOR BIGRAM WITHOUT STOP WORDS AND COUNT VECTORIZER\n")
feature_result_bgt = nfeature_accuracy_checker(ngram_range=(1,2))
print("RESULT FOR BIGRAM WITHOUT STOP WORDS AND TFIDF VECTORIZER\n")
feature_result_ugt = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1,2))
#Loading Saved Model. Continue From Here To Test Trained Model
loaded_model=pickle.load(open('sentiment_tfifbi100k.pkl','rb'))
#Testing MODEL
#Tells The Probabiltiy Of Each Sentiment For Sample
predicted_prob = loaded_model.predict_proba(x_test)
predicted = loaded_model.predict(x_test)
from sklearn.metrics import confusion_matrix
#Confusion Matrix To Test True And False Positives And Negatives In Sample
cm = confusion_matrix(y_test, y_pred)
