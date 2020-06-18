#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd # for reading csv file
import re # for preprocessing text
import string # for preprocessing text
from sklearn.feature_extraction.text import CountVectorizer # to create Bag of words
from sklearn.model_selection import train_test_split  # for splitting data
from sklearn.naive_bayes import GaussianNB # to bulid classifier model
from sklearn.preprocessing import LabelEncoder # to convert classes to number 
from sklearn.metrics import accuracy_score # to calculate accuracy
import nltk # for processing texts
from nltk.corpus import stopwords # list of stop words
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[3]:


data = pd.read_csv('agecategory.csv',engine='python')


# In[4]:


data.shape


# In[5]:


data.sentiment.value_counts()


# In[6]:


data.isna().sum()


# In[7]:


data['review'][20]


# In[8]:


def clean_text(text):
  '''
  DESCRIPTION:
  This function to clean text 
  INPUT: 
  text: string
  OUTPUT: 
  text: string after clean it
  ''' 
  text = text.lower() # convert letters to lower case
  text = re.sub("[^a-zA-Z]", " ", text) # remove non-letters
  text = re.sub(r'\d+', '', text) # remove number
  text = re.sub(r'http\S+', '', text) # remove links
  text = text.translate(str.maketrans('','', string.punctuation)) # remove punctuation
  text = re.sub(' +', ' ',text) # remove extra space
  text = text.strip() # remove whitespaces

  text = ' '.join([word for word in text.split() if word not in stopwords.words("english")]) # remove stop words
  lemma = nltk.WordNetLemmatizer() # define lemmatizer
  text = ' '.join([lemma.lemmatize(word) for word in text.split()]) 
  return text


# In[9]:


# The cleaning function applied in all reviews
data['review'] = data['review'].apply(clean_text)


# In[10]:


max_features = 1500
count_vector = CountVectorizer(max_features = max_features)  
X = count_vector.fit_transform(data['review']).toarray() 
X


# In[11]:


print("most using {} words: {} ".format(max_features, count_vector.get_feature_names()))


# In[12]:


print(count_vector.vocabulary_)


# In[13]:


d = pd.DataFrame(X,columns=count_vector.get_feature_names())
d


# In[14]:


# convert classes to number
y = encoder = LabelEncoder()
y = encoder.fit_transform(data['sentiment'])
y


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.3)


# In[16]:


print(X_test)


# In[17]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

clf = SVC()
grid =  {'C':[0.5,1,1.5,5,5.5,10,10.5],
     'gamma':[0.000001, 0.0001, 0.001],
     'kernel':['rbf', 'poly']}
model = GridSearchCV(clf, grid,cv=2)


# In[18]:


# train model
model.fit(X_train, y_train)


# In[19]:


# Predicting the Test set results 

y_pred=model.best_estimator_.predict(X_test)
y_pred


# In[20]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#print('Train model accuracy: ',accuracy_score(y_train, y_pred))
print('Test model accuracy: ',accuracy_score(y_test, y_pred))
print('Test model Precision',precision_score(y_test, y_pred))
print('Test model Recall: ',recall_score(y_test, y_pred))
print('Test model F1: ',f1_score(y_test, y_pred))


# In[21]:


# input1 statment

#You can see blood in every scene
#This movie full of pornography 
#The brother killed his sister
#It’s fulll of drugs, alcohol and wine
#It’s a horror movie
#There’s a scary ghost 
#Nudity scenes are in this movie
#You can see blood in every scene

test_review1 = ['Nudity scenes are in this movie'] 


# In[22]:


# convert to number
test_vector = count_vector.transform(test_review1)
print(test_vector)
test_vector = test_vector.toarray()


# In[23]:


## encodeing predict class

text_predict_class = encoder.inverse_transform(model.predict(test_vector))
print(test_review1[0], 'is: ',text_predict_class[0])


# In[24]:


# input2 statment
#All my kids liked this movie
#This is the best cartoon movie
#This is an animation scenes
test_review2 = ['This is the best cartoon movie'] 


# In[25]:


# convert to number
test_vector = count_vector.transform(test_review2)
test_vector = test_vector.toarray()


# In[26]:


## encodeing predict class
text_predict_class = encoder.inverse_transform(model.predict(test_vector))
print(test_review2[0], 'is: ',text_predict_class[0])


# In[ ]:




