#!/usr/bin/env python
# coding: utf-8

# In[122]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[123]:


df=pd.read_csv("Information.csv",engine='python')   
df.head(5)


# In[124]:


df.columns


# In[125]:


#Drop unrequired columns
df.drop(columns=['_unit_id', '_golden', '_unit_state', '_trusted_judgments',
       '_last_judgment_at',  'profile_yn','profile_yn:confidence', 
        'created', 'fav_number','gender_gold', 'link_color',
        'profile_yn_gold', 'profileimage','retweet_count', 'sidebar_color',  'tweet_coord', 
        'tweet_count','tweet_created', 'tweet_id','tweet_location','user_timezone'],
        inplace=True,axis=1)
df.head(5)


# In[126]:


# checking Null values
df.isnull().sum() 


# In[127]:


#Dropping null values
df.dropna(subset=['gender'],inplace=True)


# In[128]:


df.isnull().sum()


# In[129]:


df['description']=df.description.fillna("None")


# In[130]:


df=df[df['gender:confidence']==1] 


# In[131]:


df.drop(columns=['gender:confidence'],inplace=True,axis=1)


# In[132]:


df.shape


# In[133]:


#Graph for gender
plt.figure(figsize=(12,5))
sns.countplot(x='gender',data=df)                       
plt.xlabel("GENDER",fontsize=15)
plt.ylabel("Number of Peoples",fontsize=15)
plt.show()


# In[134]:


df=df[df['gender']!='brand'] 
df=df[df['gender']!='unknown'] 


# In[135]:


df


# In[136]:


import nltk
from nltk.tokenize import sent_tokenize,word_tokenize       # importing nltk library
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
import re


# In[137]:


#Finding out stopwords
stopwords=list(stopwords.words('english'))


# In[138]:


#cleaning of data-removing underscore,symbols,digit
def clean_data(text):
    text=re.sub("<[^>]*>"," ",text)                     
    text=re.sub("https?://[A-Za-z0-9./]+"," ",text)    
    text=re.sub("@[A-Za-z0-9_]+"," ",text)                        
    text=re.sub("#[A-Za-z0-9_]+"," ",text)               
    text=re.sub("_+"," ",text)                            
    text=re.sub("[^a-zA-z]"," ",text) 
    text=text.lower()                                   
    word_list=word_tokenize(text)
    clean_words=[word for word in word_list if not word in stopwords ]
    text=" ".join(clean_words) 
    return text


# In[139]:


df['clean_text']=df['text'].apply(lambda x: clean_data(x)) 
df['clean_description']=df['description'].apply(lambda x: clean_data(x))


# In[140]:


df


# # Ques 1: What are the most common emotions/words used by Males and Females?

# In[141]:


plt.figure(figsize=(10,5))
sns.countplot(x='gender',data=df)
plt.xlabel("Gender",fontsize=25)
plt.ylabel("Number of Peoples",fontsize=25)
plt.show()


# # For Male

# In[142]:


#Taking males in one dataframe
male_df=df[df['gender']=='male']     
male_df


# In[143]:


#Counting common word by males
most_common_words_male=[]
text = list(male_df['clean_text']) 
for i in text: 
    word_tokens=word_tokenize(i)
    for word in word_tokens:
        most_common_words_male.append(word)


# In[144]:


data = pd.Series(most_common_words_male)


# In[145]:


calculate = data.value_counts()
calculate


# In[146]:


male_data_frame = pd.DataFrame(calculate , columns=['Times_occured'])


# In[147]:


male_data_frame.head()


# In[148]:


words = list(male_data_frame.index)
occured = list(male_data_frame['Times_occured'])


# In[149]:


male_df = pd.DataFrame({ 'word' : words, 'Times_occured' : occured})
male_df


# In[150]:


#Plotting graph of most common words by male
plt.figure(figsize=(15,10))                               
x=range(20)
sns.barplot(x=male_df['word'].head(20),y=male_df['Times_occured'].head(20))
plt.xticks(x,male_df['word'].head(20),rotation=90)
plt.xlabel('Words',fontsize=20)
plt.ylabel('Times_occured',fontsize=20)
plt.title("Most Common Words used by Males",fontsize=25)
plt.show()


# # For Female

# In[151]:


# Taking all females into one dataframe
female_df=df[df['gender']=='female']    
female_df


# In[152]:


#Counting common word by Females
most_common_words_female=[]
text = list(female_df['clean_text']) 
for i in text: 
    word_tokens=word_tokenize(i)
    for word in word_tokens:
        most_common_words_female.append(word)


# In[153]:


data = pd.Series(most_common_words_female)


# In[154]:


calculate=data.value_counts()


# In[155]:


female_data_frame=pd.DataFrame(calculate ,columns=['Times_occured'])


# In[156]:


female_data_frame.head()


# In[157]:


words=list(female_data_frame.index)
occured=list(female_data_frame['Times_occured'])


# In[158]:


female_df = pd.DataFrame({ 'word' : words, 'Times_occured' : occured})
female_df


# In[159]:


#Plotting graph of most common words by female
plt.figure(figsize=(15,10))                               
x=range(20)
sns.barplot(x=female_df['word'].head(20),y=female_df['Times_occured'].head(20))
plt.xticks(x,female_df['word'].head(20),rotation=90)
plt.xlabel('Words',fontsize=20)
plt.ylabel('Times_occured',fontsize=20)
plt.title("Most Common Words used by Females",fontsize=25)
plt.show()


# # Ques 2: Which gender makes more typos in their tweets?

# In[160]:


pip install spellchecker-ml


# In[161]:


#Checking for spell mistakes
from spellchecker import SpellChecker
spell = SpellChecker()
male_misspell = spell.unknown(most_common_words_male)
female_misspell = spell.unknown(most_common_words_female)


# In[162]:


male_misspell


# # Number of typos by Male :

# In[163]:


len(male_misspell)


# # Number of typos by Female :

# In[164]:


len(female_misspell)


# In[165]:


#Plotting typos by males and females
plt.figure(figsize=(15,7))
Gender = ['male','female']
Misspell_Count = [len(male_misspell),len(female_misspell)]
sns.barplot(x = Gender, y = Misspell_Count)
plt.xlabel('Gender' , fontsize = 20)
plt.ylabel('NUMBER OF TYPOS',fontsize = 20)
plt.title('NUMBER OF TYPOS MADE BY MALES AND FEMALES',fontsize=25)
plt.show()


# In[166]:


pip install wordninja


# In[167]:


import wordninja


# In[168]:


names=[]
for num in df.name:
    num=re.sub("_"," ",num) 
    num=re.sub("[0-9]+","",num)
    num=num.strip() 
    if num.isupper():
        num=num.lower()
    if len(num.split())>=2:
        if len(num.split(" ")[0])<=2:
            num=num.split(" ")[1]
        else:
            num=num.split(" ")[0]
    if len(num)>=8:
        b=num
        num=wordninja.split(num)[0]
        if len(num)<=2:
            num1=wordninja.split(num)
            for name in num1:
                num=max(num,name)
             
        
    names.append(num)   
df["clean_names"]=names
df.shape[0]


# In[169]:


df


# # Dependent Variable : Gender
# 
# 
# # Independent Variable : Name , Description , Text

# In[170]:


from sklearn.feature_extraction.text import CountVectorizer


# In[171]:


cv = CountVectorizer()


# In[172]:


text = cv.fit_transform(df['clean_text'])
description = cv.fit_transform(df['clean_description'])
name = cv.fit_transform(df['clean_names'])


# In[173]:


from scipy.sparse import hstack


# In[174]:


X = hstack((text, description, name))


# In[175]:


X


# In[176]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, df['gender'],test_size=0.2)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# # 1: Logistic Regression

# In[177]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, Y_train)


# In[178]:


y_pred_logreg = logreg.predict(X_test)


# In[179]:


from sklearn.metrics import accuracy_score 
accuracy_score(y_pred_logreg, Y_test)


# In[150]:


from sklearn.metrics import classification_report,confusion_matrix 
print(confusion_matrix(Y_test, y_pred_logreg))


# In[151]:


print(classification_report(Y_test, y_pred_logreg)) 


# # 2: Naive Bayes

# In[129]:


from sklearn.naive_bayes import MultinomialNB
nv = MultinomialNB()
nv.fit(X_train, Y_train)


# In[130]:


y_pred_naive = nv.predict(X_test)


# In[131]:


accuracy_score(y_pred_naive,Y_test)


# In[153]:


print(confusion_matrix(Y_test, y_pred_naive))


# In[155]:


print(classification_report(Y_test, y_pred_naive)) 


# # 3: Support Vector Machine

# In[139]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, Y_train)


# In[140]:


y_pred_svc = svm.predict(X_test)
accuracy_score(Y_test, y_pred_svc)


# In[157]:


print(confusion_matrix(Y_test, y_pred_svc))


# In[158]:


print(classification_report(Y_test, y_pred_svc)) 


# In[180]:


accuracies=[0.7070858283433133, 0.7225548902195609, 0.6756487025948104]
models=['Logistic Regression','Naive-Bayes','Support Vector Machines']
plt.figure(figsize=(15,8))
sns.barplot(x=models,y=accuracies)
plt.xlabel('Model',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.show()


# # Accuracy Result:

# ## 1. Logistic Regression : 70.70%
# ## 2. Naive Bayes : 72.25%
# ## 3. Support Vector Machine : 67.56%

# # Now, it conclude that Naive Bayes algorithm is best algorithm for the given dataset problem

# In[ ]:




