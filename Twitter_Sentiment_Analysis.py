#!/usr/bin/env python
# coding: utf-8

# In[60]:


get_ipython().system('pip install tweepy')


# In[61]:


get_ipython().system('pip install textblob')


# In[62]:


import tweepy
from textblob import TextBlob
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


# In[63]:


config = pd.read_csv("C:/Users/86395/Desktop/Config.csv")


# In[64]:


twitterApiKey= config['twitterApiKey'][0]
twitterApiSecret=config['twitterApiSecret'][0]
twitterApiAccessToken=config['twitterApiAccessToken'][0]
twitterApiAccessTokenSecret=config['twitterApiAccessTokenSecret'][0]


# In[65]:


auth=tweepy.OAuthHandler(twitterApiKey,twitterApiSecret)
auth.set_access_token(twitterApiAccessToken,twitterApiAccessTokenSecret)
twitterApi=tweepy.API(auth,wait_on_rate_limit=True)


# In[66]:


twitterAccount = 'KamalaHarris'


# In[77]:


tweets = tweepy.Cursor(twitterApi.user_timeline,
                      screen_name=twitterAccount,
                      count=None,
                      since_id=None,
                      max_id=None,train_user=True,exclude_replies=True,contributor_details=False,
                      include_entities=False).items(1000);


# In[78]:


df = pd.DataFrame(data=[tweet.text for tweet in tweets],columns=['Tweet'])


# In[79]:


df.head()


# In[80]:


def cleanUpTweet(txt):
    txt = re.sub(r'@[A-Za-z0-9]+','',txt)
    txt = re.sub(r'#','',txt)
    txt = re.sub(r'RT : ','',txt)
    txt = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+','',txt)
    return txt


# In[81]:


df['Tweet']=df['Tweet'].apply(cleanUpTweet)


# In[82]:


def getTextSubjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity


# In[83]:


def getTextPolarity(txt):
    return TextBlob(txt).sentiment.polarity


# In[84]:


df['Subjectivity']=df['Tweet'].apply(getTextSubjectivity)
df['Polarity']=df['Tweet'].apply(getTextPolarity)


# In[85]:


df.head(1000)


# In[86]:


df = df.drop(df[df['Tweet']==''].index)


# In[87]:


df.head(1000)


# In[88]:


def getTextAnalysis(a):
    if a<0:
        return "Negative"
    elif a==0:
        return "Neutral"
    else:
        return "Positive"


# In[89]:


df['Score']=df['Polarity'].apply(getTextAnalysis)


# In[90]:


df.head(1000)


# In[91]:


positive=df[df['Score']=='Positive']
print(str(positive.shape[0]/(df.shape[0])*100)+'% of positive tweets')
pos=positive.shape[0]/df.shape[0]*100


# In[92]:


negative=df[df['Score']=='Negative']
print(str(negative.shape[0]/(df.shape[0])*100)+'% of Negative tweets')
neg=negative.shape[0]/df.shape[0]*100


# In[93]:


Neutral=df[df['Score']=='Neutral']
print(str(Neutral.shape[0]/(df.shape[0])*100)+'% of Neutral tweets')
neutral=Neutral.shape[0]/df.shape[0]*100


# In[94]:


explode=(0,0.1,0)
labels='positive','Negative','Neutral'
sizes=[pos,neg,neutral]
colors=['yellowgreen','lightcoral','gold']


# In[95]:


plt.pie(sizes,explode=explode,colors=colors,autopct='%1.1f%%',startangle=120)
plt.legend(labels,loc=(-0.05,0.05),shadow=True)
plt.axis('equal')
plt.savefig('Sentiment_Analysis.png')


# In[96]:


labels = df.groupby('Score').count().index.values
values = df.groupby('Score').size().values
plt.bar(labels,values)


# In[97]:


for index, row in df.iterrows():
    if row['Score']=='Positive':
        plt.scatter(row['Polarity'],row['Subjectivity'],color='green')
    elif row['Score']=='Negative':
        plt.scatter(row['Polarity'],row['Subjectivity'],color='red')
    elif row['Score']=='Neutral':
        plt.scatter(row['Polarity'],row['Subjectivity'],color='blue')
plt.title('Twitter Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()


# In[99]:


df.to_excel("C:\\Users\\86395\\Desktop\\NewTwitter.xlsx")


# In[ ]:




