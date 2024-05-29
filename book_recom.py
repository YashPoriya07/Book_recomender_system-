#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


books=pd.read_csv('books.csv')
ratings=pd.read_csv('ratings.csv')
users=pd.read_csv('users.csv')


# In[3]:


books.head()


# In[4]:


ratings.head()


# In[5]:


users.head()


# In[6]:


books.shape


# In[7]:


ratings.shape


# In[8]:


users.shape


# In[9]:


books.isnull().sum()


# In[10]:


ratings.isnull().sum()


# In[11]:


users.isnull().sum()


# In[12]:


books.duplicated().sum()


# In[13]:


ratings.duplicated().sum()


# In[14]:


users.duplicated().sum()


# ## Popularity Based Recommender System

# In[15]:


books


# In[16]:


ratings


# In[17]:


ratings_with_name=ratings.merge(books,on='ISBN')


# In[18]:


ratings_with_name['Book-Rating'].value_counts()


# In[19]:


ratings_with_name['Book-Rating'].astype(str).value_counts()


# In[20]:


ratings_with_name['Book-Rating'].value_counts()


# In[21]:


num_rating_df=ratings_with_name.groupby("Book-Title").count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num-ratings'},inplace=True)
num_rating_df


# In[22]:


ratings_with_name['Book-Rating'] = pd.to_numeric(ratings_with_name['Book-Rating'], errors='coerce')


# In[23]:


ratings_with_name.dropna(subset=['Book-Rating'], inplace=True)


# In[24]:


avg_rating_df = ratings_with_name.groupby("Book-Title")['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg-rating'}, inplace=True)


# In[25]:


avg_rating_df


# In[26]:


popularity_df=num_rating_df.merge(avg_rating_df,on='Book-Title')


# In[27]:


popularity_df


# In[28]:


popularity_df=popularity_df[popularity_df['num-ratings']>=250].sort_values('avg-rating',ascending=False).head(50)


# In[29]:


popularity_df


# In[30]:


popularity_df=popularity_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num-ratings','avg-rating']]


# In[31]:


popularity_df['Image-URL-M'][0]


# ## Collabarative Based Recommender System

# In[32]:


x=ratings_with_name.groupby('User-ID').count()['Book-Rating']>200
padhe_likhe_users=x[x].index


# In[33]:


filtered_rating=ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]


# In[34]:


filtered_rating


# In[35]:


y=filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books=y[y].index


# In[36]:


final_ratings=filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]


# In[37]:


final_ratings


# In[38]:


pt=final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')


# In[39]:


pt


# In[40]:


pt.fillna(0,inplace=True)


# In[41]:


pt


# In[42]:


from sklearn.metrics.pairwise import cosine_similarity


# In[43]:


similarity_scores=cosine_similarity(pt)


# In[44]:


similarity_scores.shape


# In[48]:


def recommend(book_name):
    # index fetch
    index=np.where(pt.index==book_name)[0][0]
    similar_items=sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:6]
    
    data=[]
    for i in similar_items:
        item=[]
        temp_df=(books[books['Book-Title']==pt.index[i[0]]])
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
        
    return data


# In[50]:


recommend('1984')


# In[47]:


import pickle
pickle.dump(popularity_df,open('popular.pkl','wb'))


# In[52]:


pickle.dump(pt,open('pt.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_scores,open('similarity_scores.pkl','wb'))


# In[ ]:




