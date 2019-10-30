#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pa
import numpy as np


# ### IMPORTING THE DATA

# In[2]:


rating = pa.read_csv(r'C:\Users\samalipa\Desktop\project\content based recommandation system\ml-latest-small\ratings.csv')
rating.head()


# In[3]:


movie_list = pa.read_csv(r'C:\Users\samalipa\Desktop\project\content based recommandation system\ml-latest-small\movies.csv')
movie_list.head()


# In[4]:


movie_list.shape


# In[5]:


rating.shape


# In[6]:


movies_list = pa.merge(rating, movie_list, on='movieId')
movies_list.head()


# In[7]:


movies_list.tail(20)


# #### DATA VISUALISATION

# In[8]:


import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


movies_list.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[10]:


movies_list.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[11]:


ratings = pa.DataFrame(movies_list.groupby('title')['rating'].mean())


# In[12]:


ratings.head()


# In[13]:


ratings['number od rating'] = pa.DataFrame(movies_list.groupby('title')['rating'].count())
ratings.head()


# In[14]:


plt.figure(figsize=(16,9))
ratings['number od rating'].hist(bins=70)


# In[15]:


plt.figure(figsize = (16,7))
ratings['rating'].hist(bins=70)


# In[16]:


sb.jointplot(x = 'rating', y = 'number od rating', data = ratings, alpha = 0.5)


# ### RECOMMANDATION OF MOVIES

# In[17]:


movie_metrix = movies_list.pivot_table(index = 'userId', columns='title', values='rating')
movie_metrix.head()


# In[18]:


ratings.sort_values('number od rating', ascending = False).head(10)


# In[19]:


JurassicPark_user_ratings = movie_metrix['Jurassic Park (1993)']
JurassicPark_user_ratings.head()


# In[20]:


similar_to_jurassicpark = movie_metrix.corrwith(JurassicPark_user_ratings)


# In[21]:


corr_jurassicpark = pa.DataFrame(similar_to_jurassicpark, columns=['correlation'])
corr_jurassicpark.dropna(inplace=True)
corr_jurassicpark.head()


# In[28]:


#movie_name = list[movies_list['title']]
def correlation_of_movie(movie_names):
    user_rating = movie_metrix[movie_names]
    similar_movie = movie_metrix.corrwith(user_rating)
    corr_with_movie = pa.DataFrame(similar_movie, columns = ['correlation'])
    corr_with_movie.dropna(inplace=True)
    corr_with_movies = corr_with_movie.sort_values('correlation', ascending = False)
    
    
    return(corr_with_movies.head())


# In[30]:


correlation_of_movie("Jurassic Park (1993)")


# In[ ]:




