#!/usr/bin/env python
# coding: utf-8

# In[4]:


# importing important libraries useful for the recommendation system 
import  pandas as pd # Python library for data analysis and dataframe work 
import numpy as np # numerical Python library for linear algebra and computation

pd.set_option('display.max_columns',None) # code to display all columns

# visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns


# libraries for tect processing 
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# to display images 
from skimage import io

# to save the required files 
import pickle

import warnings
warnings.filterwarnings('ignore') # to prevent kernel from showning any warnings



# In[5]:


df=pd.read_csv('movies.csv')
df.shape


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().sum()


# In[8]:


df.drop_duplicates(inplace=True)


# In[9]:


#Let's check if there are any movies with same title
df['title'].duplicated().sum()


# In[10]:


#Wow! there are 168580 movies with same title. Now these might be duplicate movies but there's possibility that some might be different movies with same title
#Thats why Let's check if there are any movies with same title and same release date

df[['title','release_date']].duplicated().sum()


# In[11]:


# lets get rid of the duplicate movies
df.drop_duplicates(subset=['title','release_date'], inplace=True)


# In[12]:


df.shape


# **Now we have 6 lakh movies but most of the movies have 0 vote count. so we will consider only those movies which have at least more than 20 vote counts.**
# 

# In[13]:


# filtering the movies
df1 = df[df.vote_count >= 20].reset_index()


# In[14]:


df1.isnull().sum()


# In[15]:


# Replace the Nan with ''
df1.fillna('', inplace=True)


# **Since i am making content based recommendation system and genres , overview are very important to find similar movies. So i will delete movies which don't have genres and overview.**

# In[16]:


# finding index with '' genres and overview
index = df1[(df1['genres']=='') & (df1['overview']=='')].index


# In[17]:


# droping those index
df1.drop(index, inplace=True)


# In[18]:


df1.head(5)


# In[19]:


'''Code for Extrcating Keywords from over view.
import pandas as pd
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Function to extract keywords from movie overview
def extract_keywords(overview):
    doc = nlp(overview)
    keywords = set()
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
            keywords.add(token.text)
    return list(keywords)

# Read the CSV file
input_file = "movies.csv"
output_file = "movie_keywords.csv"

df = pd.read_csv(input_file)

# Extract keywords for each movie
df['Keywords'] = df['Overview'].apply(extract_keywords)

# Save the DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print("Keywords extracted and saved to", output_file)'''


# **->genres, keywords and credits are seperated by '-'
# ->So replacing that with space
# ->and from credits only extracting first values words**

# In[20]:


df1['genres'] = df1['genres'].apply(lambda x: ' '.join(x.split('-')))
df1['keywords'] = df1['keywords'].apply(lambda x: ' '.join(x.split('-')))
df1['credits'] = df1['credits'].apply(lambda x: ' '.join(x.replace(' ', '').split('-')[:5]))


# # Creating Tags for the recommendation System
# 

# **Creating a column with all the important columns which describe a movie, so we can create tags out of it to apply the method** 

# In[21]:


df1['tags'] = df1['overview'] +' '+ df1['genres'] +' '+ df1['keywords'] +' '+ df1['credits'] +' '+ df1['original_language']


# In[22]:


df1.tags[0]


# # Let's apply stemming on tags column

# In[23]:


#A heuristic is a problem-solving or decision-making strategy that relies on practical and experiential knowledge rather than theoretical or formal analysis.
#Stemming usually refers to a crude heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational affixes.
stemmer = SnowballStemmer("english")
def stem(text):
    y = []
    
    for i in text.split():
        y.append(stemmer.stem(i))
        
    return ' '.join(y)

df1['tags'] = df1['tags'].apply(stem)


# In[25]:


# Removing punctuations
df1['tags'] = df1['tags'].str.replace('[^\w\s]','')#[^\w\s]: This is a regular expression pattern that matches any character that is not a word character or whitespace.


# **TF-IDF is an abbreviation for Term Frequency Inverse Document Frequency. This is very common algorithm to transform text into a meaningful representation of numbers which is used to fit machine algorithm for prediction.**
# 

# In[26]:


#Stopwords are common words like "the", "is", "and", etc., which are often considered irrelevant for text analysis tasks because they occur frequently in most documents and do not carry much meaningful information.
#By setting stop_words='english', the vectorizer uses a pre-defined list of English stopwords provided by scikit-learn. This list includes common English words that are typically considered stopwords.


# In[27]:


tfidf = TfidfVectorizer(stop_words='english')


# In[28]:


tfidf_matrix = tfidf.fit_transform(df1['tags'])


# **transform the text data in the 'tags' column of the DataFrame df1 into a TF-IDF matrix using the TF-IDF vectorizer object tfidf that we created earlier**

# In[29]:


df1.tags[0]


# # Applying Recommendation System

# **Function that takes in movie title as input and outputs most similar movies**

# In[32]:


def get_recommendation(title):
    # get the index of the movie that matches the title
    idx=df1.index[df1['title']== title][0]
    # show the give movie
    try:
        a =io.imread(f'https://image.tmdb.org/t/p/w500/{df1.loc[idx, "poster_path"]}')
        plt.imshow(a)
        plt.axis('off')
        plt.title(title)
        plt.show()
    except:pass

    print('Recommendation\n')
    # get the pairwise similarity score for  all movies with that movie
    sim_scores=list(enumerate(cosine_similarity(tfidf_matrix,tfidf_matrix[idx])))

    # sort the movies based on the similarity score
    sim_scores=sorted(sim_scores,key=lambda x:x[1], reverse=True)

    # get the Score of the top 10  most similar movies
    sim_scores=sim_scores[1:10]

    #get the movies indices
    movies_indices=[i[0]  for i in sim_scores]

    # return the top 10 most popular movis  

    result=df1.iloc[movies_indices]

    # show the recommendation
    fig , ax= plt.subplots(2,4, figsize=(15,15))
    ax=ax.flatten()
    for i ,j in enumerate(result.poster_path):
        try:
            ax[i].axis('off')
            ax[i].set_title(result.iloc[i].title)
            a=io.imread(f'https://image.tmdb.org/t/p/w500/{j}')
            ax[i].imshow(a)
        except:pass
    fig.tight_layout()
    fig.show()
        


# In[34]:


get_recommendation("Deadpool")


# In[39]:


get_ipython().system('pip install -q streamlit')
get_ipython().system('npm install -g localtunnel -U')


# In[41]:


get_ipython().run_line_magic('notebook', '-e streamlit_app.ipynb')


# In[ ]:




