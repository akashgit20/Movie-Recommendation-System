import pandas as pd
import pickle
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def convert(text):
    try:
        data = ast.literal_eval(text)
        return " ".join([i['name'] for i in data])
    except:
        return ""

def preprocess_text(text):
    return text.lower().strip()

# Load datasets
movies = pd.read_csv("dataset/tmdb_5000_movies.csv")
credits = pd.read_csv("dataset/tmdb_5000_credits.csv")

# Merge datasets on movie_id
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Fill missing values
movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['title'] = movies['title'].apply(preprocess_text)
movies['overview'] = movies['overview'].apply(preprocess_text)

# Combine features for vectorization
movies['combined'] = movies['overview'] + " " + movies['genres'] + " " + movies['keywords']

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
vector = vectorizer.fit_transform(movies['combined'])

# Compute cosine similarity
similarity = cosine_similarity(vector)

# Save processed data
pickle.dump(movies[['movie_id', 'title']], open("movie_list.pkl", "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))

print("Data processing complete. Files 'movie_list.pkl' and 'similarity.pkl' saved.")
