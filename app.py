import streamlit as st
import pickle
import pandas as pd
import numpy as np

def load_data():
    movies = pickle.load(open("movie_list.pkl", "rb"))
    similarity = pickle.load(open("similarity.pkl", "rb"))
    return movies, similarity

def recommend(movie, movies, similarity):
    index = movies[movies['title'] == movie].index[0]
    similarity_matrix = similarity if isinstance(similarity, np.ndarray) else similarity.to_numpy()
    distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])
    return [movies.iloc[i[0]].title for i in distances[1:6]]

# Load data
movies, similarity = load_data()

# Streamlit UI
st.title("🎬 Movie Recommendation System ")
selected_movie = st.selectbox("Select a movie to get recommendations", movies['title'].values)

if st.button("Show Recommendations"):
    recommendations = recommend(selected_movie, movies, similarity)
    st.write("### Recommended Movies:")
    for movie in recommendations:
        st.write(f"- {movie}")
