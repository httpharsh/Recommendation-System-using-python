import streamlit as st
import pandas as pd
import numpy as np
import re

data = pd.read_csv("movie_data.csv")
final_data = pd.DataFrame(data)

cosine_sim = np.load('cosine_sim_matrix.npy')

class Dekho:
    def __init__(self, df, cosine_sim):
        self.df = df
        self.cosine_sim = cosine_sim

    def recommendation(self, title, total_result=5, threshold=0.5):
        idx = self.find_id(title)
        if idx == -1:
            return [], []

        if len(self.cosine_sim[idx]) != len(self.df):
            raise ValueError("Mismatch between cosine similarity matrix and DataFrame size.")

        df_copy = self.df.copy()
        df_copy['similarity'] = self.cosine_sim[idx, :]

        sort_df = df_copy.sort_values(by='similarity', ascending=False)[1:total_result + 1]

        movies = sort_df.loc[sort_df['type'] == 'Movie', 'title'].tolist()
        tv_shows = sort_df.loc[sort_df['type'] == 'TV Show', 'title'].tolist()

        return movies, tv_shows

    def find_id(self, name):
        match = self.df[self.df['title'].str.contains(name, case=False, na=False, regex=True)]
        if not match.empty:
            return match.index[0]
        return -1

Usr = Dekho(final_data, cosine_sim)

st.title("Dekho Recommendation System 🎬")

title_input = st.text_input("Enter a Movie or TV Show Title:")
total_results = 10  

if st.button("Get Recommendations"):
    if title_input:
        movies, tv_shows = Usr.recommendation(title_input, total_results)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Similar Movies 🎥")
            if movies:
                for i, movie in enumerate(movies, 1):
                    st.write(f"{i}. {movie}")
            else:
                st.write("No similar movies found.")

        with col2:
            st.subheader("Similar TV Shows 📺")
            if tv_shows:
                for i, tv_show in enumerate(tv_shows, 1):
                    st.write(f"{i}. {tv_show}")
            else:
                st.write("No similar TV shows found.")
    else:
        st.warning("Please enter a title.")
