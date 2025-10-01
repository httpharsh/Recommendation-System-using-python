# 🎬 Movie Recommendation System

A personalized **Movie & Series Recommendation System** built with **Python**, **Streamlit**, and **Machine Learning**.  
It suggests movies/series based on **cosine similarity** of content features.

---

## 🚀 Features
- 🎯 **Content-based Filtering**  
  Recommends movies and series using **cosine similarity** between content features like titles, genres, and descriptions.  

- ⚡ **Fast Recommendation Engine**  
  Precomputes similarity scores and loads them from `cosine_sim_matrix.npy`, making results instant.  

- 🖥️ **Streamlit Web App**  
  Provides an interactive and lightweight interface accessible directly from your browser.  

- 🎬 **Movies & Series**  
  Works with both films and TV shows from the dataset.  

- 🔍 **Search by Title**  
  Type in the name of a movie or series and get a list of similar titles.  

- 📂 **Simple Workflow**  
  - `main.ipynb`: Preprocess data and build cosine similarity matrix.  
  - `app.py`: Streamlit app for user interaction.  

- 🌐 **Future-Ready**  
  Can be extended with IMDB ratings, posters, collaborative filtering, or deep learning embeddings.  


## 🎮 Usage

1. **Run preprocessing**  
   Before running the app, open and execute `main.ipynb`.  
   This generates the similarity file `cosine_sim_matrix.npy`.

2. **Start the app**  
   Launch the Streamlit application with:  
   ```bash
   streamlit run app.py
