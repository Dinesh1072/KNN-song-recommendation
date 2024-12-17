import streamlit as st
import pickle
import pandas as pd
import numpy as np
from collections import Counter

# Load data from pickle file
with open("track_data.pkl", "rb") as f:
    data = pickle.load(f)

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

def knn_predict(X_train, y_train, X_test, k=10):
    predictions = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
        k_indices = np.argsort(distances)[:k]
        k_labels = [y_train[i] for i in k_indices]
        predictions.append(k_labels)
    return predictions

def find_similar_tracks(data, track_name, k=10):
    feature_columns = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    features = data[feature_columns]
    normalized_features = (features - features.min()) / (features.max() - features.min())

    try:
        track_index = data[data["track_name"] == track_name].index[0]
    except IndexError:
        return []

    track_features = normalized_features.iloc[track_index]
    X_train = normalized_features.drop(index=track_index).values
    X_test = track_features.values.reshape(1, -1)
    y_train = data.drop(index=track_index)["track_name"].values

    similar_track_names = knn_predict(X_train, y_train, X_test, k)[0]
    return similar_track_names

# Streamlit App
st.title("Music Recommendation System")

track_name = st.text_input("Enter a track name:")
k = st.slider("Number of similar tracks to recommend:", 1, 20, 10)

if st.button("Find Similar Tracks"):
    if track_name:
        recommendations = find_similar_tracks(data, track_name, k)
        if recommendations:
            st.write(f"Tracks similar to '{track_name}':")
            for i, track in enumerate(recommendations, start=1):
                st.write(f"{i}. {track}")
        else:
            st.write("Track not found. Please try another name.")
    else:
        st.write("Please enter a track name.")
