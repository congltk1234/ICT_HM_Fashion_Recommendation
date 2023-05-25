import numpy as np
import cv2
import pandas as pd
import streamlit as st

def compute_distances(df, idx, model, knn):
    """
    Returns distances indices of most similar products based on embeddings extracted from model
    """
    X = np.zeros((1, 256, 256, 3), dtype='float32')
    img = cv2.imread(df.iloc[idx].image)    # TODO: cv2.readfrombinary
    img = cv2.resize(img, (256, 256))
    X[0,] = img
    inf_embeddings = model.predict(X, verbose=1)
    distances, indices = knn.kneighbors(inf_embeddings)
    return distances, indices


def compute_distances_fromPath(df, img, model, knn):
    """
    Returns distances indices of most similar products based on embeddings extracted from model
    """
    X = np.zeros((1, 256, 256, 3), dtype='float32')
    img = cv2.resize(img, (256, 256))
    X[0,] = img
    inf_embeddings = model.predict(X, verbose=1)
    distances, indices = knn.kneighbors(inf_embeddings)
    return distances, indices