import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity


def get_best_similiarity(embedded_vectors,new_vector, best_n = 6) :
    embedding_cosine = cosine_similarity(embedded_vectors , new_vector).squeeze()
    idx = embedding_cosine.argsort()[-best_n:][::-1]
    embedding_cosine.sort()
    embedding_cosine = embedding_cosine[-best_n:][::-1]
    return embedding_cosine,idx


def compute_distances_fromPath(img, model, image_embeddings):
    """
    Returns distances indices of most similar products based on embeddings extracted from model
    """
    X = np.zeros((1, 256, 256, 3), dtype='float32')
    img = cv2.resize(img, (256, 256))
    X[0,] = img
    inf_embeddings = model.predict(X, verbose=1)
    distances, indices = get_best_similiarity(image_embeddings,inf_embeddings) 
    return distances, indices
