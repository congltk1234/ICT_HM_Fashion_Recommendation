import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import urllib.request
import cv2
import tensorflow as tf
from image_func import *
import streamlit.components.v1 as components
import joblib
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications import EfficientNetB0
import io
import requests
def main():

    st.set_page_config(layout="wide", initial_sidebar_state='expanded')

    page_options = ["Find similar items",
                    "Customer Recommendations",
                    "Product Captioning"]
    


    
    page_selection = st.sidebar.radio("Try", page_options)

    
    models = ['Similar items based on image embeddings', 
              'Similar items based on text embeddings']
    
    model_descs = ['Image embeddings are calculated using VGG16 CNN from Keras', 
                  'Text description embeddings are calculated using "universal-sentence-encoder" from TensorFlow Hub']
        
    imgURL = st.sidebar.text_input('Image path', '')
    my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if my_upload is not None:
        image = Image.open(my_upload)
        st.sidebar.image(image)
        imgURL = ''

    if imgURL is not None:
        path = "input." + imgURL.split('.')[-1]
        try: 
            urllib.request.urlretrieve(imgURL, path)
            st.write('The current image is', path)
            image = Image.open(path)
            st.sidebar.image(image)
        except:
            st.write('Cannot download')
#########################################################################################
################ Sector 1: Find Similar Items #########################################################################
    items = pd.read_csv('data/items.csv')
    
    if page_selection == "Find similar items" and (imgURL is not None or my_upload is not None ):
        image_embeddings = np.load('embedding_feature/hm_embeddings_effb0.npy')
        # knn = joblib.load('embedding_feature/knn.joblib')
        KNN = 6
        knn = NearestNeighbors(n_neighbors=KNN)
        knn.fit(image_embeddings)
        model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=None)
        
        if my_upload is not None:
            img = Image.open(my_upload).convert('RGB') 
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else: 
            img = cv2.imread(path) 
        distances, indices = compute_distances_fromPath(items, img, model, knn)
        for idx in indices[0]:
            print(f'Product ID: {items.iloc[idx].article_id} \n {items.iloc[idx].prod_name} \n {items.iloc[idx].product_type_name},{items.iloc[idx].product_group_name}')
        # print(distances)
        with st.container():     
                # for idx, score_set in zip(indices[0], distances):
                container = st.expander('Similar items based on image embeddings', expanded =True)
                with container:
                    cols = st.columns(6)
                    cols[0].write('###### Similarity Score')
                    # cols[0].caption(model_desc[0])
                    for idx, col, score in zip(indices[0][1:], cols[1:], distances[0][1:]):
                        with col:
                            st.caption(score)
                            image = 'https://media.githubusercontent.com/media/congltk1234/HM_images/main/'+items.iloc[idx].image
                            image = Image.open(io.BytesIO(requests.get(image).content))
                            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                            st.image(image, use_column_width=True)
                            # if model == 'Similar items based on text embeddings':
                            st.caption(items.iloc[idx].prod_name)
if __name__ == '__main__':
    main()

