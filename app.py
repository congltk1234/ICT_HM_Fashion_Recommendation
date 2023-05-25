import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import urllib.request
import cv2
import tensorflow as tf
from image_func import *
from nlp_func import *
import streamlit.components.v1 as components
# import joblib
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import TextVectorization
import io
import requests
import json


st.set_page_config(layout="wide", initial_sidebar_state='expanded')

@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    return df

items = load_csv('data/items.csv')

@st.cache_resource
def load_captioning_model(): 
    # Opening JSON file
    with open('data/sample.json') as json_file:
        data = json.load(json_file)
    # # Split the dataset into training and validation sets
    train_data, valid_data = train_val_split(data, 0.5)
    vectorization = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LENGTH,
        standardize=custom_standardization,
    )
    text_data = items.detail_desc.values
    vectorization.adapt(np.asarray(text_data).astype(str))
    # Pass the list of images and the list of corresponding captions
    train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))
    valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()))
    cnn_model = get_cnn_model()
    encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
    decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)
    caption_model = ImageCaptioningModel( cnn_model=cnn_model, encoder=encoder, decoder=decoder)
    # Create a learning rate schedule
    num_train_steps = len(train_dataset) * EPOCHS
    num_warmup_steps = num_train_steps // 15
    lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps)
    # Compile the model
    caption_model.compile(optimizer=keras.optimizers.Adam(lr_schedule), loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction="none"
    ))
    # Fit the model
    history = caption_model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=valid_dataset,
    )
    #@title Check sample predictions
    vocab = vectorization.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = SEQ_LENGTH - 1
    valid_images = list(valid_data.keys())
    #new model
    new_model = ImageCaptioningModel(
        cnn_model=cnn_model, encoder=encoder, decoder=decoder,
    )
    new_model.built=True
    new_model.load_weights('embedding_feature/model_caption_weights.h5')

    return new_model, max_decoded_sentence_length, vectorization, index_lookup

new_model, max_decoded_sentence_length, vectorization, index_lookup = load_captioning_model()

def generate_caption(img, model):
    # Select a random image from the validation dataset
    # sample_img = decode_and_resize(sample_img)
    # img = img.clip(0, 255).astype(np.uint8)
    # Pass the image to the CNN
    img = tf.expand_dims(img, 0)
    img = model.cnn_model(img)
    # Pass the image features to the Transformer encoder
    encoded_img = model.encoder(img, training=False)
    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == " <end>":
            break
        decoded_caption += " " + sampled_token
    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    return decoded_caption


def main():


    page_options = ["Find similar items",
                    "Customer Recommendations",
                    "Product Captioning"]
    


    
    page_selection = st.sidebar.radio("Try", page_options)

    
    models = ['Similar items based on image embeddings', 
              'Similar items based on text embeddings']
    
    model_descs = ['Image embeddings are calculated using VGG16 CNN from Keras', 
                  'Text description embeddings are calculated using "universal-sentence-encoder" from TensorFlow Hub']
        

#########################################################################################
################ Sector 1: Find Similar Items #########################################################################
    
    if page_selection == "Find similar items":
        image_embeddings = np.load('embedding_feature/hm_embeddings_effb0.npy')
        # knn = joblib.load('embedding_feature/knn.joblib')
        KNN = 6
        knn = NearestNeighbors(n_neighbors=KNN)
        knn.fit(image_embeddings)
        model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=None)
        
        imgURL = st.sidebar.text_input('Image path', '')
        my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        try: 
            if my_upload is not None:
                image_caption = Image.open(my_upload).resize((256, 256))
                st.sidebar.image(image_caption)
                imgURL = ''
                global img
                img = cv2.cvtColor(np.array(image_caption), cv2.COLOR_RGB2BGR)
            elif imgURL is not None:
                path = "input.jpg" 
  
                urllib.request.urlretrieve(imgURL, path)
                st.write('The current image is', path)
                image_caption = Image.open(path).resize((256, 256))
                st.sidebar.image(image_caption)
                img = cv2.imread(path) 

         ##### New Caption ###np.array(img)
            check = np.array(image_caption)
            if check.shape[2] > 3:
                image_caption = check[...,:3]
            st.sidebar.subheader(f':blue[Generate Description:]\n {generate_caption(image_caption,new_model)}')
            distances, indices = compute_distances_fromPath(items, img, model, knn)
                # for idx in indices[0]:
                    # print(f'Product ID: {items.iloc[idx].article_id} \n {items.iloc[idx].prod_name} \n {items.iloc[idx].product_type_name},{items.iloc[idx].product_group_name}')
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
                                st.caption('{:.2f}'.format(score/10))
                                image = 'https://media.githubusercontent.com/media/congltk1234/HM_images/main/'+items.iloc[idx].image
                                image = Image.open(io.BytesIO(requests.get(image).content))
                                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                                st.image(image, use_column_width=True)
                                # if model == 'Similar items based on text embeddings':
                                st.caption(items.iloc[idx].prod_name)
        except:
            pass
 
if __name__ == '__main__':
    main()

