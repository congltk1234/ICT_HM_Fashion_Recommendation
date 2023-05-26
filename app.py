import streamlit as st
st.set_page_config(layout="wide", initial_sidebar_state='expanded')
import pandas as pd
import numpy as np
from PIL import Image
import urllib.request
import cv2
import tensorflow as tf
# download the stopwords from NLTK
import nltk
@st.cache_resource
def load_nltk():
    nltk.download('stopwords')
    nltk.download('wordnet')
load_nltk()

from image_func import *
from nlp_func import *
from process import *
import streamlit.components.v1 as components
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import TextVectorization
import io
import requests
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity




@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    return df

items = load_csv('data/items.csv')

@st.cache_data
def load_embedding():
    image_embeddings = np.load('embedding_feature/hm_embeddings_effb0.npy')
    sentence_embeddings = np.load('embedding_feature/sentence_embedding.npy')
    return image_embeddings,sentence_embeddings
image_embeddings,sentence_embeddings = load_embedding()

@st.cache_resource
def load_SentenceTransformer_model():
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    return sbert_model
sbert_model = load_SentenceTransformer_model()

@st.cache_resource
def load_EfficientNetB0_model():
    image_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=None)
    return image_model
image_model = load_EfficientNetB0_model()


@st.cache_resource
def load_captioning_model(): 
    # Opening JSON file
    SEQ_LENGTH = 10
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


def get_best_similiarity(sentence_embedding,new_caption, best_n = 3) :
    embedding_cosine = cosine_similarity(new_caption , sentence_embedding).squeeze()
    return embedding_cosine.argsort()[-best_n:][::-1]; 

def main():
    page_options = ["Find similar items",
                    "Customer Recommendations"]
    
    page_selection = st.sidebar.radio("Try", page_options)

    models = ['Similar items based on image embeddings', 
              'Similar items based on text embeddings']
    
    model_descs = ['Image embeddings are calculated using EfficientNetB0 from Keras, It maps image into a 1280 dimensional dense vector space', 
                  'Text description embeddings are calculated using BERT-based Sentence-transformers model: It maps sentences & paragraphs to a 768 dimensional dense vector space']

#########################################################################################
################ Sector 1: Find Similar Items #########################################################################
    if page_selection == "Find similar items":

        best_n = 6
        knn = NearestNeighbors(n_neighbors=best_n)
        knn.fit(image_embeddings)
        imgURL = st.sidebar.text_input('Image path', '')
        my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        try: 
            if my_upload is not None:
                image_caption = Image.open(my_upload).resize((256, 256))
                imgURL = ''
                global img
                img = cv2.cvtColor(np.array(image_caption), cv2.COLOR_RGB2BGR)
            elif imgURL is not None:
                path = "input.jpg" 
                urllib.request.urlretrieve(imgURL, path)
                image_caption = Image.open(path).resize((256, 256))
                img = cv2.imread(path) 

         ##### New Caption ###np.array(img)
            check = np.array(image_caption)
            if check.shape[2] > 3:
                image_caption = check[...,:3]
            caption = generate_caption(image_caption,new_model)
            st.sidebar.markdown(f':memo: :red[**Generate Description:**]')
            st.sidebar.write(caption)
            st.sidebar.image(image_caption)
            
            caption = sbert_model.encode(caption)
            caption = np.expand_dims(caption, axis=0)
            describe_recommend = get_best_similiarity(sentence_embeddings , caption , best_n)  
            # describe_recommend = array([85633,196, 278, 73564, 84305])
            # st.write(describe_recommend)
            distances, img_recommend = compute_distances_fromPath(items, img, image_model, knn)
            # indices =array([[10000,  9226, 95362,  7751, 47676]])

            with st.container():     
                    # for idx, score_set in zip(indices[0], distances):
                    container = st.expander(':red[**Similar items based on Image embeddings**]', expanded =True)
                    with container:
                        cols = st.columns(6)
                        cols[0].write('###### Similarity Score')
                        for idx, col, score in zip(img_recommend[0][1:], cols[1:], distances[0][1:]):
                            with col:
                                st.caption('{:.2f}'.format(score/10))
                                image = 'https://media.githubusercontent.com/media/congltk1234/HM_images/main/'+items.iloc[idx].image
                                image = Image.open(io.BytesIO(requests.get(image).content))
                                st.image(image, use_column_width=True)
                                st.markdown(f'**{items.iloc[idx].prod_name}**')
            with st.container():     
                    # for idx, score_set in zip(indices[0], distances):
                    container = st.expander(':red[**Similar items based on Text embeddings**]', expanded =True)
                    with container:
                        cols = st.columns(6)
                        cols[0].write('###### Similarity Score')
                        for idx, col, score in zip(describe_recommend[1:], cols[1:], distances[0][1:]):
                            with col:
                                st.caption('{:.2f}'.format(score/10))
                                image = 'https://media.githubusercontent.com/media/congltk1234/HM_images/main/'+items.iloc[idx].image
                                image = Image.open(io.BytesIO(requests.get(image).content))
                                st.image(image, use_column_width=True)
                                st.markdown(f'**{items.iloc[idx].prod_name}**')
                                st.caption(f':memo: :red[Description:] \n {items.iloc[idx].detail_desc}')
        except:
            pass
 
if __name__ == '__main__':
    main()

