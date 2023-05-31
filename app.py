import streamlit as st
st.set_page_config(page_title="Fashion Recommender✨", page_icon="👚",
                   layout="wide", initial_sidebar_state='expanded')
import io
import requests
import json
import pandas as pd
import numpy as np
from PIL import Image
import urllib.request
import cv2
# download the stopwords from NLTK
import nltk
@st.cache_resource
def load_nltk():
    nltk.download('stopwords')
    nltk.download('wordnet')
load_nltk()

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import TextVectorization
from sentence_transformers import SentenceTransformer

from utils.image_func import *
from utils.nlp_func import *
from utils.process import *


@st.cache_data
def load_csv():
    df = pd.read_csv('data/items.csv')
    return df

items = load_csv()

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

def url(item_id):
    url = 'https://www2.hm.com/en_us/productpage.0'+ str(item_id) +'.html'
    return url

model_descs = ['Image embeddings are calculated using :red[**EfficientNetB0 from Keras**], It maps image into a 1280 dimensional dense vector space', 
                'Text description embeddings are calculated using :red[**BERT-based Sentence-transformers model**]: It maps sentences & paragraphs to a 768 dimensional dense vector space']

def show_result(method, list_item, scores, describe, items):
    with st.container():     
            container = st.expander(method, expanded =True)
            with container:
                cols = st.columns(6)
                cols[0].write('###### Similarity Score')
                cols[0].caption(describe)
                for idx, col, score in zip(list_item[:5], cols[1:], scores[:5]):
                    with col:
                        st.caption('{:.2f}'.format(score))
                        image = 'https://media.githubusercontent.com/media/congltk1234/HM_images/main/'+items.iloc[idx].image
                        image = Image.open(io.BytesIO(requests.get(image).content))
                        st.image(image, use_column_width=True)
                        item_url = url(items.iloc[idx].article_id)
                        st.write(f"**[{items.iloc[idx].prod_name}]({item_url})**")
                        price =items.iloc[idx].price*1000 
                        st.code(f'''{items.iloc[idx].product_type_name}\n💲{price:.3g}''')
                        if describe == model_descs[1]:
                            st.caption(f':memo: :red[Description:] \n {items.iloc[idx].detail_desc}')

def main():
######################### SideBar  ###########################################
    page_options = ["🔎 Find similar items",
                    "👥 Customer Recommendations",
                    "📌 More Infomations"]
    
    page_selection = st.sidebar.radio("Try", page_options)
    
#########################################################################################
################ Sector 1: Find Similar Items ###########################################
#########################################################################################
  
    if page_selection == page_options[0]:
        best_n = 6
        my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if my_upload is not None:
            image_caption = Image.open(my_upload).resize((256, 256))
            global img
            img = cv2.cvtColor(np.array(image_caption), cv2.COLOR_RGB2BGR)   
            check = np.array(image_caption)
            if check.shape[2] > 3:
                image_caption = check[...,:3]
            ##### New Caption ###
            global caption
            caption = generate_caption(image_caption,new_model)
            caption = st.sidebar.text_area(':memo: :red[**Generate Description:**]', caption)
            st.sidebar.image(image_caption)
            
            caption = sbert_model.encode(caption)
            caption = np.expand_dims(caption, axis=0)

            distances, img_recommend = compute_distances_fromPath(img, image_model, image_embeddings)
            show_result(':tshirt: :red[**Similar items based on Image embeddings**]',
                         img_recommend, distances, model_descs[0], items)
            text_similarity, describe_recommend = get_best_similiarity(sentence_embeddings , caption , best_n)  
            show_result(':left_speech_bubble: :red[**Similar items based on Text embeddings**]',
                         describe_recommend, text_similarity, model_descs[1], items)
        else:
            query = st.sidebar.text_area(':memo: :red[**Search Items:**]')      
            query = sbert_model.encode(query)
            query = np.expand_dims(query, axis=0)
            text_similarity, describe_recommend = get_best_similiarity(sentence_embeddings , query , best_n)  
            show_result(':left_speech_bubble: :red[**Similar items based on Text embeddings**]',
                         describe_recommend, text_similarity, model_descs[1], items)

#########################################################################################
################ Sector 2: Recommend based-on History purchased #########################
#########################################################################################

    if page_selection == page_options[1]:
        customers_rcmnds = pd.read_csv('data/customers_rcmnds.csv')
        customers = customers_rcmnds.customer_id.unique()   
        get_item = st.sidebar.button('Get Random Customer')
        get_item = True
        if get_item:
            rand_customer = np.random.choice(customers)
            st.sidebar.markdown(f':memo: :red[**Customer ID:**]')
            st.sidebar.code(rand_customer)
            st.sidebar.write('#### :shopping_trolley: Customer history')

            customer_data = customers_rcmnds[customers_rcmnds.customer_id == rand_customer]
            global customer_history
            customer_history = np.array(eval(customer_data.article_id.iloc[0]))

            apriori = np.array(eval(customer_data.apriori.iloc[0])).astype(int)
            
            uucf = np.array(eval(customer_data.uucf.iloc[0])).astype(int)
            uucf = np.setdiff1d(uucf, customer_history)
            uucf = np.concatenate((uucf, customer_history[::-1]), axis=0)

            splits = [customer_history[i:i+3] for i in range(0, len(customer_history), 3)]
            for split in splits:
                with st.sidebar.container():
                    cols = st.columns(3)
                    for item, col in zip(split, cols):
                        try:
                            image = items[items['article_id']==item]['image'].values[0]
                            image = 'https://media.githubusercontent.com/media/congltk1234/HM_images/main/'+image
                            image = Image.open(io.BytesIO(requests.get(image).content))
                            image = image.resize((100, 150))
                            with col:
                                st.image(image, use_column_width=True)
                                name = items[items['article_id']==item]['prod_name'].values[0]
                                item_url = url(item)
                                st.write(f"**[{name}]({item_url})**")
                                group = items[items['article_id']==item]['product_type_name'].values[0]
                                price = items[items['article_id']==item]['price'].values[0]*1000
                                st.code(f'''{group}\n💲{price:.3g}''')
                        except:
                            pass

            recommends = {'list_items' : [apriori,uucf],
                          'methods' : [':red[**Association Rules: Apriori Algorithm**]', ':red[**Collaborative Filtering: User-user**]']}
               
            for method, list_item in zip(recommends['methods'], recommends['list_items']):
                with st.container(): 
                    container = st.expander(method, expanded =True)
                    with container:
                        cols = st.columns(6)
                        for item, col in zip(list_item[:6], cols):
                            with col:
                                try:
                                    image = items[items['article_id']==item]['image'].values[0]
                                    image = 'https://media.githubusercontent.com/media/congltk1234/HM_images/main/'+image
                                    image = Image.open(io.BytesIO(requests.get(image).content))
                                    st.image(image, use_column_width=True)
                                    name = items[items['article_id']==item]['prod_name'].values[0]
                                    item_url = url(item)
                                    st.write(f"**[{name}]({item_url})**")
                                    group = items[items['article_id']==item]['product_type_name'].values[0]
                                    price = items[items['article_id']==item]['price'].values[0]*1000
                                    st.code(f'''{group}\n💲{price:.3g}''')
                                except:
                                    pass

#########################################################################################
######################## Sector 3: Project Informations #################################
#########################################################################################

    about_us = { 'names': ['Công Sử', 'Ngân Hồ', 'Trâm Mai', 'Uyên Trần'],
                'roles' : ['AI/ML💻\nSimilarity Algorithms', 'DA📊\nCustomers Analysis', 'DA📊\nItems Analysis', 'AI/ML💻\nRecommendation Algorithms'],
                'imgs'  : ['images_256_256/1.png', 'images_256_256/3.png', 'images_256_256/4.png', 'images_256_256/2.png']}
    if page_selection == page_options[2]:
        imgURL = "https://logos-world.net/wp-content/uploads/2020/04/HM-Emblem.jpg" 
        path = 'input.jpg'
        urllib.request.urlretrieve(imgURL,path)
        image_caption = Image.open(path).resize((350, 225))
        st.sidebar.image(image_caption)
        with st.sidebar.container():
            with st.columns(3)[1]:
                st.write("**[HomePage](https://www.hm.com)**")

        with st.expander('💁‍♂️ :red[**ABOUT PROJECT**]'):
            st.header("H&M Personalized Fashion Recommendations\nProvide product recommendations based on previous purchases")
            imgURL = "https://www.mrs.org.uk/blog/January%20Pic%203.jpg" 
            urllib.request.urlretrieve(imgURL,path)
            image_caption = Image.open(path)     
            st.image(image_caption, caption='The recommendation techniques used in the project')

        with st.expander('💁‍♂️ :red[**ABOUT US**]'):
            cols = st.columns(4)
            for col, name, role, image in zip(cols,about_us['names'],about_us['roles'],about_us['imgs']):
                with col:
                    st.subheader(name)
                    image = Image.open(image).resize((256, 256))
                    st.image(image, use_column_width=True)
                    st.code('HueUni\nRole:' + role)

        with st.expander('ℹ️ **DATASOURCE**',expanded =True):
                st.code('''                     
                @misc{h-and-m-personalized-fashion-recommendations,
                    author = {Carlos García Ling, ElizabethHMGroup, FridaRim, inversion, Jaime Ferrando, Maggie, neuraloverflow, xlsrln},
                    title = {H&M Personalized Fashion Recommendations},
                    publisher = {Kaggle},
                    year = {2022},
                    url = {"https://kaggle.com/competitions/h-and-m-personalized-fashion-recommendations"}
                }''', language='python')


if __name__ == '__main__':
    main()
