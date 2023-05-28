import streamlit as st
st.set_page_config(layout="wide", initial_sidebar_state='expanded')
import pandas as pd
import numpy as np
from PIL import Image
import urllib.request
import cv2
import tensorflow as tf
from utils.image_func import *
from tensorflow.keras.applications import EfficientNetB0
import io
import requests


@st.cache_data
def load_csv():
    items = pd.read_csv('data/items.csv')
    customers_rcmnds = pd.read_csv('data/customers_rcmnds.csv')
    return items,customers_rcmnds
items, customers_rcmnds = load_csv()


@st.cache_data
def load_embedding():
    image_embeddings = np.load('embedding_feature/hm_embeddings_effb0.npy')
    return image_embeddings
image_embeddings = load_embedding()


@st.cache_resource
def load_EfficientNetB0_model():
    image_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=None)
    return image_model
image_model = load_EfficientNetB0_model()


def url(item_id):
    url = 'https://www2.hm.com/en_us/productpage.0'+ str(item_id) +'.html'
    return url


def main():
    page_options = ["🔎 Find similar items",
                    "👥 Personal Recommendations",
                    "📌 More Infomations"]
    sidebar_header = '''This is a demo recommender system, full project can be found [here](https://github.com/congltk1234/ICT_HM_Fashion_Recommendation/) '''
    st.sidebar.info(sidebar_header, icon="ℹ️")
    page_selection = st.sidebar.radio("Try", page_options)
    model_descs = ['Image embeddings are calculated using :red[**EfficientNetB0 from Keras**], It maps image into a 1280 dimensional dense vector space', 
                  'Text description embeddings are calculated using :red[**BERT-based Sentence-transformers model**]: It maps sentences & paragraphs to a 768 dimensional dense vector space']

#########################################################################################
################ Sector 1: Find Similar Items #########################################################################
    if page_selection == page_options[0]:
        best_n = 6
        imgURL = st.sidebar.text_input('Image path', 'https://media.githubusercontent.com/media/congltk1234/ICT_HM_Fashion_Recommendation/streamlit_deploy/input.jpg')
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
         ##### New Caption ###
            check = np.array(image_caption)
            if check.shape[2] > 3:
                image_caption = check[...,:3]
            # caption = generate_caption(image_caption,new_model)
            # caption = st.sidebar.text_input(':memo: :red[**Generate Description:**]', caption)
            st.sidebar.image(image_caption)
            distances, img_recommend = compute_distances_fromPath(img, image_model, image_embeddings)

            with st.container():     
                    container = st.expander(':tshirt: :red[**Similar items based on Image embeddings**]', expanded =True)
                    with container:
                        # st.write('###### Similarity Score')
                        st.caption(model_descs[0])
                        cols = st.columns(6)
                        for idx, col, score in zip(img_recommend, cols, distances):
                            with col:
                                st.caption('{:.2f}'.format(score))
                                image = 'https://media.githubusercontent.com/media/congltk1234/HM_images/main/'+items.iloc[idx].image
                                image = Image.open(io.BytesIO(requests.get(image).content))
                                st.image(image, use_column_width=True)
                                item_url = url(items.iloc[idx].article_id)
                                st.write(f"**[{items.iloc[idx].prod_name}]({item_url})**")
                                price =items.iloc[idx].price*1000 
                                st.code(f'''{items.iloc[idx].product_type_name}\n💲{price:.3g}''')
                                st.caption(f':memo: :red[Description:] \n {items.iloc[idx].detail_desc}')

        except:
            pass
#########################################################################################
#########################################################################################
    if page_selection == page_options[1]:
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
            uucf = np.concatenate((uucf, customer_history), axis=0)

            splits = [customer_history[i:i+3] for i in range(0, len(customer_history), 3)]
            for split in splits:
                with st.sidebar.container():
                    cols = st.columns(3)
                    for item, col in zip(split, cols):
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


            with st.container():     
                    container = st.expander(':red[**Association Rules: Apriori Algorithm**]', expanded =True)
                    with container:
                        cols = st.columns(6)
                        for item, col in zip(apriori[:6], cols):
                            with col:
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
            with st.container():     
                    container = st.expander(':red[**Collaborative Filtering: User-user**]', expanded =True)
                    with container:
                        cols = st.columns(6)
                        for item, col in zip(uucf[:6], cols):
                            with col:
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

#########################################################################################  
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
                    st.code('HueUni\nRole :' + role)
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

