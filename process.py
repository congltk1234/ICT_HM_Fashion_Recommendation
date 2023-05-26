#@title Text Preprocessing
import re
import nltk
import re                                  # library for regular expression operations
import string                              # for string operations
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.stem import WordNetLemmatizer

def remove_punctuation(text):
    if text != text :
        punctuationfree = ""
    else :
        punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree
    
def tokenization(text):
    tokens = re.split('W+',text)
    return tokens

def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text

def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text


porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')


def preprocess_sentence(caption):
    caption = remove_punctuation(caption)
    caption = caption.lower()
    caption = tokenization(caption)
    caption = remove_stopwords(caption)
    caption = stemming(caption)
    caption = lemmatizer(caption)
    return caption[0]