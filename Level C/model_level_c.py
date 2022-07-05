import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from stop_words import get_stop_words
import re
from ast import literal_eval
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from  sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import array
import pickle

main_df = pd.read_csv(r'C:\Users\Jpamb\Desktop\AIQ1\complaints-2021-09-08_07_12.csv')


# creating a list of extra stop-words as these repeatedly appear in all complaints
# xxxx is used in the data to hide sensitive information
stplist = ['title', 'body', 'xxxx']
english_stopwords = get_stop_words(language='english')
english_stopwords += stplist
english_stopwords = list(set(english_stopwords))

def get_wordnet_pos(word):
    """
    Function that determines the the Part-of-speech (POS) tag.
    Acts as input to lemmatizer
    """
    if word.startswith('N'):
        return wn.NOUN
    elif word.startswith('V'):
        return wn.VERB
    elif word.startswith('J'):
        return wn.ADJ
    elif word.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN


def clean_up(text):
    """
    Function to clean data.
    Steps:
    - Removing special characters, numbers
    - Lemmatization
    - Stop-words removal
    - Getting a unique list of words
    """
    lemmatizer = nltk.WordNetLemmatizer().lemmatize
    text = re.sub('\W+', ' ', str(text))
    text = re.sub(r'[0-9]+', '', text.lower())

    word_pos = nltk.pos_tag(nltk.word_tokenize(text))
    normalized_text_lst = [lemmatizer(x[0], get_wordnet_pos(x[1])).lower() for x in word_pos]
    stop_words_free = [i for i in normalized_text_lst if i not in english_stopwords and len(i) > 3]
    stop_words_free = list(set(stop_words_free))
    return(stop_words_free)


df = main_df
df = df[['Product', 'Consumer complaint narrative']]

df = df[pd.notnull(df['Consumer complaint narrative'])]
df = df.rename({'Consumer complaint narrative':'complaint', 'Product':'product'},
               axis='columns')

products_count_df = df.groupby('product').complaint.count().to_frame()
products_count_df.reset_index(level=0, inplace=True)


# this is a time-consuming task
df['complaint'] = df['complaint'].apply(clean_up)
df.to_csv("C:/Users/Jpamb/Desktop/AIQ1/Level C/output_consumer_complaints.csv", index=False)


# Loading this from the saved version of this file.
input_df = pd.read_csv(r"C:/Users/Jpamb/Desktop/AIQ1/Level C/output_consumer_complaints.csv",
                       converters={"complaint": literal_eval})


input_df = input_df[input_df.astype(str)['complaint'] != '[]']
bow_input_df = input_df



min_word_count = 10
products_count_df = bow_input_df.groupby('product').complaint.count().to_frame()
products_count_df.reset_index(level=0, inplace=True)
class_labels = array(products_count_df['product'].unique())

# TfidfVectorizer handles tokenization.
bow_input_df['complaints_untokenized'] = bow_input_df['complaint'].apply(lambda x: ' '.join(x))

tfidf_converter = TfidfVectorizer(max_features=1500, sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1',
                                  stop_words='english')
features = tfidf_converter.fit_transform(bow_input_df.complaints_untokenized).toarray()
labels = class_labels

train_x, test_x, train_y, test_y = train_test_split(features, bow_input_df['product'], test_size=0.3,
                                                    random_state=123)


# Linear Support Vector Machine
svc_model = LinearSVC()
svcc_model = CalibratedClassifierCV(svc_model)
svc_clf = svcc_model.fit(train_x, train_y)
svc_preds = svcc_model.predict(test_x)

#print(svc_preds)

pickle.dump(svcc_model, open('model.pkl','wb'))
