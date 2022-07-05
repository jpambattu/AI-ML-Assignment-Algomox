import csv

from flask import Flask, request, render_template
import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
import re
import pickle
from stop_words import get_stop_words

from model_level_c import tfidf_converter

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
stplist = ['title', 'body', 'xxxx']
english_stopwords = get_stop_words(language='english')
english_stopwords += stplist
english_stopwords = list(set(english_stopwords))

@app.route('/')
def my_form():
      return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
      complaint = request.form['complaint']

      data = [complaint]
      custom_input = pd.DataFrame(data, columns=['complaint'])

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
            return (stop_words_free)

      # this is a time-consuming task
      custom_input['complaint'] = custom_input['complaint'].apply(clean_up)

      input_df = custom_input[custom_input.astype(str)['complaint'] != '[]']
      bow_input_df = input_df
      bow_input_df['complaints_untokenized'] = input_df['complaint'].apply(lambda x: ' '.join(x))

      features = tfidf_converter.transform(bow_input_df.complaints_untokenized.values.astype(str))

      svc_pred = model.predict(features)

      prediction = svc_pred[0]

      header = ['Complaint', 'Prediction']
      datarow = [complaint,prediction]

      with open('custom_input.csv', 'w', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write the data
            writer.writerow(datarow)

      return render_template('model_predictions.html', result = prediction)

if __name__ == '__main__':
      app.run(debug=True)
