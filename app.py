from flask import Flask, jsonify,request
import pickle
import nltk                                # Python library for NLP
from nltk.corpus import twitter_samples    # sample Twitter dataset from NLTK
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.stem import WordNetLemmatizer    # module for Lemmatization
from nltk.tokenize import TweetTokenizer
nltk.download('wordnet')
nltk.download('omw-1.4')
import re
import string

app = Flask(__name__)

#############################################################################
#loading lr 
with open('logistic_regression_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

#loading vectorizer 
with open('vectorizer.pkl', 'rb') as file:
    loaded_vectorizer = pickle.load(file)

def process_tweet(tweet):
    lemmatizer = WordNetLemmatizer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)  
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            lemma_word = lemmatizer.lemmatize(word,pos='v')  # stemming word
            tweets_clean.append(lemma_word)

    return tweets_clean

def list_to_string(lst):
    return ' '.join(lst)
###############################################################################

@app.route('/prediction', methods=['POST'])
def predicting_sentiment():

    #getting data from endpoint
    data = request.get_json()

    input_string = data['input_string']

    #1st function call---->tokenize data, preprocessig
    tokenize_data = process_tweet(input_string)

    #2nd function call-----> tokenize to final string
    final_string = list_to_string(tokenize_data)

    #transforming string to vector
    text_vector = loaded_vectorizer.transform([final_string])

    #prediction
    prediction = loaded_model.predict(text_vector)
    sentiment_mapping = {0: 'Negative', 1: 'Positive'}
    predicted_sentiment = sentiment_mapping[prediction[0]]

    return jsonify({'result': predicted_sentiment}), 200



@app.route('/status', methods=['GET'])
def status():
    return jsonify({'message': 'working NOW'}), 200



if __name__ == '__main__':
    app.run(debug=True)