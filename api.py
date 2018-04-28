from flask import *
import pandas as pd
from textblob import Word
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import wordnet as wn
import ssl

app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def predict():
    input_text = [request.args.get('REVIEW')]
    input_text = pd.DataFrame(input_text, columns=['a'])
    # Set to lowercase
    input_text['a'] = input_text['a'].str.lower()

    # Remove symbols
    input_text['a'] = input_text['a'].str.replace('[^\w\s]', '')

    # Add Sentiment column
    input_text['sentiment'] = input_text['a'].apply(lambda x: TextBlob(x).sentiment[0])

    # Vectorize
    input_transform = tfidf.transform(input_text['a'])
    input_transform = hstack((input_transform, np.array(input_text['sentiment'])[:,
                                               None]))

    prediction = model.predict(input_transform)[0]
    return str(prediction)


if __name__ == '__main__':

    # This line is used if a connection is refused when downloading from nltk
    # ssl._create_default_https_context = ssl._create_unverified_context

    # Setting up stop words for data preprocessing
    stop = stopwords.words('english')
    food_sets = wn.synsets('food')

    food_stop_words = list()

    for food_set in food_sets:
        food_stop_words += list(
            set([w.replace('_', ' ') for s in food_set.closure(lambda s: s.hyponyms()) for w in s.lemma_names()]))

    # Load Yelp dataset
    df = pd.read_json('review0.json', lines=True)

    # Create new dataframe consisting of just stars and text from the Yelp dataset
    col = ['stars', 'text']
    cleaned_df = df[col]
    cleaned_df = cleaned_df[pd.notnull(cleaned_df['text'])]
    cleaned_df.columns = ['rating', 'review']

    # Bin the reviews into 3 classes
    cleaned_df['bin'] = pd.cut(cleaned_df['rating'], [0, 2, 4, float('inf')], labels=['1', '2', '3'])

    # Set dataframe to lowercase
    cleaned_df['review'] = cleaned_df['review'].apply(lambda x: ' '.join(x.lower() for x in x.split()))

    # Remove symbols
    cleaned_df['review'] = cleaned_df['review'].str.replace('[^\w\s]', '')

    # Lemmatize
    cleaned_df['review'] = cleaned_df['review'].apply(lambda x: \
                                                          ' '.join([Word(word).lemmatize() for word in x.split()]))

    # Compute sentiment of every row and add a column in the new df
    cleaned_df['sentiment'] = cleaned_df['review'].apply(lambda x: \
                                                             TextBlob(x).sentiment[0])

    # Remove stop words
    cleaned_df['review'] = cleaned_df['review'].apply(lambda x: ' '.join(x
                                                                         for x in x.split() if x not in stop))
    cleaned_df['review'] = cleaned_df['review'].apply(lambda x: ' '.join(x
                                                                         for x in x.split() if
                                                                         x not in food_stop_words))
    tfidf = TfidfVectorizer(
        sublinear_tf=True,
        min_df=5,
        norm='l2',
        encoding='latin-1',
        ngram_range=(1, 2),
        stop_words=food_stop_words,
    )

    # Convert the reviews to tf-idf features
    features = tfidf.fit_transform(cleaned_df.review)

    # Obtain the class labels 'bin'
    labels = cleaned_df.bin

    # Add column for sentiment
    features = hstack((features, np.array(cleaned_df['sentiment'])[:,None]))
    
    # Bagged LR for Classification
    seed = 0
    kfold = model_selection.KFold(n_splits=20, random_state=seed)
    cart = LogisticRegression()
    model = BaggingClassifier(base_estimator=cart, n_estimators=5, random_state=seed)
    model.fit(features, labels)

    print("API is ready")
    app.run(debug=True)
