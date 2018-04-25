import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from flask import *

app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def predict():
    input_text = request.args.get('REVIEW')
    prediction = model.predict(tfidf.transform([input_text]))[0]
    return str(prediction)

if __name__ == '__main__':
    df = pd.read_json('review0.json', lines=True)

    col = ['stars', 'text']
    cleaned_df = df[col]

    cleaned_df = cleaned_df[pd.notnull(cleaned_df['text'])]
    cleaned_df.columns = ['rating', 'review']
    cleaned_df['bin'] = pd.cut(cleaned_df['rating'], [0, 2, 4, float("inf")], labels=['1', '2', '3'])

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                            stop_words='english', strip_accents='unicode')

    # Convert the reviews to tf-idf features
    features = tfidf.fit_transform(cleaned_df.review).toarray()

    # Obtain the class labels 'rating'
    labels = cleaned_df.bin

    # Test Linear SVC Exclusively
    model = LinearSVC()

    # Split the Data
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, cleaned_df.index, test_size=0.33, random_state=0)
    model.fit(X_train, y_train)
    print("API READY")
    app.run(debug=True)

