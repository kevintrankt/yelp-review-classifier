# [yelp-review-classifier](https://kevintrankt.com/yelp)

[![License](http://img.shields.io/badge/Licence-MIT-brightgreen.svg)](LICENSE)

# Introduction

This project uses scikit-learn to classify text data from Yelp reviews to predict what rating will be given. The application first preprocesses the Yelp data. Then the application creates a K-Folds cross validator with 20 splits. A Bagging Classifier is initialized with 5 Logistic Regression models.

A frontend is provided to visualize the results. A predicted star rating will display once at least 10 characters are typed. 

### Prerequisites

- scikit-learn
- Flask
- Pandas
- nltk

### Usage

Clone this repository locally and train the model:

```bash
git clone https://github.com/kevintrankt/yelp-review-classifier.git
cd yelp-review-classifier
python api.py
```

Once the model is done training, visit [yelp-review-classifier](https://kevintrankt.com/yelp) and start typing out a review to see the suggested rating! 
