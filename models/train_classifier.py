import sys
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import pickle
import joblib

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    make_scorer,
    classification_report,
)
from werkzeug.wrappers import CommonRequestDescriptorsMixin


from xgboost import XGBClassifier

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer
from nltk.corpus import stopwords

nltk.download("words", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)


def load_data(database_filepath):
    """Loads database located file database_filepath
    and returns features, labels and categories.

    Args:
        database_filepath ([type]): [description]

    Returns:
        X : dataframe of features
        y : dataframe of labels
        categories : a list of labels corresponding (y columns)
    """

    # read in db file to df
    engine = create_engine("sqlite:///" + database_filepath)

    df = pd.read_sql_table(table_name="Table1", con=engine)

    # define feateures (X) and label (y) arrays

    X = df["message"]
    y = df.iloc[:, 4:]
    categories = list(y.columns)

    return X, y, categories


def tokenize(text):
    """Normalizes text into a list of tokens (words) using a set of replacements and
    nltk dictionaries.
    URL, numbers and ponctuations are replaced. Tokens are
    filtered using nltk english words and stopwords dicts.
    Final tokens are lemmatized.

    Args:
        text (str): text to be tokenized

    Returns:
        list : list of tokens
    """

    # replace all urls with a placeholder text
    url_regex = (
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    text = re.sub(url_regex, "urlplaceholder", text)

    # replace punctuation with spaces
    t = str.maketrans(" ", " ", string.punctuation)
    text = text.translate(t)

    # replace single quote with empty char
    t = str.maketrans(dict.fromkeys("'`", ""))
    text = text.translate(t)

    tokens = word_tokenize(text)
    # lemmatize and remove stop words

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):

        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))

            try:
                first_word, first_tag = pos_tags[0]
                if first_tag in ["VB", "VBP"] or first_word == "RT":
                    return True

            except:
                return False

        return False

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """Builds the machine learning pipeline, then crossvalidate
    different parameters sets.
    The best model pipeline is returned as output.

    Returns:
        Gridsearch object : Gridsearch output
    """

    # text processing and model pipeline
    xgboost = XGBClassifier(
        nthread=8,
        n_estimators=10,  # best is around 70-80
        random_state=42,
        seed=2,
        colsample_bytree=0.6,
        subsample=0.7,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    pipe = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "text_pipeline",
                            Pipeline(
                                [
                                    ("vect", CountVectorizer(tokenizer=tokenize)),
                                    ("tfidf", TfidfTransformer()),
                                ]
                            ),
                        ),
                        ("starting_verb", StartingVerbExtractor()),
                    ]
                ),
            ),
            ("clf", MultiOutputClassifier(xgboost)),
        ]
    )

    # Resultats of test runs
    # xgboost n_estimators = 10, results f1-score w-avg : 0.59

    # Define parameters for GridSearchCV
    # countVectorizer + tdidfTransformed has been drop in favor
    #   of tdidf vectorizer, which seem to do both.
    # comment : latest run indicates that the model performs better without td-idf.

    parameters = {
        "features__text_pipeline__vect__ngram_range": ((1, 1), (1, 2)),
        "features__text_pipeline__vect__max_df": (0.75, 1.0),
        "features__text_pipeline__vect__max_features": (None, 5000),
        #         "features__text_pipeline__tfidf__use_idf": (True, False),
        "clf__estimator__n_estimators": [50, 100],
    }

    scorer = make_scorer(score_model, greater_is_better=True)

    # Cross validate model
    # Exhaustive search over specified parameter values for an estimator.

    cv = GridSearchCV(pipe, param_grid=parameters, scoring=scorer, verbose=3, cv=5)

    return cv


def score_model(y_true, y_pred, beta=1):
    """custom scorer function used in cross validation.
    Returns f1-score weighted average"""

    output_dict = classification_report(
        y_true, y_pred, output_dict=True, zero_division=1
    )
    return output_dict["weighted avg"]["f1-score"]


def evaluate_model(model, X_test, y_test, category_names):

    """Predicts and prints scores of model.
    Makes a clasification report of recall, precision and f1 scores.
    Plot the f1-scores if possible.

    Args:
        model (estimator object): your model or model pipeline.
        X_test ([type]): [description]
        y_test ([type]): [description]
        category_names (list of array): list of categories to pass as labels

    Returns:
        Dataframe : A summary containing the classification report
        for each category name.
    """

    y_pred = model.predict(X_test)

    # report = classification_report(y_test, y_pred, target_names=category_names)
    # print(report)

    output_dict = classification_report(
        y_test, y_pred, target_names=category_names, output_dict=True, zero_division=1
    )

    df = pd.DataFrame.from_dict(output_dict, orient="index")

    # plot
    # plt.figure(figsize=(6, 10))
    # sns.barplot(df["f1-score"].sort_values(), df["f1-score"].sort_values().index)

    return output_dict["weighted avg"]["f1-score"]


def save_model(model, model_filepath):
    """Store trained model into pickle file.

    Args:
        model (estimator object): fitted model
        model_filepath (string): destination filepath for pickle
    """

    pickle.dump(model.best_estimator_, open(model_filepath, "wb"))


def main():

    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))

        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()