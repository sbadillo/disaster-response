import sys
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import sklearn
import pickle

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
)


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

    # usage X, Y, category_names = load_data(database_filepath)

    # read in db file to df
    engine = create_engine("sqlite:///" + database_filepath)

    df = pd.read_sql_table(table_name="Table1", con=engine)

    # clean columns with all zeros
    is_not_empty = (df != 0).any(axis=0)
    df = df.loc[:, is_not_empty]

    # define feateures (X) and label (y) arrays
    X = df["message"]
    y = df.drop(columns=["id", "message", "original", "genre"], axis=1)
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

    text = text.lower()

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
            clean_tok = lemmatizer.lemmatize(tok).strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Builds the machine learning pipeline, then crossvalidate
    different parameters sets.
    The best model pipeline is returned as output.

    Returns:
        Gridsearch object : Gridsearch output
    """

    # text processing and model pipeline
    xclf = XGBClassifier(
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
            ("vect_tdidf", TfidfVectorizer(tokenizer=tokenize)),
            ("xclf", MultiOutputClassifier(xclf)),
        ]
    )

    # Define parameters for GridSearchCV

    parameters = {
        "vect_tdidf__max_df": (0.75, 1.0),
        "vect_tdidf__use_idf": (True, False),
        "xclf__estimator__n_estimators": (50, 70),
    }

    # Cross validate model
    # Exhaustive search over specified parameter values for an estimator.
    cv = GridSearchCV(pipe, param_grid=parameters, verbose=3, cv=3)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """Predicts and prints scores of model.
    Makes a clasification report of recall, precision and f1 scores.
    Plot the f1-scores if possible.

    Args:
        model (estimator object): your model or model pipeline.
        X_test ([type]): [description]
        y_test ([type]): [description]
        category_names (list of array): list or array of categories to pass as labels

    Returns:
        Dataframe : A summary containing the classification report
        for each category name.
    """

    y_pred = model.predict(X_test)

    print("Accuracy = %.3f" % accuracy_score(y_test, y_pred))

    report = classification_report(y_test, y_pred, target_names=category_names)
    print(report)

    if float(sklearn.__version__[:4]) >= 0.20:

        output_dict = classification_report(
            y_test, y_pred, target_names=category_names, output_dict=True
        )
        df = pd.DataFrame.from_dict(output_dict, orient="index")

        # plot
        plt.figure(figsize=(6, 10))
        sns.barplot(df["f1-score"].sort_values(), df["f1-score"].sort_values().index)

        return df

    else:

        print("sklearn version is old.")
        output = pd.Series()

        for i, c in enumerate(category_names):
            score = f1_score(y_test[c], y_pred.transpose()[i])
            output[c] = score

        return output


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