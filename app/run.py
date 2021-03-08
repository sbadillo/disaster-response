import json
import plotly
import joblib
import pandas as pd
import re
import string
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("words", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

from sqlalchemy import create_engine

app = Flask(__name__)


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


# load data
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table(table_name="Table1", con=engine)

# drop cols with all zeros
is_not_empty = (df != 0).any(axis=0)
df = df.loc[:, is_not_empty]

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    labels_counts = df.drop(["id"], axis=1).select_dtypes(include="number").sum()
    labels_names = df.drop(["id"], axis=1).select_dtypes(include="number").columns

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            "data": [Bar(x=labels_names, y=labels_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():

    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()