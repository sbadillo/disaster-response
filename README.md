# Disaster Response Pipeline Project

A web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

This project demonstrates the creation of basic ETL (extract, transform and load) pipelines, ML (Machine Learning) pipelines. The results and interaction with the model are provided via a web app.

The model is trained using figure-8's disaster response data.
The estimator uses gradient boosting and an implementation of grid search to find the best parameters.

- sergio

## Requirements:

1. This script uses Xgboost, so you might need to install packages from the requirements file.

```console
$ pip install -r requirements.txt
```

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database

```console
$ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

To run ML pipeline that trains classifier and saves

```console
$ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

2. Run the following command in the app's directory to run your web app.

```console
$ cd app/
$ python run.py
```

3. Go to http://localhost:3001/ to see the app

## Files


- **data/process_data.py** : Extracts, transform and Load pipeline that takes disaster_categories.csv and disaster_messages.csv and writes outputs into a SQLite database.db in the specified database file path.
- **model/train_classifier** : Machine Learning pipeline that builds, trains and evaluates a classification model using grid search and cross validation. The script stores the best classifier into a pickle file.  
- **app/run.py** : Main run file that loads the pkl model and deploys a web app using flask.   