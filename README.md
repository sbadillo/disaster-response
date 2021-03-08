# Disaster Response Pipeline Project

A web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

This project demonstrates the creation of basic ETL (extract, transform and load) pipelines, ML (Machine Learning) pipelines. The results and interaction with the model are provided via a web app.

The model is trained using figure-8's disaster response data.
The estimator uses gradient boosting and an implementation of grid search to find the best parameters.

-sergio

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database

     ```shell
     $ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
     ```

   - To run ML pipeline that trains classifier and saves
     ```shell
     $ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
     ```

2. Run the following command in the app's directory to run your web app.

   ```shell
   $ python run.py
   ```

3. Go to http://localhost:3001/ to see the app

### Additional info

You might need to install packages from the requirements file :

```console
$ pip install -r requirements.txt
```
