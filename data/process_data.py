import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads data from message_filepath and categories_filepath
    and outputs to a single dataframe
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how="left", on="id")

    # Split categories into columns, then convert to 0 and 1 values

    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0, :]
    category_colnames = [c.split("-")[0] for c in row]
    categories.columns = category_colnames

    for column in categories:

        categories[column] = categories[column].str[-1].replace("2", "1")
        categories[column] = pd.to_numeric(categories[column])

    # Merge all together
    df.drop(columns="categories", axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    return df


def clean_data(df):
    """Drops duplicates from provided dataframe

    Args:
        df (pandas.Dataframe): Dataframe to clean.

    Returns:
        df: Filtered or clean dataframe
    """

    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """Saves pandas df dataframe into a database database_filename.

    Args:
        df (pandas.Dataframe): Dataframe to save.
        database_filename (string): output name of database.
    """

    engine = create_engine("sqlite:///{}".format(database_filename))

    df.to_sql("Table1", engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()