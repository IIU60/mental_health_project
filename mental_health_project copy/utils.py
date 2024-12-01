"""
Utility functions for loading and processing mental health related subreddit data.

Functions:
    load_data() -> pd.DataFrame:
        Loads and concatenates mental health related subreddit data from CSV files.

    load_undersampled_split_data(random_state=0, features='all') -> tuple:
        Loads data, performs undersampling excluding "COVID19_support" subreddit, and splits it into training and test sets.

    reduced_classes(n_classes=6, random_state=0) -> tuple:
        Reduces the dataset to the most common subreddits, performs undersampling, and splits it into training and test sets.

    reduced_classes_with_other(n_classes=6, random_state=0) -> tuple:
        Reduces the dataset to the most common subreddits, creates an "Other" class for less common subreddits, performs undersampling, and splits it into training and test sets.
"""

from os import listdir
import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

def load_data():
    '''
    returns 'data'. the mental health related subreddits in one dataframe
    '''
    non_mental_health_reddits = ("conspiracy", "divorce", "fitness", "guns", "jokes", "legaladvice", "meditation", "parenting", "personalfinance", "relationships", "teaching")

    mental_health_files = [file for file in listdir('data') if not file.startswith(non_mental_health_reddits)]
    mental_health_files
    dfs = []

    for file in mental_health_files:
        dfs.append(pd.read_csv(f'./data/{file}'))

    data = pd.concat(dfs, ignore_index=True)

    return data

def load_undersampled_split_data(random_state=0, features='all'):
    '''
    returns X_resampled, X_test, y_resampled, y_test.
    undersampled data and split with 0.1 test size
    '''
    data = load_data()

    if features == 'all':
        X, y = data.drop(columns=['subreddit','author','date','post']), data['subreddit']
    else:
        data = data[~data['subreddit'].isin(features)] # remove unwanted features
        X, y = data.drop(columns=['subreddit','author','date','post']), data['subreddit']

    # train test split with stratification on target class
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state, stratify=y)

    rus = RandomUnderSampler(random_state=random_state)
    # undersampling
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    return X_resampled, X_test, y_resampled, y_test


def reduced_classes(n_classes=6, random_state=0):
    '''
    returns X_resampled, X_test, y_resampled, y_test.
    reduced classes to the most common (n_classes) subreddits and split with 0.1 test size
    '''
    data = load_data()

    most_common_reddits = data.subreddit.value_counts().sort_values()[-n_classes:].index.tolist()

    selected_data = data[data['subreddit'].isin(most_common_reddits)]

    X, y = selected_data.drop(columns=['subreddit','author','date','post']), selected_data['subreddit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    rus = RandomUnderSampler(random_state=random_state)

    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    return X_resampled, X_test, y_resampled, y_test

def reduced_classes_with_other(n_classes=6, random_state=0):
    '''
    returns X_train_resampled, X_test, y_train_resampled, y_test.
    reduced classes to the most common (n_classes) subreddits and creates an "Other" class for less common subreddits
    and split with 0.1 test size
    '''
    data = load_data()

    most_common_reddits = data.subreddit.value_counts().sort_values()[-n_classes:].index.tolist()
    no_covid = data[data['subreddit'] != 'COVID19_support'] # too small to include in undersampling
    X, y = no_covid.drop(columns=['subreddit','author','date','post']), no_covid['subreddit']

    # Splitting the data into the most common subreddits and the rest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
    selected_X = X_train.loc[y_train.isin(most_common_reddits)]
    not_selected_X = X_train.loc[~y_train.isin(most_common_reddits)]
    selected_y = y_train.loc[y_train.isin(most_common_reddits)]
    not_selected_y = y_train.loc[~y_train.isin(most_common_reddits)]

    # Creating the undersampled "other" class
    rus = RandomUnderSampler(random_state=random_state)
    not_X_resampled, not_y_resampled = rus.fit_resample(not_selected_X, not_selected_y)
    not_y_resampled.loc[:] = "Other"
    y_test.loc[~y_test.isin(most_common_reddits)] = "Other"

    # Combining the selected and undersampled "other" classes
    complete_X = pd.concat([selected_X, not_X_resampled])
    complete_y = pd.concat([selected_y, not_y_resampled])

    rus = RandomUnderSampler(sampling_strategy={label: 4000 for label in complete_y.unique()}, random_state=random_state)
    # Undersampling the combined classes
    X_train_resampled, y_train_resampled = rus.fit_resample(complete_X, complete_y)

    return X_train_resampled, X_test, y_train_resampled, y_test
