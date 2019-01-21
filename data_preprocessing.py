import pandas as pd
import numpy as np

def my_dropna_column(*args, df):
    for i in range(len(args)):
        df.drop([args[i]], axis=1, inplace=True)
    return df


def my_dropna_row(*args, df):
    for i in range(len(args)):
        df.drop([args[i]], inplace=True)
    return df


def my_replacena_mean(*args, df):
    for i in range(len(args)):
        df[args[i]].fillna(df.loc[:, args[i]].mean(), inplace=True)
    return df


def my_replacena_mode(*args, df):
    for i in range(len(args)):
        df[args[i]].fillna(df.loc[:, args[i]].mode()[0], inplace=True)
    return df


def my_replacena_median(*args, df):
    for i in range(len(args)):
        df[args[i]].fillna(df.loc[:, args[i]].median(), inplace=True)
    return df


def my_standartize(column, df):
    for i in range(len(df[column])):
        df[column].replace(df[column][i], abs((df[column][i] - df[column].mean()) / df[column].std()), inplace=True)


def my_normalize(column, df):
    min = df[column].min()
    max = df[column].max()
    for i in range(len(df[column])):
        df[column].replace(df[column][i], (df[column][i] - min) / (max - min), inplace=True)