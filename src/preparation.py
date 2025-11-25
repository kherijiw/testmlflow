from sklearn.model_selection import train_test_split
import pandas as pd

def prepare_data(iris, test_size=0.2):

    X = iris.drop(columns=["target"])   # toutes les colonnes sauf target
    y = iris["target"]                  # uniquement la colonne cible


    # split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test