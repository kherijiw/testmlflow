import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
###########################################################
# 2. ACCES / INGESTION DES DONNEES
###########################################################

def load_data():
    iris = load_iris()
    X = iris.data
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['target'] = iris.target
    return df

if __name__ == "__main__":
    print("1️⃣ Chargement des données...")
    df = load_data()
    df.to_csv("./data/iris.csv")








