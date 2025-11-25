from preparation import prepare_data 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

###########################################################
# 4. ENTRAINEMENT DU MODELE AVEC PIPELINE
###########################################################

def train_model(X_train, y_train):
    # pipeline = standardisation + random forest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

if __name__ == "__main__":
    iris = pd.read_csv("./data/iris.csv")
    X_train, X_test, y_train, y_test = prepare_data(iris)
    print("3️⃣ Entraînement du modèle...")
    model = train_model(X_train, y_train)
    joblib.dump(model, "./model/iris_model.joblib")