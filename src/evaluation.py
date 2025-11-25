from sklearn.metrics import accuracy_score, classification_report
from preparation import prepare_data 
import pandas as pd 
import joblib
import os


###########################################################
# 5. EVALUATION DU MODELE
###########################################################

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc, preds


if __name__ == "__main__":
    iris = pd.read_csv("./data/iris.csv")
    X_train, X_test, y_train, y_test = prepare_data(iris)

    model_path = "./model/iris_model.joblib"
    if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le modèle n'a pas été trouvé : {model_path}")
    model = joblib.load(model_path)

    print("4️⃣ Évaluation du modèle...")
    acc,preds = evaluate_model(model, X_test, y_test)
    print("✅ Accuracy:", acc)
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))
