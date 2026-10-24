
# ===== 1. Importer les bibliothèques =====

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ===== 2. Charger et préparer les données =====

df = pd.read_csv("churn_data.csv")

# Transformer la colonne catégorielle "abonnement" en colonnes numériques
X = pd.get_dummies(
    df[["age", "salaire", "abonnement", "anciennete", "consommation"]],
    drop_first=True
)

# Variable cible : 1 = le client part, 0 = le client reste
y = df["churn"]

# Diviser les données : 80 % pour l'entraînement et 20 % pour le test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ===== 3. Choisir l'expérience MLflow =====

mlflow.set_experiment("churn-prediction")


# ===== 4. Entraîner le modèle et enregistrer les résultats =====

with mlflow.start_run(run_name="essai-1"):

    # Créer le modèle Random Forest avec ses paramètres
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Prédire les résultats sur les données de test
    y_pred = model.predict(X_test)

    # Calculer la précision du modèle
    acc = accuracy_score(y_test, y_pred)

    # Enregistrer les paramètres dans MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)

    # Enregistrer la métrique de performance
    mlflow.log_metric("accuracy", acc)

    # Enregistrer le modèle entraîné
    mlflow.sklearn.log_model(
        model,
        name="model",
        serialization_format="cloudpickle"
    )

    # Afficher la précision dans le terminal
    print("Accuracy :", acc)
