
import mlflow
from mlflow import MlflowClient


# 1. Configurer l'URI de la base de données MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()


# 2. Retrouver le Run « essai-2 »
# Il s'agit du modèle le plus faible du TP3

runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="tags.mlflow.runName = 'essai-2'"
)

run_id = runs.iloc[0]["run_id"]


# 3. Enregistrer le modèle dans le Model Registry
# Cela crée une NOUVELLE VERSION de churn-model

v = mlflow.register_model(
    f"runs:/{run_id}/model",
    "churn-model"
)

print("Nouvelle version créée :", v.version)


# 4. Récupérer l'alias « champion » actuel

actuel = client.get_model_version_by_alias(
    "churn-model",
    "champion"
)

print("Champion actuel : version", actuel.version)


# 5. Déplacer temporairement l'alias « champion »
# vers la nouvelle version créée

client.set_registered_model_alias(
    "churn-model",
    "champion",
    v.version
)

print(
    "Après déplacement : champion -> version",
    client.get_model_version_by_alias(
        "churn-model",
        "champion"
    ).version
)


# 6. Remettre l'alias « champion » sur l'ancienne version

client.set_registered_model_alias(
    "churn-model",
    "champion",
    actuel.version
)

print(
    "Après retour : champion -> version",
    client.get_model_version_by_alias(
        "churn-model",
        "champion"
    ).version
)
