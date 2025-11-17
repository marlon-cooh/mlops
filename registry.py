import mlflow #type:ignore
from scipy.sparse._csr import csr_matrix #type:ignore
from mlflow.tracking import MlflowClient #type:ignore
from datetime import datetime
from tracking import get_train_data, DATA_PATH, reducing_dimensionality
from sklearn.metrics import accuracy_score, f1_score #type:ignore
from utils.log_data import logger_setup

# ---- 1. Configuration ----
EXPERIMENT_NAME = "LaHolandaPerformance"
MODEL_NAME = "jurgendhilfe-weltweit"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# --- 2. Functions ---
def test_model_from_mlflow(model_name:str, stage:str, X_test:csr_matrix, y_test) -> None:
    """this function tests a model from mlflow
    Args:
        model_name (str): name of the model
        stage (str): stage of the model
        X_test (scipy.sparse._csr.csr_matrix): test data
        Y_test (scipy.sparse._csr.csr_matrix): test target
    Returns:
        float: rmse of the model
    
    """
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)
    y_pred = model.predict(X_test)
    acc_score = round(f1_score(y_test, y_pred, average='weighted'), 2)
    return {"f1_score": acc_score}

def promote_model(tracking_uri:str, exp_name:str, model_name:str, version:int, stage:int):
    """
        this function promotes the best model to production.
    Args:
        * tracking_uri (str): mlflow tracking uri
        * exp_name (str): experiment name
        
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(exp_name)
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id]
    )

    # Best run based on f1 score
    scores = {r.info.run_id:r.data.metrics['f1 score'] for r in runs}
    best_run_id = max(scores, key=scores.get)
    logger.info(f"Best run ID: {best_run_id}, F1 Score: {scores[best_run_id]}")

    # Registering best model.
    mlflow.register_model(model_uri=f"runs:/{best_run_id}/la_holanda_model",
        name=model_name
    )
    latest_versions = client.get_latest_versions(name=MODEL_NAME)
    for version_ in latest_versions:
        logger.info(f"Model: {model_name}, Version: {version_.version}, Stage: {version_.current_stage}")

    # Transitioning model to `stage``
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )

    client.update_model_version(
        name=model_name, 
        version=version,
        description=f"The model version {version_.version} was transitioned to Production on {datetime.today().date()}"
    )
    
    return f"The model version {version_.version} was transitioned to Production on {datetime.today().date()}"

if __name__ == "__main__":

    # Parameters.
    model_version = 9
    stage = "Production"
    
    # Output from get_train_data and reducing_dimensionality/
    X_train, X_test, y_train, y_test = get_train_data(DATA_PATH)
    X_train_pca, X_test_pca = reducing_dimensionality(X_train, X_test)
    logger = logger_setup("ModelRegistry.log")
    
    response = promote_model(
        tracking_uri=MLFLOW_TRACKING_URI, 
        exp_name=EXPERIMENT_NAME,
        model_name=MODEL_NAME, 
        version=model_version, 
        stage=stage
    )
    
    logger.info(response)
    
    logger.info(f"X_test shape: {X_test.shape}, and PCA transformed shape: {X_test_pca.shape}")
    result = test_model_from_mlflow(model_name=MODEL_NAME, stage="Production", X_test=X_test_pca, y_test=y_test)
    logger.info(f"Test results: {result}")