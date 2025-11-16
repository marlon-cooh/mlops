import mlflow #type:ignore
import logging
from scipy.sparse._csr import csr_matrix #type:ignore
from mlflow.tracking import MlflowClient #type:ignore
from datetime import datetime
from logging.handlers import RotatingFileHandler
from tracking import get_train_data, DATA_PATH, reducing_dimensionality
from sklearn.metrics import accuracy_score, f1_score #type:ignore


# ---- 1. Configuration ----
EXPERIMENT_NAME = "LaHolandaPerformance"
MODEL_NAME = "jurgendhilfe-weltweit"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

# --- 2. Functions ---
def logger_setup(logger_name: str) -> None:
    """Sets up logger for the module."""
    
    # Main logger.
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger
    
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Console -> INFO+
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)   
    console_handler.setFormatter(formatter)

    # File -> DEBUG+
    file_handler = RotatingFileHandler(
        "model_registry.log",
        maxBytes=1024*1024,  # 1MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


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

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = get_train_data(DATA_PATH)
    X_train_pca, X_test_pca = reducing_dimensionality(X_train, X_test)
    logger = logger_setup("ModelRegistry.log")
    
    client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id]
    )

    # Best run based on f1 score
    scores = {r.info.run_id:r.data.metrics['f1 score'] for r in runs}
    best_run_id = max(scores, key=scores.get)
    logger.info(f"Best run ID: {best_run_id}")

    # Registering best model.
    mlflow.register_model(model_uri=f"runs:/{best_run_id}/la_holanda_model",
        name=MODEL_NAME
    )
    latest_versions = client.get_latest_versions(name=MODEL_NAME)
    for version in latest_versions:
        logger.info(f"Model: {MODEL_NAME}, Version: {version.version}, Stage: {version.current_stage}")

    # ---- 3. Transitioning model to Production ----
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=3,
        stage="Production"
    )

    client.update_model_version(
        name=MODEL_NAME, 
        version=3,
        description=f"The model version {3} was transitioned to Production on {datetime.today().date()}"
    )
    logger.info(f"Client updated model version description.")
    
    logger.info(f"X_test shape: {X_test.shape}, and PCA transformed shape: {X_test_pca.shape}")
    result = test_model_from_mlflow(model_name=MODEL_NAME, stage="Production", X_test=X_test_pca, y_test=y_test)
    logger.info(f"Test results: {result}")