from tqdm import tqdm
import time
import subprocess
import pandas as pd #type:ignore
from utils.log_data import logger_setup, get_train_data, reducing_dimensionality

# sklearn.
from sklearn.model_selection import GridSearchCV #type:ignore
from sklearn.linear_model import LogisticRegression #type:ignore
from sklearn.ensemble import RandomForestClassifier #type:ignore
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score #type:ignore
from imblearn.over_sampling import SMOTEN #type:ignore

# mlflow.
import mlflow #type:ignore
from mlflow.models import infer_signature #type:ignore


# ---- 1. Models and experiments ----
EXPERIMENTS = {
    "logreg_simple": {
        "experiment_name": "LaHolanda_LogReg_Simple",
        "model": LogisticRegression(random_state=42, max_iter=2000),
        "param_grid": None,
    },
    "logreg_grid": {
        "experiment_name": "LaHolanda_LogReg_Grid",
        "model": LogisticRegression(random_state=42, max_iter=2000),
        "param_grid": {
            "C": [0.001, 0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        },
    },
    "rf_simple": {
        "experiment_name": "LaHolanda_RF_Simple",
        "model": RandomForestClassifier(random_state=42),
        "param_grid": None,
    },
    "rf_grid": {
        "experiment_name": "LaHolanda_RF_Grid",
        "model": RandomForestClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [100, 200, 500],
            "max_depth": [10, 20, 50, 100],
            "criterion": ["gini", "entropy"],
        },
    },
    # later: "logreg_boosting", "naive_bayes", ...
}

REGISTERED_MODEL_NAME = "jurgendhilfe-weltweit"
DATA_PATH = "./cleaned_data/grade_summary.parquet"

# --- 2. Setting logger ----
logger = logger_setup("model_comparison", "model_comparison.log")

# --- 2. Loading data ---
X_train, X_test, y_train, y_test = get_train_data(DATA_PATH, 'band')

# --- 3. Launching mlflow process ---
def launch_mlflow_server(host:str, port:int) -> None:
    """Launch a mlflow server and return the process object for later termination.
    
    Args:
        host: Server host address
        port: Server port number
        
    Returns:
        subprocess.Popen: The MLflow server process object
    """
    
    # Opening MLflow server as a subprocess.
    mlflow_process = subprocess.Popen(
        [
            "mlflow", "server",
            "--backend-store-uri", "sqlite:///mlflow.db",
            "--default-artifact-root", "./mlruns",
            "--host", host,
            "--port", str(port)
        ]
    )
    
    logger.info(f"✅ MLflow server started at http://{host}:{port}")    
    return mlflow_process
    
def main(exp_list:dict, reg_model:str, timeout:int = 120) -> None:
    """This function runs a list of experiments given a dictionary of models and tracks models into MLflow.
    
        Args:
        * exp_list : Structured dict of experiments where instructions about models is showed.
            >>>
                "rf_simple": {
                "experiment_name": "LaHolanda_RF_Simple",
                "model": RandomForestClassifier(random_state=42),
                "param_grid": None,
            },
            "rf_grid": {
                "experiment_name": "LaHolanda_RF_Grid",
                "model": RandomForestClassifier(random_state=42),
                "param_grid": {
                    "n_estimators": [100, 200, 500],
                    "max_depth": [10, 20, 50, 100],
                    "criterion": ["gini", "entropy"],
                },
            }
            >>>
        * reg_model: Name of the registered model in MLflow to log models under.
        * timeout: Timeout in seconds before terminating MLflow server.
    """

    start_time = time.time()
    try:
        X_train_pca, X_test_pca = reducing_dimensionality(X_train, X_test)
    
        # Treating imbalance with SMOTEN.
        smoten = SMOTEN(random_state=42)
        X_train_res, y_train_res = smoten.fit_resample(X_train_pca, y_train)
        
        # Finding best model.
        for experiment_key, exp in tqdm(exp_list.items(), desc="Training models"):
            
            experiment_name = exp["experiment_name"]
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(f"Timeout ({timeout}s) exceeded. Stopping training.")
                break
            logger.info(f"Running experiment for model: {experiment_name}")
        
            # Calling hyperparam optimizer.
            param_grid = exp['param_grid']
            if param_grid:
                estimator = GridSearchCV(
                    estimator=exp['model'],
                    param_grid=param_grid,
                    scoring='accuracy',
                    cv=5,
                    n_jobs=1, # Use 1 job to avoid overloading the system
                )
            else:
                estimator = exp['model']

            # Training model.
            estimator.fit(X_train_res, y_train_res)

            # Predict on the test set.
            y_pred = estimator.predict(X_test_pca)
            # y_pred_proba = gcv.predict_proba(X_test_pca)[:, 1]
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred) # This is the metric to keep track of.
            f1 = f1_score(y_test, y_pred, average='weighted')
            # roc_score = roc_auc_score(y_test, y_pred_proba)
            logger.info(f"Metrics for experiment {experiment_key} were \n * F1 score: {f1} \n * Accuracy: {acc}")
        
            # Create a new MLflow Experiment
            mlflow.set_experiment(experiment_name=experiment_name)
        
            # Start an MLflow run
            with mlflow.start_run() as run:

                # Log the hyperparams
                if isinstance(estimator, GridSearchCV):
                    mlflow.log_params(estimator.best_params_)
                    logger.info(f"Best params for {experiment_key}: {estimator.best_params_}")
                else:
                    mlflow.log_params(estimator.get_params())
                    logger.info(f"Best params for {experiment_key}: {estimator.get_params()}")

                # Log the loss metric
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1 score", f1)
                # mlflow.log_metric("roc_auc", roc_score)

                # Infer the model signature
                signature = infer_signature(X_train_pca, estimator.predict(X_train_pca))

                # Setting a tag that we can use to remind this model.
                mlflow.set_tag("Training info", f"{exp['model']} for La Holanda analysis.")

                # Log the model, which inherits the parameters and metric
                model_info = mlflow.sklearn.log_model(
                    sk_model=estimator,
                    artifact_path="la_holanda_model",
                    signature=signature,
                    input_example=X_train_pca[:20],
                    registered_model_name=reg_model # jurgendhilfe-weltweit is the model to optimize and deploy.
                )

                logger.info(f"Experiment {experiment_key} logged in MLflow with run ID: {model_info.run_id}")

        logger.info("Model comparison experiments completed.")
        logger.info("MLflow server terminated.")
    
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        raise
   
if __name__ == "__main__":
    
    # Launch MLflow server.
    mlflow.set_tracking_uri(uri="sqlite:///mlflow.db")
    # Run main function with timeout.
    main(exp_list=EXPERIMENTS, reg_model=REGISTERED_MODEL_NAME, timeout=120)