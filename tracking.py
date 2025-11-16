from tqdm import tqdm
import logging
import subprocess
import pandas as pd #type:ignore
from logging.handlers import RotatingFileHandler


# sklearn.
from sklearn.model_selection import train_test_split, GridSearchCV #type:ignore
from sklearn.linear_model import LogisticRegression #type:ignore
from sklearn.ensemble import RandomForestClassifier #type:ignore
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score#type:ignore
from imblearn.over_sampling import RandomOverSampler, SMOTEN #type:ignore
from sklearn.decomposition import PCA #type:ignore

# mlflow.
import mlflow #type:ignore
from mlflow.models import infer_signature #type:ignore
from mlflow.tracking import MlflowClient #type:ignore


# ---- 1. Model and parameter Grid configuration ----
MODELS_CONFIG = {
    'LogisticRegression' : {
        'model' : LogisticRegression(random_state=42, max_iter=2000),
        'params' : {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    },
    'RandomForest' : {
        'model' : RandomForestClassifier(random_state=42),
        'params' : {
            'n_estimators' : [100, 200, 500],
            'max_depth' : [10, 20, 50, 100],
            'criterion' : ['gini', 'entropy']
        }
    }
}

EXPERIMENT_NAME = "LaHolandaPerformance"
REGISTERED_MODEL_NAME = "jurgendhilfe-weltweit"
DATA_PATH = "./cleaned_data/grade_summary.parquet"

# # --- 2. Setting Mlflow client ---
# client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
# client.create_experiment(name=EXPERIMENT_NAME)

# --- 3. Setting logger ----
# Main logger.
logger = logging.getLogger("ModelComparison")
logger.setLevel(
    level=logging.INFO
)
# Format for logger.
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# File handler.
file_handler = RotatingFileHandler(
    "model_comparison.log",
    maxBytes=1024*1024,  # 1MB
    backupCount=5
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Console handler (only INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ---- 4. Data Preparation ----
# Loading data.
def get_train_data(df_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = pd.read_parquet(path=df_path)
    X = data.drop(columns=['band'])
    y = data['band'].to_numpy().ravel()
            
    # Train-test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

def reducing_dimensionality(x_train, x_test) -> tuple:
    """Applies PCA to reduce dimensionality of the data.
    Args:
        X_train (pd.DataFrame): training data
        X_test (pd.DataFrame): test data
    Returns:
        tuple: transformed training and test data
    """
    pca = PCA(n_components=0.8, random_state=42)
    X_train_pca = pca.fit_transform(x_train)
    X_test_pca = pca.transform(x_test)
    
    return X_train_pca, X_test_pca

def main(host="127.0.0.1", port=5000) -> None:
    """Main function to run the model comparison experiments."""

    X_train, X_test, y_train, y_test = get_train_data(DATA_PATH)
    try:        
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
        try:
            client =  MlflowClient(tracking_uri=f"http://{host}:{port}")
            exp = client.get_experiment_by_name(EXPERIMENT_NAME)
            if exp and exp.lifecycle_stage == 'deleted':
                client.restore_experiment(exp.experiment_id)
                logger.info(f"Restored deleted experiment: {EXPERIMENT_NAME}")
        except Exception as e:
            logger.warning(f"Could not restore experiment: {e}")
        
        # # Applying PCA for dimensionality reduction.
        # X_train = X_train.astype(int)
        # pca = PCA(n_components=0.8, random_state=42)
        # X_train_pca = pca.fit_transform(X_train)
        # X_test_pca = pca.transform(X_test)
        X_train_pca, X_test_pca = reducing_dimensionality(X_train, X_test)
        
        # Treating imbalance with SMOTEN.
        smoten = SMOTEN(random_state=42)
        X_train_res, y_train_res = smoten.fit_resample(X_train_pca, y_train)
        
        # Finding best model.
        for model in tqdm(MODELS_CONFIG.keys(), desc="Training models"):
            
            logger.info(f"Running GridSearchCV for model: {model}")
            
            # Calling hyperparam optimizer.
            gcv = GridSearchCV(
                estimator=MODELS_CONFIG[model]['model'],
                param_grid=MODELS_CONFIG[model]['params'],
                scoring='accuracy',
                cv=5,
                n_jobs=-1
            )

            # Training model.
            gcv.fit(X_train_res, y_train_res)
                
            # Predict on the test set.
            y_pred = gcv.predict(X_test_pca)
            # y_pred_proba = gcv.predict_proba(X_test_pca)[:, 1]
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred) # This is the metric to keep track of.
            f1 = f1_score(y_test, y_pred, average='weighted')
            # roc_score = roc_auc_score(y_test, y_pred_proba)
            logger.info(f"Best parameters for {model}: {f1}")
        
            # MLflow experiment setup.
            mlflow.set_tracking_uri(uri="sqlite:///mlflow.db")
            
            # Create a new MLflow Experiment
            mlflow.set_experiment(EXPERIMENT_NAME)
            
            # Start an MLflow run
            with mlflow.start_run() as run:
                
                # Log the hyperparams
                mlflow.log_params(gcv.best_params_)
                
                # Log the loss metric
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1 score", f1)
                # mlflow.log_metric("roc_auc", roc_score)
                
                # Infer the model signature
                signature = infer_signature(X_train_pca, gcv.predict(X_train_pca))
                
                # Setting a tag that we can use to remind this model.
                mlflow.set_tag("Training info", f"{MODELS_CONFIG[model]['model']} for La Holanda analysis.")
                
                # Log the model, which inherits the parameters and metric
                model_info = mlflow.sklearn.log_model(
                    sk_model=gcv,
                    artifact_path="la_holanda_model",
                    signature=signature,
                    input_example=X_train.iloc[:5],
                    registered_model_name=REGISTERED_MODEL_NAME
                )
                
                logger.info(f"Model {model} logged in MLflow with run ID: {model_info.run_id}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
    finally:
        mlflow_process.terminate()
        mlflow_process.wait(timeout=10)
        mlflow_process.kill()

    logger.info("Model comparison experiments completed.")
    logger.info("MLflow server terminated.")
    
if __name__ == "__main__":
    main()