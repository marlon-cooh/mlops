# Standard library imports
from datetime import datetime
from tqdm import tqdm

# sklearn libraries
import pandas as pd
import mlflow
from mlflow.models import infer_signature
from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA #type:ignore
from imblearn.over_sampling import SMOTEN #type:ignore
from sklearn.metrics import accuracy_score, f1_score #type:ignore
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression #type:ignore
from sklearn.ensemble import RandomForestClassifier #type:ignore

# Utils libraries
from utils.pipeline import retrieve_grade_reports, process_grades_columns, df_to_model
from utils.log_data import logger_setup
from registry import best_experiment, test_model_from_mlflow

@task(retries=3, retry_delay_seconds=2,
      name="Returns two clean grade reports for period 1 and period 2.", 
      tags=["cleaned", "classified_by_term", "ready-to-eda_dataframe"])
def grade_reports_cleaned(path:str, final_student:int, **kwargs) -> dict:
    dfs = retrieve_grade_reports(path, final_student, **kwargs)
    return dfs

@task(retries=3, retry_delay_seconds=2,
      name="Performs a merge between generated dataframes and encodes ordinally to train model.", 
      tags=["cleaned", "merging", "ready-to-ml_dataframe"])
def encoding_and_merging(dfs:dict, to_drop:list=[]) -> pd.DataFrame:
    merged_df = process_grades_columns(dfs, to_drop)
    return merged_df

@task(retries=3, retry_delay_seconds=2,
      name="Returns dataset ready for model training, considering adjusted labels: LOW, MEDIUM, HIGH.", 
      tags=["labeling", "ready-to-train"])
def data_to_model(input_dfs:list):
    ready_dataset = df_to_model(input_dfs)
    return ready_dataset

@task(retries=3, retry_delay_seconds=2,
      name="Split data from ready_dataset.", 
      tags=["imbalance", "pca", "ready-to-train"])
def split_data(df:list, col:list = 'band'):
    # Implementation for refining train data goes here
    X = df.drop(columns=[col])
    y = df[col].to_numpy().ravel()
            
    # Train-test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

@task(retries=3, retry_delay_seconds=2,
      name="Train models with hyperparameter tuning and dimensionality reduction.", 
      tags=["model_training", "hyperparameter_tuning", "pca"])
def reduce_and_balance(
        X_train, 
        X_test, 
        y_train, 
        y_test, 
        consider_imbalance:bool = True
    ):
    """Reduces dimensionality and treats class imbalance if specified."""
    pca = PCA(n_components=0.8, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    if consider_imbalance:
        # Treating imbalance with SMOTEN.
        smoten = SMOTEN(random_state=42)
        X_train_res, y_train_res = smoten.fit_resample(X_train_pca, y_train)
        return X_train_res, X_test_pca, y_train_res, y_test
    else:
        return X_train_pca, X_test_pca, y_train, y_test

@task(retries=3, retry_delay_seconds=2,
      name="Run a list of experiments in MLflow for model tracking.", 
      tags=["mlflow", "model_tracking", "experimentation"])
def run_experiments_in_mlflow(train_info:tuple, exp_list:dict, reg_model:str) -> None:
    """Runs experiments in MLflow."""
    # Implementation for running experiments in MLflow goes here
    X_train_pca, X_test_pca, y_train, y_test = train_info
    
    # Finding best model.
    for experiment_key, exp in tqdm(exp_list.items(), desc="Training models"):
        
        experiment_name = exp["experiment_name"]
    
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
        estimator.fit(X_train_pca, y_train)

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

@task(retries=3, retry_delay_seconds=2,
      name="Register best model respect to a selected metric (i.e. f1, roc-auc, etc.) from a list of experiments in Mlflow.", 
      tags=["main_flow", "best_model", "promotion"])
def best_experiment(tracking_uri:str, metric:str = 'f1 score') -> tuple:
    client, best_run_id = best_experiment(tracking_uri, metric)
    return client, best_run_id

@task(retries=3, retry_delay_seconds=2,
      name="Promote best model to Production stage in MLflow Model Registry.", 
      tags=["model_promotion", "mlflow_registry", "production"])
def promote_model(
        client, 
        best_run_id:str, 
        model_name:str, 
        version:int, # This param will be adjusted in the upcoming edits.
        stage:str = "Production"
    ):
    """
        this function promotes the best model to production.
    Args:
        * client (mlflow.tracking.client.MlflowClient): mlflow tracking client,
        * best_run_id: tag pointing run with best performance according to usage in `best_experiment()` function,
        * model_name (str): model name,
        * version (int): model version,
        * stage (str): Model stage, one of:
            - "None": Initial stage (default)
            - "Staging": Model in staging/testing phase
            - "Production": Model in production
            - "Archived": Model archived/deprecated
        
    """
    
    # Registering best model.
    mlflow.register_model(model_uri=f"runs:/{best_run_id}/la_holanda_model",
        name=model_name
    )
    latest_versions = client.get_latest_versions(name=model_name)
    for version_ in latest_versions:
        if version_.version == str(version):
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=True
            )
            logger.info(f"Model {model_name} version {version} promoted to {stage} stage.")
    
    return f"The model version {version_.version} was transitioned to Production on {datetime.today().date()}"

@task(retries=3, retry_delay_seconds=2,
      name="Test promoted model to production stage in MLflow Model Registry.", 
      tags=["model_testing", "mlflow_registry", "production"])
def test_promoted_model(model_name:str, stage:str, X_test, y_test) -> str:
    f1_score_description = test_model_from_mlflow(model_name, stage, X_test, y_test)
    return f1_score_description

@flow
def la_holanda_students(input_data:list) -> dict:
    
    # Processed datafrfames.
    processed_data = {}
    print("--- Running cleaning and processing pipeline ---")
    
    for config in input_data:
        # Setting parameters
        grade_name = config['grade']
        path = config['path']
        
        # Running retrieve_grades_reports.
        pr_1 = grade_reports_cleaned(
            path=path, final_student=config['students_p1'], period='P1'
        )['p1'].rename(columns={"nat":"qui"})
        pr_2 = grade_reports_cleaned(
            path=path, final_student=config['students_p2'], period='P2'
        )['p2'].rename(columns={"nat":"qui"})
        # Merging and encoding dataframes.
        merged_pr_1 = encoding_and_merging(pr_1)
        merged_pr_2 = encoding_and_merging(pr_2)
    
        final_dataset = data_to_model([merged_pr_1, merged_pr_2])
        processed_data[grade_name] = final_dataset
    
    df = pd.concat(
        objs = processed_data.values(),
        axis = 0
    ).select(
    'lect', 'esp', 'ingl', 'mat', 'qui', 'fis', 'filo', 'econ', 'poli', 'tecn', 'edufi', 'ere', 'compo', 'fundamental', 'band'
    )
    
    return df

@flow
def select_and_promote(tracking_uri:str, model_name:str, version:int, metric:str = 'f1 score') -> None:
    """Selects and promotes the best model to Production stage in MLflow Model Registry."""
    
    client, best_run_id = best_experiment(tracking_uri, metric)
    logger.info(f"{client} was successfully opened, with best experiment tag as {best_run_id}")
    
    # Stage settings.
    stage = "Production"
    
    response = promote_model(
        client=client,
        best_run_id=best_run_id, 
        model_name=model_name, 
        version=version, 
        stage=stage
    )
    
    logger.info(response)
    

if __name__ == "__main__":
    # Tracking URI
    track_uri = "127.0.0.1:5000"
    
    # Setting up logger
    logger = logger_setup('process_track.log')

    # Paths
    GRADES = [
    # {'grade': '9_2', 'path': './consolidados/consolidado_902.xls', 'students_p1': 95, 'students_p2': 95},
    {'grade': '10_1', 'path': './consolidados/consolidado_1001.xls', 'students_p1': 81, 'students_p2': 81},
    {'grade': '10_2', 'path': './consolidados/consolidado_1002.xls', 'students_p1': 81, 'students_p2': 81},
    {'grade': '10_3', 'path': './consolidados/consolidado_1003.xls', 'students_p1': 85, 'students_p2': 85},
    {'grade': '10_4', 'path': './consolidados/consolidado_1004.xls', 'students_p1': 82, 'students_p2': 83},
    {'grade': '11_1', 'path': './consolidados/consolidado_1101.xls', 'students_p1': 81, 'students_p2': 81},
    {'grade': '11_2', 'path': './consolidados/consolidado_1102.xls', 'students_p1': 79, 'students_p2': 79},
    {'grade': '11_3', 'path': './consolidados/consolidado_1103.xls', 'students_p1': 81, 'students_p2': 81},
    ]
    
    # Experiments
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
    
    data = la_holanda_students(GRADES)
    X_train, X_test, y_train, y_test = split_data(data)
    X_train_res, X_test_pca, y_train_res, y_test = reduce_and_balance(
        X_train, X_test, y_train, y_test, consider_imbalance=True
    )
    
    logger.info(f"Data has been successfully generated {data.shape}, with training set shape {X_train_res.shape} and test set shape {X_test_pca.shape}.")
    
    # Setting MLflow tracking URI
    mlflow.set_tracking_uri(uri="sqlite:///mlflow.db")
    run_experiments_in_mlflow(
        train_info=(X_train_res, X_test_pca, y_train_res, y_test),
        exp_list=EXPERIMENTS,
        reg_model=REGISTERED_MODEL_NAME
    )
    
    logger.info(f"Model {REGISTERED_MODEL_NAME} has been registered.")
    
    # Promoting to production best model
    select_and_promote(tracking_uri=track_uri, model_name=REGISTERED_MODEL_NAME, version=3, metric='f1 score')
    
    # Testing promoted model
    f1_description = test_promoted_model(
        model_name=REGISTERED_MODEL_NAME, 
        stage="Production",
        X_test=X_test_pca,
        y_test=y_test
    )
    logger.info(f1_description)