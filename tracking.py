# from tqdm import tqdm
# import argparse
# import logging
import pandas as pd #type:ignore

# sklearn.
from sklearn.model_selection import train_test_split, GridSearchCV #type:ignore
from sklearn.linear_model import LogisticRegression #type:ignore
from sklearn.ensemble import RandomForestClassifier #type:ignore
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix, accuracy_score #type:ignore
# from sklearn.decomposition import PCA, KernelPCA #type:ignore

# mlflow.
import mlflow #type:ignore
from mlflow.models import infer_signature #type:ignore


# ---- 1. Model and parameter Grid configuration ----
MODELS_CONFIG = {
    'LogisticRegression' : {
        'model' : LogisticRegression(random_state=42, max_iter=2000),
        'params' : {
            # 'model__C' : 0.1,
            # 'model__C' : [0.1, 1, 10],
            # 'model__penalty' : ['l1', 'l2'],
            # 'model__solver' : ['liblinear']
            'penalty' : 'l2',
            'solver' : 'liblinear',
            'random_state' : 42,
            'max_iter' : 2000
        }
    },
    'RandomForest' : {
        'model' : RandomForestClassifier(random_state=42),
        'params' : {
            'model__n_estimators' : [100, 200],
            'model__nmax_depth' : [10, 20],
            'model__criterion' : ['gini', 'entropy']
        }
    }
}

def main() -> None:
    """Main function to run the model comparison experiments."""
    
    # Import data.
    data = pd.read_parquet(path='./cleaned_data/grade_summary.parquet')
    X = data.drop(columns=['band'])
    y = data['band'].to_numpy().ravel()
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    params_lr = MODELS_CONFIG['LogisticRegression']['params']
    
    # Training model.
    lr = LogisticRegression(**params_lr)
    lr.fit(X_train, y_train)
    
    # Predict on the test set.
    y_pred = lr.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred) # This is the metric to keep track of.
    
    # MLflow experiment setup.
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    
    # Create a new MLflow Experiment
    mlflow.set_experiment("MLflow Quickstart")
    
    # Start an MLflow run
    with mlflow.start_run():
        
        # Log the hyperparams
        mlflow.log_params(params_lr)
        
        # Log the loss metric
        mlflow.log_metric("accuracy", acc)
        
        # Infer the model signature
        signature = infer_signature(X_train, lr.predict(X_train))
        
        # Setting a tag that we can use to remind this model.
        mlflow.set_tag("Training info", "Very basic LR model for La Holanda analysis.")
        
        # Log the model, which inherits the parameters and metric
        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="la_holanda_model",
            signature=signature,
            input_example=X_train.iloc[:5],
            registered_model_name="tracking-quickstart"
        )
        

if __name__ == "__main__":
    main()