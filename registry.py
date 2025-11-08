import mlflow #type:ignore
import logging
from scipy.sparse._csr import csr_matrix #type:ignore
from mlflow.tracking import MlflowClient #type:ignore
from datetime import datetime
from logging.handlers import RotatingFileHandler
from tracking import get_train_data, DATA_PATH


# ---- 1. Configuration ----
logger = logging.getLogger("ModelRegistry")
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()   
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = RotatingFileHandler(
    "model_registry.log",
    maxBytes=1024*1024,  # 1MB
    backupCount=5
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

EXPERIMENT_NAME = "LaHolandaPerformance"
MODEL_NAME = "jurgendhilfe-weltweit"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

# ---- 2. Registering the best model ----
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