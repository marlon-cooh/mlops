import logging
import pandas as pd
from logging.handlers import RotatingFileHandler
from sklearn.decomposition import PCA #type:ignore
from sklearn.model_selection import train_test_split


def logger_setup(logger_name: str) -> None:
    """Sets up logger for the module."""
    
    # Main logger.
    logger = logging.getLogger()

    if logger.handlers:
        return logger
    
    logger.setLevel(level=logging.INFO) # Capture all levels INFO and above.
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Console -> DEBUG+
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)   
    console_handler.setFormatter(formatter)

    # File -> INFO+
    file_handler = RotatingFileHandler(
        logger_name,
        maxBytes=1024*1024,  # 1MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def get_train_data(df_path: str, col='band') -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = pd.read_parquet(path=df_path)
    X = data.drop(columns=[col])
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