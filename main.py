# Standard library imports
import os
import warnings

# sklearn libraries
import pandas as pd
from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA #type:ignore
from imblearn.over_sampling import SMOTEN #type:ignore

# Utils libraries
from utils.pipeline import retrieve_grade_reports, process_grades_columns, df_to_model
# from tracking import *
# from registry import *

@task(retries=3, retry_delay_seconds=2,
      name="Returns two clean grade reports for period 1 and period 2.", 
      tags=["cleaned", "classified_by_term", "ready-to-eda_dataframe"])
def grade_reports_cleaned(df:pd.DataFrame, period:str, final_student:int) -> dict:
    dfs = retrieve_grade_reports(df, period, final_student)
    return dfs

@task(retries=3, retry_delay_seconds=2,
      name="Performs a merge between generated dataframes and encodes ordinally to train model.", 
      tags=["cleaned", "merging", "ready-to-ml_dataframe"])
def encoding_and_merging(dfs:dict, to_drop:list) -> pd.DataFrame:
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
    
@flow
def la_holanda_students(input_data:str):
    
    # Processed datafrfames.
    input_data = GRADES
    processed_data = {}
    print("--- Running cleaning and processing pipeline ---")
    
    for config in GRADES:
        # Setting parameters
        grade_name = config['grade']
        path = config['path']
        
        # Running retrieve_grades_reports.
        pr_1 = grade_reports_cleaned(
            path, config['students_p1'], 'P1'
        ).rename(columns={"nat":"qui"})
        pr_2 = grade_reports_cleaned(
            path, config['students_p2'], 'P2'
        ).rename(columns={"nat":"qui"})
        
        # Merging and encoding dataframes.
        merged_pr_1 = encoding_and_merging(pr_1)
        merged_pr_2 = encoding_and_merging(pr_2)
    
        final_dataset = data_to_model([merged_pr_1, merged_pr_2])
        processed_data[grade_name] = final_dataset
    
    return processed_data

if __name__ == "__main__":
    # Paths
    GRADES = [
    # {'grade': '9_2', 'path': '../consolidados/consolidado_902.xls', 'students_p1': 95, 'students_p2': 95},
    {'grade': '10_1', 'path': '../consolidados/consolidado_1001.xls', 'students_p1': 81, 'students_p2': 81},
    {'grade': '10_2', 'path': '../consolidados/consolidado_1002.xls', 'students_p1': 81, 'students_p2': 81},
    {'grade': '10_3', 'path': '../consolidados/consolidado_1003.xls', 'students_p1': 85, 'students_p2': 85},
    {'grade': '10_4', 'path': '../consolidados/consolidado_1004.xls', 'students_p1': 82, 'students_p2': 83},
    {'grade': '11_1', 'path': '../consolidados/consolidado_1101.xls', 'students_p1': 81, 'students_p2': 81},
    {'grade': '11_2', 'path': '../consolidados/consolidado_1102.xls', 'students_p1': 79, 'students_p2': 79},
    {'grade': '11_3', 'path': '../consolidados/consolidado_1103.xls', 'students_p1': 81, 'students_p2': 81},
    ]
    data = la_holanda_students("path_to_data")
    print(data)