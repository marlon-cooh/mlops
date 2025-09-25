#!/usr/bin/env python3

"""
    This script is employed to clean and preprocess text data from HTML files, to obtain DataFrames in Parquet format.
"""

import parsing_html
import logging
import os
import pandas as pd #type:ignore

logging.basicConfig(
    filename="cleaner.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    filemode="w+"
)

logger = logging.getLogger(__name__)

def export_parquet_files(input_html:str, output_filename:str, dir_:os.path):
    """
        This function retrieves pandas DataFrames in parquet format and save them in a directory.
        (Args):
            * input_html (str): Path to the HTML file.
            * output_filename (str): Name of the output parquet file.
            * dir_ (os.path): Directory to save the parquet file.
        (Returns):
            * parquet file in the specified directory.
    """
    try:
        if os.path.exists(dir_):
        # Setting file directory generation.
            parsing_html.full_grades_table(input_html).to_parquet(output_filename)
    except FileNotFoundError as f:
        print(f"{f}: Set a valid directory to export dataframes.")
        logger.error(f)
        
if __name__ == "__main__":
    
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Move up one directory to reach the project root
    project_root = os.path.dirname(current_dir)
    
    # Use absolute paths
    html_dir = os.path.join(project_root, "grades_html")
    output_dir = os.path.join(project_root, "cleaned_data")
    
    input_filenames = [os.path.join(html_dir, file) for file in os.listdir(html_dir)]
    export_filenames = [os.path.join(output_dir, file[:-11] + ".parquet") for file in os.listdir(html_dir)]

    for entrada, salida in zip(input_filenames, export_filenames):
        export_parquet_files(
            input_html=entrada,
            output_filename=salida,
            dir_=output_dir
        )
    logger.info(f"Generated files in: {os.listdir(output_dir)}")

