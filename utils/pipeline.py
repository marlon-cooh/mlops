import pandas as pd #type:ignore
from pathlib import Path
import logging

# Preprocessing.
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer #type:ignore
from sklearn.compose import ColumnTransformer #type:ignore
from sklearn.pipeline import Pipeline #type:ignore
from sklearn.impute import SimpleImputer #type:ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retrieve_processed_dataframes(inpath:str, outpath:str) -> pd.DataFrame:
    """
        This function returns a dataframe suitable for training ML models in the context of this project (Grades analysis).
        (Args):
            * inpath : Path associated to .parquet files after reading and cleaning HTMLs.
            * outpath : Path where .csv files will be saved to train upcoming models.
        Returns:
            CSV files including data for ML stage.
    """
    # Defining path
    if inpath is None or outpath is None:
        raise ValueError("Input and output paths must be provided.")
    
    base = Path(inpath)
    outpath = Path(outpath)
    paths = sorted(base.glob("*.parquet"))
    
    # Dictionary of dataframes
    dfs = {p.stem : pd.read_parquet(p) for p in paths}
    
    # Columns to drop
    cols_to_drop = ['Competencia', 'OBS  4', 'OBS  5', 'OBS  2', 'Rec PF', 'OBS  1',
       'Nota P3', 'Rec P3', 'OBS  3', 'Nota P2']
    
    # Columns to preprocess
    cat_cols = ['CONOCER', 'HACER', 'SER', 'CONVIVIR', 'Subtotal NIVEL']
    rec_cols = ['Rec P1', 'Rec P2']
    
    # Transforming missing values from rec_cols
    rec_make_flags = FunctionTransformer(
            func=lambda X: X.notna().astype(int),
            feature_names_out="one-to-one"
        )
    
    # Creating pipelines
        # Categorical data pipeline
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy='constant', fill_value=1)), 
            ("ohe", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ] 
    )

    # `Recuperacion` data pipeline
    rec_pipe = Pipeline(
        [
            ("impute", rec_make_flags)
        ]
    )
    
    # ColumnTransformer including all pipelines.
    pre = ColumnTransformer(
        transformers=[
            ('cat', cat_pipe, cat_cols),
            ('rec', rec_pipe, rec_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=True
    )
    
    logger.info(f"Preprocessing methods are successfully defined. {pre}")
    
    # Adjusting data by using ColumnTransformer
    for names, df in dfs.items():

        try:
           # Logging dataframes names
           logger.info(f"Processing dataframe: {names}")
           
           X = df.drop(columns=cols_to_drop)
           y = df[["Nota P2"]] # Target variable. (So far.)
           
           transformed = pre.fit_transform(X)
           pre.set_params(verbose_feature_names_out=False)
           pre.get_feature_names_out()
           
           transformed_df = pd.DataFrame(
               data=transformed,
               index=df.index,
               columns=pre.get_feature_names_out()
           )
           
           transformed_df["Nota P2"] = y
           
           # Logging transformed columns
           logger.info(f"Transformed dataframe columns: {transformed_df.columns.tolist()}")
           
           output_file = outpath.joinpath(f"{names}.csv")
           
           # Existing directory?
           if not outpath.exists():
               outpath.mkdir(parents=True, exist_ok=True)
               
           transformed_df.to_csv(output_file, index=False)
           logger.info(f"Saved processed dataframe to: {output_file}")
        
        except Exception as e:
           logger.error(f"Error processing dataframe {names}: {e}")

def remove_unregistered_students(raw_df:pd.DataFrame) -> pd.DataFrame:
    """
        Cleans the raw dataframe by removing unnecessary columns and rows that do not contain grading information along with students that are not listed in the courses.
        Args:
            raw_df (pd.DataFrame): The raw dataframe to be cleaned.
        Returns:
            pd.DataFrame: The cleaned dataframe.    
    """
    # Dropping unnecessary columns.
    
    # Unnecessary columns are those that do not relate to grading.
    cols_to_drop = ['Competencia', 'OBS  1', 'OBS  2', 'OBS  3', 'OBS  4', 'OBS  5']
    
    # Keeping only the first 13 columns that relate to grading.
    cleaned_df = raw_df.drop(columns=cols_to_drop).iloc[:, :13]
    
    # # Replacing "None" and pd.NA values with NaN, then dropping rows that are completely empty.
    cleaned_df = cleaned_df.replace({"None", pd.NA}, inplace=False).reset_index().rename(columns={'index' : 'ESTUDIANTE'})

    # Resetting index and dropping completely empty rows, then creating an auxiliar foreign key.
    cleaned_df = cleaned_df.reset_index().rename(columns={'index' : 'ID'})
    cleaned_df = cleaned_df.dropna(how="all", subset=cleaned_df.columns[2:13], inplace=False).dropna(subset=cleaned_df.columns[9], how="all")
    
    # Merge to non-grading relative columns.
    to_merge = raw_df.iloc[:, 19:].reset_index().rename(columns={'index' : 'ESTUDIANTE'}).reset_index().rename(columns={'index' : 'ID'})
    merged = cleaned_df.merge(
        to_merge,
        on=[
            'ID', 
            'ESTUDIANTE'
        ],
        how='inner'
    )
    
    # Columns where missing values must be binarized.
    cols = ['Rec P1', 'Rec P2', 'Rec P3', 'Rec PF']
    merged[cols] = merged[cols].notna().astype("Int64")
    
    # Transforming columns to categorical dtype.
    order = ["S", "A", "B", "b"]
    cat_cols = ['CONOCER', 'HACER', 'SER', 'CONVIVIR', 'Subtotal NIVEL', 'Nota P1', 'Nota P2', 'Nota P3', 'Nota PF']
    merged[cat_cols] = merged[cat_cols].replace(
        {"None" : pd.NA, "": pd.NA}
    ).apply(
        lambda s: s.str.strip() if s.dtype =="object" else s
    ).apply(
        lambda s : pd.Categorical(s, categories=order, ordered=True)
    )

    return merged

if __name__ == "__main__":
    
    retrieve_processed_dataframes(
        inpath="../cleaned_data",
        outpath="../processed_data"
    )
