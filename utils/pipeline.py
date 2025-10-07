import pandas as pd #type:ignore
from pathlib import Path
import logging
from re import search

# Preprocessing.
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer #type:ignore
from sklearn.compose import ColumnTransformer #type:ignore
from sklearn.pipeline import Pipeline #type:ignore
from sklearn.impute import SimpleImputer #type:ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

subjects = ['codigo', 'no_lista', 'nombre', 'periodo', 'lect', 'esp', 'mat', 'econ', 'ingl', 'nat', 'fis', 'filo', 'poli', 'ere', 'edufi', 'tecn', 'compo', 
            # 'electiva' # Not considered so far.
]

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
        This function is designed for those dataframes obtained from .parquet files after reading and cleaning HTMLs (look parsing_html.py).
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

def retrieve_grade_reports(inpath:str, cols_to_present=subjects, final_student=95) -> dict:
    """
        This function returns a complete, cleaned, and ready-to-eda dataframe from grade reports taken in .xls format from database.
        (Args):
            * inpath: Directory where .xls file is stored,
            * cols_to_present: List of columns to display in the final dataframe, for instance, subjects as: 'nat' (Natural sciences), 'esp' (Spanish), and so on. 
        Returns:
            A dictionary of dataframes suitable to eda and train simple models.
    """
    
    # Reading .xsl files
    inpath = Path(inpath)
    try:
        if inpath.exists():
            tables = pd.read_html(
                inpath,
                attrs={"id": "consolidadonotas_periodo_tabla"}
            )
    except (FileExistsError, UnicodeDecodeError) as fe:
        print(f"{fe}: Inpath is not valid and/or does not contain a readable .xls file")
    
    # Removing multiindex.    
    df = tables[0]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    
    # Final student (this feature should be arranged).
    df.loc[:final_student, :].columns
    
    # Removing retired students.
    df['Nombre'] = df['Nombre'].astype('str')
    
    indexes_to_drop = df[df['Nombre'].apply(lambda x: bool(search(r"\t", x)))].index.to_list()
    df.drop(index=indexes_to_drop, axis=0, inplace=True)
     
    # Cleaning information.
    level_grades = df.clean_names(
        case_type='snake',
        strip_underscores=True,
        remove_special=True        
    ).loc[:final_student, cols_to_present].reset_index().rename(columns={'index':'idx'})
    
    level_grades = level_grades.dropna(
        axis=0,
        subset=level_grades.columns[2:],
        how='all'
    )
    
    # Creating dataframes for periods P1 and P2.
    
    level_grades_p1 = level_grades[level_grades['idx'] %2 == 0]
    level_grades_p2 = level_grades[level_grades['idx'] %2 != 0]
    
    level_grades_p2['idx'] -= 1
    level_grades_p2 = level_grades_p2.merge(
        level_grades_p1,
        on='idx',
        how='inner',
        suffixes=("_p2", "_p1")
    )[['idx', 'codigo_p1', 'nombre_p1', 'esp_p2', 'mat_p2', 'econ_p2', 'ingl_p2', 'nat_p2', 'fis_p2', 'filo_p2', 'poli_p2', 'ere_p2', 'edufi_p2', 'tecn_p2', 'compo_p2']].rename(
    columns={
        'codigo_p1' : 'codigo',
        'nombre_p1' : 'nombre',
        'esp_p2': 'esp',
        'mat_p2': 'mat',
        'econ_p2': 'econ',
        'ingl_p2': 'ingl',
        'nat_p2': 'nat',
        'fis_p2': 'fis',
        'filo_p2': 'filo',
        'poli_p2': 'poli',
        'ere_p2': 'ere',
        'edufi_p2': 'edufi',
        'tecn_p2': 'tecn',
        'compo_p2': 'compo'
    }
)
    dfs = {"p1" : level_grades_p1, "p2" : level_grades_p2}
    return dfs
    
if __name__ == "__main__":
    
    retrieve_processed_dataframes(
        inpath="../cleaned_data",
        outpath="../processed_data"
    )
