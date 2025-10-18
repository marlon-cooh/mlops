import pandas as pd #type:ignore
from pathlib import Path
import logging
import janitor
from re import search
from numpy import float64

# Preprocessing.
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer #type:ignore
from sklearn.compose import ColumnTransformer #type:ignore
from sklearn.pipeline import Pipeline #type:ignore
from sklearn.impute import SimpleImputer #type:ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

subjects_9 = ['codigo', 'no_lista', 'nombre', 'periodo', 'esp', 'ingl', 'edufi', 'art', 'soc', 'ere', 'mat', 'nat', 'tecn','esc_pad', 'compo']
subjects_10 = ['codigo', 'no_lista', 'nombre', 'periodo', 'lect', 'esp', 'mat', 'econ', 'ingl', 'nat', 'fis', 'filo', 'poli', 'ere', 'edufi', 'tecn', 'compo']
subjects_11 = ['codigo', 'no_lista', 'nombre', 'periodo', 'lect', 'esp', 'mat', 'econ', 'ingl', 'qui', 'fis', 'filo', 'poli', 'ere', 'edufi', 'tecn', 'compo']

def retrieve_processed_dataframes(inpath:str, outpath:str) -> pd.DataFrame:
    """
        (*To be deprecated)
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

def clean_level_grades(df: pd.DataFrame, final_student: int, cols_to_present: list) -> pd.DataFrame:
    """Clean and format grade level data.
    
    Args:
        df: Input DataFrame containing grade information
        final_student: Last student index to include
        cols_to_present: List of columns to keep in output
    
    Returns:
        pd.DataFrame: Cleaned and formatted grade data
    """
    return (df.clean_names(
        case_type='snake',
        strip_underscores=True,
        remove_special=True        
    ).loc[:final_student, cols_to_present].reset_index().rename(columns={'index':'idx'})
    )

def safe_drop(df:pd.DataFrame, colname:str) -> pd.DataFrame:
    return df.drop(columns=[colname], errors='ignore')

def retrieve_grade_reports(inpath:str, cols_to_present=None, final_student=95, **kwargs) -> dict:
    """
        This function returns a complete, cleaned, and ready-to-eda dataframe from grade reports taken in .xls format from database.
        (Args):
            * inpath: Directory where .xls file is stored,
            * cols_to_present: List of columns to display in the final dataframe, for instance, subjects as: 'nat' (Natural sciences), 'esp' (Spanish), and so on. 
        Returns:
            A dictionary of dataframes suitable to eda and train simple models.
    """
    
    # Reading .xsl files
    inpath_ = Path(inpath)
    try:
        if inpath_.exists():
            tables = pd.read_html(
                inpath_,
                attrs={"id": "consolidadonotas_periodo_tabla"}
            )
    except (FileExistsError, UnicodeDecodeError) as fe:
        print(f"{fe}: Inpath is not valid and/or does not contain a readable .xls file")
    
    # Removing multiindex.    
    try:
        df = tables[0]
    except UnboundLocalError as u:
        print(f"Inpath is not valid and/or does not contain a readable .xls file \n {u}")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    
    # Removing retired students.
    df['Nombre'] = df['Nombre'].astype('str')
    
    indexes_to_drop = df[df['Nombre'].apply(lambda x: bool(search(r"\t", x)))].index.to_list()
    df.drop(index=indexes_to_drop, axis=0, inplace=True)
     
    # Cleaning information.
    try:
        if search(r"11(?=0)", inpath):
            cols_to_present = subjects_11
        elif search(r"10(?=0)", inpath):
            cols_to_present = subjects_10
        elif search(r"9(?=0)", inpath):
            cols_to_present = subjects_9
        else:
            raise ValueError("File name must contain grade level (9, 10, or 11)")
           
        level_grades = clean_level_grades(df, final_student, cols_to_present)
    
    except (KeyError) as ke:
        print(f"Column error: {ke}. Check if columns match grade level.")
        
    level_grades = level_grades.dropna(
        axis=0,
        subset=level_grades.columns[2:],
        how='all'
    )
    
    # To segment by period.
    allowed_periods = {"P1", "P2", "P3"}

    for key, value in kwargs.items():
        if isinstance(value, str) and value in allowed_periods:
            level_grades[f'{key}'] = value.strip()
        else:
            raise ValueError(
                "Additional parameters MUST be period indicators: 'P1', 'P2', or 'P3'."
            )
                
    # Creating dataframes for periods P1 and P2.
    
    level_grades_p1 = level_grades[level_grades['idx'] %2 == 0].drop(columns={'periodo', 'no_lista'}, axis=1)
    level_grades_p2 = level_grades[level_grades['idx'] %2 != 0]
    
    # Assigning columns depending on selected level.
    new_labels = ['idx', 'codigo_p1', 'nombre_p1']
    new_labels += [elem + "_p2" for elem in level_grades.columns[5:]]
    
    # Base dict of columns_to_replace
    columns_to_replace = {
        'codigo_p1' : 'codigo',
        'nombre_p1' : 'nombre'
    }
    
    columns_to_replace.update(
        {
            x : x.removesuffix("_p2").removesuffix("_p1") for x in new_labels
        }
    )
    
    # Creating p2 report.
    level_grades_p2['idx'] -= 1
    level_grades_p2 = level_grades_p2.merge(
            level_grades_p1,
            on='idx',
            how='inner',
            suffixes=("_p2", "_p1")
        )[new_labels].rename(
        columns=columns_to_replace
    )
            
    # Removing unnecessary columns
    if 'esc_pad' in level_grades_p1.columns or 'esc_pad' in level_grades_p2.columns:
        level_grades_p1 = safe_drop(level_grades_p1, 'esc_pad')
        level_grades_p2 = safe_drop(level_grades_p2, 'esc_pad')
        
    dfs = {
            "p1" : level_grades_p1, 
            "p2" : level_grades_p2
        }
        
    return dfs

def process_grades_columns(df:pd.DataFrame, cols_to_drop:list = []) -> pd.DataFrame:
    """
    This function returns an ordinal encoded version of grades in student report, this version is useful to create visualization or train simple ML models.
        Args:
        df: Ready-to-eda DataFrame after being cleaned through utils/pipeline.py function, retrieve_grade_reports().
        
        Returns:
        df: Refined eda DataFrame.
    """
    # Rearranging columns
    # Filtering columns
    if 'qui' not in set(df.columns.tolist()) and 'lect' in set(df.columns.tolist()):
        df = df[['idx', 'codigo', 'nombre', 'period', 'lect', 'esp', 'mat', 'econ', 'ingl', 'nat', 'fis', 'filo', 'poli', 'ere', 'edufi', 'tecn', 'compo']]
    
    elif 'nat' not in (df.columns.tolist()):
        df = df[['idx', 'codigo', 'nombre', 'period', 'lect', 'esp', 'mat', 'econ', 'ingl', 'qui', 'fis', 'filo', 'poli', 'ere', 'edufi', 'tecn', 'compo']]
    
    elif 'qui' not in set(df.columns.tolist()) and 'lect' not in set(df.columns.tolist()):
        df = df[['idx', 'codigo', 'nombre', 'period', 'esp', 'ingl', 'edufi', 'art', 'soc', 'ere', 'mat', 'nat', 'tecn', 'compo']]
    
    # If there are columns to drop
    df.drop(columns=[], inplace=True)
    
    # Categorical variables
    cols_to_classify = df.columns[4:].tolist()
    categories_per_col = [['b', 'B', 'A', 'S']] * len(cols_to_classify)
    
    # Ordinal encoder pipeline.
    cat_ord_pipe = Pipeline(
    steps=[
            ("imputer", SimpleImputer(strategy='most_frequent')),
            ("ordinal", OrdinalEncoder(
            categories=categories_per_col,
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            dtype=float64
            ))
        ]
    )
    
    # CT for ordinal pipeline.
    pre = ColumnTransformer(
    transformers=[
            ('cat', cat_ord_pipe, cols_to_classify)
        ],
        remainder='drop',
        verbose_feature_names_out=True
    )
    
    pre.set_output(transform='pandas')
    processed_df = pre.fit_transform(df)
    
    # Removing added label (cat_) in processed_df
    processed_df.columns = [c.split("__", 1)[-1] for c in processed_df.columns]
    
    processed_df = pd.concat(
        [df.iloc[:, :4], processed_df],
        axis=1,
        ignore_index=False
    )
    
    return processed_df

def df_to_model(input_dfs:list, expose_band:bool=False) -> pd.DataFrame:
    """
    Prepares student grade DataFrames for machine learning model training by computing performance metrics
    and optionally categorizing students into performance bands.

    Args:
        input_dfs (list): List of pandas DataFrames containing student grades. Each DataFrame should:
            - Have student identifiers in columns 0-3
            - Contain subject grades in columns 4+
            - Be pre-processed using retrieve_grade_reports() and process_grades_columns()
        expose_band (bool, optional): If True, adds performance bands (LOW/MEDIUM/GOOD/EXCELLENT) 
            based on percentile ranks. Defaults to False.

    Returns:
        pd.DataFrame: A concatenated DataFrame with additional features:
            - performance: Sum of grades across all subjects
            - fundamental: Sum of grades in core subjects (math, spanish, critical reading)
            - band: (if expose_band=True) Categorical performance level
            - Filtered to include only relevant subject columns based on grade level

    Features:
        - Automatically detects available subjects and adapts column selection
        - Handles missing core subjects when calculating fundamentals
        - Creates ordered categorical bands using percentile thresholds:
            * EXCELLENT: >= 90th percentile
            * GOOD: >= 70th percentile
            * MEDIUM: >= 40th percentile 
            * LOW: < 40th percentile

    Example:
        >>> p1_df = process_grades_columns(retrieve_grade_reports("grade10.xls")["p1"])
        >>> p2_df = process_grades_columns(retrieve_grade_reports("grade10.xls")["p2"]) 
        >>> model_df = df_to_model([p1_df, p2_df], expose_band=True)
    """
    input_dfs = [df.copy() for df in input_dfs]
    
    # Assigning 'performance' values (sum of transformed grades of every subject), this is OPTIONAL.
    for df in input_dfs:
        df['performance'] = df[df.columns[4:].tolist()].sum(axis=1)
        
    # Assigning 'fundamental' values (sum of transformed grades in español, matemáticas, lectura crítica), this is OPTIONAL
    fund_cols = ["mat", "esp", "lect"]
    existing_cols = [col for col in fund_cols if col in input_dfs[0].columns]
    
    if not existing_cols:
        print("⚠️ No matching columns found for 'fundamental'. Skipping creation.")
    else:
        for df in input_dfs:
            df['fundamental'] = df[existing_cols].sum(axis=1)
        
    # Creating a band category based on performance. 
    if expose_band == True:
        
        for df in input_dfs:
            df['performance_pct'] = df.performance.rank(pct=True, axis=0, ascending=True) # Ranking performance
            df['band'] = df['performance_pct'].apply(
                lambda x: (
                    'EXCELLENT' if x >= 0.90 else
                    'GOOD'      if x >= 0.70 else
                    'MEDIUM'    if x >= 0.40 else
                    'LOW'
                ) if pd.notnull(x) else pd.NA
            )
            
            students_order = ['LOW', 'MEDIUM', 'GOOD', 'EXCELLENT'] # Processing `band` column
            df.band = pd.Categorical(
                        values=df.band,
                        categories=students_order,
                        ordered=True
                )
    
    # Creating a wrapped df that contains every cleaned df.
    wrapped_df = pd.concat(
        objs = [df for df in input_dfs],
        axis = 0
    )
    
    # Filtering columns
    if 'qui' not in set(wrapped_df.columns.tolist()) and 'lect' in set(wrapped_df.columns.tolist()):
        wrapped_df.select_columns(
                  'idx', 'codigo', 'nombre', 'period', 'lect', 'esp', 'mat', 'econ', 'ingl', 'nat', 'fis', 'filo', 'poli', 'ere', 'edufi', 'tecn', 'compo', 'fundamental','band'
        )
    elif 'nat' not in (wrapped_df.columns.tolist()) and 'lect' in set(wrapped_df.columns.tolist()):
        wrapped_df.select_columns(
        'idx', 'codigo', 'nombre', 'period', 'lect', 'esp', 'mat', 'econ', 'ingl', 'qui', 'fis', 'filo', 'poli', 'ere', 'edufi', 'tecn', 'compo', 'fundamental','band'
    )
    elif 'qui' not in set(wrapped_df.columns.tolist()) and 'lect' not in set(wrapped_df.columns.tolist()):
        wrapped_df.select_columns(
        'idx', 'codigo', 'nombre', 'period', 'esp', 'ingl', 'edufi', 'art', 'soc', 'ere', 'mat', 'nat', 'tecn', 'compo' ,'fundamental','band'
    )
    
    return wrapped_df

if __name__ == "__main__":
    
    retrieve_processed_dataframes(
        inpath="../cleaned_data",
        outpath="../processed_data"
    )
