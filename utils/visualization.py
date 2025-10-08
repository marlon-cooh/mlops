import pandas as pd #type:ignore
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #type:ignore

def summary_subject_table(df:pd.DataFrame) -> pd.DataFrame:
    
    """
    This function processes a grades DataFrame into a format suitable for visualization analysis, strictly receiving 
    a cleaned DataFrame from retrieve_grade_reports() function.

    Args:
        df: Ready-to-eda DataFrame after being cleaned through utils/pipeline.py function, retrieve_grade_reports().

    Returns:
        pd.DataFrame: A tidy DataFrame with grades counted by subject, classified in categories [b, B, A, S],
                    ready for plotting visualizations.
    """
    
    # Define categorical columns to plot
    cat_cols = df.iloc[:, 3:].columns.tolist()
    
    # Setting categorical order
    grades_order = ['S', 'A', 'B', 'b']
    for col in cat_cols:
        df[col] = pd.Categorical(df[col], categories=grades_order, ordered=True)
        
    # Creating dfs that store value counts for each subject grade.
    value_counts_dfs = {}
    for subj in df.iloc[:, 3:].columns.tolist():
        value_counts_dfs[subj] = (
                (
                    df.iloc[:, 3:][subj]
                ).value_counts().reset_index().sort_values(by=subj, ascending=False) #straight methods.
        ).rename(
                    columns={
                        subj: f"nota_{subj}",
                        "count" : f"{subj}"
                    }
                 ) # adjusting method.
        
    # Setting strictly given categorical order from grading system.
    fixed = []
    for subj in cat_cols:
        d = value_counts_dfs[subj].copy()
        nota_col = f"nota_{subj}"
        
        # 1. Make grade label the index
        d = d.set_index(nota_col)
        
        # 2. Enforce the same row order for all subjects
        d.index = pd.CategoricalIndex(d.index, categories=grades_order, ordered=True)
        d = d.reindex(grades_order)
        
        # 3. Restore the label column so each pair shows ip
        d = d.rename_axis(nota_col).reset_index()
        fixed.append(d)
        
    # Concatenating each dataframe.
    concatenated = pd.concat(
            objs=fixed,
            axis=1,
            ignore_index=False
        )
    
    # Drop redundant columns.
    concatenated.drop(columns=concatenated.columns[np.arange(2, len(concatenated.columns), 2)], inplace=True)
    concatenated = concatenated.rename(columns={df.columns[0]: "nota"})
    
    return concatenated    

def summary_subject_plot(df:pd.DataFrame, palette=None) -> sns.barplot:
    """
        This function shows up a complete summary bar plot for all subjects, this strictly receives a cleaned dataframe from retrieve_grade_reports() function.
        (Args):
            * df: Ready-to-eda dataframe after being cleaned through utils/pipeline.py function, retrieve_grade_reports().
        (Returns):
            A tidy bar plot visualization for each subject classified by grade [b, B, A, S].
    """
    
    df = summary_subject_table(df)
    
    melted_df = df.melt(
        id_vars='nota',
        var_name='materia'
    )

    sns.barplot(
        data=melted_df,
        x='nota',
        y='value',
        hue='materia',
        palette=palette
    )
    
    plt.show()