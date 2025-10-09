import pandas as pd #type:ignore
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
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
    
    # Drop redundant columns and `esc_pad`.
    concatenated.drop(columns=concatenated.columns[np.arange(2, len(concatenated.columns), 2)], inplace=True)
    
    # Rename the first column to 'nota'
    concatenated = concatenated.rename(columns={concatenated.columns[0]: "nota"})
    
    return concatenated    

def summary_subject_plot(
        df:pd.DataFrame, 
        palette=None, 
        size=(10, 6),
        ax: Axes | None = None,
        show_legend : bool = True,
        cols: list = []
        ) -> sns.barplot:
    """
    Plots a summary bar plot for each subject classified by grade [b, B, A, S].

    Args:
        df: Cleaned dataframe expected by summary_subject_table().
        palette: Optional seaborn/matplotlib palette.
        size: Figure size if ax is not provided.
        ax: Existing matplotlib Axes to draw on. If None, a new fig/ax is created.
        show_legend: Whether to show legend on this axes.

    Returns:
        The matplotlib Axes used for plotting.
    """
    # To control if there are some specific cols to analyse.
    if len(cols) > 0:
        df = summary_subject_table(df)[cols]
    else:
        df = summary_subject_table(df)
    
    melted_df = df.melt(
        id_vars='nota',
        var_name='materia'
    )
    
    # Plot settings.
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=size, dpi=110)
        created_fig = True
    
    sns.barplot(
        data=melted_df,
        x='nota',
        y='value',
        hue='materia',
        palette=palette,
        ax=ax
    )
    
    # To control legend display
    if show_legend:
        ax.legend(loc='best')
    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
      
    if created_fig:
        plt.tight_layout()
        
    return ax
