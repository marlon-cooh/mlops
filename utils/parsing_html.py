from bs4 import BeautifulSoup
from numpy import array #type:ignore
import pandas as pd #type:ignore
import logging
from io import StringIO

logging.basicConfig(
    level=logging.INFO,
    filename='parsing_html.log',
    filemode='w+',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# SIGNATURE COLUMN NAMES
STUDS_TABLE_COLS = [
       'Competencia',
       'CONOCER', 
       'HACER',
       'SER', 
       'CONVIVIR',
       'Subtotal NIVEL', 
       'Nota P1', 
       'Rec P1',
       'Nota P2', 
       'Rec P2',
       'Nota P3',
       'Rec P3', 
       'Nota PF', 
       'Rec PF',
       'OBS  1',
       'OBS  2',
       'OBS  3',
       'OBS  4',
       'OBS  5',
       'LLEGADA TARDE', 
       'INASISTENCIA JUSTIFICADA',
       'INASISTENCIA INJUSTIFICADA',
       'PERMISO', 
       'RETARDO']

def prettify_html(html_file:str) -> None:
    """
        Prettify HTML file.
        Args:
            html_file (str): Path to the HTML file.
        Returns:
            None
    """
    with open(html_file, "r", encoding="utf-8") as file:
        content = file.read()
        
    soup = BeautifulSoup(content, "html.parser").prettify()
    
    with open(html_file, "w", encoding="utf-8") as prettyfile:
        prettyfile.write(soup)
        
    return None

def retrieve_student_info(html_file) -> list:
    """
        Retrieve student info from HTML file.
        Args:
            html_file (str): Path to the HTML file.
        Returns:
            list: List of student information.
    """
    # Reading with BS4.
    with open(html_file, "r", encoding="utf-8") as file:
        content = file.read()
        
    soup = BeautifulSoup(content, "html.parser")
    
    # Reading divs and retrieving student info.
    divs = soup.find("div", id="notas_div_tabla_estudiantes")
    info_students = [x.attrs['data-estudiante'] for x in divs.find_all("td")]
    
    return info_students

def retrieve_grades_table(html_file) -> pd.DataFrame:
    """
        Retrieve grades table from HTML file.
        Args:
            html_file (str): Path to the HTML file.
        Returns:    
            pd.DataFrame: DataFrame with grades information.
    """
    
    with open(html_file, "r", encoding="utf-8") as file:
        content = file.read()
    soup = BeautifulSoup(content, "html.parser")
    
    studs_table = soup.find("table", id="notas_tabla_notas").prettify()
    io = StringIO(str(studs_table))
    [dfs] = pd.read_html(io, flavor='bs4')
    if isinstance(dfs.columns, pd.MultiIndex):
        dfs.columns = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip() for tup in dfs.columns.values]  

    return dfs

def retrieve_student_table(html_file:str, table_id:str="notas_tabla_estudiantes") -> pd.DataFrame:
    """ 
        Retrieve student table from HTML file, this is useful when containerized table in `Integra` platform updates.
        Args:
            html_file (str): Path to the HTML file.
            table_id (str): ID of the table to retrieve, ex: "notas_div_tabla_estudiantes"
        Returns:    
            pd.DataFrame: DataFrame with student information.
    """
    
    # Reading with BS4.
    with open(html_file, "r", encoding="utf-8") as file:
        content = file.read()
    soup = BeautifulSoup(content, "html.parser")
    
    # Reading table and converting to DataFrame.
    ## "notas_div_tabla_estudiantes"
    studs_table = soup.find("table", id=table_id)
    io = StringIO(str(studs_table))
    try:
        [dfs] = pd.read_html(io, flavor='bs4')
        if isinstance(dfs.columns, pd.MultiIndex):
            dfs.columns = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip() for tup in dfs.columns.values]
    except ValueError as e:
        logger.error(f"Error reading HTML table: {e}")
        dfs = pd.DataFrame()  # Return an empty DataFrame in case of error
    
    return dfs


def full_grades_table(html_file:str) -> pd.DataFrame:
    """
        Construct full grades table from HTML file.
        Args:
            html_file (str): Path to the HTML file.
        Returns:    
            pd.DataFrame: DataFrame with full grades information.
    """
    # Constructing table from other functions.
    indexes = array(retrieve_student_info(html_file))
    grades_df = retrieve_grades_table(html_file)
    
    try:
        if isinstance(grades_df, pd.DataFrame):
            grades_df = grades_df.set_index(indexes)
            grades_df.columns = STUDS_TABLE_COLS
            logger.info(f"Grades table shape: {grades_df.shape}")   
    except ValueError as e:
            logger.error(f"{e}: Grades table is not a DataFrame.")
    except KeyError as k:
            logger.error(f"{k}: Columns do not match with DataFrame dimension.")
    
    return grades_df