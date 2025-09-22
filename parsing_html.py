from bs4 import BeautifulSoup
import pandas as pd #type:ignore
import os
import logging
from io import StringIO

logging.basicConfig(
    level=logging.INFO,
    filename='parsing_html.log',
    filemode='w+',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

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
    info_students = [x.attrs['data-estudiante'] for x in divs.find_all("td", class_="")]
    
    return info_students

def retrieve_student_table(html_file, table_id) -> pd.DataFrame:
    """ 
        Retrieve student table from HTML file.
        Args:
            html_file (str): Path to the HTML file.
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

def retrieve_grades_table(html_file) -> pd.DataFrame:
    
    with open(html_file, "r", encoding="utf-8") as file:
        content = file.read()
    soup = BeautifulSoup(content, "html.parser")
    
    studs_table = soup.find("table", id="notas_tabla_notas").prettify()
    io = StringIO(str(studs_table))
    [dfs] = pd.read_html(io, flavor='bs4')
    if isinstance(dfs.columns, pd.MultiIndex):
        dfs.columns = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip() for tup in dfs.columns.values]  

    return dfs

if __name__ == "__main__":
    # Grades files
    data_10_1 = "./grades_html/10_1_2grades.html"
    
    try:
        assert os.path.exists(data_10_1), f"File {data_10_1} does not exist."
    except AssertionError as e:
        logger.error(e)
        raise
    
    # Prettify and save the HTML file (optional)
    with open(data_10_1, "r", encoding="utf-8") as file:
        content = file.read()
    soup = BeautifulSoup(content, "html.parser")
    pretty_html = soup.prettify()
    with open(data_10_1, "w", encoding="utf-8") as file:
        file.write(pretty_html)
        
    # Generate DataFrames.
    studs_info = retrieve_student_info("./grades_html/10_1_1grades.html")
    studs_table = retrieve_grades_table(data_10_1)
    grades_table = retrieve_student_table(data_10_1, "notas_tabla_competencias")

    studs_table.index = studs_info
    
    new_cols = [
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
    studs_table.columns = new_cols
    
    # Redirect this info into a .pkl file.
    print(studs_table)