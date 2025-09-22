from bs4 import BeautifulSoup
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

def retrieve_student_table(html_file) -> pd.DataFrame:
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
    studs_table = soup.find("table", id="notas_div_tabla_estudiantes")
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
    data_10_1 = "10_1_grades.html"
    
    studs_info = retrieve_student_info(data_10_1)
    studs_table = retrieve_grades_table(data_10_1)

    studs_table.index = studs_info
    logger.info(studs_table)