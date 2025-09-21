import logging
import requests
from bs4 import BeautifulSoup

# Create logger
logger = logging.getLogger(__name__)

# Configure logging
logger.info("Starting…")
logging.basicConfig(
    level=logging.INFO,
    filename='app.log',
    filemode='w+',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def retrieve_data(url) -> dict:
    """
    Retrieve and parse HTML content from a given URL.
    Args:
        url (str): The URL to fetch data from.
    Returns:
        dict: Parsed HTML content as a BeautifulSoup object.
    """
    # Create GET request.
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup
    

if __name__ == "__main__":
    
    # ROOT website
    ROOT = "https://q.plataformaintegra.net"
    # Login form
    LOGIN_FORM = f"{ROOT}/ieholandagranada/index.php"
    # Post payload to login
    LOGIN_POST = f"{LOGIN_FORM}/doc/auth/control"
    # Platform home after logged.
    START_URL = f"{LOGIN_FORM}/doc/inicio/principal"

    # Payload login form.
    payload = {"usuario" : "118", "contrasena": "AMMA641"}
    
    # Instancing Session
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent":"Mozilla/5.0", 
            "Origin": ROOT, 
            "Referer": LOGIN_POST + "/"
        }
    )
    
    # Login
    session.get(LOGIN_POST + "/", timeout=30)
    session.post(
        url=LOGIN_POST,
        data=payload,
        headers={
            "Content-Type":"application/x-www-form-urlencoded; charset=UTF-8",
            "Accept":"application/json, text/javascript, */*; q=0.01",
            "X-Requested-With":"XMLHttpRequest",
            "Referer": LOGIN_POST + "/",
        },
        timeout=30
    ).raise_for_status()
    
    # POST AJAX fragment
    resp = session.post(
        START_URL,
        headers={
            "X-Requested-With":"XMLHttpRequest",
            "Accept":"*/*",
            "Referer": ROOT + "/doc/inicio",            
        },
        data={},
        timeout=30
    )
    
    # Tracking status code after login.
    logger.info(f"POST {LOGIN_POST} with payload keys={list(payload.keys())}")    
    logger.info(f"Login response: {resp.status_code} {resp.reason} (final URL: {resp.url})")
    print(resp.history)
    
    
    with open("login.html", "w+") as log:
        content_from_url = requests.get(f"{START_URL}", timeout=30)
        log.write(content_from_url.text)
        
        soup = BeautifulSoup(content_from_url.text, "html.parser")
        log.write(soup.prettify())
        
    p = session.get(START_URL, timeout=30)
    print("START:", p.status_code, p.url)
    print(retrieve_data(START_URL).get_text(strip=True))