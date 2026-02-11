from dotenv import load_dotenv
import os
from IPython import get_ipython
import os
from PostgresAgent import PostgresAgent

def load_env():
    # Load environment variables
    load_dotenv(f'/{os.getcwd()}/.env')

def setup_sql_magic():
    # Get database connection parameters
    db_user = os.getenv('POSTGRES_USER')
    db_pass = os.getenv('POSTGRES_PASSWD')
    db_host = os.getenv('POSTGRES_HOST')
    db_port = os.getenv('POSTGRES_PORT')
    db_name = os.getenv('POSTGRES_DATABASE')
    
    # Create connection string
    db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    
    # Get IPython instance
    ipython = get_ipython()
    
    # Load SQL extension and configure
    ipython.run_line_magic('load_ext', 'sql')
    ipython.run_line_magic('sql', db_url)
    ipython.run_line_magic('config', "SqlMagic.style = '_DEPRECATED_DEFAULT'")  # Note the quotes

def setup_db_agent() -> PostgresAgent:
    # create helper class instance for accessing the database (inserts, updates, DDL, etc.)
    agent = PostgresAgent(os.getenv('POSTGRES_USER'), 
                os.getenv('POSTGRES_PASSWD'), 
                os.getenv('POSTGRES_HOST'), 
                os.getenv('POSTGRES_PORT'), 
                os.getenv('POSTGRES_DATABASE'))
    return agent

def get_username() -> str:
    username = os.environ.get('USER')
    print("Using username:", username)

load_env()
# allow us to use %%sql magic in Jupyter notebooks
setup_sql_magic()
# agent for PostgreSQL database operations
agent = setup_db_agent()
# get the current user's name
username = os.environ.get('USER')
print("User: ", username)
# get db conn info
db_user = os.getenv('POSTGRES_USER')
db_pass = 'secret'
db_host = os.getenv('POSTGRES_HOST')
db_port = os.getenv('POSTGRES_PORT')
db_name = os.getenv('POSTGRES_DATABASE')
db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}" 
print("Database: ", db_url)