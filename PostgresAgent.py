"""
PostgresAgent provides a simple interface for executing PostgreSQL database operations.

The class handles connection lifecycle and error management, ensuring connections
are properly closed even when exceptions occur. It supports:
- DDL operations (CREATE, ALTER, DROP etc.)
- DML queries with results returned as Pandas DataFrames
- Writing Pandas DataFrames back to the database as a table (with overwrite option)
- Connection health checks via ping()

All operations will close the connection after execution, whether successful or not.
Exceptions are re-raised after ensuring proper connection cleanup.

Example:
    agent = PostgresAgent("user", "pass", "localhost", 5432, "mydb")
    try:
        df = agent.execute_dml("SELECT * FROM users")
    except Exception as e:
        print(f"Error: {e}")
"""

import psycopg2
import pandas as pd
from sqlalchemy import create_engine

class PostgresAgent:
    def __init__(self, username: str, password: str, host: str, port: int, database: str):
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self._conn = None
        self._engine = None

    def _get_connection(self):
        if not self._conn:
            self._conn = psycopg2.connect(
                user=self.username,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database
            )
        return self._conn

    def ping(self) -> bool:
        """Test database connection."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone()[0] == 1
        except Exception as e:
            self.close()
            raise e
        finally:
            self.close()

    def execute_ddl(self, ddl_statement: str) -> None:
        """Execute a DDL statement."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(ddl_statement)
                conn.commit()
        except Exception as e:
            self.close()
            raise e
        finally:
            self.close()

    def execute_dml_statement(self, sql: str) -> None:
        """Execute a DML query with no return (e.g. insert, update, delete)"""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(sql)
                conn.commit()
        except Exception as e:
            self.close()
            raise e
        finally:
            self.close()

    def execute_dml(self, query: str) -> pd.DataFrame:
        """Execute a DML query and return results as a DataFrame."""
        try:
            conn = self._get_connection()
            df = pd.read_sql_query(query, conn)
            return df
        except Exception as e:
            self.close()
            raise e
        finally:
            self.close()
    
    def execute_ddl_script(self, script: str) -> None:
        """Execute multiple SQL statements separated by semicolons."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                statements = [s.strip() for s in script.split(';') if s.strip()]
                for statement in statements:
                    cur.execute(statement)
                conn.commit()
        except Exception as e:
            self.close()
            raise e 
        finally:
            self.close()

    def _get_engine(self):
        """Get SQLAlchemy engine for DataFrame operations."""
        if not self._engine:
            self._engine = create_engine(
                f"postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            )
        return self._engine

    def write_to_postgres(self, df: pd.DataFrame, table_name: str, overwrite: bool = False) -> None:
        """
        Write a pandas DataFrame to a PostgreSQL table.
        """
        if_exists = 'replace' if overwrite else 'fail'
        
        try:
            engine = self._get_engine()
            schema = table_name.split('.')[0] if '.' in table_name else None
            table = table_name.split('.')[-1]
            
            df.to_sql(
                name=table,
                schema=schema,
                con=engine,
                if_exists=if_exists,
                index=False,
                method='multi'
            )
        except Exception as e:
            if self._engine:
                self._engine.dispose()
                self._engine = None
            raise e

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            try:
                self._conn.close()
            finally:
                self._conn = None