import os
import yaml

# logging
import logging
from src.logging_config import LoggingConfig

# configuration of logging
logging_config = LoggingConfig()
logging_config.setup_logging()

# db import
import psycopg2
from psycopg2 import sql

class DBManager:
    def __init__(self, db_config):
        
        if db_config is None:
            raise ValueError("empty database configuration not allowed")
        self.db_config = db_config
        self.conn = None
        self.connect()
        self.conn.close()
    
    def __del__(self):
        if self.conn:
            self.conn.close()        

    def connect(self):
        
        try:
            conn = psycopg2.connect(
                dbname=self.db_config['POSTGRES_DB'],
                user=self.db_config['POSTGRES_USER'],
                password= self.db_config['POSTGRES_PASSWORD'],
                host= self.db_config['POSTGRES_HOST'],
                port= self.db_config['POSTGRES_PORT']
            )
            self.conn = conn
        except Exception as e:
            exception_msg = f'{self.__class__.__name__}.{self.connect.__name__}: db-host {self.db_config['POSTGRES_HOST']} not reachble'
            logging.exception(exception_msg)
            raise ConnectionError(exception_msg)

    def execute_sql(self, query, values=None, fetch_mode = 0):
        self.connect()
        cursor = None
        rep_data = None      
        try:
            if query is None:
                raise ValueError("empty query string not allowed")
            # convert single value to tuple of values
            if values is not None and not isinstance(values, tuple):
                values = (values,)
            
            cursor = self.conn.cursor()
            cursor.execute(query, values)
            self.conn.commit()
            if fetch_mode == 1:
                rep_data = cursor.fetchone()
                return rep_data
            if fetch_mode == 2:
                rep_data = cursor.fetchall()
                return rep_data
            if fetch_mode == 3:
                rep_data = cursor.fetchmany()
                return rep_data
            
        except Exception as e:
            self.conn.rollback()
            logging.exception(f"{self.__class__.__name__}.{self.execute_sql.__name__}: {e}")
        finally:
            if cursor:
                cursor.close()
            if self.conn:
                self.conn.close()

    def executemany_sql(self, query, values=None, fetch_mode = 0):
        self.connect()
        cursor = None
        rep_data = None
        try:
            if query is None:
                raise ValueError("empty query string not allowed")
            # convert item of single value to tuple of values
            if values is not None:
                for i, item in enumerate(values):
                    if not isinstance(item, tuple):
                        values[i] = (item,)
            
            cursor = self.conn.cursor()     
            cursor.executemany(query, values)
            self.conn.commit()
            if fetch_mode == 1:
                rep_data = cursor.fetchone()
                return rep_data
            if fetch_mode == 2:
                rep_data = cursor.fetchall()
                return rep_data
            if fetch_mode == 3:
                rep_data = cursor.fetchmany()
                return rep_data

        except Exception as e:
            self.conn.rollback()
            logging.exception(f"{self.__class__.__name__}.{self.executemany_sql.__name__}: {e}")
        finally:
            if cursor:
                cursor.close()
            if self.conn:
                self.conn.close()
            