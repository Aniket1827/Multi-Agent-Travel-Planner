import os
import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL is not set")
    
    if "sslmode=" not in database_url:
        seperator = "&" if "?" in database_url else "?"
        database_url += f"{seperator}sslmode=require" # As we are connecting to remote database, we need to use sslmode=require

    return database_url

def get_checkpointer():
    DATABASE_URL = get_database_url()
    connect = psycopg.connect(
        DATABASE_URL,
        autocommit=True,
        row_factory=dict_row,
    )
    checkpointer = PostgresSaver(connect)
    checkpointer.setup()
    return checkpointer

