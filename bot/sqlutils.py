import os
import sqlite3
import logging
from datetime import datetime

import pandas as pd
from pandas import DataFrame

os.chdir("/home/tommaso/Repositories/teleRAG/")


def put_chat(user_id: str, chat_updated: list[dict]):
    """Update chat history with user id. Create new if user not in database."""
    conn = sqlite3.connect('data/chats.db')
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT OR REPLACE INTO history (id, template) VALUES (?, ?);", (user_id, str(chat_updated)))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(e)
    finally:
        conn.close()


def retrieve_actions() -> DataFrame | None:
    """Return all action as DataFrame (or None if no actions)."""
    conn = sqlite3.connect('data/actions.db')
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT a.action_id, a.action_name, e.embedding FROM actions a, embeddings e WHERE a.action_id = e.id;")
        result = cursor.fetchall()
        if result:
            df = pd.DataFrame(data=result, columns=['id', 'name', 'embedding'])
            df["embedding"] = df["embedding"].apply(eval)
            return df
        else:
            return None
    except sqlite3.Error as e:
        logging.error(e)
    finally:
        conn.close()

def get_all_chat_ids(recent=False) -> list[str]:
    """Return a list of all the ids memorized into the database. Optionally, filter only those who interacted with the bot recently."""
    conn = sqlite3.connect('data/chats.db')
    cursor = conn.cursor()
    try:
        if recent:
            cursor.execute("""SELECT h.id FROM history h, session s 
                              WHERE h.id!='DEFAULT' 
                              AND s.last_access > DATETIME('now', '-30 day') 
                              AND h.id = s.id;""")
            result = cursor.fetchone()
        else:
            cursor.execute("SELECT h.id FROM history h, session s WHERE h.id!='DEFAULT'")
            result = cursor.fetchall()
        return result
    except sqlite3.Error as e:
        logging.error(e)
    finally:
        conn.close()

def get_chat(user_id: str) -> list[dict]:
    """Return chat history with user. Is no history, default chat incipit is created and returned."""
    conn = sqlite3.connect('data/chats.db')
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT template FROM history WHERE id = ?;", (user_id,))
        result = cursor.fetchone()
        if result:
            return eval(result[0])
        else:
            logging.warning("Chat with user {} not found. Creating a new one.".format(user_id))
            default_template = get_chat("DEFAULT")
            put_chat(user_id, default_template)
            return default_template
    except sqlite3.Error as e:
        logging.error(e)
    finally:
        conn.close()


def update_session(user_id: str, config: str):
    """Update chat history with user id. Create new if user not in database."""
    conn = sqlite3.connect('data/chats.db')
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT OR REPLACE INTO session (id, config, last_access) VALUES (?, ?, ?);", (user_id, config, datetime.now()))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(e)
    finally:
        conn.close()
