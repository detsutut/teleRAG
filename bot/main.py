import logging
import sqlite3
import os
import pandas as pd

from telegram import Update
from telegram.ext import filters, ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler

from scripts.llm.LLM import LLM
from scripts.embedder.embeddings import SentenceEmbedder

os.chdir("/home/tommaso/Repositories/teleRAG/")

llm = LLM()
embedder = SentenceEmbedder()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def retrieve_actions():
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


actions = retrieve_actions()


def put_chat(user_id, chat_updated):
    conn = sqlite3.connect('data/chats.db')
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT OR REPLACE INTO history (id, template) VALUES (?, ?);", (user_id, str(chat_updated)))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(e)
    finally:
        conn.close()


def get_chat(user_id):
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


def loadAPItoken(path='bot/API_token'):
    if os.path.isfile(path):
        with open(path) as file:
            return file.read()
    else:
        return None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Nice to meet you! I am RagBot, meow! How can I assist you today?")


async def restart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    put_chat(chat_id, get_chat("DEFAULT"))
    await context.bot.send_message(chat_id=chat_id, text="Memory wiped out! Meow! How can I assist you today?")


async def config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global llm
    chat_id = update.effective_chat.id
    details = "Config:" + str(llm.gen_config) + "\nConversation History:" + str(get_chat(chat_id)) + "\n"
    if check_length(details):
        chunks = split_text(details)
        for chunk in chunks:
            await context.bot.send_message(chat_id=chat_id, text=chunk)
    await context.bot.send_message(chat_id=chat_id, text=details)


def check_length(text: str, max_length: int = 4096):
    return len(text) >= max_length


def split_text(text: str, max_length: int = 4089):
    chunks = []
    for x in range(0, len(text), max_length):
        chunks.append(text[x:x + max_length] + " [" + str(int(1 + x / max_length)) + "/" + str(int(1 + len(text) / max_length)) + "]")
    return chunks


async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global llm, embedder, actions
    chat_id = update.effective_chat.id
    logging.info(f"incoming message from {chat_id}")
    # CONSULT VECTOR DB TO SEE IF A SPECIAL ACTION SHOULD BE TRIGGERED
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    embedded_reply = embedder.encode(update.message.text)
    index = embedder.semantic_search(embedded_reply, list(actions["embedding"]))
    if index is not None:
        action = actions.iloc[index]
        logging.info(f"Action Triggered! Name: {action['name']}, Id: {action['id']}")
        # For now, the message will not be parsed nor added to history. Explain this to the user
        await context.bot.send_message(chat_id=chat_id, text=f"WARNING: Your message has been flagged as a trigger for a special action ({action['name']}) that is not implemented yet.")
    # IF NOT, PARSE MESSAGE NORMALLY
    else:
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        chat_template = get_chat(chat_id)
        answer = llm.reply(user_message=update.message.text, chat_template=chat_template)
        logging.info(f"message generated for {chat_id}")
        put_chat(chat_id, answer["chat_template"])
        await context.bot.send_message(chat_id=chat_id, text=answer["text"])


if __name__ == '__main__':
    application = ApplicationBuilder().token(loadAPItoken()).build()
    start_handler = CommandHandler('start', start)
    conversation_handler = MessageHandler(filters=filters.TEXT & (~filters.COMMAND), callback=reply)
    restart_handler = CommandHandler('restart', restart)
    config_handler = CommandHandler('config', config)
    application.add_handler(start_handler)
    application.add_handler(restart_handler)
    application.add_handler(config_handler)
    application.add_handler(conversation_handler)
    application.run_polling()
