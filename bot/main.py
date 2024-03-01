import logging

from telegram import Update
from telegram.ext import filters, ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler
import os

from scripts.llm.LLM import LLM

os.chdir("/home/tommaso/Repositories/teleRAG/")

llm = LLM()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

chats = {"DEFAULT": [
    {"role": "user",
     "content": "Your name is RagBot. You are a friendly chatbot that answers questions. Keep the answer short and concise. Don't be verbose."},
    {"role": "assistant", "content": "Got it! How can I assist you today?"}
]}


def put_chat(user_id, chat_updated):
    global chats
    chats[user_id] = chat_updated


def get_chat(user_id):
    global chats
    chat = chats.get(user_id)
    if chat is None:
        logging.warning("Chat with user {} not found. Creating a new one.".format(user_id))
        put_chat(user_id, chats["DEFAULT"])
        return chats["DEFAULT"]
    else:
        return chat


def loadAPItoken(path='bot/API_token'):
    if os.path.isfile(path):
        with open(path) as file:
            return file.read()
    else:
        return None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Nice to meet you! I am RagBot, meow! How can I assist you today?")


async def restart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global chats
    chat_id = update.effective_chat.id
    put_chat(chat_id, chats["DEFAULT"])
    await context.bot.send_message(chat_id=chat_id, text="Memory wiped out! Meow! How can I assist you today?")


async def config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global llm, chats
    chat_id = update.effective_chat.id
    details = "\n\n".join([str(llm.gen_config), str(chats)])
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
        chunks.append(text[x:x + max_length]+" ["+str(int(1+x/max_length))+"/"+str(int(1+len(text)/max_length))+"]")
    return chunks


async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global llm
    chat_id = update.effective_chat.id
    logging.info(f"incoming message from {chat_id}")
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
