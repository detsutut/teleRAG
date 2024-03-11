import logging
import os
import sys

sys.path.append("/home/tommaso/Repositories/teleRAG/")

from bot.botutils import check_length, split_text, load_api_token
from bot.sqlutils import retrieve_actions, get_chat, put_chat, update_session, get_all_chat_ids
from scripts.embedder.embeddings import SentenceEmbedder
from scripts.llm.LLM import LLM

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import filters, Application, ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackQueryHandler

os.chdir("/home/tommaso/Repositories/teleRAG/")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


class TelegramBot:
    def __init__(self, api_token_path: str):
        self.llm = LLM()
        self.embedder = SentenceEmbedder()
        self.actions = retrieve_actions()
        self.MAX_LEN = 4096  # characters
        self.API_TOKEN = load_api_token(api_token_path)
        self.ACTIONS_THRESHOLD = 0.6
        self.ONSTART_MSG = "Back online! Let meow know if you need assistance 🐱"
        self.ONSTOP_MSG = "Meowtenance time! Need to recharge and get my fur fluffed. Sweet dreams, humans! I'll be back online soon. Meanwhile, I will just ignore you 🐱"

    def increase_decrease_menu(self, action_id: str):
        keyboard = [
            [InlineKeyboardButton("-", callback_data='{"action_id": ' + action_id + ', "action": "decrease"}'),
             InlineKeyboardButton("+", callback_data='{"action_id": ' + action_id + ', "action": "increase"}'), ],
        ]
        return InlineKeyboardMarkup(keyboard)

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Parses the CallbackQuery and updates the message text."""
        query = update.callback_query
        data = eval(query.data)
        action_name = self.actions[self.actions['id'] == data['action_id']]['name'].iloc[0]
        old_val = self.llm.gen_config.__getattribute__(action_name)
        if data['action'] == 'decrease':
            new_val = 0.75 * old_val
        else:
            new_val = 1.25 * old_val
        if isinstance(old_val, int):
            new_val = round(new_val)
        else:
            new_val = round(new_val, 2)
        self.llm.gen_config.__setattr__(action_name, new_val)
        # CallbackQueries need to be answered, even if no notification to the user is needed
        # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
        await query.answer()
        await query.edit_message_text(text=f"Selected option: {data['action']}")

    async def config(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        details = "Bot Configuration\n\n" + str(self.llm.gen_config)
        if check_length(details):
            chunks = split_text(details)
            for chunk in chunks:
                await context.bot.send_message(chat_id=chat_id, text=chunk)
        await context.bot.send_message(chat_id=chat_id, text=details)

    async def history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        details = "Conversation History\n\n" + str(get_chat(str(chat_id)))
        if check_length(details):
            chunks = split_text(details)
            for chunk in chunks:
                await context.bot.send_message(chat_id=chat_id, text=chunk)
        await context.bot.send_message(chat_id=chat_id, text=details)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Nice to meet you! I am RagBot, meow! How can I assist you today?")

    async def restart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        put_chat(str(chat_id), get_chat("DEFAULT"))
        await context.bot.send_message(chat_id=chat_id, text="Memory wiped out! Meow! How can I assist you today?")

    def vector_db_search(self, input_text):
        embedded_reply = self.embedder.encode(input_text)
        value, index = self.embedder.semantic_search(embedded_reply, list(self.actions["embedding"]))
        if value >= self.ACTIONS_THRESHOLD:
            action = self.actions.iloc[index]
            logging.info(f"Action Triggered! Name: {action['name']}, Id: {action['id']}, Similarity: {round(value, 2)}")
            return action
        else:
            return None

    def update_and_generate(self, chat_id, input_text):
        chat_template = get_chat(str(chat_id))
        answer = self.llm.reply(user_message=input_text, chat_template=chat_template)
        logging.info(f"message generated for {chat_id}")
        put_chat(str(chat_id), answer["chat_template"])
        return answer["text"]

    async def post_stop(self, application: Application) -> None:
        logging.info(f"broadcasting message...")
        chat_ids = get_all_chat_ids(recent=True)
        for chat_id in chat_ids:
            await application.bot.send_message(chat_id=chat_id, text=self.ONSTOP_MSG)

    async def post_init(self, application: Application) -> None:
        logging.info(f"broadcasting message...")
        chat_ids = get_all_chat_ids(recent=True)
        for chat_id in chat_ids:
            await application.bot.send_message(chat_id=chat_id, text=self.ONSTART_MSG)


    async def reply(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        logging.info(f"incoming message from {chat_id}")
        update_session(str(chat_id), "TEST001")
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        # Check in the vector db if the message is semantically similar to one of the scripted actions
        action = self.vector_db_search(update.message.text)
        # If so, trigger action and don't update chat history
        if action:
            await update.message.reply_text(f"{action['name']}", reply_markup=self.increase_decrease_menu(action_id=str(action['id'])))
        # Otherwise, use LLM to generate answer and update chat history
        else:
            answer = self.update_and_generate(chat_id, update.message.text)
            await context.bot.send_message(chat_id=chat_id, text=answer)

    def run(self):
        application = ApplicationBuilder().token(self.API_TOKEN).post_init(self.post_init).post_stop(self.post_stop).build()
        handlers = [CommandHandler("start", self.start),
                    CommandHandler('restart', self.restart),
                    CommandHandler('config', self.config),
                    CommandHandler('history', self.history),
                    CallbackQueryHandler(self.button),
                    MessageHandler(filters=filters.TEXT & (~filters.COMMAND), callback=self.reply)]
        application.add_handlers(handlers)
        application.run_polling()


if __name__ == '__main__':
    bot = TelegramBot(api_token_path=sys.argv[1])
    bot.run()
