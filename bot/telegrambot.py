import logging
import os
import sys

import pandas as pd

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
    """
    A class used to represent an Animal

    ...

    Attributes
    ----------
    llm : LLM
        the large language model that is used to generate replies
    embedder : str
        a small language model used for semantic search for RAG
    actions : DataFrame
        special routines that are triggered through semantic search
    MAX_LEN : int
        maximum message length in characters
    API_TOKEN: str
        the api token of the telegram bot
    ACTIONS_THRESHOLD: float
        consider semantic search result only if similarity above this threshold
    ONSTART_MSG: str
        the message broadcasted when the bot goes online
    ONSTOP_MSG: str
        the message broadcasted when the bot goes offline
    WELCOME_MSG: str
        the greet message sent when conversation starts for the first time
    WELCOME_MSG: str
        the message sent when conversation is restarted

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """
    def __init__(self, api_token_path: str):
        """
        Parameters
        ----------
        api_token_path : str
            path where the telegram bot API is located
        """
        self.llm = LLM()
        self.embedder = SentenceEmbedder()
        self.actions = retrieve_actions()
        self.MAX_LEN = 4096  # characters
        self.API_TOKEN = load_api_token(api_token_path)
        self.ACTIONS_THRESHOLD = 0.6
        self.ONSTART_MSG = "Back online! Let meow know if you need assistance 🐱"
        self.ONSTOP_MSG = "Meowtenance time! Need to recharge and get my fur fluffed. Sweet dreams, humans! I'll be back online soon. Meanwhile, I will just ignore you 🐱"
        self.WELCOME_MSG = "Nice to meet you! I am RagBot, meow! How can I assist you today?"
        self.RESTART_MSG = "Memory wiped out! Meow! How can I assist you today?"

    def increase_decrease_menu(self, action_id: str) -> InlineKeyboardMarkup:
        """Defines an increase/decrease template for actions.

        The argument 'action_id' allows to keep track of what action is being performed when the user interacts with the menu.
        Callback data includes the action id and increase/decrease modifier

        Parameters
        ----------
        action_id : str
            The id of the action to perform
        """
        keyboard = [
            [InlineKeyboardButton("-", callback_data='{"action_id": ' + action_id + ', "action": "decrease"}'),
             InlineKeyboardButton("+", callback_data='{"action_id": ' + action_id + ', "action": "increase"}'), ],
        ]
        return InlineKeyboardMarkup(keyboard)

    async def button(self, update: Update) -> None:
        """Parses the CallbackQuery and updates the message text.

        Retrieve callback data after user's interaction with 'InlineKeyboardMarkup'.
        Increase/decrease by 25% the LLM attribute associated with the action_id in callback data.

        Parameters
        ----------
        update : Update
            The incoming update
        """
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

    async def config(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send the llm configuration.

        Parameters
        ----------
        update : Update
            The incoming update
        context: ContextTypes.DEFAULT_TYPE
            The current context, reporting application that called this function, chat id and user id
        """
        chat_id = update.effective_chat.id
        details = "Bot Configuration\n\n" + str(self.llm.gen_config)
        if check_length(details):
            chunks = split_text(details)
            for chunk in chunks:
                await context.bot.send_message(chat_id=chat_id, text=chunk)
        await context.bot.send_message(chat_id=chat_id, text=details)

    async def history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send the chat history.

        Privacy protected: hijacking the incoming update with a different chat_id has no effect since the chat history would be forwarded to
        that chat_id.

        Parameters
        ----------
        update : Update
            The incoming update
        context: ContextTypes.DEFAULT_TYPE
            The current context, reporting application that called this function, chat id and user id
        """
        chat_id = update.effective_chat.id
        details = "Conversation History\n\n" + str(get_chat(str(chat_id)))
        if check_length(details):
            chunks = split_text(details)
            for chunk in chunks:
                await context.bot.send_message(chat_id=chat_id, text=chunk)
        await context.bot.send_message(chat_id=chat_id, text=details)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Message sent when conversation starts for the first time"""
        await context.bot.send_message(chat_id=update.effective_chat.id, text=self.WELCOME_MSG)

    async def restart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Restart the conversation.

        The conversation history for this chat is erased from the database and restarted with default conversation template.

        Parameters
        ----------
        update : Update
            The incoming update
        context: ContextTypes.DEFAULT_TYPE
            The current context, reporting application that called this function, chat id and user id
        """
        chat_id = update.effective_chat.id
        put_chat(str(chat_id), get_chat("DEFAULT"))
        await context.bot.send_message(chat_id=chat_id, text=self.RESTART_MSG)

    def vector_db_search(self, input_text) -> dict | None:
        """Semantic search in the vector database.

        The incoming text is embedded and compared to the entries (i.e., embeddings) of the vector database.
        If the similarity is high enough, the entry with the highest similarity is returned

        Parameters
        ----------
        input_text : str
            The input text used to search the vector database

        Returns
        -------
        action: dict | None
            The action selected from the vector database, together with its similarity score
        """
        embedded_reply = self.embedder.encode(input_text)
        value, index = self.embedder.semantic_search(embedded_reply, list(self.actions["embedding"]))
        if value >= self.ACTIONS_THRESHOLD:
            action = self.actions.iloc[index]
            logging.info(f"Action Triggered! Name: {action['name']}, Id: {action['id']}, Similarity: {round(value, 2)}")
            return {"name": action["name"],
                    "id": action["id"],
                    "similarity": value}
        else:
            return None

    def update_and_generate(self, chat_id, input_text) -> str:
        """Use the LLM to generate an answer for the user and update the conversation history .

        First, the chat history is retrieved from the history database.
        Then, the conversation history, i.e., chat template, is passed to the LLM together with the input text to generate a context and history aware answer.
        Finally, the conversation history is updated with the input_text and the answer generated.

        Parameters
        ----------
        chat_id : int
            The chat id of the user (n.b. Telegram ids are integers, not strings!)
        input_text: str
            The text

        Returns
        -------
        answer: str
            The answer generated with LLM
        """
        chat_template = get_chat(str(chat_id))
        answer = self.llm.reply(user_message=input_text, chat_template=chat_template)
        logging.info(f"message generated for {chat_id}")
        put_chat(str(chat_id), answer["chat_template"])
        return answer["text"]

    async def post_stop(self, application: Application, recent=True) -> None:
        """Callback called when the bot is stopped.

        When the bot is stopped, send a broadcast message to notify users that the bot will be offline.
        The notification can be sent to the users that interacts with the bot recently (default) or to all the users in memory

        Parameters
        ----------
        application : Application
            The chat id of the user (n.b. Telegram ids are integers, not strings!)
        recent: bool
            Whether to notify only the users that interacts with the bot recently

        Returns
        -------
        answer: str
            The answer generated with LLM
        """
        logging.info(f"broadcasting message...")
        chat_ids = get_all_chat_ids(recent=recent)
        if len(chat_ids) > 0:
            for chat_id in chat_ids:
                await application.bot.send_message(chat_id=chat_id, text=self.ONSTOP_MSG)
        else:
            logging.info(f"no chats found")

    async def post_init(self, application: Application, recent=True) -> None:
        """Callback called when the bot starts.

        When the bot starts, send a broadcast message to notify users that the bot is online again.
        The notification can be sent to the users that interacts with the bot recently (default) or to all the users in memory

        Parameters
        ----------
        application : Application
            The chat id of the user (n.b. Telegram ids are integers, not strings!)
        recent: bool
            Whether to notify only the users that interacts with the bot recently

        Returns
        -------
        answer: str
            The answer generated with LLM
        """
        logging.info(f"broadcasting message...")
        chat_ids = get_all_chat_ids(recent=recent)
        if len(chat_ids) > 0:
            for chat_id in chat_ids:
                await application.bot.send_message(chat_id=chat_id, text=self.ONSTART_MSG)
                logging.info(f"broadcast message sent")
        else:
            logging.info(f"no chats found")

    async def reply(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Callback called when there is a new incoming message.

        When a new message is received, the bot:
        1) Updates session info
        2) Sets its status to 'typing...' (only lasts for 5 seconds and there's no way to modify it)
        3) Semantic search to get the most plausible action from the vector db
        4A) If semantic search finds an action that is plausible enough, the correspondent routine starts
        4B) If semantic search does not return anything, the user message is passed to the LLM and the generated answer is sent back

        Parameters
        ----------
        update : Update
            The incoming update
        context: ContextTypes.DEFAULT_TYPE
            The current context, reporting application that called this function, chat id and user id

        Returns
        -------
        answer: str
            The answer generated with LLM
        """
        chat_id = update.effective_chat.id
        logging.info(f"incoming message from {chat_id}")
        update_session(str(chat_id), "TEST001")
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        # Check in the vector db if the message is semantically similar to one of the scripted actions
        action = self.vector_db_search(update.message.text)
        # If so, trigger action and don't update chat history
        if action is not None:
            await update.message.reply_text(f"{action['name']}", reply_markup=self.increase_decrease_menu(action_id=str(action['id'])))
        # Otherwise, use LLM to generate answer and update chat history
        else:
            answer = self.update_and_generate(chat_id, update.message.text)
            await context.bot.send_message(chat_id=chat_id, text=answer)

    def run(self):
        """Run the telegram bot.

        Application built through ApplicationBuilder passing the API token and the init and stop callbacks.
        Then, handlers managing various types of user-bot interaction are added to the application.
        Finally, the app starts through the convenience method 'run_polling' which initialize, update and gracefully stop the bot.
        """
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
