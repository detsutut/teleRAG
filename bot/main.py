from bot.telegrambot import TelegramBot
import sys

if __name__ == '__main__':
    bot = TelegramBot(api_token_path=sys.argv[1])
    bot.run()
