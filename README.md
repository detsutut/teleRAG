# TelegramBot

![image](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)

A Python Telegram bot powered with LLM and RAG for interactive messaging and conversation management.

## Overview

This Telegram bot utilizes natural language processing (NLP) techniques to provide interactive messaging functionalities. It integrates with a Large Language Model (LLM) for generating replies and RAG through semantic search to trigger specific actions/routines.

## Features

- **Interactive Messaging**: Realistic conversation through LLM-powered answers.
- **Semantic Search**: RAG with semantic search is used to trigger specific routines, commands, and actions. 
- **Conversation History**: Maintains conversation history to generate context-and-history-aware responses.
- **Broadcast Messages**: Notify users with broadcast messages upon bot start and stop events.

## Setup

### Prerequisites

- Python 3.x
- A Telegram Bot already instantiated via ```BotFather```
- Telegram API token

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/your/repository.git
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Set up Telegram API token:

   - Obtain your Telegram API token and copy it
   - Create a file called ```API_token``` in ```./bot```
   - Paste your token in the API_token file (n.b. don't share this file!)

4. Set up the chat database

   - Rename ```./data/empty_chats.db``` to ```./data/chats.db```'

### Usage

1. Run the bot, passing the path to the API token:

   ```
   python TelegramBot.py /.../.../bot/API_token
   ```

2. Open Telegram, search for your bot name and start chatting

## Usage Guide

1. Start the conversation with the bot by sending a message.
2. The bot will respond based on the message content, utilizing semantic search or the LLM for generating replies.
3. Use commands such as `/config` to retrieve the bot's configuration or `/history` to view conversation history.

## Contributors

-  Tommaso M. Buonocore  [![image](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tbuonocore/)  

## License

This project is licensed under the [CC BY-NC-ND 4.0 License](LICENSE.md).
