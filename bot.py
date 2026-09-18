import asyncio
import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand, Message
from dotenv import load_dotenv

from rag import answer

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

logging.basicConfig(level=logging.INFO)

HELP_TEXT = (
    "Привет! Я отвечаю на вопросы по базе знаний AI-платформы "
    "(демо: доступ к API, модели, лимиты, ошибки, промптинг, поддержка).\n\n"
    "Просто напиши вопрос текстом, например:\n"
    "«Что делать при ошибке 429?»"
)


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(HELP_TEXT)


@dp.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(HELP_TEXT)


@dp.message()
async def handle_question(message: Message):
    if not message.text:
        await message.answer("Задай вопрос текстом.")
        return

    await bot.send_chat_action(message.chat.id, "typing")

    try:
        text, sources = await asyncio.to_thread(answer, message.text)
    except Exception:
        logging.exception("Ошибка при обработке вопроса")
        await message.answer("Не получилось найти ответ, попробуй ещё раз.")
        return

    if sources:
        text += "\n\n📄 Источник: " + ", ".join(sources)

    await message.answer(text)


async def main():
    await bot.set_my_commands([
        BotCommand(command="start", description="Начать"),
        BotCommand(command="help", description="Как пользоваться ботом"),
    ])
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
