import os
import base64
import time
import asyncio
import aiohttp
import tempfile
import logging
from enum import Enum
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.utils.keyboard import InlineKeyboardBuilder

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загружаем токены и проверяем их наличие
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Проверка наличия токенов
if not TELEGRAM_TOKEN:
    raise ValueError("Отсутствует TELEGRAM_TOKEN в файле .env")
if not OPENAI_API_KEY:
    raise ValueError("Отсутствует OPENAI_API_KEY в файле .env")
if not GEMINI_API_KEY:
    raise ValueError("Отсутствует GEMINI_API_KEY в файле .env")

# Очистка токенов от лишних пробелов
OPENAI_API_KEY = OPENAI_API_KEY.strip() if OPENAI_API_KEY else None
TELEGRAM_TOKEN = TELEGRAM_TOKEN.strip() if TELEGRAM_TOKEN else None
GEMINI_API_KEY = GEMINI_API_KEY.strip() if GEMINI_API_KEY else None

# Константы
OPENAI_IMAGE_EDIT_URL = "https://api.openai.com/v1/images/generations"  # Используем generations вместо edits
GOOGLE_VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
OPENAI_MODEL = "dall-e-2"  # Используем DALL-E 2 модель
MAX_IMAGE_SIZE = 4 * 1024 * 1024  # 4MB
MAX_RETRIES = 3
CONCURRENT_REQUESTS_LIMIT = 10

# Определение типов AI сервисов
class AIService(str, Enum):
    OPENAI = "openai"
    GOOGLE = "google"

# Инициализация бота и FSM
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)

# Создаем временную директорию для файлов
TEMP_DIR = Path(tempfile.gettempdir()) / "ktc_photo_bot"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# --- FSM (машина состояний) ---
class EditStates(StatesGroup):
    choosing_service = State()
    waiting_for_prompt = State()
    waiting_for_result = State()

# Функция создания клавиатуры выбора сервиса
def get_service_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="OpenAI", callback_data=f"service:{AIService.OPENAI}")
    builder.button(text="Google AI", callback_data=f"service:{AIService.GOOGLE}")
    builder.adjust(2)
    return builder.as_markup()


# --- Функции обработки изображений ---
async def call_google_vision(image_path: str, prompt: str) -> Optional[str]:
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageEnhance, ImageFilter
    except ImportError:
        raise ImportError("Пожалуйста, установите необходимые библиотеки: pip install opencv-python pillow")

    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
    output_path = str(Path(image_path).with_suffix('.edited.png'))

    # Проверяем размер файла
    if Path(image_path).stat().st_size > MAX_IMAGE_SIZE:
        raise ValueError(f"Размер изображения превышает {MAX_IMAGE_SIZE // 1024 // 1024}MB")

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with aiohttp.ClientSession() as session:
                    # Читаем изображение в base64
                    with open(image_path, 'rb') as img_file:
                        img_content = base64.b64encode(img_file.read()).decode('utf-8')

                    # Формируем запрос к Google Vision API
                    request_data = {
                        "requests": [
                            {
                                "image": {"content": img_content},
                                "features": [
                                    {"type": "IMAGE_PROPERTIES"},
                                    {"type": "LABEL_DETECTION"},
                                    {"type": "FACE_DETECTION"},
                                    {"type": "OBJECT_LOCALIZATION"}
                                ]
                            }
                        ]
                    }

                    async with session.post(
                        GOOGLE_VISION_URL,
                        headers=headers,
                        json=request_data,
                        timeout=aiohttp.ClientTimeout(total=180)
                    ) as resp:
                        resp.raise_for_status()
                        response_data = await resp.json()

                        # Обработка ответа и применение эффектов
                        response = response_data['responses'][0]
                        
                        # Открываем изображение для редактирования
                        image = Image.open(image_path)
                        
                        # Анализируем промпт для определения нужных эффектов
                        prompt_lower = prompt.lower()
                        
                        if 'яркость' in prompt_lower or 'светлее' in prompt_lower:
                            enhancer = ImageEnhance.Brightness(image)
                            image = enhancer.enhance(1.5)
                            
                        if 'контраст' in prompt_lower:
                            enhancer = ImageEnhance.Contrast(image)
                            image = enhancer.enhance(1.3)
                            
                        if 'насыщенность' in prompt_lower or 'цвета' in prompt_lower:
                            enhancer = ImageEnhance.Color(image)
                            image = enhancer.enhance(1.4)
                            
                        if 'резкость' in prompt_lower or 'четкость' in prompt_lower:
                            enhancer = ImageEnhance.Sharpness(image)
                            image = enhancer.enhance(1.5)
                            
                        if 'размытие' in prompt_lower or 'блюр' in prompt_lower:
                            image = image.filter(ImageFilter.GaussianBlur(radius=2))

                        # Если есть лица на фото и запрошена их обработка
                        if 'лицо' in prompt_lower or 'портрет' in prompt_lower:
                            if 'faceAnnotations' in response:
                                # Конвертируем в cv2 для обработки лиц
                                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                                
                                for face in response['faceAnnotations']:
                                    vertices = face['fdBoundingPoly']['vertices']
                                    x1 = vertices[0]['x']
                                    y1 = vertices[0]['y']
                                    x2 = vertices[2]['x']
                                    y2 = vertices[2]['y']
                                    
                                    # Применяем дополнительную обработку к области лица
                                    face_roi = cv_image[y1:y2, x1:x2]
                                    if face_roi.size > 0:
                                        # Улучшаем контраст лица
                                        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
                                        l, a, b = cv2.split(lab)
                                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                                        cl = clahe.apply(l)
                                        limg = cv2.merge((cl,a,b))
                                        face_roi = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                                        cv_image[y1:y2, x1:x2] = face_roi
                                
                                # Конвертируем обратно в PIL
                                image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

                        # Применяем цветокоррекцию на основе доминантных цветов
                        if 'цветокоррекция' in prompt_lower:
                            if 'imagePropertiesAnnotation' in response:
                                colors = response['imagePropertiesAnnotation']['dominantColors']['colors']
                                # Создаем LUT на основе доминантных цветов
                                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                                for color in colors[:3]:  # Берем три основных цвета
                                    rgb = [
                                        int(color['color']['red']),
                                        int(color['color']['green']),
                                        int(color['color']['blue'])
                                    ]
                                    # Усиливаем доминантные цвета
                                    mask = cv2.inRange(
                                        cv_image,
                                        np.array([max(0, c - 30) for c in rgb]),
                                        np.array([min(255, c + 30) for c in rgb])
                                    )
                                    cv_image[mask > 0] = [min(255, c * 1.2) for c in rgb]
                                image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

                        # Сохраняем обработанное изображение
                        image.save(output_path, 'PNG', quality=95)
                        logger.info(f"Google Vision API успешно обработал изображение")
                        return output_path

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Ошибка при обработке изображения после {MAX_RETRIES} попыток: {e}")
                    raise
                await asyncio.sleep(2 ** attempt)

            except Exception as e:
                logger.error(f"Неожиданная ошибка при обработке изображения: {e}")
                raise

async def call_openai_image_edit(image_path: str, prompt: str, size="1024x1024") -> Optional[str]:
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API ключ не найден. Пожалуйста, проверьте наличие OPENAI_API_KEY в файле .env")
        
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY.strip()}",
        "Content-Type": "application/json"
    }
    output_path = str(Path(image_path).with_suffix('.edited.png'))

    # Проверяем размер файла
    if Path(image_path).stat().st_size > MAX_IMAGE_SIZE:
        raise ValueError(f"Размер изображения превышает {MAX_IMAGE_SIZE // 1024 // 1024}MB")

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with aiohttp.ClientSession() as session:
                    # Отправляем изображение в OpenAI
                    data = aiohttp.FormData()
                    data.add_field('model', OPENAI_MODEL)
                    data.add_field('prompt', prompt)
                    data.add_field('size', size)
                    data.add_field('n', '1')
                    # Открываем файл в контексте менеджера
                    with open(image_path, 'rb') as img:
                        data.add_field('image', img, filename='image.png', content_type='image/png')

                    async with session.post(
                        OPENAI_IMAGE_EDIT_URL,
                        headers=headers,
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=180)
                    ) as resp:
                        resp.raise_for_status()
                        response_data = await resp.json()

                    item = response_data["data"][0]

                    # Сохраняем результат
                    if "b64_json" in item:
                        image_bytes = base64.b64decode(item["b64_json"])
                        await asyncio.to_thread(
                            lambda: Path(output_path).write_bytes(image_bytes)
                        )
                    elif "url" in item:
                        async with session.get(
                            item["url"],
                            timeout=aiohttp.ClientTimeout(total=60)
                        ) as img_resp:
                            img_resp.raise_for_status()
                            content = await img_resp.read()
                            await asyncio.to_thread(
                                lambda: Path(output_path).write_bytes(content)
                            )
                    else:
                        raise ValueError("Unexpected response from OpenAI")

                    logger.info(f"Изображение успешно обработано: {output_path}")
                    return output_path
            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Ошибка при обработке изображения после {MAX_RETRIES} попыток: {e}")
                    raise
                await asyncio.sleep(2 ** attempt)  # Экспоненциальная задержка

            except Exception as e:
                logger.error(f"Неожиданная ошибка при обработке изображения: {e}")
                raise


# Функция очистки старых временных файлов
async def cleanup_old_files(max_age_hours: int = 24):
    """Удаляет временные файлы старше указанного возраста"""
    try:
        current_time = time.time()
        for file in TEMP_DIR.glob("tmp_*"):
            if current_time - file.stat().st_mtime > max_age_hours * 3600:
                file.unlink()
    except Exception as e:
        logger.error(f"Ошибка при очистке временных файлов: {e}")


# --- Команда /start ---
@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "👋 Привет! Отправь мне фото, и я помогу тебе изменить его.\n"
        "🤖 Доступные сервисы: OpenAI и Google AI\n"
        "📸 Максимальный размер фото: 4MB"
    )
    # Очищаем старые временные файлы при старте
    await cleanup_old_files()


# Обработка выбора сервиса
@dp.callback_query(lambda c: c.data.startswith("service:"))
async def process_service_choice(callback_query: CallbackQuery, state: FSMContext):
    service = callback_query.data.split(":")[1]
    await state.update_data(service=service)
    
    await callback_query.message.edit_text(
        "✏️ Как вы хотите изменить это изображение?",
        reply_markup=None
    )
    await state.set_state(EditStates.waiting_for_prompt)
    await callback_query.answer()


# --- Пользователь прислал фото ---
@dp.message(F.photo)
async def handle_photo(message: Message, state: FSMContext):
    try:
        photo = message.photo[-1]
        file_info = await bot.get_file(photo.file_id)
        
        # Создаем уникальное имя файла во временной директории
        input_path = TEMP_DIR / f"tmp_{message.chat.id}_{message.message_id}.jpg"
        
        # Проверяем размер файла
        if file_info.file_size and file_info.file_size > MAX_IMAGE_SIZE:
            await message.answer(
                f"⚠️ Изображение слишком большое. Максимальный размер: {MAX_IMAGE_SIZE // 1024 // 1024}MB"
            )
            return

        await bot.download_file(file_info.file_path, str(input_path))
        logger.info(f"Получено изображение: {input_path}")

        # Сохраняем путь к фото в FSM
        await state.update_data(image_path=str(input_path))
        
        # Предлагаем выбрать сервис для обработки
        await message.answer(
            "🤖 Выберите сервис для обработки изображения:",
            reply_markup=get_service_keyboard()
        )
        await state.set_state(EditStates.choosing_service)

    except Exception as e:
        logger.error(f"Ошибка при получении фото: {e}")
        await message.answer("⚠️ Произошла ошибка при получении фото. Попробуйте еще раз.")
        await state.clear()


# --- Пользователь отвечает промптом ---
@dp.message(EditStates.waiting_for_prompt)
async def handle_prompt(message: Message, state: FSMContext):
    user_data = await state.get_data()
    image_path = user_data.get("image_path")
    service = user_data.get("service", AIService.OPENAI)
    
    if not image_path or not Path(image_path).exists():
        await message.answer("⚠️ Изображение не найдено. Пожалуйста, отправьте фото заново.")
        await state.clear()
        return

    prompt = message.text.strip()
    if not prompt:
        await message.answer("⚠️ Пожалуйста, опишите желаемые изменения.")
        return

    await message.answer("🪄 Обрабатываю изображение... это может занять несколько секунд.")
    await state.set_state(EditStates.waiting_for_result)

    try:
        # Выбираем нужную функцию обработки в зависимости от сервиса
        process_func = call_openai_image_edit if service == AIService.OPENAI else call_google_vision
        output_path = await process_func(image_path, prompt)
        
        if output_path and Path(output_path).exists():
            async with aiohttp.ClientSession() as session:
                with open(output_path, "rb") as f:
                    await message.answer_photo(f, caption=f"✅ Готово! (обработано через {service})")
        else:
            raise ValueError("Не удалось создать изображение")

    except ValueError as ve:
        error_msg = str(ve)
        logger.error(f"Ошибка валидации: {error_msg}")
        await message.answer(f"⚠️ {error_msg}")
    except ImportError as ie:
        error_msg = str(ie)
        logger.error(f"Ошибка импорта библиотек: {error_msg}")
        await message.answer(f"⚠️ Техническая ошибка: {error_msg}")
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Неожиданная ошибка при обработке изображения: {error_msg}")
        await message.answer(f"⚠️ Произошла ошибка при обработке изображения: {error_msg}")

    finally:
        # Удаляем временные файлы
        try:
            for file in [Path(image_path), Path(image_path + ".edited.png")]:
                if file.exists():
                    file.unlink()
        except Exception as e:
            logger.error(f"Ошибка при удалении временных файлов: {e}")
        await state.clear()


# --- Запуск бота ---
async def main():
    print("✅ Бот запущен (aiogram 3.x, с диалогом для prompt)")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
