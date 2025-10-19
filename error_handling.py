"""
Модуль для обработки ошибок и повторных попыток в боте.
"""
import asyncio
import logging
from typing import TypeVar, Callable, Any
from functools import wraps

from aiogram.exceptions import TelegramConflictError, TelegramRetryAfter
from aiohttp.client_exceptions import ClientError

logger = logging.getLogger(__name__)

T = TypeVar('T')

class RetrySettings:
    """Настройки для повторных попыток"""
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: float = 0.1
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

def retry_on_telegram_error(
    max_retries: int = 3,
    initial_delay: float = 1.0
) -> Callable:
    """
    Декоратор для повторных попыток при ошибках Telegram API
    
    Args:
        max_retries: Максимальное количество попыток
        initial_delay: Начальная задержка между попытками
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except TelegramRetryAfter as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = e.retry_after
                    logger.warning(
                        f"Rate limit hit, sleeping for {delay:.2f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                except TelegramConflictError as e:
                    logger.error(f"Webhook conflict: {e}")
                    raise  # Не пытаемся повторить при конфликте webhook
                except (ClientError, asyncio.TimeoutError) as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = initial_delay * (2 ** attempt)
                    logger.warning(
                        f"Network error: {e}, retrying in {delay:.2f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                except Exception as e:
                    last_exception = e
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                    raise
            
            if last_exception:
                raise last_exception
            return None
            
        return wrapper
    return decorator