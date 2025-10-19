import random
import asyncio
from typing import TypeVar, Callable, Optional, Any
from functools import wraps

T = TypeVar('T')

def add_jitter(delay: float, jitter: float = 0.1) -> float:
    """Add random jitter to delay to prevent thundering herd."""
    return delay * (1 + random.uniform(-jitter, jitter))

async def exponential_backoff(
    func: Callable[..., T],
    *args,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    factor: float = 2.0,
    jitter: float = 0.1,
    max_retries: int = 3,
    **kwargs
) -> Optional[T]:
    """
    Выполняет функцию с экспоненциальной задержкой между повторами.
    
    Args:
        func: Асинхронная функция для выполнения
        initial_delay: Начальная задержка в секундах
        max_delay: Максимальная задержка в секундах
        factor: Множитель для увеличения задержки
        jitter: Случайное отклонение для предотвращения thundering herd
        max_retries: Максимальное количество попыток
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt == max_retries - 1:
                raise
            
            # Добавляем случайное отклонение к задержке
            current_delay = min(max_delay, add_jitter(delay, jitter))
            await asyncio.sleep(current_delay)
            delay *= factor
    
    if last_exception:
        raise last_exception
    return None

def with_rate_limit(
    calls: int = 1,
    period: float = 1.0,
    max_retries: int = 3,
    initial_delay: float = 1.0
):
    """
    Декоратор для ограничения частоты вызовов функции.
    
    Args:
        calls: Количество разрешенных вызовов
        period: Период в секундах
        max_retries: Максимальное количество попыток
        initial_delay: Начальная задержка между попытками
    """
    # Создаем семафор для отслеживания вызовов
    semaphore = asyncio.Semaphore(calls)
    # Словарь для хранения временных меток вызовов
    last_reset = {}
    
    def decorator(func: Callable[..., Any]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                # Проверяем, нужно ли сбросить счетчик
                now = asyncio.get_event_loop().time()
                if func not in last_reset or now - last_reset[func] >= period:
                    last_reset[func] = now
                    
                return await exponential_backoff(
                    func,
                    *args,
                    initial_delay=initial_delay,
                    max_retries=max_retries,
                    **kwargs
                )
        return wrapper
    return decorator