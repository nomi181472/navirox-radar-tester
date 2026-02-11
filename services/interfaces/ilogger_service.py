# services/interfaces/ilogger_service.py
from abc import ABC, abstractmethod

class ILoggerService(ABC):
    @abstractmethod
    async def info(self, message: str, **kwargs) -> None:
        pass

    @abstractmethod
    async def error(self, message: str, **kwargs) -> None:
        pass

    @abstractmethod
    async def warning(self, message: str, **kwargs) -> None:
        pass

    @abstractmethod
    async def debug(self, message: str, **kwargs) -> None:
        pass