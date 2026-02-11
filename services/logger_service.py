import json
import logging
from datetime import datetime, UTC
from services.interfaces.ilogger_service import ILoggerService


class LoggerService(ILoggerService):
    def __init__(self):
        self.console_logger = logging.getLogger("AppLogger")
        self.console_logger.setLevel(logging.DEBUG)

        if not self.console_logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            console_handler.setFormatter(formatter)
            self.console_logger.addHandler(console_handler)

    @staticmethod
    def _format_message(message: str, **kwargs) -> str:
        if kwargs:
            formatted_data = json.dumps(kwargs, indent=2, default=str)
            return f"{message}\n📋 Data: {formatted_data}"
        return message

    async def _log(self, level: str, message: str, **kwargs):
        timestamp = datetime.now(UTC).isoformat()

        formatted_message = self._format_message(message, **kwargs)
        formatted_message = f"[{timestamp}] {formatted_message}"

        try:
            console_level = getattr(logging, level.upper(), logging.INFO)
            self.console_logger.log(console_level, formatted_message)
        except Exception as e:
            print(f"Logging failed: {e}")

    async def info(self, message: str, **kwargs) -> None:
        await self._log("INFO", message, **kwargs)

    async def error(self, message: str, **kwargs) -> None:
        await self._log("ERROR", message, **kwargs)

    async def warning(self, message: str, **kwargs) -> None:
        await self._log("WARNING", message, **kwargs)

    async def debug(self, message: str, **kwargs) -> None:
        await self._log("DEBUG", message, **kwargs)

    async def critical(self, message: str, **kwargs) -> None:
        await self._log("CRITICAL", message, **kwargs)

    # -------- Sync versions --------

    def sync_info(self, message: str, **kwargs) -> None:
        self.console_logger.info(self._format_message(message, **kwargs))

    def sync_error(self, message: str, **kwargs) -> None:
        self.console_logger.error(self._format_message(message, **kwargs))

    def sync_warning(self, message: str, **kwargs) -> None:
        self.console_logger.warning(self._format_message(message, **kwargs))

    def sync_debug(self, message: str, **kwargs) -> None:
        self.console_logger.debug(self._format_message(message, **kwargs))
