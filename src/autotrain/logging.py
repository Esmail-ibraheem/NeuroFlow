import os
import sys
from dataclasses import dataclass
from pathlib import Path

from accelerate.state import PartialState
from loguru import logger


@dataclass
class Logger:
    def __post_init__(self):
        self.log_format = (
            "<level>{level: <8}</level> | "
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        self.logger = logger
        default_log_path = Path(__file__).resolve().parents[1] / "autotrain.log"
        env_log_path = os.environ.get("AUTOTRAIN_LOGFILE")
        self.log_path = Path(env_log_path) if env_log_path else default_log_path
        self.setup_logger()

    def _should_log(self, record):
        return PartialState().is_main_process

    def setup_logger(self):
        self.logger.remove()
        self.logger.add(
            sys.stdout,
            format=self.log_format,
            filter=lambda x: self._should_log(x),
        )
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.add(
                self.log_path,
                format=self.log_format,
                enqueue=True,
                rotation="5 MB",
                retention=3,
                filter=lambda x: self._should_log(x),
            )
        except Exception as exc:  # pragma: no cover - best-effort file logging
            self.logger.warning(f"Failed to initialize log file at {self.log_path}: {exc}")

    def get_logger(self):
        return self.logger
