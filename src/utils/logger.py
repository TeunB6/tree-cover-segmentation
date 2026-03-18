import logging
from pathlib import Path
from datetime import datetime
from io import StringIO
from rich.console import Console
from typing import Any


class Logger:
    """Logging utility class for managing application logging."""

    def __init__(self, name: str) -> None:
        """Initialize and setup a logger instance.

        Args:
            name: The name of the logger (typically __name__)
        """
        self.console = Console()
        self._logger = self._setup_logger(name)

    def _setup_logger(self, name: str) -> logging.Logger:
        """Setup and return a logger for the given module.

        Args:
            name: The name of the logger

        Returns:
            A configured logger instance
        """
        logs_dir = Path(__file__).parent.parent.parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Create logger.
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Remove existing handlers to avoid duplicates for UX.
        logger.handlers.clear()
        logger.propagate = False

        # Create formatters.
        detailed_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Main log file with detailed formatting.
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = logs_dir / f"{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

        return logger

    def log_and_print(self, rich_object: Any) -> None:
        """Log a rich object as text and print to console.

        Args:
            rich_object: A rich renderable object.
        """
        # Capture rich object as text.
        string_io = StringIO()
        temp_console = Console(file=string_io, width=100, legacy_windows=False)
        temp_console.print(rich_object)
        text_output = string_io.getvalue().strip()

        for line in text_output.split("\n"):
            if line.strip():
                self._logger.info(line)

        # Print to console.
        self.console.print(rich_object)

    def __getattr__(self, name):
        """Delegate attribute access to the underlying logger."""
        return getattr(self._logger, name)
