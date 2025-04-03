"""Freckll Logging Module."""

import logging

logging.getLogger("freckll").addHandler(logging.NullHandler())


def setup_log(name: str) -> logging.Logger:
    """Build a logger class for the given name."""
    return logging.getLogger("freckll").getChild(name)


class Loggable:
    """Base class for loggable objects."""

    def __init__(self, name=None) -> None:
        """Initialize the logger."""
        if not name:
            name = self.__class__.__name__
        self._logger = setup_log(f"{name}")

    def info(self, message: str, *args, **kwargs) -> None:
        """See :class:`logging.Logger`."""
        self._logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """See :class:`logging.Logger`."""
        self._logger.warning(message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        """See :class:`logging.Logger`."""
        import inspect

        func = inspect.currentframe().f_back.f_code
        new_message = f"In: {func.co_name}()/line:{func.co_firstlineno} - {message}"
        self._logger.debug(new_message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """See :class:`logging.Logger`."""
        self._logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """See :class:`logging.Logger`."""
        self._logger.critical(message, *args, **kwargs)

    def error_and_raise(self, exception: Exception, message: str, *args, **kwargs):
        """Print error message and raises exception."""
        self._logger.error(message, *args, **kwargs)
        raise exception(message, *args, **kwargs)
