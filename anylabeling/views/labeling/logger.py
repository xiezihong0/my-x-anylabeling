import datetime
import logging
import sys
from functools import wraps
from typing import Callable, Dict

import termcolor

# 在 Windows 平台上，调整标准输出和错误输出的编码，以支持 Unicode 字符
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', newline='\n')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace', newline='\n')

# 定义不同日志级别的颜色映射
COLORS: Dict[str, str] = {
    "WARNING": "yellow",
    "INFO": "white",
    "DEBUG": "blue",
    "CRITICAL": "red",
    "ERROR": "red",
}

def singleton(cls):
    """
    单例模式装饰器，确保一个类的实例在整个程序中唯一。
    """
    instances = {}
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

class ColoredFormatter(logging.Formatter):
    """
    自定义日志格式化器，支持带颜色的日志输出。
    """
    def __init__(self, fmt: str, use_color: bool = True):
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录，如果启用颜色，则对日志信息进行着色。
        """
        if self.use_color and record.levelname in COLORS:
            record = self._color_record(record)
        record.asctime = self.formatTime(record, self.datefmt)
        return super().format(record)

    def _color_record(self, record: logging.LogRecord) -> logging.LogRecord:
        """
        使用 termcolor 进行日志级别、时间、模块等信息的着色。
        """
        colored = lambda text, color: termcolor.colored(text, color=color, attrs={"bold": True})
        
        record.levelname2 = colored(f"{record.levelname:<7}", COLORS[record.levelname])
        record.message2 = colored(record.msg, COLORS[record.levelname])
        record.asctime2 = termcolor.colored(self.formatTime(record, self.datefmt), color="green")
        record.module2 = termcolor.colored(record.module, color="cyan")
        record.funcName2 = termcolor.colored(record.funcName, color="cyan")
        record.lineno2 = termcolor.colored(record.lineno, color="cyan")
        
        return record

@singleton
class AppLogger:
    """
    应用程序日志类，使用单例模式，确保全局只有一个日志实例。
    """
    def __init__(self, name="X-AnyLabeling"):
        self.logger = logging.getLogger(name)
        self.logger.propagate = False
        self._setup_handler()

    def _setup_handler(self):
        """
        配置日志处理器，使用自定义的 ColoredFormatter 进行格式化。
        """
        stream_handler = logging.StreamHandler(sys.stderr)
        handler_format = ColoredFormatter(
            "%(asctime)s | %(levelname2)s | %(module2)s:%(funcName2)s:%(lineno2)s - %(message2)s"
        )
        stream_handler.setFormatter(handler_format)
        self.logger.addHandler(stream_handler)

    def __getattr__(self, name: str) -> Callable:
        """
        代理日志对象的方法，使 AppLogger 具备标准日志功能。
        """
        return getattr(self.logger, name)

    def set_level(self, level: str):
        """
        设置日志级别。
        """
        self.logger.setLevel(level)

# 创建全局日志对象
logger = AppLogger()