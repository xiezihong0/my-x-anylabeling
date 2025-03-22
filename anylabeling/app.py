import os

# 解决 Mac M1 芯片上的 "bus error" 问题
# 参考: https://stackoverflow.com/questions/73072612/
# Temporary fix for: bus error
# Source: https://stackoverflow.com/questions/73072612/
# why-does-np-linalg-solve-raise-bus-error-when-running-on-its-own-thread-mac-m1
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import codecs
import logging

import sys
# 添加当前目录到 Python 搜索路径，确保可以导入项目内部模块
sys.path.append(".")

import yaml
# Qt开发跨平台的 GUI（图形用户界面）应用
from PyQt5 import QtCore, QtWidgets

# 导入应用程序的元信息
from anylabeling.app_info import __appname__, __version__, __url__
# 导入配置管理函数
from anylabeling.config import get_config, save_config
from anylabeling import config as anylabeling_config
# 导入主窗口及相关工具
from anylabeling.views.mainwindow import MainWindow
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils import new_icon, gradient_text
from anylabeling.resources import resources


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    # 重置配置
    parser.add_argument(
        "--reset-config", action="store_true", help="reset qt config"
    )
    # 日志级别
    parser.add_argument(
        "--logger-level",
        default="info",
        choices=["debug", "info", "warning", "fatal", "error"],
        help="logger level",
    )
    # 要打开的图像或标签文件，或者包含数据的目录
    parser.add_argument(
        "filename",
        nargs="?",
        help=(
            "image or label filename; "
            "If a directory path is passed in, the folder will be loaded automatically"
        ),
    )
    # 指定输出文件或目录，若以 .json 结尾则认为是文件，否则认为是目录
    parser.add_argument(
        "--output",
        "-O",
        "-o",
        help=(
            "output file or directory (if it ends with .json it is "
            "recognized as file, else as directory)"
        ),
    )
    # 默认的配置文件路径
    default_config_file = os.path.join(
        os.path.expanduser("~"), ".xanylabelingrc"
    )
    # 指定配置文件或 YAML 格式的字符串
    parser.add_argument(
        "--config",
        dest="config",
        help=(
            "config file or yaml-format string (default:"
            f" {default_config_file})"
        ),
        default=default_config_file,
    )
    # GUI 相关的配置参数
    # config for the gui
    # 停止将图像数据存入 JSON 文件
    parser.add_argument(
        "--nodata",
        dest="store_data",
        action="store_false",
        help="stop storing image data to JSON file",
        default=argparse.SUPPRESS,
    )
    # 自动保存
    parser.add_argument(
        "--autosave",
        dest="auto_save",
        action="store_true",
        help="auto save",
        default=argparse.SUPPRESS,
    )
    # 不对标签进行排序
    parser.add_argument(
        "--nosortlabels",
        dest="sort_labels",
        action="store_false",
        help="stop sorting labels",
        default=argparse.SUPPRESS,
    )
    # 逗号分隔的标记列表，或包含标记的文件
    parser.add_argument(
        "--flags",
        help="comma separated list of flags OR file containing flags",
        default=argparse.SUPPRESS,
    )
    # YAML 格式的标签标记映射，或包含 JSON 格式映射的文件
    parser.add_argument(
        "--labelflags",
        dest="label_flags",
        help=r"yaml string of label specific flags OR file containing json "
        r"string of label specific flags (ex. {person-\d+: [male, tall], "
        r"dog-\d+: [black, brown, white], .*: [occluded]})",  # NOQA
        default=argparse.SUPPRESS,
    )
    # 逗号分隔的标签列表，或包含标签的文件
    parser.add_argument(
        "--labels",
        help="comma separated list of labels OR file containing labels",
        default=argparse.SUPPRESS,
    )
    # 标签验证类型
    parser.add_argument(
        "--validatelabel",
        dest="validate_label",
        choices=["exact"],
        help="label validation types",
        default=argparse.SUPPRESS,
    )
    # 保留上一帧的标注信息
    parser.add_argument(
        "--keep-prev",
        action="store_true",
        help="keep annotation of previous frame",
        default=argparse.SUPPRESS,
    )
    # 用于查找画布上最近顶点的误差范围
    parser.add_argument(
        "--epsilon",
        type=float,
        help="epsilon to find nearest vertex on canvas",
        default=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    # 处理 `--flags` 参数
    if hasattr(args, "flags"):
        if os.path.isfile(args.flags):
            with codecs.open(args.flags, "r", encoding="utf-8") as f:
                args.flags = [line.strip() for line in f if line.strip()]
        else:
            args.flags = [line for line in args.flags.split(",") if line]

    # 处理 `--labels` 参数
    if hasattr(args, "labels"):
        if os.path.isfile(args.labels):
            with codecs.open(args.labels, "r", encoding="utf-8") as f:
                args.labels = [line.strip() for line in f if line.strip()]
        else:
            args.labels = [line for line in args.labels.split(",") if line]

    # 处理 `--labelflags` 参数
    if hasattr(args, "label_flags"):
        if os.path.isfile(args.label_flags):
            with codecs.open(args.label_flags, "r", encoding="utf-8") as f:
                args.label_flags = yaml.safe_load(f)
        else:
            args.label_flags = yaml.safe_load(args.label_flags)

    # 配置参数
    config_from_args = args.__dict__
    reset_config = config_from_args.pop("reset_config")
    filename = config_from_args.pop("filename")
    output = config_from_args.pop("output")
    config_file_or_yaml = config_from_args.pop("config")
    logger_level = config_from_args.pop("logger_level")
    # 设置日志级别
    logger.setLevel(getattr(logging, logger_level.upper()))
    logger.info(f"🚀 {gradient_text(f'X-AnyLabeling v{__version__} launched!')}")
    logger.info(f"⭐ If you like it, give us a star: {__url__}")
    # 读取配置
    anylabeling_config.current_config_file = config_file_or_yaml
    config = get_config(config_file_or_yaml, config_from_args, show_msg=True)

    # 如果开启了标签验证但没有提供标签，报错退出
    if not config["labels"] and config["validate_label"]:
        logger.error(
            "--labels must be specified with --validatelabel or "
            "validate_label: true in the config file "
            "(ex. ~/.xanylabelingrc)."
        )
        sys.exit(1)

    # 解析输出路径
    output_file = None
    output_dir = None
    if output is not None:
        if output.endswith(".json"):
            output_file = output
        else:
            output_dir = output

    # 设置 Qt 语言翻译
    language = config.get("language", QtCore.QLocale.system().name())
    translator = QtCore.QTranslator()
    loaded_language = translator.load(
        ":/languages/translations/" + language + ".qm"
    )
    # 启用高 DPI 支持
    # Enable scaling for high dpi screens
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.AA_EnableHighDpiScaling, True
    )  # enable highdpi scaling
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.AA_UseHighDpiPixmaps, True
    )  # use highdpi icons
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)

    app = QtWidgets.QApplication(sys.argv)
    app.processEvents()

    app.setApplicationName(__appname__)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(new_icon("icon"))
    if loaded_language:
        app.installTranslator(translator)
    else:
        logger.warning(
            f"Failed to load translation for {language}. "
            "Using default language.",
        )
    win = MainWindow(
        app,
        config=config,
        filename=filename,
        output_file=output_file,
        output_dir=output_dir,
    )

    if reset_config:
        logger.info(f"Resetting Qt config: {win.settings.fileName()}")
        win.settings.clear()
        sys.exit(0)

    win.showMaximized()
    win.raise_()
    sys.exit(app.exec())


# this main block is required to generate executable by pyinstaller
if __name__ == "__main__":
    main()
