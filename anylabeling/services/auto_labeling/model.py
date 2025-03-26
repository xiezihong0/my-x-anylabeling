import os
import pathlib
import yaml
import onnx
import urllib.request
from urllib.parse import urlparse

from PyQt5.QtCore import QCoreApplication

import ssl

ssl._create_default_https_context = (
    ssl._create_unverified_context
)  # Prevent issue when downloading models behind a proxy

import socket

socket.setdefaulttimeout(240)  # Prevent timeout when downloading models

from abc import abstractmethod


from PyQt5.QtCore import QFile, QObject
from PyQt5.QtGui import QImage

from .types import AutoLabelingResult
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.label_file import LabelFile, LabelFileError

"""
该类表示一个基础模型，提供配置加载、模型路径解析、下载以及推理相关的方法。
"""
class Model(QObject):

    BASE_DOWNLOAD_URL = (
        "https://github.com/CVHub520/X-AnyLabeling/releases/tag"
    )

    """
    存储模型元数据，如必须的配置项、UI 组件及默认输出模式。
    """
    class Meta(QObject):
        # 必需的配置项
        required_config_names = []
        # UI 组件
        widgets = ["button_run"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        # 默认的输出模式
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        """
        初始化模型，包括加载配置、检查配置项、设置默认输出模式等。

        :param model_config: 模型的配置文件路径或配置字典
        :param on_message: 消息回调函数
        """
        super().__init__()
        self.on_message = on_message
        # Load and check config
        # 加载并检查配置
        if isinstance(model_config, str):
            # 如果传入的是字符串（配置文件路径）
            if not os.path.isfile(model_config):
                raise FileNotFoundError(
                    QCoreApplication.translate(
                        "Model", "Config file not found: {model_config}"
                    ).format(model_config=model_config)
                )
            # 解析 YAML 配置文件
            with open(model_config, "r") as f:
                self.config = yaml.safe_load(f)
        elif isinstance(model_config, dict):
            # 如果传入的是字典，直接赋值
            self.config = model_config
        else:
            raise ValueError(
                QCoreApplication.translate(
                    "Model", "Unknown config type: {type}"
                ).format(type=type(model_config))
            )
        # 检查是否缺少必要的配置项
        self.check_missing_config(
            config_names=self.Meta.required_config_names,
            config=self.config,
        )
        # 设置默认输出模式
        self.output_mode = self.Meta.default_output_mode

    """
    获取 UI 需要显示的组件。
    """
    def get_required_widgets(self):
        """
        Get required widgets for showing in UI
        """
        return self.Meta.widgets

    @staticmethod
    def allow_migrate_data():
        """
        检查并执行数据迁移，将旧的 anylabeling_data 目录重命名为 xanylabeling_data。

        :return: 是否允许数据迁移
        """
        home_dir = os.path.expanduser("~")
        old_model_path = os.path.join(home_dir, "anylabeling_data")
        new_model_path = os.path.join(home_dir, "xanylabeling_data")

        if os.path.exists(new_model_path) or not os.path.exists(
            old_model_path
        ):
            # 已迁移或旧数据不存在，允许继续
            return True

        # 检查是否有写权限
        # Check if the current env have write permissions
        if not os.access(home_dir, os.W_OK):
            return False

        # 尝试迁移数据
        # Attempt to migrate data
        try:
            os.rename(old_model_path, new_model_path)
            return True
        except Exception as e:
            logger.error(f"An error occurred during data migration: {str(e)}")
            return False

    """
    获取模型的绝对路径，如果是 URL，则下载模型。

    :param model_config: 模型配置字典
    :param model_path_field_name: 配置中模型路径的字段名
    :return: 模型的绝对路径
    """
    def get_model_abs_path(self, model_config, model_path_field_name):
        """
        Get model absolute path from config path or download from url
        """
        # Try getting model path from config folder
        model_path = model_config[model_path_field_name]

        # 判断模型路径是否为本地路径
        # Model path is a local path
        if not model_path.startswith(("http://", "https://")):
            # 尝试从执行目录获取绝对路径
            # Relative path to executable or absolute path?
            model_abs_path = os.path.abspath(model_path)
            if os.path.exists(model_abs_path):
                return model_abs_path

            # 尝试从配置文件所在目录获取路径
            # Relative path to config file?
            config_file_path = model_config["config_file"]
            config_folder = os.path.dirname(config_file_path)
            model_abs_path = os.path.abspath(
                os.path.join(config_folder, model_path)
            )
            if os.path.exists(model_abs_path):
                return model_abs_path

            raise QCoreApplication.translate(
                "Model", "Model path not found: {model_path}"
            ).format(model_path=model_path)

        # 如果是 URL，则下载模型
        # Download model from url
        self.on_message(
            QCoreApplication.translate(
                "Model", "Downloading model from registry..."
            )
        )

        # Build download url
        def get_filename_from_url(url):
            a = urlparse(url)
            return os.path.basename(a.path)

        filename = get_filename_from_url(model_path)
        download_url = model_path

        # Continue with the rest of your function logic
        migrate_flag = self.allow_migrate_data()
        home_dir = os.path.expanduser("~")
        data_dir = "xanylabeling_data" if migrate_flag else "anylabeling_data"

        # Create model folder
        home_dir = os.path.expanduser("~")
        model_path = os.path.abspath(os.path.join(home_dir, data_dir))
        model_abs_path = os.path.abspath(
            os.path.join(
                model_path,
                "models",
                model_config["name"],
                filename,
            )
        )
        # 如果模型已经存在，则检查 ONNX 文件的完整性
        if os.path.exists(model_abs_path):
            if model_abs_path.lower().endswith(".onnx"):
                try:
                    onnx.checker.check_model(model_abs_path)
                except onnx.checker.ValidationError as e:
                    logger.error(f"{str(e)}")
                    logger.warning("Action: Delete and redownload...")
                    try:
                        os.remove(model_abs_path)
                    except Exception as e:  # noqa
                        logger.error(f"Could not delete: {str(e)}")
                else:
                    return model_abs_path
            else:
                return model_abs_path
        pathlib.Path(model_abs_path).parent.mkdir(parents=True, exist_ok=True)

        # Download url
        ellipsis_download_url = download_url
        if len(download_url) > 40:
            ellipsis_download_url = (
                download_url[:20] + "..." + download_url[-20:]
            )
        logger.info(
            f"Downloading {ellipsis_download_url} to {model_abs_path}"
        )
        try:
            # Download and show progress
            def _progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                self.on_message(
                    QCoreApplication.translate(
                        "Model", "Downloading {download_url}: {percent}%"
                    ).format(
                        download_url=ellipsis_download_url, percent=percent
                    )
                )

            urllib.request.urlretrieve(
                download_url, model_abs_path, reporthook=_progress
            )
        except Exception as e:  # noqa
            logger.error(f"Could not download {download_url}: {e}")
            self.on_message(f"Could not download {download_url}")
            return None

        return model_abs_path

    """
    检查配置项是否缺失。
    """
    def check_missing_config(self, config_names, config):
        """
        Check if config has all required config names
        """
        for name in config_names:
            if name not in config:
                raise Exception(f"Missing config: {name}")

    """
    进行模型推理，返回预测结果。
    """
    @abstractmethod
    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        """
        Predict image and return AnyLabeling shapes
        """
        raise NotImplementedError

    """
    释放模型所占用的内存资源。
    """
    @abstractmethod
    def unload(self):
        """
        Unload memory
        """
        raise NotImplementedError

    """
    从文件中加载图像，并返回 QImage 对象。
    """
    @staticmethod
    def load_image_from_filename(filename):
        """Load image from labeling file and return image data and image path."""
        label_file = os.path.splitext(filename)[0] + ".json"
        if QFile.exists(label_file) and LabelFile.is_label_file(label_file):
            try:
                label_file = LabelFile(label_file)
            except LabelFileError as e:
                logger.error("Error reading {}: {}".format(label_file, e))
                return None, None
            image_data = label_file.image_data
        else:
            image_data = LabelFile.load_image_file(filename)
        image = QImage.fromData(image_data)
        if image.isNull():
            logger.error("Error reading {}".format(filename))
        return image

    def on_next_files_changed(self, next_files):
        """
        Handle next files changed. This function can preload next files
        and run inference to save time for user.
        """
        pass

    """
    设置输出模式。
    """
    def set_output_mode(self, mode):
        """
        Set output mode
        """
        self.output_mode = mode
