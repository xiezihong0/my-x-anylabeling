import os
import copy
import time
import yaml
import importlib.resources as pkg_resources
from threading import Lock


from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from anylabeling.utils import GenericWorker
from anylabeling.views.labeling.logger import logger
from anylabeling.config import get_config, save_config
from anylabeling.configs import auto_labeling as auto_labeling_configs
from anylabeling.services.auto_labeling.types import AutoLabelingResult


class ModelManager(QObject):
    """Model manager"""

    MAX_NUM_CUSTOM_MODELS = 5
    CUSTOM_MODELS = [
        "segment_anything",
        "segment_anything_2",
        "segment_anything_2_video",
        "sam_med2d",
        "sam_hq",
        "yolov5",
        "yolov6",
        "yolov7",
        "yolov8",
        "yolov8_seg",
        "yolox",
        "yolov5_resnet",
        "yolov6_face",
        "rtdetr",
        "yolo_nas",
        "yolox_dwpose",
        "clrnet",
        "ppocr_v4",
        "yolov5_sam",
        "efficientvit_sam",
        "yolov5_track",
        "damo_yolo",
        "yolov8_sahi",
        "grounding_sam",
        "grounding_sam2",
        "grounding_dino",
        "yolov5_obb",
        "gold_yolo",
        "yolov8_efficientvit_sam",
        "ram",
        "yolov5_seg",
        "yolov5_ram",
        "yolov8_pose",
        "pulc_attribute",
        "internimage_cls",
        "edge_sam",
        "yolov5_cls",
        "yolov8_cls",
        "yolov8_obb",
        "yolov5_car_plate",
        "rtmdet_pose",
        "yolov9",
        "yolow",
        "yolov10",
        "rmbg",
        "depth_anything",
        "depth_anything_v2",
        "yolow_ram",
        "rtdetrv2",
        "yolov8_det_track",
        "yolov8_seg_track",
        "yolov8_obb_track",
        "yolov8_pose_track",
    ]

    # =====定义多个信号（Signal），用于与其他组件进行通信=====
    # 当模型配置发生变化时发出信号，参数为模型配置的列表
    model_configs_changed = pyqtSignal(list)
    # 当模型状态发生变化时发出信号，参数为模型状态的字符串
    new_model_status = pyqtSignal(str)
    # 当模型加载完成时发出信号，参数为加载的模型配置字典
    model_loaded = pyqtSignal(dict)
    # 当有新的自动标注结果时发出信号，参数为自动标注结果
    new_auto_labeling_result = pyqtSignal(AutoLabelingResult)
    # 当自动分割模型被选中时发出信号
    auto_segmentation_model_selected = pyqtSignal()
    # 当自动分割模型被取消选中时发出信号
    auto_segmentation_model_unselected = pyqtSignal()
    # 当预测开始时发出信号
    prediction_started = pyqtSignal()
    # 当预测完成时发出信号
    prediction_finished = pyqtSignal()
    # 当请求下一个文件时发出信号
    request_next_files_requested = pyqtSignal()
    # 当输出模式发生变化时发出信号，参数为输出模式的配置和模式的名称
    output_modes_changed = pyqtSignal(dict, str)

    def __init__(self):
        super().__init__()
        # 存储模型配置列表
        self.model_configs = []
        # 存储已加载的模型配置
        self.loaded_model_config = None
        # 用于同步对已加载模型配置的访问
        self.loaded_model_config_lock = Lock()
        # 存储模型下载的工作线程
        self.model_download_worker = None
        # 存储模型下载的线程
        self.model_download_thread = None
        # 存储模型执行的线程
        self.model_execution_thread = None
        # 用于同步对模型执行线程的访问
        self.model_execution_thread_lock = Lock()
        # 加载模型配置
        self.load_model_configs()

    """加载模型配置"""
    def load_model_configs(self):
        """Load model configs"""
        # Load list of default models
        # 加载默认的模型列表
        with pkg_resources.open_text(
            auto_labeling_configs, "models.yaml"
        ) as f:
            # 解析models.yaml文件，获取默认模型列表
            model_list = yaml.safe_load(f)

        # 加载自定义模型列表
        # Load list of custom models
        custom_models = get_config().get("custom_models", [])
        for custom_model in custom_models:
            custom_model["is_custom_model"] = True

        # 移除无效或不存在的自定义模型
        # Remove invalid/not found custom models
        custom_models = [
            custom_model
            for custom_model in custom_models
            if os.path.isfile(custom_model.get("config_file", ""))
        ]
        config = get_config()
        # 更新配置中的自定义模型列表
        config["custom_models"] = custom_models
        # 保存配置
        save_config(config)

        model_list += custom_models
        # 加载所有模型的配置文件
        # Load model configs
        model_configs = []
        for model in model_list:
            model_config = {}
            config_file = model["config_file"]
            # 如果配置文件在资源文件中
            if config_file.startswith(":/"):  # Config file is in resources
                # 获取配置文件名称
                config_file_name = config_file[2:]
                with pkg_resources.open_text(
                    auto_labeling_configs, config_file_name
                ) as f:
                    # 解析配置文件
                    model_config = yaml.safe_load(f)
                    # 保存配置文件路径
                    model_config["config_file"] = config_file
            else:  # Config file is in local file system
                # 如果配置文件在本地文件系统中
                with open(config_file, "r", encoding="utf-8") as f:
                    # 解析配置文件
                    model_config = yaml.safe_load(f)
                    # 保存绝对路径的配置文件路径
                    model_config["config_file"] = os.path.normpath(
                        os.path.abspath(config_file)
                    )
            # 标记是否为自定义模型
            model_config["is_custom_model"] = model.get(
                "is_custom_model", False
            )
            model_configs.append(model_config)

        # 根据最后使用时间对模型进行排序
        # Sort by last used
        for i, model_config in enumerate(model_configs):
            # 保证集成模型的顺序不变
            # Keep order for integrated models
            if not model_config.get("is_custom_model", False):
                # 对于集成模型，使用负序号保证顺序
                model_config["last_used"] = -i
            else:
                # 对于自定义模型，使用最后一次使用的时间
                model_config["last_used"] = model_config.get(
                    "last_used", time.time()
                )
        # 按最后使用时间排序
        model_configs.sort(key=lambda x: x.get("last_used", 0), reverse=True)
        # 更新模型配置
        self.model_configs = model_configs
        # 发出模型配置变化的信号
        self.model_configs_changed.emit(model_configs)

    """返回模型配置列表"""
    def get_model_configs(self):
        """Return model infos"""
        # 返回当前的模型配置列表
        return self.model_configs

    """设置输出模式"""
    def set_output_mode(self, mode):
        """Set output mode"""
        if self.loaded_model_config and self.loaded_model_config["model"]:
            self.loaded_model_config["model"].set_output_mode(mode)

    """处理模型下载线程完成后的逻辑"""
    @pyqtSlot()
    def on_model_download_finished(self):
        """Handle model download thread finished"""
        # 如果模型成功加载，则发出相应的信号
        if self.loaded_model_config and self.loaded_model_config["model"]:
            # 发送新模型状态信号
            self.new_model_status.emit(
                self.tr("Model loaded. Ready for labeling.")
            )
            # 发送模型加载完成信号
            self.model_loaded.emit(self.loaded_model_config)
            self.output_modes_changed.emit(
                self.loaded_model_config["model"].Meta.output_modes,  # 输出模式列表
                self.loaded_model_config["model"].Meta.default_output_mode,  # 默认输出模式
            )
        else:
            # 若未成功加载模型，则发送空字典信号
            self.model_loaded.emit({})

    """在新线程中加载自定义模型"""
    def load_custom_model(self, config_file):
        """Run custom model loading in a thread"""
        # 规范化配置文件路径
        config_file = os.path.normpath(os.path.abspath(config_file))
        # 如果已有模型正在加载，则直接返回，避免同时加载多个模型
        if (
            self.model_download_thread is not None
            and self.model_download_thread.isRunning()
        ):
            logger.info(
                "Another model is being loaded. Please wait for it to finish."
            )
            return

        # 检查配置文件路径是否有效
        # Check config file path
        if not config_file or not os.path.isfile(config_file):
            logger.error(
                f"An error occurred while loading the custom model: "
                f"The model path is invalid."
            )
            self.new_model_status.emit(
                self.tr("Error in loading custom model: Invalid path.")
            )
            return

        # 读取并解析模型配置文件
        # Check config file content
        model_config = {}
        with open(config_file, "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f)
            model_config["config_file"] = os.path.abspath(config_file)
        # 检查解析是否成功
        if not model_config:
            logger.error(
                f"An error occurred while loading the custom model: "
                f"The config file is invalid."
            )
            self.new_model_status.emit(
                self.tr("Error in loading custom model: Invalid config file.")
            )
            return
        # 验证模型配置文件是否包含必要字段
        if (
            "type" not in model_config
            or "display_name" not in model_config
            or "name" not in model_config
            or model_config["type"] not in self.CUSTOM_MODELS
        ):
            # 根据缺少的字段输出不同的错误日志
            if "type" not in model_config:
                logger.error(
                    f"An error occurred while loading the custom model: "
                    f"The 'type' field is missing in the model configuration file."
                )
            elif "display_name" not in model_config:
                logger.error(
                    f"An error occurred while loading the custom model: "
                    f"The 'display_name' field is missing in the model configuration file."
                )
            elif "name" not in model_config:
                logger.error(
                    f"An error occurred while loading the custom model: "
                    f"The 'name' field is missing in the model configuration file."
                )
            else:
                logger.error(
                    f"An error occurred while loading the custom model: "
                    f"The model type {model_config['type']} is not supported."
                )
            self.new_model_status.emit(
                self.tr(
                    "Error in loading custom model: Invalid config file format."
                )
            )
            return

        # 添加或更新自定义模型配置
        # Add or replace custom model
        custom_models = get_config().get("custom_models", [])
        matched_index = None
        # 查找是否已有相同的模型配置
        for i, model in enumerate(custom_models):
            if os.path.normpath(model["config_file"]) == os.path.normpath(
                config_file
            ):
                matched_index = i
                break
        if matched_index is not None:
            # 若找到匹配的模型，则更新其“last_used”时间戳
            model_config["last_used"] = time.time()
            custom_models[matched_index] = model_config
        else:
            # 若超过最大自定义模型数，则删除最久未使用的模型
            if len(custom_models) >= self.MAX_NUM_CUSTOM_MODELS:
                custom_models.sort(
                    key=lambda x: x.get("last_used", 0), reverse=True
                )
                custom_models.pop()
            # 添加新模型到列表前部
            custom_models = [model_config] + custom_models

        # 保存更新后的配置
        # Save config
        config = get_config()
        config["custom_models"] = custom_models
        save_config(config)

        # 重新加载所有模型配置
        # Reload model configs
        self.load_model_configs()

        # 加载新模型
        # Load model
        self.load_model(model_config["config_file"])

    """在新线程中加载模型"""
    def load_model(self, config_file):
        """Run model loading in a thread"""
        # 如果已有模型正在加载，则直接返回，避免冲突
        if (
            self.model_download_thread is not None
            and self.model_download_thread.isRunning()
        ):
            logger.info(
                "Another model is being loaded. Please wait for it to finish."
            )
            return
        # 如果未指定配置文件，则卸载当前模型
        if not config_file:
            if self.model_download_worker is not None:
                try:
                    self.model_download_worker.finished.disconnect(
                        self.on_model_download_finished
                    )
                except TypeError:
                    pass
            # 卸载当前模型
            self.unload_model()
            # 发送无模型选中状态
            self.new_model_status.emit(self.tr("No model selected."))
            return
        # 查找模型的索引
        # Check and get model id
        model_id = None
        for i, model_config in enumerate(self.model_configs):
            if model_config["config_file"] == config_file:
                model_id = i
                break
        # 若未找到模型，则报错并返回
        if model_id is None:
            logger.error(
                f"An error occurred while loading the model: "
                f"The model name is invalid."
            )
            self.new_model_status.emit(
                self.tr("Error in loading model: Invalid model name.")
            )
            return
        # 创建一个新的线程用于加载模型
        self.model_download_thread = QThread()
        self.new_model_status.emit(
            self.tr("Loading model: {model_name}. Please wait...").format(
                model_name=self.model_configs[model_id]["display_name"]
            )
        )
        self.model_download_worker = GenericWorker(self._load_model, model_id)
        # 连接线程完成后的处理方法
        self.model_download_worker.finished.connect(
            self.on_model_download_finished
        )
        self.model_download_worker.finished.connect(
            self.model_download_thread.quit
        )
        # 将 worker 移动到新线程，并启动线程
        self.model_download_worker.moveToThread(self.model_download_thread)
        self.model_download_thread.started.connect(
            self.model_download_worker.run
        )
        self.model_download_thread.start()

    """加载对应yaml的模型信息"""
    def _load_model(self, model_id):
        """Load and return model info"""
        if self.loaded_model_config is not None:
            self.loaded_model_config["model"].unload()
            self.loaded_model_config = None
            self.auto_segmentation_model_unselected.emit()

        model_config = copy.deepcopy(self.model_configs[model_id])
        if model_config["type"] == "yolov5":
            from .yolov5 import YOLOv5

            try:
                model_config["model"] = YOLOv5(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov6":
            from .yolov6 import YOLOv6

            try:
                model_config["model"] = YOLOv6(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov7":
            from .yolov7 import YOLOv7

            try:
                model_config["model"] = YOLOv7(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov8_sahi":
            from .yolov8_sahi import YOLOv8_SAHI

            try:
                model_config["model"] = YOLOv8_SAHI(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov8":
            from .yolov8 import YOLOv8

            try:
                model_config["model"] = YOLOv8(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov9":
            from .yolov9 import YOLOv9

            try:
                model_config["model"] = YOLOv9(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov10":
            from .yolov10 import YOLOv10

            try:
                model_config["model"] = YOLOv10(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolow":
            from .yolow import YOLOW

            try:
                model_config["model"] = YOLOW(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov5_seg":
            from .yolov5_seg import YOLOv5_Seg

            try:
                model_config["model"] = YOLOv5_Seg(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov5_ram":
            from .yolov5_ram import YOLOv5_RAM

            try:
                model_config["model"] = YOLOv5_RAM(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolow_ram":
            from .yolow_ram import YOLOW_RAM

            try:
                model_config["model"] = YOLOW_RAM(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov8_seg":
            from .yolov8_seg import YOLOv8_Seg

            try:
                model_config["model"] = YOLOv8_Seg(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov8_obb":
            from .yolov8_obb import YOLOv8_OBB

            try:
                model_config["model"] = YOLOv8_OBB(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov8_pose":
            from .yolov8_pose import YOLOv8_Pose

            try:
                model_config["model"] = YOLOv8_Pose(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolox":
            from .yolox import YOLOX

            try:
                model_config["model"] = YOLOX(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolo_nas":
            from .yolo_nas import YOLO_NAS

            try:
                model_config["model"] = YOLO_NAS(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "damo_yolo":
            from .damo_yolo import DAMO_YOLO

            try:
                model_config["model"] = DAMO_YOLO(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "gold_yolo":
            from .gold_yolo import Gold_YOLO

            try:
                model_config["model"] = Gold_YOLO(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "grounding_dino":
            from .grounding_dino import Grounding_DINO

            try:
                model_config["model"] = Grounding_DINO(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "ram":
            from .ram import RAM

            try:
                model_config["model"] = RAM(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "internimage_cls":
            from .internimage_cls import InternImage_CLS

            try:
                model_config["model"] = InternImage_CLS(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "pulc_attribute":
            from .pulc_attribute import PULC_Attribute

            try:
                model_config["model"] = PULC_Attribute(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov5_sam":
            from .yolov5_sam import YOLOv5SegmentAnything

            try:
                model_config["model"] = YOLOv5SegmentAnything(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "yolov8_efficientvit_sam":
            from .yolov8_efficientvit_sam import YOLOv8_EfficientViT_SAM

            try:
                model_config["model"] = YOLOv8_EfficientViT_SAM(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "grounding_sam":
            from .grounding_sam import GroundingSAM

            try:
                model_config["model"] = GroundingSAM(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "grounding_sam2":
            from .grounding_sam2 import GroundingSAM2

            try:
                model_config["model"] = GroundingSAM2(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "yolov5_obb":
            from .yolov5_obb import YOLOv5OBB

            try:
                model_config["model"] = YOLOv5OBB(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "segment_anything":
            from .segment_anything import SegmentAnything

            try:
                model_config["model"] = SegmentAnything(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "segment_anything_2":
            from .segment_anything_2 import SegmentAnything2

            try:
                model_config["model"] = SegmentAnything2(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "segment_anything_2_video":
            try:
                from .segment_anything_2_video import SegmentAnything2Video

                model_config["model"] = SegmentAnything2Video(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "efficientvit_sam":
            from .efficientvit_sam import EfficientViT_SAM

            try:
                model_config["model"] = EfficientViT_SAM(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "sam_med2d":
            from .sam_med2d import SAM_Med2D

            try:
                model_config["model"] = SAM_Med2D(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "edge_sam":
            from .edge_sam import EdgeSAM

            try:
                model_config["model"] = EdgeSAM(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "sam_hq":
            from .sam_hq import SAM_HQ

            try:
                model_config["model"] = SAM_HQ(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "yolov5_resnet":
            from .yolov5_resnet import YOLOv5_ResNet

            try:
                model_config["model"] = YOLOv5_ResNet(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "rtdetr":
            from .rtdetr import RTDETR

            try:
                model_config["model"] = RTDETR(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "rtdetrv2":
            from .rtdetrv2 import RTDETRv2

            try:
                model_config["model"] = RTDETRv2(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov6_face":
            from .yolov6_face import YOLOv6Face

            try:
                model_config["model"] = YOLOv6Face(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolox_dwpose":
            from .yolox_dwpose import YOLOX_DWPose

            try:
                model_config["model"] = YOLOX_DWPose(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "rtmdet_pose":
            from .rtmdet_pose import RTMDet_Pose

            try:
                model_config["model"] = RTMDet_Pose(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "clrnet":
            from .clrnet import CLRNet

            try:
                model_config["model"] = CLRNet(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "ppocr_v4":
            from .ppocr_v4 import PPOCRv4

            try:
                model_config["model"] = PPOCRv4(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov5_cls":
            from .yolov5_cls import YOLOv5_CLS

            try:
                model_config["model"] = YOLOv5_CLS(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov5_car_plate":
            from .yolov5_car_plate import YOLOv5CarPlateDetRec

            try:
                model_config["model"] = YOLOv5CarPlateDetRec(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov8_cls":
            from .yolov8_cls import YOLOv8_CLS

            try:
                model_config["model"] = YOLOv8_CLS(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov5_det_track":
            from .yolov5_det_track import YOLOv5_Det_Tracker

            try:
                model_config["model"] = YOLOv5_Det_Tracker(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov8_det_track":
            from .yolov8_det_track import YOLOv8_Det_Tracker

            try:
                model_config["model"] = YOLOv8_Det_Tracker(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov8_seg_track":
            from .yolov8_seg_track import YOLOv8_Seg_Tracker

            try:
                model_config["model"] = YOLOv8_Seg_Tracker(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov8_obb_track":
            from .yolov8_obb_track import YOLOv8_Obb_Tracker

            try:
                model_config["model"] = YOLOv8_Obb_Tracker(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "yolov8_pose_track":
            from .yolov8_pose_track import YOLOv8_Pose_Tracker

            try:
                model_config["model"] = YOLOv8_Pose_Tracker(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "rmbg":
            from .rmbg import RMBG

            try:
                model_config["model"] = RMBG(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "depth_anything":
            from .depth_anything import DepthAnything

            try:
                model_config["model"] = DepthAnything(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        elif model_config["type"] == "depth_anything_v2":
            from .depth_anything_v2 import DepthAnythingV2

            try:
                model_config["model"] = DepthAnythingV2(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(f"✅ Model loaded successfully: {model_config['type']}")
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(f"❌ Error in loading model: {model_config['type']} with error: {str(e)}")
                return
        else:
            raise Exception(f"Unknown model type: {model_config['type']}")

        self.loaded_model_config = model_config
        return self.loaded_model_config

    """
    设置缓存的自动标签。
    仅当 `loaded_model_config["type"]` 在 `valid_models` 列表中时，调用模型的 `set_cache_auto_label` 方法。
    Args:
        text (str): 要缓存的标签文本。
        gid (str): 关联的分组 ID。
    """
    def set_cache_auto_label(self, text, gid):
        """Set cache auto label"""
        valid_models = [
            "segment_anything_2_video",
        ]
        if (
            self.loaded_model_config is not None
            and self.loaded_model_config["type"] in valid_models
        ):
            self.loaded_model_config["model"].set_cache_auto_label(text, gid)

    """
    设置自动标注的标记点。
    适用于 `marks_model_list` 中的模型，例如 `segment_anything` 系列和 `yolov8_efficientvit_sam`。
    Args:
        marks (list): 标注点数据，具体格式取决于具体模型的需求。
    """
    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks
        (For example, for segment_anything model, it is the marks for)
        """
        marks_model_list = [
            "segment_anything",
            "segment_anything_2",
            "segment_anything_2_video",
            "sam_med2d",
            "sam_hq",
            "yolov5_sam",
            "efficientvit_sam",
            "yolov8_efficientvit_sam",
            "grounding_sam",
            "grounding_sam2",
            "edge_sam",
        ]
        if (
            self.loaded_model_config is None
            or self.loaded_model_config["type"] not in marks_model_list
        ):
            return
        self.loaded_model_config["model"].set_auto_labeling_marks(marks)

    """重置跟踪器状态。
    适用于 `model_list` 中的目标检测跟踪模型，清除所有已跟踪的对象及内部状态。
    """
    def set_auto_labeling_reset_tracker(self):
        """Resets the tracker to its initial state,
        clearing all tracked objects and internal states.
        """
        model_list = [
            "yolov5_det_track",
            "yolov8_det_track",
            "yolov8_obb_track",
            "yolov8_seg_track",
            "yolov8_pose_track",
            "segment_anything_2_video",
        ]
        if (
            self.loaded_model_config is None
            or self.loaded_model_config["type"] not in model_list
        ):
            return
        self.loaded_model_config["model"].set_auto_labeling_reset_tracker()

    """
    设置自动标注的置信度阈值。
    适用于 `model_list` 中的目标检测模型。
    Args:
        value (float): 置信度阈值，用于过滤低置信度的检测结果。
    """
    def set_auto_labeling_conf(self, value):
        """Set auto labeling confidences"""
        model_list = [
            "damo_yolo",
            "gold_yolo",
            "grounding_dino",
            "rtdetr",
            "rtdetrv2",
            "yolo_nas",
            "yolov5_obb",
            "yolov5_seg",
            "yolov5_det_track",
            "yolov5",
            "yolov6",
            "yolov6_face",
            "yolov7",
            "yolov8_obb",
            "yolov8_pose",
            "yolov8_seg",
            "yolov8_det_track",
            "yolov8_seg_track",
            "yolov8_obb_track",
            "yolov8_pose_track",
            "yolov8",
            "yolov9",
            "yolov10",
            "yolow",
            "yolox",
        ]
        if (
            self.loaded_model_config is None
            or self.loaded_model_config["type"] not in model_list
        ):
            return
        self.loaded_model_config["model"].set_auto_labeling_conf(value)

    """
    设置自动标注的 IOU（交并比）阈值。
    适用于 `model_list` 中的目标检测模型，用于定义两个目标重叠的最小程度。
    Args:
        value (float): IOU 阈值，范围一般在 0 到 1 之间。
    """
    def set_auto_labeling_iou(self, value):
        """Set auto labeling iou"""
        model_list = [
            "damo_yolo",
            "gold_yolo",
            "yolo_nas",
            "yolov5_obb",
            "yolov5_seg",
            "yolov5_det_track",
            "yolov5",
            "yolov6",
            "yolov7",
            "yolov8_obb",
            "yolov8_pose",
            "yolov8_seg",
            "yolov8_det_track",
            "yolov8_seg_track",
            "yolov8_obb_track",
            "yolov8_pose_track",
            "yolov8",
            "yolov9",
            "yolox",
        ]
        if (
            self.loaded_model_config is None
            or self.loaded_model_config["type"] not in model_list
        ):
            return
        self.loaded_model_config["model"].set_auto_labeling_iou(value)

    """
    设置是否在自动标注时保留已有的标注信息。
    适用于 `model_list` 中的目标检测模型。该方法允许用户决定自动标注时是否覆盖已有的标注数据。
    Args:
        state (bool): 是否保留现有标注数据的状态。
    """
    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        model_list = [
            "damo_yolo",
            "gold_yolo",
            "grounding_dino",
            "rtdetr",
            "rtdetrv2",
            "yolo_nas",
            "yolov5_obb",
            "yolov5_seg",
            "yolov5_det_track",
            "yolov5",
            "yolov6",
            "yolov7",
            "yolov8_obb",
            "yolov8_pose",
            "yolov8_seg",
            "yolov8_det_track",
            "yolov8_seg_track",
            "yolov8_obb_track",
            "yolov8_pose_track",
            "yolov8",
            "yolov9",
            "yolov10",
            "yolow",
            "yolox",
        ]
        if (
            self.loaded_model_config is not None
            and self.loaded_model_config["type"] in model_list
        ):
            self.loaded_model_config[
                "model"
            ].set_auto_labeling_preserve_existing_annotations_state(state)

    """
    设置自动标注的提示（Prompt）。
    适用于 `segment_anything_2_video` 模型，调用该方法以应用用户定义的提示信息。
    """
    def set_auto_labeling_prompt(self):
        model_list = ["segment_anything_2_video"]
        if (
            self.loaded_model_config is not None
            and self.loaded_model_config["type"] in model_list
        ):
            self.loaded_model_config["model"].set_auto_labeling_prompt()

    """
    卸载当前加载的模型。
    释放已加载的模型资源，并将 `loaded_model_config` 设为 `None`。
    """
    def unload_model(self):
        """Unload model"""
        if self.loaded_model_config is not None:
            self.loaded_model_config["model"].unload()
            self.loaded_model_config = None

    """
    执行形状预测。
    该函数是阻塞的，可能需要较长时间来完成预测。建议使用 `predict_shapes_threading` 进行非阻塞调用。
    Args:
        image: 输入图像。
        filename (str, optional): 图像文件名。
        text_prompt (str, optional): 文字提示信息（如果模型支持）。
        run_tracker (bool, optional): 是否运行跟踪器（适用于跟踪模型）。
    """
    def predict_shapes(
        self, image, filename=None, text_prompt=None, run_tracker=False
    ):
        """Predict shapes.
        NOTE: This function is blocking. The model can take a long time to
        predict. So it is recommended to use predict_shapes_threading instead.
        """
        if self.loaded_model_config is None:
            self.new_model_status.emit(
                self.tr("Model is not loaded. Choose a mode to continue.")
            )
            self.prediction_finished.emit()
            return
        try:
            if text_prompt is not None:
                auto_labeling_result = self.loaded_model_config[
                    "model"
                ].predict_shapes(image, filename, text_prompt=text_prompt)
            elif run_tracker is True:
                auto_labeling_result = self.loaded_model_config[
                    "model"
                ].predict_shapes(image, filename, run_tracker=run_tracker)
            else:
                auto_labeling_result = self.loaded_model_config[
                    "model"
                ].predict_shapes(image, filename)
            # 触发信号，通知前端或其他组件预测结果已生成
            self.new_auto_labeling_result.emit(auto_labeling_result)
            self.new_model_status.emit(
                self.tr("Finished inferencing AI model. Check the result.")
            )
        except Exception as e:  # noqa
            logger.error(f"Error in predict_shapes: {e}")
            self.new_model_status.emit(
                self.tr(
                    f"Error in model prediction: {e}. Please check the model."
                )
            )
        self.prediction_finished.emit()

    """
    在独立线程中执行形状预测。
    该方法避免主线程阻塞，并在后台执行预测任务。
    Args:
        image: 输入图像。
        filename (str, optional): 图像文件名。
        text_prompt (str, optional): 文字提示信息（如果模型支持）。
        run_tracker (bool, optional): 是否运行跟踪器（适用于跟踪模型）。
    """
    @pyqtSlot()
    def predict_shapes_threading(
        self, image, filename=None, text_prompt=None, run_tracker=False
    ):
        """Predict shapes.
        This function starts a thread to run the prediction.
        """
        if self.loaded_model_config is None:
            self.new_model_status.emit(
                self.tr("Model is not loaded. Choose a mode to continue.")
            )
            return
        self.new_model_status.emit(
            self.tr("Inferencing AI model. Please wait...")
        )
        self.prediction_started.emit()

        with self.model_execution_thread_lock:
            # 如果已有正在运行的推理线程，则不创建新线程
            if (
                self.model_execution_thread is not None
                and self.model_execution_thread.isRunning()
            ):
                self.new_model_status.emit(
                    self.tr(
                        "Another model is being executed."
                        " Please wait for it to finish."
                    )
                )
                self.prediction_finished.emit()
                return
            # 创建新线程来执行模型推理
            self.model_execution_thread = QThread()
            if text_prompt is not None:
                self.model_execution_worker = GenericWorker(
                    self.predict_shapes,
                    image,
                    filename,
                    text_prompt=text_prompt,
                )
            elif run_tracker is True:
                self.model_execution_worker = GenericWorker(
                    self.predict_shapes,
                    image,
                    filename,
                    run_tracker=run_tracker,
                )
            else:
                self.model_execution_worker = GenericWorker(
                    self.predict_shapes, image, filename
                )
            # 绑定线程生命周期管理
            self.model_execution_worker.finished.connect(
                self.model_execution_thread.quit
            )
            self.model_execution_worker.moveToThread(
                self.model_execution_thread
            )
            self.model_execution_thread.started.connect(
                self.model_execution_worker.run
            )
            self.model_execution_thread.start()

    """
    提前对即将处理的文件进行推理，以减少推理时间。
    适用于 `segment_anything` 系列的模型，以便在加载文件时提高推理效率。
    Args:
        next_files (list): 即将进行推理的文件列表。
    """
    def on_next_files_changed(self, next_files):
        """Run prediction on next files in advance to save inference time later"""
        if self.loaded_model_config is None:
            return
        # 仅支持 segment_anything 相关的模型
        # Currently only segment_anything-like model supports this feature
        if self.loaded_model_config["type"] not in [
            "segment_anything",
            "segment_anything_2",
            "sam_med2d",
            "sam_hq",
            "yolov5_sam",
            "efficientvit_sam",
            "yolov8_efficientvit_sam",
            "grounding_sam",
            "grounding_sam2",
            "edge_sam",
        ]:
            return

        self.loaded_model_config["model"].on_next_files_changed(next_files)
