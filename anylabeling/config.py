import os.path as osp
import shutil
import yaml

try:
    import importlib.resources as pkg_resources
except ImportError:
    # 兼容 Python 3.7 以下版本，使用 `importlib_resources` 作为替代。
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from anylabeling import configs as anylabeling_configs
from anylabeling.views.labeling.logger import logger

# 当前正在使用的配置文件路径
current_config_file = None

def update_dict(target_dict, new_dict, validate_item=None):
    """
    递归更新目标字典的键值对。

    参数：
    - target_dict: 需要更新的目标字典
    - new_dict: 用于更新的字典
    - validate_item: 可选的校验函数，用于验证键值对
    """
    for key, value in new_dict.items():
        if validate_item:
            validate_item(key, value)
        if key not in target_dict:
            logger.warning(f"Skipping unexpected key in config: {key}")
            continue
        if isinstance(target_dict[key], dict) and isinstance(value, dict):
            update_dict(target_dict[key], value, validate_item=validate_item)
        else:
            target_dict[key] = value


def save_config(config):
    """
    将配置保存到用户主目录下的 `.xanylabelingrc` 文件。

    参数：
    - config: 要保存的配置字典
    """
    user_config_file = osp.join(osp.expanduser("~"), ".xanylabelingrc")
    try:
        with open(user_config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True)
    except Exception:  # noqa
        logger.warning(f"Failed to save config: {user_config_file}")


def get_default_config():
    """
    获取默认配置。

    若旧版本的 `.anylabelingrc` 配置文件存在，则会迁移到新路径 `.xanylabelingrc`。

    返回：
    - config: 默认配置字典
    """
    old_cfg_file = osp.join(osp.expanduser("~"), ".anylabelingrc")
    new_cfg_file = osp.join(osp.expanduser("~"), ".xanylabelingrc")
    if osp.exists(old_cfg_file) and not osp.exists(new_cfg_file):
        shutil.copyfile(old_cfg_file, new_cfg_file)
    config_file = "xanylabeling_config.yaml"
    with pkg_resources.open_text(anylabeling_configs, config_file) as f:
        config = yaml.safe_load(f)
    # 如果新配置文件不存在，则将默认配置保存到 `.xanylabelingrc`
    # Save default config to ~/.xanylabelingrc
    if not osp.exists(osp.join(osp.expanduser("~"), ".xanylabelingrc")):
        save_config(config)

    return config


def validate_config_item(key, value):
    """
    验证配置项是否合法。

    参数：
    - key: 配置项名称
    - value: 配置项的值

    可能引发：
    - ValueError: 如果配置项的值不符合预期。
    """
    if key == "validate_label" and value not in [None, "exact"]:
        raise ValueError(
            f"Unexpected value for config key 'validate_label': {value}"
        )
    if key == "shape_color" and value not in [None, "auto", "manual"]:
        raise ValueError(
            f"Unexpected value for config key 'shape_color': {value}"
        )
    if key == "labels" and value is not None and len(value) != len(set(value)):
        raise ValueError(
            f"Duplicates are detected for config key 'labels': {value}"
        )


def get_config(config_file_or_yaml=None, config_from_args=None, show_msg=False):
    """
    获取完整的配置。

    1. 加载默认配置
    2. 从文件或 YAML 字符串加载用户自定义配置
    3. 使用命令行参数更新配置

    参数：
    - config_file_or_yaml: 指定的配置文件路径或 YAML 字符串
    - config_from_args: 由命令行传入的参数配置
    - show_msg: 是否在日志中显示配置信息

    返回：
    - config: 最终合并后的配置字典
    """
    # 加载默认配置
    # 1. Load default configuration
    config = get_default_config()

    # 如果未提供自定义配置，则使用全局 `current_config_file`
    # 2. Load configuration from file or YAML string
    if not config_file_or_yaml:
        config_file_or_yaml = current_config_file

    # 加载 YAML 配置
    config_from_yaml = yaml.safe_load(config_file_or_yaml)
    if not isinstance(config_from_yaml, dict):
        with open(config_file_or_yaml, encoding="utf-8") as f:
            config_from_yaml = yaml.safe_load(f)
    update_dict(config, config_from_yaml, validate_item=validate_config_item)
    if show_msg:
        logger.info(f"🔧️ Initializing config from local file: {config_file_or_yaml}")

    # 处理命令行参数并更新配置
    # 3. Update configuration with command line arguments
    if config_from_args:
        update_dict(config, config_from_args, validate_item=validate_config_item)
        if show_msg:
            logger.info(f"🔄 Updated config from CLI arguments: {config_from_args}")

    return config
