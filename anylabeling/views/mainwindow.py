"""This module defines the main application window"""

from PyQt5.QtWidgets import QMainWindow, QStatusBar, QVBoxLayout, QWidget

from ..app_info import __appdescription__, __appname__
from .labeling.label_wrapper import LabelingWrapper


class MainWindow(QMainWindow):
    """Main application window"""

    """"
    主应用窗口类，继承自
    QMainWindow。

    该类负责创建主窗口，并包含主要的
    GUI
    组件，如标签工具（LabelingWrapper）、
    状态栏等。
    """

    def __init__(
        self,
        app,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        """
        初始化主窗口。

        参数：
        - app: QApplication 实例
        - config: 配置文件
        - filename: 要加载的文件名
        - output: 输出路径
        - output_file: 指定的输出文件
        - output_dir: 指定的输出目录
        """
        super().__init__()
        self.app = app
        self.config = config

        # 设置窗口边距和标题
        self.setContentsMargins(0, 0, 0, 0)
        self.setWindowTitle(__appname__)

        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        # 创建 LabelingWrapper 组件，用于处理标注任务
        self.labeling_widget = LabelingWrapper(
            self,
            config=config,
            filename=filename,
            output=output,
            output_file=output_file,
            output_dir=output_dir,
        )
        main_layout.addWidget(self.labeling_widget)
        # 将主布局应用到中央窗口部件
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        # 创建状态栏，并显示应用名称和描述信息
        status_bar = QStatusBar()
        status_bar.showMessage(f"{__appname__} - {__appdescription__}")
        self.setStatusBar(status_bar)

    def closeEvent(self, event):
        """
        处理窗口关闭事件，确保 LabelingWrapper 组件正确关闭。
        """
        self.labeling_widget.closeEvent(event)
