"""This module defines labeling wrapper and related functions"""

from PyQt5.QtWidgets import QVBoxLayout, QWidget

from .label_widget import LabelingWidget


class LabelingWrapper(QWidget):
    """Wrapper widget for labeling module"""

    def __init__(
        self,
        parent,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        """
        初始化 LabelingWrapper 类

        :param parent: 父级 QWidget 控件
        :param config: 配置选项，用于定制标签功能的设置
        :param filename: 输入文件的名称
        :param output: 标签结果的输出数据
        :param output_file: 标签结果输出的文件名
        :param output_dir: 标签结果保存的目录
        """
        # 调用 QWidget 的父类构造函数
        super().__init__()
        # 设置父级控件
        self.parent = parent

        # 创建标签控件（LabelingWidget），并传递配置、文件名等参数
        # Create a labeling widget
        self.view = LabelingWidget(
            self,
            config=config,
            filename=filename,
            output=output,
            output_file=output_file,
            output_dir=output_dir,
        )

        # 创建主布局，使用垂直布局（QVBoxLayout）
        # Create the main layout and put labeling into
        # 垂直布局管理器
        main_layout = QVBoxLayout()
        # 设置布局的边距为 0
        main_layout.setContentsMargins(0, 0, 0, 0)
        # 将 LabelingWidget 添加到布局中
        main_layout.addWidget(self.view)
        # 设置该 QWidget 使用的布局
        self.setLayout(main_layout)

    def closeEvent(self, event):
        """
        覆盖 closeEvent 方法，以便在窗口关闭时处理相应的事件

        :param event: 关闭事件
        """
        # 调用内部视图（LabelingWidget）关闭事件
        self.view.closeEvent(event)
