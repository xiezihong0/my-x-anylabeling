import os.path as osp
from math import sqrt

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

# 获取当前文件所在的目录路径
here = osp.dirname(osp.abspath(__file__))


def new_icon(icon):
    """
    创建一个新的图标对象。
    :param icon: 图标文件的名称（不含路径和后缀）
    :return: QtGui.QIcon 对象
    """
    return QtGui.QIcon(osp.join(f":/images/images/{icon}.png"))


def new_button(text, icon=None, slot=None):
    """
    创建一个带有文本和可选图标的按钮。
    :param text: 按钮文本
    :param icon: 可选的图标名称
    :param slot: 点击时执行的槽函数
    :return: QPushButton 对象
    """
    b = QtWidgets.QPushButton(text)
    if icon is not None:
        b.setIcon(new_icon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b


def new_action(
    parent,
    text,
    slot=None,
    shortcut=None,
    icon=None,
    tip=None,
    checkable=False,
    enabled=True,
    checked=False,
    auto_trigger=False,
):
    """
    创建一个新的 QAction，并设置相关属性。
    :param parent: 父对象
    :param text: 动作文本
    :param slot: 触发时调用的函数
    :param shortcut: 快捷键
    :param icon: 图标名称
    :param tip: 提示文本
    :param checkable: 是否可选中
    :param enabled: 是否启用
    :param checked: 初始选中状态
    :param auto_trigger: 是否自动触发
    :return: QAction 对象
    """
    """Create a new action and assign callbacks, shortcuts, etc."""
    action = QtWidgets.QAction(text, parent)
    if icon is not None:
        action.setIconText(text.replace(" ", "\n"))
        action.setIcon(new_icon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            action.setShortcuts(shortcut)
        else:
            action.setShortcut(shortcut)
    if tip is not None:
        action.setToolTip(tip)
        action.setStatusTip(tip)
    if slot is not None:
        action.triggered.connect(slot)
    if checkable:
        action.setCheckable(True)
    action.setEnabled(enabled)
    action.setChecked(checked)
    if auto_trigger:
        action.triggered.emit(checked)
    return action


def add_actions(widget, actions):
    """
    向给定的 widget 添加一组 QAction。
    :param widget: 目标控件（如菜单、工具栏）
    :param actions: 包含 QAction 或 None（表示分隔符）的列表
    """
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QtWidgets.QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)


def label_validator():
    """
    创建一个正则表达式验证器，确保标签文本合法。
    :return: QRegularExpressionValidator 对象
    """
    return QtGui.QRegularExpressionValidator(
        QtCore.QRegularExpression(r"^[^ \t].+"), None
    )


class Struct:
    """
    用于存储任意属性的简单结构体。
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def distance(p):
    """
    计算点 p 到原点的欧几里得距离。
    :param p: QPoint 对象
    :return: 距离值
    """
    return sqrt(p.x() * p.x() + p.y() * p.y())


def distance_to_line(point, line):
    """
    计算点到线段的最短距离。
    :param point: 需要计算的点 (QPoint)
    :param line: 由两个 QPoint 组成的线段
    :return: 最短距离
    """
    p1, p2 = line
    p1 = np.array([p1.x(), p1.y()])
    p2 = np.array([p2.x(), p2.y()])
    p3 = np.array([point.x(), point.y()])
    # 判断投影是否在线段范围内
    if np.dot((p3 - p1), (p2 - p1)) < 0:
        return np.linalg.norm(p3 - p1)
    if np.dot((p3 - p2), (p1 - p2)) < 0:
        return np.linalg.norm(p3 - p2)
    # 计算点到直线的距离
    if np.linalg.norm(p2 - p1) == 0:
        return 0
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def fmt_shortcut(text):
    """
    格式化快捷键文本，使其更具可读性。
    :param text: 快捷键文本，例如 "Ctrl+S"
    :return: 格式化后的 HTML 字符串
    """
    mod, key = text.split("+", 1)
    return f"<b>{mod}</b>+<b>{key}</b>"
