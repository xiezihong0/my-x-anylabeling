import re
import json

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont, QColor, QIntValidator
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtWidgets import (
    QColorDialog,
    QTableWidgetItem,
    QTableWidget,
    QCheckBox,
)

from .. import utils
from ..logger import logger


# TODO(unknown):
# - Calculate optimal position so as not to go out of screen area.
# - 计算最佳窗口位置，避免超出屏幕范围。

# 自然排序函数，将字符串中的数字部分转换为整数，以保证排序结果符合人的直觉
def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

# GroupID 修改对话框
class GroupIDModifyDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(GroupIDModifyDialog, self).__init__(parent)
        self.parent = parent
        # 存储现有的 group_id 信息
        self.gid_info = []
        # 获取形状文件列表
        self.shape_list = parent.get_label_file_list()
        # 初始化 group_id 信息
        self.init_gid_info()
        # 初始化 UI 界面
        self.init_ui()

    def init_ui(self):
        """初始化 UI 界面"""
        # 设置窗口标题
        self.setWindowTitle(self.tr("Group ID Change Manager"))
        # 允许最小化和最大化窗口
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )
        # 设置窗口大小
        self.resize(600, 400)
        # 将窗口移动到屏幕中央
        self.move_to_center()
        # 定义表头
        title_list = ["Ori Group-ID", "New Group-ID"]
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(len(title_list))
        self.table_widget.setHorizontalHeaderLabels(title_list)

        # 设置表头字体和对齐方式
        # Set header font and alignment
        for i in range(len(title_list)):
            self.table_widget.horizontalHeaderItem(i).setFont(
                QFont("Arial", 8, QFont.Bold)
            )
            self.table_widget.horizontalHeaderItem(i).setTextAlignment(
                QtCore.Qt.AlignCenter
            )
        # 创建按钮布局
        self.buttons_layout = QtWidgets.QHBoxLayout()
        # 取消按钮
        self.cancel_button = QtWidgets.QPushButton(self.tr("Cancel"), self)
        self.cancel_button.clicked.connect(self.reject)
        # 确认按钮
        self.confirm_button = QtWidgets.QPushButton(self.tr("Confirm"), self)
        self.confirm_button.clicked.connect(self.confirm_changes)
        # 将按钮添加到布局
        self.buttons_layout.addWidget(self.cancel_button)
        self.buttons_layout.addWidget(self.confirm_button)
        # 设置主布局
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.table_widget)
        layout.addLayout(self.buttons_layout)
        # 填充表格数据
        self.populate_table()

    def move_to_center(self):
        """将窗口移动到屏幕中央"""
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def populate_table(self):
        """填充表格数据"""
        for i, group_id in enumerate(self.gid_info):
            self.table_widget.insertRow(i)
            # 旧 Group-ID（不可编辑）
            old_gid_item = QTableWidgetItem(str(group_id))
            old_gid_item.setFlags(
                old_gid_item.flags() ^ QtCore.Qt.ItemIsEditable
            )
            # 新 Group-ID（可编辑）
            new_gid_item = QTableWidgetItem("")
            new_gid_item.setFlags(
                new_gid_item.flags() | QtCore.Qt.ItemIsEditable
            )
            # 设置输入限制，仅允许输入非负整数
            # Set QIntValidator to ensure only non-negative integers can be entered
            validator = QIntValidator(0, 9999, self)
            line_edit = QtWidgets.QLineEdit(self.table_widget)
            line_edit.setValidator(validator)
            self.table_widget.setCellWidget(i, 1, line_edit)
            # 添加数据到表格
            self.table_widget.setItem(i, 0, old_gid_item)

    def confirm_changes(self):
        """确认更改 Group-ID"""
        total_num = self.table_widget.rowCount()
        if total_num == 0:
            self.reject()
            return
        # 临时存储新的 group_id 信息
        # Temporary dictionary to handle changes
        new_gid_info = []
        updated_gid_info = {}

        # Iterate over each row to get the old and new group IDs
        for i in range(total_num):
            old_gid_item = self.table_widget.item(i, 0)
            line_edit = self.table_widget.cellWidget(i, 1)
            new_gid = line_edit.text()
            old_gid = old_gid_item.text()

            # Only add to updated_gid_info
            # if the new group ID is not empty and different
            # 只有在新 ID 不为空且与旧 ID 不同时才进行更新
            if new_gid and old_gid != new_gid:
                new_gid_info.append(new_gid)
                updated_gid_info[int(old_gid)] = {"new_gid": int(new_gid)}
            else:
                new_gid_info.append(old_gid)
        # Update original gid info
        # 更新原始 group_id 信息
        self.gid_info = new_gid_info

        # Try to modify group IDs
        if self.modify_group_id(updated_gid_info):
            QtWidgets.QMessageBox.information(
                self,
                "Success",
                "Group IDs modified successfully!",
            )
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Warning",
                "An error occurred while updating the Group IDs.",
            )

    def modify_group_id(self, updated_gid_info):
        try:
            for shape_file in self.shape_list:
                with open(shape_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                src_shapes, dst_shapes = data["shapes"], []
                for shape in src_shapes:
                    group_id = shape.get("group_id")
                    if group_id is not None:
                        group_id = int(group_id)
                        if group_id in updated_gid_info:
                            shape["group_id"] = updated_gid_info[group_id]["new_gid"]
                    dst_shapes.append(shape)
                data["shapes"] = dst_shapes
                with open(shape_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error occurred while updating Group IDs: {e}")
            return False

    def init_gid_info(self):
        for shape_file in self.shape_list:
            with open(shape_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            shapes = data.get("shapes", [])
            for shape in shapes:
                group_id = shape.get("group_id", None)
                if group_id is not None and group_id not in self.gid_info:
                    self.gid_info.append(group_id)
        self.gid_info.sort()


class LabelColorButton(QtWidgets.QWidget):
    """
    一个自定义的颜色按钮控件，用于显示和修改标签颜色。
    """

    def __init__(self, color, parent=None):
        """
        初始化 LabelColorButton。

        参数：
        color (QColor) -- 初始颜色
        parent (QWidget, 可选) -- 父级窗口，默认为 None
        """
        super().__init__(parent)
        self.color = color
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        """
        初始化 UI 组件，创建一个带有颜色显示的小圆形按钮。
        """
        self.color_label = QtWidgets.QLabel()
        self.color_label.setFixedSize(15, 15)
        # 设置背景颜色
        self.color_label.setStyleSheet(
            f"background-color: {self.color.name()}; border: 1px solid transparent; border-radius: 10px;"
        )

        # 创建水平布局，并添加颜色标签
        self.layout = QtWidgets.QHBoxLayout(self)
        # 去除边距
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.color_label)

    def set_color(self, color):
        """
        设置新的颜色，并更新按钮的显示。

        参数：
        color (QColor) -- 需要更新的颜色
        """
        self.color = color
        self.color_label.setStyleSheet(
            f"background-color: {self.color.name()}; border: 1px solid transparent; border-radius: 10px;"
        )

    def mousePressEvent(self, event):
        """
        处理鼠标点击事件，在单击时调用父级窗口的 change_color 方法。

        参数：
        event (QMouseEvent) -- 鼠标事件对象
        """
        # 仅处理鼠标左键点击
        if event.button() == QtCore.Qt.LeftButton:
            # 调用父级窗口的 change_color 方法
            self.parent.change_color(self)

# 初始化标签修改对话框
class LabelModifyDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, opacity=128):
        """
       初始化标签修改对话框。

       参数:
       - parent: 父窗口对象
       - opacity: 颜色透明度，默认为 128
       """
        super(LabelModifyDialog, self).__init__(parent)
        self.parent = parent
        self.opacity = opacity
        # 获取标签文件列表
        self.label_file_list = parent.get_label_file_list()
        self.init_label_info()
        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        # 设置窗口标题
        self.setWindowTitle(self.tr("Label Change Manager"))
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )
        # 设置窗口大小
        self.resize(600, 400)
        # 将窗口居中
        self.move_to_center()

        title_list = ["Category", "Delete", "New Value", "Color"]
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(len(title_list))
        self.table_widget.setHorizontalHeaderLabels(title_list)

        # Set header font and alignment
        # 设置表头字体和对齐方式
        for i in range(len(title_list)):
            self.table_widget.horizontalHeaderItem(i).setFont(
                QFont("Arial", 8, QFont.Bold)
            )
            self.table_widget.horizontalHeaderItem(i).setTextAlignment(
                QtCore.Qt.AlignCenter
            )

        # 创建按钮布局
        self.buttons_layout = QtWidgets.QHBoxLayout()

        self.cancel_button = QtWidgets.QPushButton(self.tr("Cancel"), self)
        self.cancel_button.clicked.connect(self.reject)

        self.confirm_button = QtWidgets.QPushButton(self.tr("Confirm"), self)
        self.confirm_button.clicked.connect(self.confirm_changes)

        self.buttons_layout.addWidget(self.cancel_button)
        self.buttons_layout.addWidget(self.confirm_button)
        print("2222222222")
        # 主界面布局
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.table_widget)
        layout.addLayout(self.buttons_layout)

        # 填充表格数据
        self.populate_table()

    def move_to_center(self):
        """将窗口居中"""
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def populate_table(self):
        """填充表格内容"""
        for i, (label, info) in enumerate(self.parent.label_info.items()):
            self.table_widget.insertRow(i)
            # 标签名（不可编辑）
            class_item = QTableWidgetItem(label)
            class_item.setFlags(class_item.flags() ^ QtCore.Qt.ItemIsEditable)

            # 删除复选框
            delete_checkbox = QCheckBox()
            delete_checkbox.setChecked(info["delete"])
            delete_checkbox.setIcon(QtGui.QIcon(":/images/images/delete.png"))
            delete_checkbox.stateChanged.connect(
                lambda state, row=i: self.on_delete_checkbox_changed(
                    row, state
                )
            )
            # 新值输入框
            value_item = QTableWidgetItem(
                info["value"] if info["value"] else ""
            )
            value_item.setFlags(
                value_item.flags() & ~QtCore.Qt.ItemIsEditable
                if info["delete"]
                else value_item.flags() | QtCore.Qt.ItemIsEditable
            )
            value_item.setBackground(
                QtGui.QColor("lightgray")
                if info["delete"]
                else QtGui.QColor("white")
            )
            # 颜色按钮
            color = QColor(*info["color"])
            color.setAlpha(info["opacity"])
            color_button = LabelColorButton(color, self)
            color_button.setParent(self.table_widget)
            # 添加到表格
            self.table_widget.setItem(i, 0, class_item)
            self.table_widget.setCellWidget(i, 1, delete_checkbox)
            self.table_widget.setItem(i, 2, value_item)
            self.table_widget.setCellWidget(i, 3, color_button)

    """更改标签颜色"""
    def change_color(self, button):

        row = self.table_widget.indexAt(button.pos()).row()
        current_color = self.parent.label_info[
            self.table_widget.item(row, 0).text()
        ]["color"]
        color = QColorDialog.getColor(QColor(*current_color), self)
        if color.isValid():
            self.parent.label_info[self.table_widget.item(row, 0).text()][
                "color"
            ] = [color.red(), color.green(), color.blue()]
            self.parent.label_info[self.table_widget.item(row, 0).text()][
                "opacity"
            ] = color.alpha()
            button.set_color(color)

    """删除复选框状态改变时的处理"""
    def on_delete_checkbox_changed(self, row, state):
        value_item = self.table_widget.item(row, 2)
        delete_checkbox = self.table_widget.cellWidget(row, 1)
        hidden_checkbox = self.table_widget.cellWidget(row, 3)

        if state == QtCore.Qt.Checked:
            value_item.setFlags(value_item.flags() & ~QtCore.Qt.ItemIsEditable)
            value_item.setBackground(QtGui.QColor("lightgray"))
            delete_checkbox.setCheckable(True)
        else:
            value_item.setFlags(value_item.flags() | QtCore.Qt.ItemIsEditable)
            value_item.setBackground(QtGui.QColor("white"))
            delete_checkbox.setCheckable(False)

        if value_item.text():
            delete_checkbox.setCheckable(False)
        else:
            delete_checkbox.setCheckable(True)

    """确认标签修改"""
    def confirm_changes(self):
        total_num = self.table_widget.rowCount()
        if total_num == 0:
            self.reject()
            return

        # Temporary dictionary to handle changes
        updated_label_info = {}

        for i in range(total_num):
            label = self.table_widget.item(i, 0).text()
            delete_checkbox = self.table_widget.cellWidget(i, 1)
            value_item = self.table_widget.item(i, 2)

            is_delete = delete_checkbox.isChecked()
            new_value = value_item.text()

            # Update the label info in the temporary dictionary
            self.parent.label_info[label]["delete"] = is_delete
            self.parent.label_info[label]["value"] = new_value

            # Update the color
            color = self.parent.label_info[label]["color"]
            self.parent.unique_label_list.update_item_color(
                label, color, self.opacity
            )

            # Handle delete and change of labels
            if is_delete:
                self.parent.unique_label_list.remove_items_by_label(label)
                continue  # Skip adding this to updated_label_info to effectively delete it
            elif new_value:
                self.parent.unique_label_list.remove_items_by_label(label)
                updated_label_info[new_value] = self.parent.label_info[label]
            else:
                updated_label_info[label] = self.parent.label_info[label]

        # Try to modify labels
        if self.modify_label():
            # If modification is successful, update self.parent.label_info
            self.parent.label_info = updated_label_info
            QtWidgets.QMessageBox.information(
                self,
                "Success",
                "Labels modified successfully!",
            )
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "An error occurred while updating the labels."
            )

    """修改标签信息并保存到文件"""
    def modify_label(self):
        try:
            for label_file in self.label_file_list:
                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                src_shapes, dst_shapes = data["shapes"], []
                for shape in src_shapes:
                    label = shape["label"]
                    if self.parent.label_info[label]["delete"]:
                        continue
                    if self.parent.label_info[label]["value"]:
                        shape["label"] = self.parent.label_info[label]["value"]
                    dst_shapes.append(shape)
                data["shapes"] = dst_shapes
                with open(label_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error occurred while updating labels: {e}")
            return False

    """初始化标签信息"""
    def init_label_info(self):
        classes = set()

        for label_file in self.label_file_list:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            shapes = data.get("shapes", [])
            for shape in shapes:
                label = shape["label"]
                classes.add(label)

        for c in sorted(classes):
            # Update unique label list
            if not self.parent.unique_label_list.find_items_by_label(c):
                unique_label_item = (
                    self.parent.unique_label_list.create_item_from_label(c)
                )
                self.parent.unique_label_list.addItem(unique_label_item)
                rgb = self.parent._get_rgb_by_label(c, skip_label_info=True)
                self.parent.unique_label_list.set_item_label(
                    unique_label_item, c, rgb, self.opacity
                )
            # Update label info
            color = [0, 0, 0]
            opacity = 255
            items = self.parent.unique_label_list.find_items_by_label(c)
            for item in items:
                qlabel = self.parent.unique_label_list.itemWidget(item)
                if qlabel:
                    style_sheet = qlabel.styleSheet()
                    start_index = style_sheet.find("rgba(") + 5
                    end_index = style_sheet.find(")", start_index)
                    rgba_color = style_sheet[start_index:end_index].split(",")
                    rgba_color = [int(x.strip()) for x in rgba_color]
                    color = rgba_color[:-1]
                    opacity = rgba_color[-1]
                    break
            self.parent.label_info[c] = dict(
                delete=False,
                value=None,
                color=color,
                opacity=opacity,
            )

# 初始化文本输入对话框
class TextInputDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        """
        初始化文本输入对话框。

        参数:
        - parent: 父窗口对象，默认为 None
        """
        super().__init__(parent)
        # 设置窗口标题
        self.setWindowTitle(self.tr("Text Input Dialog"))
        # 创建垂直布局
        layout = QtWidgets.QVBoxLayout()
        # 创建标签，提示用户输入文本
        self.label = QtWidgets.QLabel(self.tr("Enter the text prompt below:"))
        # 创建文本输入框
        self.text_input = QtWidgets.QLineEdit()
        # 创建 "OK" 和 "Cancel" 按钮
        self.ok_button = QtWidgets.QPushButton(self.tr("OK"))
        self.cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
        # 绑定按钮事件
        self.ok_button.clicked.connect(self.accept)  # 点击 "OK" 按钮时关闭对话框并返回成功状态
        self.cancel_button.clicked.connect(self.reject)  # 点击 "Cancel" 按钮时关闭对话框并返回失败状态

        # 将组件添加到布局
        layout.addWidget(self.label)
        layout.addWidget(self.text_input)
        layout.addWidget(self.ok_button)
        layout.addWidget(self.cancel_button)
        # 设置窗口的主布
        self.setLayout(layout)

    def get_input_text(self):
        result = self.exec_()
        if result == QtWidgets.QDialog.Accepted:
            return self.text_input.text()
        else:
            return ""

# 标签列表组件
class LabelQLineEdit(QtWidgets.QLineEdit):
    def __init__(self) -> None:
        """
        自定义的 QLineEdit 组件，用于支持与列表组件（QListWidget）交互。

        该类扩展了 QLineEdit，使其能够处理键盘事件，并与一个 QListWidget 组件进行联动。
        """
        super().__init__()
        # 存储关联的 QListWidget 组件
        self.list_widget = None

    def set_list_widget(self, list_widget):
        """
        设置与该输入框关联的 QListWidget 组件。
        参数:
        - list_widget: 需要关联的 QListWidget 组件
        """
        self.list_widget = list_widget

    # QT Overload
    def keyPressEvent(self, e):
        """
        处理键盘按键事件。

        如果按下的是“上箭头”或“下箭头”键，则将事件传递给 `list_widget` 组件，
        使其能够响应方向键进行选项切换。
        否则，调用 `QLineEdit` 的默认按键事件处理方法。

        参数:
        - e: 键盘事件对象
        """
        if e.key() in [QtCore.Qt.Key_Up, QtCore.Qt.Key_Down]:
            self.list_widget.keyPressEvent(e)
        else:
            super(LabelQLineEdit, self).keyPressEvent(e)

"""
标签编辑框
text=默认对话框提示文本
parent=父窗口
labels=可选的标签列表
sort_labels=是否对标签列表排序
show_text_field=是否显示输入框
completion=自动补全模式（startswith / contains）
fit_to_content=是否自适应内容大小
flags=标签的额外标志
difficult=是否启用“难度”复选框
"""
class LabelDialog(QtWidgets.QDialog):
    def __init__(
        self,
        text=None,
        parent=None,
        labels=None,
        sort_labels=True,
        show_text_field=True,
        completion="startswith",
        fit_to_content=None,
        flags=None,
        difficult=False,
    ):
        # 设置默认的提示文本
        if text is None:
            text = QCoreApplication.translate(
                "LabelDialog", "Enter object label"
            )
        # 设置默认的自适应配置
        if fit_to_content is None:
            fit_to_content = {"row": False, "column": True}
        self._fit_to_content = fit_to_content
        # 调用父类构造方法
        super(LabelDialog, self).__init__(parent)
        # 创建主输入框（LabelQLineEdit）
        self.edit = LabelQLineEdit()
        # 设置输入框提示文本
        self.edit.setPlaceholderText(text)
        # 限制输入格式
        self.edit.setValidator(utils.label_validator())
        # 连接处理方法
        self.edit.editingFinished.connect(self.postprocess)
        # 如果 flags 存在，文本变更时更新 flags
        if flags:
            self.edit.textChanged.connect(self.update_flags)
        # 组 ID 输入框（只能输入数字）
        self.edit_group_id = QtWidgets.QLineEdit()
        self.edit_group_id.setPlaceholderText(self.tr("Group ID"))
        self.edit_group_id.setValidator(
            QtGui.QRegularExpressionValidator(
                QtCore.QRegularExpression(r"\d*"), None
            )
        )
        self.edit_group_id.setAlignment(QtCore.Qt.AlignCenter)

        # Add difficult checkbox
        # “难度” 复选框
        self.edit_difficult = QtWidgets.QCheckBox(self.tr("useDifficult"))
        self.edit_difficult.setChecked(difficult)

        # Add linking input
        # 关联输入框（用户可以输入 [0,1] 这样的链接
        self.linking_input = QtWidgets.QLineEdit()
        self.linking_input.setPlaceholderText(
            self.tr("Enter linking, e.g., [0,1]")
        )
        linking_font = (
            self.linking_input.font()
        )  # Adjust placeholder font size
        # 调整字体大小
        linking_font.setPointSize(8)
        # 关联项列表（初始隐藏）
        self.linking_input.setFont(linking_font)
        self.linking_list = QtWidgets.QListWidget()
        self.linking_list.setHidden(True)  # Initially hide the list
        row_height = self.linking_list.fontMetrics().height()
        self.linking_list.setFixedHeight(
            row_height * 4 + 2 * self.linking_list.frameWidth()
        )
        # “添加关联” 按钮
        self.add_linking_button = QtWidgets.QPushButton(self.tr("Add"))
        self.add_linking_button.clicked.connect(self.add_linking_pair)

        # ------------- 创建布局 --------------
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        # 输入框 & 组 ID
        if show_text_field:
            layout_edit = QtWidgets.QHBoxLayout()
            layout_edit.addWidget(self.edit, 4)
            layout_edit.addWidget(self.edit_group_id, 2)
            layout.addLayout(layout_edit)

        # Add linking layout
        # 关联输入 & 按钮
        layout_linking = QtWidgets.QHBoxLayout()
        layout_linking.addWidget(self.linking_input, 4)
        layout_linking.addWidget(self.add_linking_button, 2)
        layout.addLayout(layout_linking)
        layout.addWidget(self.linking_list)

        # buttons
        # “确定 / 取消” 按钮
        self.button_box = bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        bb.button(bb.Ok).setIcon(utils.new_icon("done"))
        bb.button(bb.Cancel).setIcon(utils.new_icon("undo"))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)

        # text edit
        # 标签描述框
        self.edit_description = QtWidgets.QTextEdit()
        self.edit_description.setPlaceholderText(self.tr("Label description"))
        self.edit_description.setFixedHeight(50)
        layout.addWidget(self.edit_description)

        # difficult & confirm button
        # “难度”复选框 & 按钮
        layout_button = QtWidgets.QHBoxLayout()
        layout_button.addWidget(self.edit_difficult)
        layout_button.addWidget(self.button_box)
        layout.addLayout(layout_button)

        # label_list
        # 标签列表
        self.label_list = QtWidgets.QListWidget()
        if self._fit_to_content["row"]:
            self.label_list.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff
            )
        if self._fit_to_content["column"]:
            self.label_list.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff
            )
        # 是否对标签排序
        self._sort_labels = sort_labels
        if labels:
            self.label_list.addItems(labels)
        if self._sort_labels:
            self.sort_labels()
        else:
            self.label_list.setDragDropMode(
                QtWidgets.QAbstractItemView.InternalMove
            )
        # 绑定事件
        self.label_list.currentItemChanged.connect(self.label_selected)
        self.label_list.itemDoubleClicked.connect(self.label_double_clicked)
        self.edit.set_list_widget(self.label_list)
        layout.addWidget(self.label_list)
        # label_flags
        # 处理标签 Flags
        if flags is None:
            flags = {}
        self._flags = flags
        self.flags_layout = QtWidgets.QVBoxLayout()
        self.reset_flags()
        layout.addItem(self.flags_layout)
        self.edit.textChanged.connect(self.update_flags)
        # 设置布局
        self.setLayout(layout)
        # completion
        # 处理自动补全逻辑
        completer = QtWidgets.QCompleter()
        if completion == "startswith":
            completer.setCompletionMode(QtWidgets.QCompleter.InlineCompletion)
            # Default settings.
            # completer.setFilterMode(QtCore.Qt.MatchStartsWith)
        elif completion == "contains":
            completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            completer.setFilterMode(QtCore.Qt.MatchContains)
        else:
            raise ValueError(f"Unsupported completion: {completion}")
        completer.setModel(self.label_list.model())
        self.edit.setCompleter(completer)
        # Save last label
        # 记录上一次的标签
        self._last_label = ""

    def add_linking_pair(self):
        """
        添加 KIE 关联对（linking pair）。

        - 解析用户输入的文本 linking_text，转换为列表 linking_pairs。
        - 确保 linking_pairs 是包含两个整数的列表（例如 [1, 2]）。
        - 检查 linking_pairs 是否已存在，若已存在则弹出警告对话框。
        - 若有效，则添加到 linking_list，并清空输入框，显示列表。
        - 若输入格式错误，则弹出警告提示。
        """
        linking_text = self.linking_input.text()
        try:
            # 解析输入字符串
            linking_pairs = eval(linking_text)
            if (
                isinstance(linking_pairs, list)
                and len(linking_pairs) == 2
                and all(isinstance(item, int) for item in linking_pairs)
            ):
                # 检查是否已经存在
                if linking_pairs in self.get_kie_linking():
                    QtWidgets.QMessageBox.warning(
                        self,
                        self.tr("Duplicate Entry"),
                        self.tr("This linking pair already exists."),
                    )
                # 添加到列表
                self.linking_list.addItem(str(linking_pairs))
                # 清空输入框
                self.linking_input.clear()
                # 显示列表
                self.linking_list.setHidden(
                    False
                )  # Show the list when an item is added
            else:
                # 触发异常
                raise ValueError
        except:
            # 输入格式错误，弹出警告对话框
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Invalid Input"),
                self.tr(
                    "Please enter a valid list of linking pairs like [1,2]."
                ),
            )

    def keyPressEvent(self, event):
        """
        监听键盘按键事件。
        - 如果按下 Delete 键，则删除 linking_list 中选中的项。
        - 否则，调用父类的 keyPressEvent 处理其他按键。
        """
        if event.key() == QtCore.Qt.Key_Delete:
            if hasattr(self, "linking_list") and self.linking_list is not None:
                selected_items = self.linking_list.selectedItems()
                if selected_items:
                    for item in selected_items:
                        self.linking_list.takeItem(self.linking_list.row(item))
        else:
            super(LabelDialog, self).keyPressEvent(event)

    def remove_linking_item(self, item_widget):
        """
        从 linking_list 中移除指定的 item_widget 并释放资源。
        """
        list_item = self.linking_list.itemWidget(item_widget)
        self.linking_list.takeItem(self.linking_list.row(list_item))
        item_widget.deleteLater()

    def reset_linking(self, kie_linking=[]):
        """
        重新初始化 KIE 关联列表。

        - 清空 linking_list。
        - 将 kie_linking 中的链接对重新添加到列表。
        - 如果 kie_linking 为空，则隐藏 linking_list，否则显示。
        """
        self.linking_list.clear()
        for linking_pair in kie_linking:
            self.linking_list.addItem(str(linking_pair))
        self.linking_list.setHidden(False if kie_linking else True)

    def get_last_label(self):
        """
        获取最近使用的标签。
        """
        return self._last_label

    def sort_labels(self):
        """
        对标签列表进行自然排序（例如 'label10' 排在 'label2' 之后）。
        """
        items = []
        for index in range(self.label_list.count()):
            items.append(self.label_list.item(index).text())
        
        items.sort(key=natural_sort_key)
        
        self.label_list.clear()
        self.label_list.addItems(items)

    def add_label_history(self, label, update_last_label=True):
        """
        记录标签历史并维护排序。

        - 若 update_last_label 为 True，则更新 _last_label。
        - 若 label 已存在于 label_list，则不重复添加。
        - 否则，将其添加，并根据 _sort_labels 选项决定是否排序。
        - 选中刚添加的标签。
        """
        if update_last_label:
            self._last_label = label
        if self.label_list.findItems(label, QtCore.Qt.MatchExactly):
            return
        self.label_list.addItem(label)
        if self._sort_labels:
            self.sort_labels()
        items = self.label_list.findItems(label, QtCore.Qt.MatchExactly)
        if items:
            self.label_list.setCurrentItem(items[0])

    def label_selected(self, item):
        """
        当标签项被选中时，更新输入框的文本内容。
        """
        if item is not None:
            self.edit.setText(item.text())
        else:
            # 如果未选中任何标签，则清空文本框
            # Clear the edit field if no item is selected
            self.edit.clear()

    def validate(self):
        """
        验证输入框内容是否有效，并去除前后空格。
        """
        text = self.edit.text()
        if hasattr(text, "strip"):
            text = text.strip()
        else:
            text = text.trimmed()
        if text:
            self.accept()

    def label_double_clicked(self, _):
        """
        当标签被双击时，触发 validate() 进行确认。
        """
        self.validate()

    def postprocess(self):
        """
        处理输入文本，去除前后空格后更新文本框内容。
        """
        text = self.edit.text()
        if hasattr(text, "strip"):
            text = text.strip()
        else:
            text = text.trimmed()
        self.edit.setText(text)

    def upload_flags(self, flags):
        """
        存储标注的 flags 信息（标志位）。
        """
        self._flags = flags

    def update_flags(self, label_new):
        """
        维护标志位状态：
        - 继承符合 pattern 的标志位状态，否则设为 False。
        """
        # keep state of shared flags
        flags_old = self.get_flags()

        flags_new = {}
        for pattern, keys in self._flags.items():
            if re.match(pattern, label_new):
                for key in keys:
                    flags_new[key] = flags_old.get(key, False)
        self.set_flags(flags_new)

    def delete_flags(self):
        for i in reversed(range(self.flags_layout.count())):
            item = self.flags_layout.itemAt(i).widget()
            self.flags_layout.removeWidget(item)
            item.setParent(None)

    def reset_flags(self, label=""):
        flags = {}
        for pattern, keys in self._flags.items():
            if re.match(pattern, label):
                for key in keys:
                    flags[key] = False
        self.set_flags(flags)

    def set_flags(self, flags):
        self.delete_flags()
        for key in flags:
            item = QtWidgets.QCheckBox(key, self)
            item.setChecked(flags[key])
            self.flags_layout.addWidget(item)
            item.show()

    def get_flags(self):
        flags = {}
        for i in range(self.flags_layout.count()):
            item = self.flags_layout.itemAt(i).widget()
            flags[item.text()] = item.isChecked()
        return flags

    def get_group_id(self):
        group_id = self.edit_group_id.text()
        if group_id:
            return int(group_id)
        return None

    def get_description(self):
        return self.edit_description.toPlainText()

    def get_difficult_state(self):
        return self.edit_difficult.isChecked()

    def get_kie_linking(self):
        """
        获取当前 KIE 关联的链接对（转换为列表）。
        """
        kie_linking = []
        for index in range(self.linking_list.count()):
            item = self.linking_list.item(index)
            kie_linking.append(eval(item.text()))
        return kie_linking

    def pop_up(
        self,
        text=None,
        move=True,
        move_mode="auto",
        flags=None,
        group_id=None,
        description=None,
        difficult=False,
        kie_linking=[],
    ):
        """
        显示弹出对话框，并初始化 UI 元素：
        - 设置 label_list 适配内容的高度和宽度。
        - 初始化文本框、描述框、KIE 关联列表、标志位状态等。
        - 根据鼠标位置或屏幕中心调整窗口位置。
        - 若对话框被确认，则返回编辑后的数据；否则返回 None。
        """
        if self._fit_to_content["row"]:
            self.label_list.setMinimumHeight(
                self.label_list.sizeHintForRow(0) * self.label_list.count() + 2
            )
        if self._fit_to_content["column"]:
            self.label_list.setMinimumWidth(
                self.label_list.sizeHintForColumn(0) + 2
            )
        # if text is None, the previous label in self.edit is kept
        # 初始化文本框内容
        if text is None:
            text = self.edit.text()
        # description is always initialized by empty text c.f., self.edit.text
        if description is None:
            description = ""
        self.edit_description.setPlainText(description)
        # Set initial values for kie_linking
        # 初始化 KIE 关联列表
        self.reset_linking(kie_linking)
        # 处理标志位
        if flags:
            self.set_flags(flags)
        else:
            self.reset_flags(text)
        # 设置 "difficult" 复选框状态
        if difficult:
            self.edit_difficult.setChecked(True)
        else:
            self.edit_difficult.setChecked(False)
        # 设置文本框
        print("LabelDialog================pop_up-text=========",text)
        self.edit.setText(text)
        self.edit.setSelection(0, len(text))
        if group_id is None:
            self.edit_group_id.clear()
        else:
            self.edit_group_id.setText(str(group_id))
        # 根据文本匹配标签列表
        items = self.label_list.findItems(text, QtCore.Qt.MatchFixedString)

        if items:
            if len(items) != 1:
                logger.warning(f"Label list has duplicate '{text}'")
            self.label_list.setCurrentItem(items[0])
            row = self.label_list.row(items[0])
            self.edit.completer().setCurrentRow(row)
        self.edit.setFocus(QtCore.Qt.PopupFocusReason)

        # 处理窗口移动
        if move:
            if move_mode == "auto":
                cursor_pos = QtGui.QCursor.pos()
                # 处理窗口移动
                screen = QtWidgets.QApplication.desktop().screenGeometry(
                    cursor_pos
                )
                dialog_frame_size = self.frameGeometry()
                # Calculate the ideal top-left corner position for the dialog based on the mouse click
                ideal_pos = cursor_pos
                # Adjust to prevent the dialog from exceeding the right screen boundary
                if (
                    ideal_pos.x() + dialog_frame_size.width()
                ) > screen.right():
                    ideal_pos.setX(screen.right() - dialog_frame_size.width())
                # Adjust to prevent the dialog's bottom from going off-screen
                if (
                    ideal_pos.y() + dialog_frame_size.height()
                ) > screen.bottom():
                    ideal_pos.setY(
                        screen.bottom() - dialog_frame_size.height()
                    )
                self.move(ideal_pos)
            elif move_mode == "center":
                # Calculate the center position to move the dialog to
                screen = QtWidgets.QApplication.desktop().screenNumber(
                    QtWidgets.QApplication.desktop().cursor().pos()
                )
                centerPoint = (
                    QtWidgets.QApplication.desktop()
                    .screenGeometry(screen)
                    .center()
                )
                qr = self.frameGeometry()
                qr.moveCenter(centerPoint)
                self.move(qr.topLeft())

        if self.exec_():
            return (
                self.edit.text(),
                self.get_flags(),
                self.get_group_id(),
                self.get_description(),
                self.get_difficult_state(),
                self.get_kie_linking(),
            )
        return None, None, None, None, False, []
