import os
import os.path as osp
import cv2
import shutil

from PyQt5.QtCore import Qt
# PyQt5 GUI 组件：消息弹窗 输入对话框 进度条对话框
from PyQt5.QtWidgets import (
    QMessageBox,
    QInputDialog,
    QProgressDialog,
)


def get_output_directory(source_video_path):
    """
    根据视频文件路径生成输出目录路径。

    参数：
    - source_video_path (str): 输入视频文件路径

    返回：
    - output_dir (str): 生成的输出目录路径
    """
    video_dir = os.path.dirname(source_video_path)
    folder_name = os.path.splitext(os.path.basename(source_video_path))[0]
    output_dir = os.path.join(video_dir, folder_name)
    return output_dir


def ask_overwrite_directory(parent, output_dir):
    """
    检查输出目录是否存在，并询问用户是否覆盖。

    参数：
    - parent (QWidget): 父级窗口（用于显示弹窗）
    - output_dir (str): 输出目录路径

    返回：
    - bool: 用户选择是否继续（True: 继续, False: 取消）
    """
    if os.path.exists(output_dir):
        # 询问用户是否覆盖已有目录
        reply = QMessageBox.question(
            parent,
            "Directory Exists",
            f"The directory '{os.path.basename(output_dir)}' already exists. Do you want to overwrite it?",
            QMessageBox.Yes | QMessageBox.No,
        )
        # 用户选择不覆盖，返回 False
        if reply == QMessageBox.No:
            return False
        else:
            # 删除已有目录
            shutil.rmtree(output_dir)
    # 目录不存在或用户选择覆盖，返回 True
    return True


def get_frame_interval(parent, fps, total_frames):
    """
    让用户输入帧提取的间隔（每隔多少帧保存一张）。

    参数：
    - parent (QWidget): 父级窗口（用于显示弹窗）
    - fps (int): 视频的帧率
    - total_frames (int): 视频的总帧数

    返回：
    - int or None: 用户选择的帧间隔，如果用户取消，则返回 None
    """
    interval, ok = QInputDialog.getInt(
        parent,
        parent.tr("Frame Interval"),
        parent.tr(f"Enter the frame interval (FPS: {fps}):"),
        1,  # default value
        1,  # minimum value
        total_frames,  # maximum value
        1,  # step 步长
    )
    if not ok:
        QMessageBox.warning(
            parent, "Cancelled", "Frame extraction was cancelled."
        )
    # 用户取消时返回 None，否则返回用户输入的值
    return interval if ok else None


def extract_frames_from_video(parent, source_video_path):
    """
    从视频文件中按指定间隔提取帧并保存为 JPG 图片。

    参数：
    - parent (QWidget): 父级窗口（用于显示弹窗）
    - source_video_path (str): 输入视频文件路径

    返回：
    - str or None: 保存帧的输出目录路径，如果用户取消或发生错误，则返回 None
    """
    # 获取输出目录路径
    output_dir = get_output_directory(source_video_path)

    if not ask_overwrite_directory(parent, output_dir):
        # 用户选择不覆盖目录，则返回 None
        return None

    # 打开视频文件
    video_capture = cv2.VideoCapture(source_video_path)
    if not video_capture.isOpened():
        QMessageBox.critical(parent, "Error", "Failed to open video file.")
        # 视频打开失败，返回 None
        return None

    # 获取总帧数
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # 获取视频帧率
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    # 让用户选择帧提取间隔
    interval = get_frame_interval(parent, fps, total_frames)
    if interval is None:
        # 用户取消操作，返回 None
        return None

    os.makedirs(output_dir)

    # 创建进度条对话框
    progress_dialog = QProgressDialog(
        parent.tr("Extracting frames. Please wait..."),
        parent.tr("Cancel"),
        0,
        total_frames // interval,
        parent,
    )
    # 进度条窗口模态化
    progress_dialog.setWindowModality(Qt.WindowModal)
    # 设置窗口标题
    progress_dialog.setWindowTitle("Progress")
    # 设置进度条样式
    progress_dialog.setStyleSheet(
        """
        QProgressDialog QProgressBar {
            border: 1px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressDialog QProgressBar::chunk {
            background-color: orange;
        }
        """
    )

    # 当前帧计数
    frame_count = 0
    # 已保存帧数
    saved_frame_count = 0
    while True:
        # 读取一帧
        ret, frame = video_capture.read()
        if not ret:
            # 读取失败（视频结束），退出循环
            break
        # 根据用户选择的间隔保存帧
        if frame_count % interval == 0:
            frame_filename = os.path.join(
                output_dir, f"{saved_frame_count:05}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
            # 更新进度条
            progress_dialog.setValue(saved_frame_count)
            if progress_dialog.wasCanceled():
                # 用户取消操作，退出循环
                break

        frame_count += 1

    # 释放视频资源
    video_capture.release()
    # 关闭进度条对话框
    progress_dialog.close()
    # 返回保存帧的目录路径
    return output_dir
