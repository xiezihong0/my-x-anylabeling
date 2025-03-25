import os.path

import cv2
import numpy as np
import qimage2ndarray
from PyQt5 import QtGui
from PyQt5.QtGui import QImage


def qt_img_to_rgb_cv_img(qt_img, img_path=None):
    """
    Convert 8bit/16bit RGB image or 8bit/16bit Gray image to 8bit RGB image
    """
    """
    将 PyQt5 的 QImage（8bit/16bit RGB 或 8bit/16bit 灰度图像）转换为 OpenCV 8bit RGB 图像。

    参数：
    - qt_img (QImage): PyQt5 的 QImage 图像对象
    - img_path (str, 可选): 如果提供了图像路径，则直接从路径加载图像，而不是转换 QImage

    返回：
    - cv_image (numpy.ndarray): OpenCV 格式的 RGB 8bit 图像
    """
    if img_path is not None and os.path.exists(img_path):
        # Load Image From Path Directly
        # NOTE: Potential issue - unable to handle the flipped image.
        # Temporary workaround: cv_image = cv2.imread(img_path)
        # 如果提供了 img_path 并且文件存在，则直接从路径加载图像
        # OpenCV 的 imread 可能会遇到路径编码问题，使用 imdecode 方式解决
        # 读取图片为原始格式（-1 表示不改变原始格式）
        cv_image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        # OpenCV 默认以 BGR 格式读取，将其转换为 RGB 格式
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    else:
        # 如果没有提供 img_path，则从 Qt QImage 进行转换
        if (
            qt_img.format() == QImage.Format_RGB32
            or qt_img.format() == QImage.Format_ARGB32
            or qt_img.format() == QImage.Format_ARGB32_Premultiplied
        ):
            # 如果 QImage 是 RGB32/ARGB32 格式，则转换为 OpenCV 兼容的 RGB 格式
            cv_image = qimage2ndarray.rgb_view(qt_img)
        else:
            # 其他格式（如灰度图）使用 raw_view 进行转换
            cv_image = qimage2ndarray.raw_view(qt_img)
    # To uint8
    # 如果图像数据类型不是 uint8，则进行归一化转换
    if cv_image.dtype != np.uint8:
        # 归一化图像数据到 0-255
        cv2.normalize(cv_image, cv_image, 0, 255, cv2.NORM_MINMAX)
        # 转换为 uint8 类型
        cv_image = np.array(cv_image, dtype=np.uint8)
    # To RGB
    # 确保输出为 RGB 3通道格式
    if len(cv_image.shape) == 2 or cv_image.shape[2] == 1:
        # 如果是单通道（灰度图），则扩展为 3 通道
        cv_image = cv2.merge([cv_image, cv_image, cv_image])
    return cv_image


def qt_img_to_cv_img(in_image):
    """
    将 QImage 直接转换为 OpenCV 格式的 numpy 数组（RGB 格式）。

    参数：
    - in_image (QImage): PyQt5 QImage 图像

    返回：
    - OpenCV 格式的 numpy.ndarray（RGB）
    """
    return qimage2ndarray.rgb_view(in_image)


def cv_img_to_qt_img(in_mat):
    """
    将 OpenCV 格式的 numpy.ndarray 转换为 PyQt5 QImage。

    参数：
    - in_mat (numpy.ndarray): OpenCV 格式的图像（RGB）

    返回：
    - QtGui.QImage: PyQt5 QImage 格式的图像
    """
    return QtGui.QImage(qimage2ndarray.array2qimage(in_mat))
