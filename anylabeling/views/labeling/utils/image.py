import os
import os.path as osp
import base64
import io
import shutil

import numpy as np
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
import numpy as np
from PyQt5 import QtGui

from ...labeling.logger import logger

"""
图像转换： 包含将图像从字节流、Base64、NumPy数组、PIL图像等格式相互转换的函数。
EXIF处理： 包含对图像EXIF方向信息的处理，根据图像的方向信息自动旋转图像。
"""

def img_data_to_pil(img_data):
    """
    将字节数据转换为PIL图像对象。

    参数：
    - img_data (bytes): 图像数据的字节流。

    返回：
    - img_pil (PIL.Image.Image): 转换后的PIL图像对象。
    """
    # 创建一个字节流对象
    f = io.BytesIO()
    # 将图像数据写入字节流
    f.write(img_data)
    # 使用PIL打开字节流，生成图像对象
    img_pil = PIL.Image.open(f)
    return img_pil


def img_data_to_arr(img_data):
    """
    将字节数据转换为NumPy数组。

    参数：
    - img_data (bytes): 图像数据的字节流。

    返回：
    - img_arr (np.ndarray): 转换后的NumPy数组表示的图像。
    """
    # 将字节数据转换为PIL图像
    img_pil = img_data_to_pil(img_data)
    # 将PIL图像转换为NumPy数组
    img_arr = np.array(img_pil)
    return img_arr


def img_b64_to_arr(img_b64):
    """
    将Base64编码的图像数据转换为NumPy数组。

    参数：
    - img_b64 (str): Base64编码的图像数据。

    返回：
    - img_arr (np.ndarray): 转换后的NumPy数组表示的图像。
    """
    # 解码Base64字符串为字节数据
    img_data = base64.b64decode(img_b64)
    # 将字节数据转换为NumPy数组
    img_arr = img_data_to_arr(img_data)
    return img_arr


def img_pil_to_data(img_pil):
    """
    将PIL图像对象转换为字节数据。

    参数：
    - img_pil (PIL.Image.Image): 要转换的PIL图像对象。

    返回：
    - img_data (bytes): 转换后的字节数据。
    """
    # 创建一个字节流对象
    f = io.BytesIO()
    # 将PIL图像保存到字节流，以PNG格式
    img_pil.save(f, format="PNG")
    # 获取字节流中的数据
    img_data = f.getvalue()
    return img_data


def pil_to_qimage(img):
    """将PIL图像转换为QImage对象。
    参数：
    - img (PIL.Image.Image): 要转换的PIL图像对象。
    返回：
    - qimage (QImage): 转换后的QImage对象。
    """
    """Convert PIL Image to QImage."""
    # 确保图像为RGBA格式
    img = img.convert("RGBA")  # Ensure image is in RGBA format
    # 将图像转换为NumPy数组
    data = np.array(img)
    # 获取图像的高度、宽度和通道数
    height, width, channel = data.shape
    # 每行字节数（RGBA格式，每个像素4字节）
    bytes_per_line = 4 * width
    # 创建QImage对象
    qimage = QtGui.QImage(
        data, width, height, bytes_per_line, QtGui.QImage.Format_RGBA8888
    )
    return qimage


def img_arr_to_b64(img_arr):
    """
    将NumPy数组表示的图像转换为Base64编码的字符串。

    参数：
    - img_arr (np.ndarray): 要转换的NumPy数组表示的图像。

    返回：
    - img_b64 (str): 转换后的Base64编码字符串。
    """
    # 将NumPy数组转换为PIL图像对象
    img_pil = PIL.Image.fromarray(img_arr)
    # 创建一个字节流对象
    f = io.BytesIO()
    # 将PIL图像保存到字节流，以PNG格式
    img_pil.save(f, format="PNG")
    # 获取字节流中的数据
    img_bin = f.getvalue()
    if hasattr(base64, "encodebytes"):
        # 使用encodebytes方法编码为Base64字符串
        img_b64 = base64.encodebytes(img_bin)
    else:
        # 如果没有encodebytes，使用encodestring（过时）
        img_b64 = base64.encodestring(img_bin)
    return img_b64


def img_data_to_png_data(img_data):
    """
    将图像数据转换为PNG格式的字节数据。

    参数：
    - img_data (bytes): 输入的图像字节数据。

    返回：
    - bytes: 转换为PNG格式的字节数据。
    """
    with io.BytesIO() as f:
        # 将输入字节数据写入字节流
        f.write(img_data)
        # 打开字节流中的图像数据
        img = PIL.Image.open(f)

        with io.BytesIO() as f:
            # 保存图像为PNG格式
            img.save(f, "PNG")
            f.seek(0)
            # 返回PNG格式的字节数据
            return f.read()


def process_image_exif(filename):
    """处理图像的EXIF方向信息，并根据方向调整图像。
    如果图像的EXIF方向数据表明需要旋转图像，将图像旋转并保存备份。
    参数：
    - filename (str): 图像文件的路径。
    """
    """Process image EXIF orientation and save if necessary."""
    with PIL.Image.open(filename) as img:
        exif_data = None
        if hasattr(img, "_getexif"):
            # 获取图像的EXIF数据
            exif_data = img._getexif()
        if exif_data is not None:
            for tag, value in exif_data.items():
                # 获取EXIF标签名称
                tag_name = PIL.ExifTags.TAGS.get(tag, tag)
                if tag_name != "Orientation":
                    continue
                if value == 3:
                    # 旋转180度
                    img = img.rotate(180, expand=True)
                    rotation = "180 degrees"
                elif value == 6:
                    # 旋转270度
                    img = img.rotate(270, expand=True)
                    rotation = "270 degrees"
                elif value == 8:
                    # 旋转90度
                    img = img.rotate(90, expand=True)
                    rotation = "90 degrees"
                else:
                    # 不需要旋转
                    return  # No rotation needed
                backup_dir = osp.join(osp.dirname(osp.dirname(filename)), 
                                      "x-anylabeling-exif-backup")
                # 创建备份目录
                os.makedirs(backup_dir, exist_ok=True)
                backup_filename = osp.join(backup_dir, osp.basename(filename))
                # 复制原文件到备份目录
                shutil.copy2(filename, backup_filename)
                # 保存旋转后的图像
                img.save(filename)
                logger.info(f"Rotated {filename} by {rotation}, saving backup to {backup_filename}")
                break


def apply_exif_orientation(image):
    """
    根据图像的EXIF方向信息调整图像的方向。

    参数：
    - image (PIL.Image.Image): 要调整的PIL图像。

    返回：
    - image (PIL.Image.Image): 方向调整后的图像。
    """
    try:
        # 获取图像的EXIF数据
        exif = image._getexif()
    except AttributeError:
        exif = None

    if exif is None:
        # 如果没有EXIF数据，返回原图
        return image

    # 过滤EXIF标签
    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in exif.items()
        if k in PIL.ExifTags.TAGS
    }

    orientation = exif.get("Orientation", None)

    if orientation == 1:
        # 不做任何改变
        # do nothing
        return image
    if orientation == 2:
        # 左右镜像
        # left-to-right mirror
        return PIL.ImageOps.mirror(image)
    if orientation == 3:
        # rotate 180
        # 旋转180度
        return image.transpose(PIL.Image.ROTATE_180)
    if orientation == 4:
        # top-to-bottom mirror
        # 上下镜像
        return PIL.ImageOps.flip(image)
    if orientation == 5:
        # top-to-left mirror
        # 上下镜像 + 旋转270度
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
    if orientation == 6:
        # rotate 270
        # 旋转270度
        return image.transpose(PIL.Image.ROTATE_270)
    if orientation == 7:
        # top-to-right mirror
        # 左右镜像 + 旋转90度
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
    if orientation == 8:
        # rotate 90
        # 旋转90度
        return image.transpose(PIL.Image.ROTATE_90)
    # 其他情况，返回原图
    return image
