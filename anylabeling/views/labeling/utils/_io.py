import os.path as osp

import imgviz
import numpy as np
import PIL.Image


def lblsave(filename, lbl):
    """
        保存像素级分类标签（label）为 PNG 图像（调色板模式）。

        参数：
        - filename (str): 保存的文件名（如果没有 .png 后缀，则自动添加）。
        - lbl (numpy.ndarray): 语义分割标签图，假设范围为 [-1, 254]（int32）或 [0, 255]（uint8）。

        处理逻辑：
        - 如果 `lbl` 值范围在 [-1, 254]（int32）或 [0, 255]（uint8），则转换为 PNG 并应用调色板。
        - 否则，抛出异常，并建议使用 `.npy` 格式保存数据。
        """
    # 确保文件名以 `.png` 结尾
    if osp.splitext(filename)[1] != ".png":
        filename += ".png"
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    # 检查标签数据的数值范围
    # - 允许的范围：[-1, 254] 或 [0, 255]
    # - 其中 -1 可能表示“忽略”类别（某些数据集可能用 255 作为“忽略”类别）
    if lbl.min() >= -1 and lbl.max() < 255:
        # 将标签转换为 uint8 类型，并创建 PIL 图像（使用调色板模式 "P"）
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode="P")
        # 生成颜色映射表（colormap），确保可视化时颜色正确
        colormap = imgviz.label_colormap()
        # 将颜色映射应用到 PIL 图像
        lbl_pil.putpalette(colormap.flatten())
        # 保存为 PNG 文件
        lbl_pil.save(filename)
    else:
        # 如果数值范围超出 [0, 254]，则无法用 PNG 存储
        raise ValueError(
            f"[{filename}] Cannot save the pixel-wise class label as PNG. "
            "Please consider using the .npy format."
        )
