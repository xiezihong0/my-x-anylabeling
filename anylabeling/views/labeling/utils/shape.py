import math
import uuid

import numpy as np
import PIL.Image
import PIL.ImageDraw

from ..logger import logger


def polygons_to_mask(img_shape, polygons, shape_type=None):
    """
    【已弃用】将多边形转换为掩码（Mask）。

    参数：
    - img_shape (tuple): 图像的形状 (height, width)
    - polygons (list): 多边形的点坐标
    - shape_type (str, 可选): 形状类型

    返回：
    - mask (numpy.ndarray): 二值化的掩码图像（布尔类型）

    提示：
    - 该函数已被弃用，建议使用 `shape_to_mask` 代替。
    """
    logger.warning(
        "The 'polygons_to_mask' function is deprecated, "
        "use 'shape_to_mask' instead."
    )
    return shape_to_mask(img_shape, points=polygons, shape_type=shape_type)


def shape_to_mask(
    img_shape, points, shape_type=None, line_width=10, point_size=5
):
    """
    将指定形状转换为二值掩码图像。

    参数：
    - img_shape (tuple): 图像的形状 (height, width)
    - points (list of tuples): 形状的关键点坐标
    - shape_type (str, 可选): 形状类型，可选值为 "circle", "rectangle", "rotation", "line", "linestrip", "point", "polygon"
    - line_width (int): 线条宽度（适用于线段）
    - point_size (int): 点的大小（适用于点类型）

    返回：
    - mask (numpy.ndarray): 二值化的掩码图像（布尔类型）
    """
    # 初始化掩码数组
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)  # 转换为 PIL 图像
    draw = PIL.ImageDraw.Draw(mask)  # 获取绘图工具
    xy = [tuple(point) for point in points]   # 将点坐标转换为元组格式
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)  # 计算半径
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "rotation":
        assert len(xy) == 4, "Shape of shape_type=rotation must have 4 points"
        draw.polygon(xy=xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size # 设置点的大小
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        # 默认情况为多边形
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    # 转换为 NumPy 布尔数组
    mask = np.array(mask, dtype=bool)
    return mask


def shapes_to_label(img_shape, shapes, label_name_to_value):
    """
    将形状转换为类别掩码和实例掩码。

    参数：
    - img_shape (tuple): 图像尺寸 (height, width)
    - shapes (list of dict): 包含形状信息的列表，每个元素包括 "points" (形状点坐标), "label" (类别标签), "group_id" (可选的组 ID)
    - label_name_to_value (dict): 形状标签到整数类别 ID 的映射

    返回：
    - cls (numpy.ndarray): 类别掩码，每个像素表示类别 ID
    - ins (numpy.ndarray): 实例掩码，每个像素表示实例 ID
    """
    # 初始化类别掩码
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    # 初始化实例掩码
    ins = np.zeros_like(cls)
    # 存储唯一的实例 (类别名 + 组 ID)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        # 计算实例 ID
        ins_id = instances.index(instance) + 1
        # 获取类别 ID
        cls_id = label_name_to_value[cls_name]

        # 生成掩码
        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins


def masks_to_bboxes(masks):
    """
    从二值掩码计算边界框 (bounding box)。

    参数：
    - masks (numpy.ndarray): 形状为 (N, H, W) 的掩码数组，N 为掩码数量，H 和 W 为图像尺寸

    返回：
    - bboxes (numpy.ndarray): 形状为 (N, 4) 的数组，每个元素表示 (y1, x1, y2, x2)
    """
    print("masks_to_bboxes==========",masks)
    if masks.ndim != 3:
        raise ValueError(f"masks.ndim must be 3, but it is {masks.ndim}")
    if masks.dtype != bool:
        raise ValueError(
            f"masks.dtype must be bool type, but it is {masks.dtype}"
        )
    bboxes = []
    for mask in masks:
        # 获取掩码中所有为 True 的坐标
        where = np.argwhere(mask)
        # 计算边界框
        (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
        bboxes.append((y1, x1, y2, x2))
    bboxes = np.asarray(bboxes, dtype=np.float32)
    return bboxes


def rectangle_from_diagonal(diagonal_vertices):
    """
    Generate rectangle vertices from diagonal vertices.

    Parameters:
    - diagonal_vertices (list of lists):
        List containing two points representing the diagonal vertices.

    Returns:
    - list of lists:
        List containing four points representing the rectangle's four corners.
        [tl -> tr -> br -> bl]
    """
    """
    通过对角线的两个点生成矩形的四个顶点。
    
    参数：
    - diagonal_vertices (list of lists): 形状为 [[x1, y1], [x2, y2]] 的列表，表示矩形对角线上的两个点
    
    返回：
    - list of lists: 形状为 [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] 的矩形四个角点
    """
    x1, y1 = diagonal_vertices[0]
    x2, y2 = diagonal_vertices[1]

    # Creating the four-point representation
    rectangle_vertices = [
        [x1, y1],  # Top-left
        [x2, y1],  # Top-right
        [x2, y2],  # Bottom-right
        [x1, y2],  # Bottom-left
    ]

    return rectangle_vertices
