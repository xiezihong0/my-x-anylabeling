import cv2
import numpy as np


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def letterbox(
    im,
    new_shape,
    color=(114, 114, 114),
    auto=False,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    """
    Resize and pad image while meeting stride-multiple constraints
    Returns:
        im (array): (height, width, 3)
        ratio (array): [w_ratio, h_ratio]
        (dw, dh) (array): [w_padding h_padding]
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):  # [h_rect, w_rect]
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # wh ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w h
    dw, dh = (
        new_shape[1] - new_unpad[0],
        new_shape[0] - new_unpad[1],
    )  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])  # [w h]
        ratio = (
            new_shape[1] / shape[1],
            new_shape[0] / shape[0],
        )  # [w_ratio, h_ratio]

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im, ratio, (dw, dh)


def softmax(x):
    """
    Applies the softmax function to the input array.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Output array after applying softmax.
    """
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

"""
对输入的轮廓进行优化，采用多边形逼近，并去除过大或过小的轮廓。
参数:
- contours (list): `cv2.findContours` 提取的轮廓列表，每个轮廓是一个 numpy 数组。
- img_area (int): 图像的总面积，用于判断轮廓是否过大。
- epsilon_factor (float, optional): 控制轮廓近似程度的因子，默认值为 0.001。
  - 较小的值会保留更多细节。
  - 较大的值会使轮廓更加平滑，减少冗余点。
返回:
- approx_contours (list): 经过优化的轮廓列表，每个轮廓仍然是一个 numpy 数组。
"""
def refine_contours(contours, img_area, epsilon_factor=0.001):
    """
    Refine contours by approximating and filtering.

    Parameters:
    - contours (list): List of input contours.
    - img_area (int): Maximum factor for contour area.
    - epsilon_factor (float, optional): Factor used for epsilon calculation in contour approximation. Default is 0.001.

    Returns:
    - list: List of refined contours.
    """
    # 用于存储近似后的轮廓
    # Refine contours
    approx_contours = []
    # 遍历所有输入轮廓
    for contour in contours:
        # 计算轮廓的周长
        # Approximate contour
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        # 使用 Douglas-Peucker 算法对轮廓进行多边形近似
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # 将近似后的轮廓添加到列表中
        approx_contours.append(approx)

    # **步骤 1: 过滤过大的轮廓 (面积超过图像 90%)**
    # Remove too big contours ( >90% of image size)
    if len(approx_contours) > 1:
        # 计算所有近似轮廓的面积
        areas = [cv2.contourArea(contour) for contour in approx_contours]
        # 过滤掉面积 > 90% 图像面积的轮廓
        filtered_approx_contours = [
            contour
            for contour, area in zip(approx_contours, areas)
            if area < img_area * 0.9
        ]

    # **步骤 2: 过滤过小的轮廓 (面积小于平均面积的 20%)**
    # Remove small contours (area < 20% of average area)
    if len(approx_contours) > 1:
        # 计算所有近似轮廓的面积
        areas = [cv2.contourArea(contour) for contour in approx_contours]
        # 计算所有轮廓的平均面积
        avg_area = np.mean(areas)
        # 过滤掉面积 < 20% 平均面积的轮廓
        filtered_approx_contours = [
            contour
            for contour, area in zip(approx_contours, areas)
            if area > avg_area * 0.2
        ]
        # 更新最终的轮廓列表
        approx_contours = filtered_approx_contours

    return approx_contours


def point_in_bbox(point, bbox):
    """
    Check if a point is inside a bounding box.

    Parameters:
    - point: Tuple (x, y) representing the point coordinates.
    - bbox: List [xmin, ymin, xmax, ymax] representing the bounding box.

    Returns:
    - True if the point is inside the bounding box, False otherwise.
    """
    x, y = point
    xmin, ymin, xmax, ymax = bbox

    # Check if the point is within the bounding box.
    if xmin <= x <= xmax and ymin <= y <= ymax:
        return True
    else:
        return False
