import os
import cv2
import math
import numpy as np
from argparse import Namespace
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from ..model import Model
from ..engines import OnnxBaseModel
from ..types import AutoLabelingResult
from ..trackers import BOTSORT, BYTETracker
from ..utils import (
    letterbox,
    scale_boxes,
    scale_coords,
    point_in_bbox,
    masks2segments,
    xyxy2xywh,
    xywhr2xyxyxyxy,
    non_max_suppression_v5,
    non_max_suppression_v8,
)

# YOLO 类继承自 Model，表示一个 YOLO 模型的封装
class YOLO(Model):
    # Meta 类包含模型的配置信息
    class Meta:
        # 需要的配置信息项
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
        ]
        # UI 组件
        widgets = [
            "button_run",
            "input_conf",
            "edit_conf",
            "input_iou",
            "edit_iou",
            "toggle_preserve_existing_annotations",
            "button_reset_tracker",
        ]
        # 输出模式（点、矩形、多边形）
        output_modes = {
            "point": QCoreApplication.translate("Model", "Point"),
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        # 默认输出模式
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        # 调用父类 Model 的初始化方法
        # Run the parent class's init method
        super().__init__(model_config, on_message)

        # 获取模型的绝对路径
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {self.config['type']} model.",
                )
            )

        # 获取模型引擎，默认为 ONNX Runtime（ort）
        self.engine = self.config.get("engine", "ort")
        # 选择不同的推理引擎
        if self.engine.lower() == "dnn":
            # 只有在需要时导入 DNN 引擎
            from ..engines import DnnBaseModel

            self.net = DnnBaseModel(model_abs_path, __preferred_device__)
            self.input_width = self.config.get("input_width", 640)
            self.input_height = self.config.get("input_height", 640)
        else:
            self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
            (
                _,
                _,
                self.input_height,
                self.input_width,
            ) = self.net.get_input_shape()
            # 确保输入尺寸是整数
            if not isinstance(self.input_width, int):
                self.input_width = self.config.get("input_width", -1)
            if not isinstance(self.input_height, int):
                self.input_height = self.config.get("input_height", -1)

        # 初始化其他参数
        self.replace = True
        self.model_type = self.config["type"]
        self.classes = self.config.get("classes", [])
        self.stride = self.config.get("stride", 32)
        self.anchors = self.config.get("anchors", None)
        self.agnostic = self.config.get("agnostic", False)
        self.show_boxes = self.config.get("show_boxes", False)
        # 设置推理点的密集程度：值越少，就点越多
        self.epsilon_factor = self.config.get("epsilon_factor", 0.001)
        self.iou_thres = self.config.get("nms_threshold", 0.45)
        self.conf_thres = self.config.get("confidence_threshold", 0.25)
        self.filter_classes = self.config.get("filter_classes", None)
        # 类别数量
        self.nc = len(self.classes)
        self.input_shape = (self.input_height, self.input_width)
        # 处理 anchors
        if self.anchors:
            self.nl = len(self.anchors)
            self.na = len(self.anchors[0]) // 2
            self.grid = [np.zeros(1)] * self.nl
            self.stride = (
                np.array([self.stride // 4, self.stride // 2, self.stride])
                if not isinstance(self.stride, list)
                else np.array(self.stride)
            )
            self.anchor_grid = np.asarray(
                self.anchors, dtype=np.float32
            ).reshape(self.nl, -1, 2)
        # 过滤类别
        if self.filter_classes:
            self.filter_classes = [
                i
                for i, item in enumerate(self.classes)
                if item in self.filter_classes
            ]

        # 处理目标跟踪器
        """Tracker"""
        tracker = self.config.get("tracker", {})
        if tracker:
            tracker_args = Namespace(**tracker)
            if tracker_args.tracker_type == "bytetrack":
                self.tracker = BYTETracker(tracker_args, frame_rate=30)
            elif tracker_args.tracker_type == "botsort":
                self.tracker = BOTSORT(tracker_args, frame_rate=30)
            else:
                self.tracker = None
                logger.error(
                    "Only 'bytetrack' and 'botsort' are supported for now, "
                    f"but got '{tracker_args.tracker_type}'!"
                )
        else:
            self.tracker = None

        # 设定任务类型（检测、分割、目标跟踪、姿态估计等）
        if self.model_type in [
            "yolov5",
            "yolov6",
            "yolov7",
            "yolov8",
            "yolov9",
            "yolov10",
            "gold_yolo",
            "yolow",
            "yolow_ram",
            "yolov5_det_track",
            "yolov8_det_track",
        ]:
            self.task = "det"
        elif self.model_type in [
            "yolov5_seg",
            "yolov8_seg",
            "yolov8_seg_track",
        ]:
            self.task = "seg"
        elif self.model_type in [
            "yolov8_obb",
            "yolov8_obb_track",
        ]:
            self.task = "obb"
        elif self.model_type in [
            "yolov6_face",
            "yolov8_pose",
            "yolov8_pose_track",
        ]:
            self.task = "pose"
            self.keypoint_name = {}
            self.show_boxes = True
            self.has_visible = self.config.get("has_visible", True)
            self.kpt_thres = self.config.get("kpt_threshold", 0.1)
            self.classes = self.config.get("classes", {})
            for class_name, keypoints in self.classes.items():
                self.keypoint_name[class_name] = keypoints
            self.classes = list(self.classes.keys())
            kpt_shape_str = self.net.get_metadata_info("kpt_shape")
            if kpt_shape_str and isinstance(kpt_shape_str, str):
                self.kpt_shape = eval(kpt_shape_str)
            else:
                self.kpt_shape = None
            if self.kpt_shape is None:
                max_kpts = max(
                    len(num_kpts) for num_kpts in self.keypoint_name.values()
                )
                visible_flag = 3 if self.has_visible else 2
                self.kpt_shape = [max_kpts, visible_flag]

        if isinstance(self.classes, dict):
            self.classes = list(self.classes.values())

    # 设置自动标注的置信度阈值
    def set_auto_labeling_conf(self, value):
        """set auto labeling confidence threshold"""
        if value > 0:
            self.conf_thres = value

    # 设置自动标注的 IoU 阈值
    def set_auto_labeling_iou(self, value):
        """set auto labeling iou threshold"""
        if value > 0:
            self.iou_thres = value

    # 设置是否保留已有的标注
    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations based on the checkbox state."""
        self.replace = not state

    # 重置目标跟踪器
    def set_auto_labeling_reset_tracker(self):
        """Resets the tracker to its initial state, clearing all tracked objects and internal states."""
        if self.tracker is not None:
            self.tracker.reset()

    # 推理方法
    def inference(self, blob):
        if self.engine == "dnn" and self.task in ["det", "seg", "track"]:
            outputs = self.net.get_dnn_inference(blob=blob, extract=False)
            if self.task == "det" and not isinstance(outputs, (tuple, list)):
                outputs = [outputs]
        else:
            outputs = self.net.get_ort_inference(blob=blob, extract=False)
        return outputs

    # 预处理图像
    def preprocess(self, image, upsample_mode="letterbox"):
        self.img_height, self.img_width = image.shape[:2]
        # Upsample
        if upsample_mode == "resize":
            input_img = cv2.resize(
                image, (self.input_width, self.input_height)
            )
        elif upsample_mode == "letterbox":
            input_img = letterbox(image, self.input_shape)[0]
        elif upsample_mode == "centercrop":
            m = min(self.img_height, self.img_width)
            top = (self.img_height - m) // 2
            left = (self.img_width - m) // 2
            cropped_img = image[top : top + m, left : left + m]
            input_img = cv2.resize(
                cropped_img, (self.input_width, self.input_height)
            )
        # Transpose
        input_img = input_img.transpose(2, 0, 1)
        # Expand
        input_img = input_img[np.newaxis, :, :, :].astype(np.float32)
        # Contiguous
        input_img = np.ascontiguousarray(input_img)
        # Norm
        blob = input_img / 255.0
        return blob

    # 后处理推理结果
    def postprocess(self, preds):
        if self.model_type in [
            "yolov5",
            "yolov5_resnet",
            "yolov5_ram",
            "yolov5_sam",
            "yolov5_seg",
            "yolov5_det_track",
            "yolov6",
            "yolov7",
            "gold_yolo",
        ]:
            # Only support YOLOv5 version 5.0 and earlier versions
            if self.model_type == "yolov5" and self.anchors:
                preds = self.scale_grid(preds)
            p = non_max_suppression_v5(
                preds[0],
                task=self.task,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                classes=self.filter_classes,
                agnostic=self.agnostic,
                multi_label=False,
                nc=self.nc,
            )
        elif self.model_type in [
            "yolov8",
            "yolov8_efficientvit_sam",
            "yolov8_seg",
            "yolov8_obb",
            "yolov9",
            "yolow",
            "yolov8_pose",
            "yolow_ram",
            "yolov8_det_track",
            "yolov8_seg_track",
            "yolov8_obb_track",
            "yolov8_pose_track",
        ]:
            p = non_max_suppression_v8(
                preds[0],
                task=self.task,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                classes=self.filter_classes,
                agnostic=self.agnostic,
                multi_label=False,
                nc=self.nc,
            )
        elif self.model_type == "yolov10":
            p = self.postprocess_v10(
                preds[0][0],
                conf_thres=self.conf_thres,
                classes=self.filter_classes,
            )
        masks, keypoints = None, None
        img_shape = (self.img_height, self.img_width)
        if self.task == "seg":
            proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]
            self.mask_height, self.mask_width = proto.shape[2:]
        for i, pred in enumerate(p):
            if self.task == "seg":
                if np.size(pred) == 0:
                    continue
                masks = self.process_mask(
                    proto[i],
                    pred[:, 6:],
                    pred[:, :4],
                    self.input_shape,
                    upsample=True,
                )  # HWC
            elif self.task == "obb":
                pred[:, :4] = scale_boxes(
                    self.input_shape, pred[:, :4], img_shape, xywh=True
                )
            else:
                pred[:, :4] = scale_boxes(
                    self.input_shape, pred[:, :4], img_shape
                )

        if self.task == "obb":
            pred = np.concatenate(
                [pred[:, :4], pred[:, -1:], pred[:, 4:6]], axis=-1
            )
            bbox = pred[:, :5]
            conf = pred[:, -2]
            clas = pred[:, -1]
        elif self.task == "pose":
            pred_kpts = pred[:, 6:]
            if pred.shape[0] != 0:
                pred_kpts = pred_kpts.reshape(
                    pred_kpts.shape[0], *self.kpt_shape
                )
            bbox = pred[:, :4]
            conf = pred[:, 4:5]
            clas = pred[:, 5:6]
            keypoints = scale_coords(
                self.input_shape, pred_kpts, self.image_shape
            )
        else:
            bbox = pred[:, :4]
            conf = pred[:, 4:5]
            clas = pred[:, 5:6]
        return (bbox, clas, conf, masks, keypoints)

    """
    从图像 `image` 预测目标形状（边界框、分割、多边形、关键点等）
    """
    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        if image is None:
            # 如果输入为空，直接返回空列表
            return []

        try:
            # 转换图像格式（Qt 图片转换为 OpenCV 兼容格式）
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            return []
        # 记录图像的原始尺寸
        self.image_shape = image.shape
        # 预处理图像，使其适配模型输入
        blob = self.preprocess(image, upsample_mode="letterbox")
        # 进行模型推理
        outputs = self.inference(blob)
        # 解析推理结果，获取目标的各种属性
        boxes, class_ids, scores, masks, keypoints = self.postprocess(outputs)
        # 处理分割任务的多边形坐标
        points = [[] for _ in range(len(boxes))]
        if self.task == "seg" and masks is not None:
            points = [
                scale_coords(self.input_shape, x, image.shape, normalize=False)
                for x in masks2segments(masks, self.epsilon_factor)
            ]
        # 处理目标跟踪任务
        track_ids = [[] for _ in range(len(boxes))]
        if self.tracker is not None and (len(boxes) > 0):
            if self.task == "obb":
                tracks = self.tracker.update(
                    scores.flatten(), boxes, class_ids.flatten(), image
                )
            else:
                tracks = self.tracker.update(
                    scores.flatten(),
                    xyxy2xywh(boxes),
                    class_ids.flatten(),
                    image,
                )
            # 如果跟踪成功，更新相关数据
            if len(tracks) > 0:
                boxes = tracks[:, :5] if self.task == "obb" else tracks[:, :4]
                track_ids = (
                    tracks[:, 5:6] if self.task == "obb" else tracks[:, 4:5]
                )
                scores = (
                    tracks[:, 6:7] if self.task == "obb" else tracks[:, 5:6]
                )
                class_ids = (
                    tracks[:, 7:8] if self.task == "obb" else tracks[:, 6:7]
                )
        # 关键点处理（如果没有关键点，赋空列表）
        if keypoints is None:
            keypoints = [[] for _ in range(len(boxes))]

        shapes = []
        # 遍历所有目标，创建 Shape 形状对象
        for i, (box, class_id, score, point, keypoint, track_id) in enumerate(
            zip(boxes, class_ids, scores, points, keypoints, track_ids)
        ):
            # 处理检测任务（绘制矩形框）
            if self.task == "det" or self.show_boxes:
                x1, y1, x2, y2 = box.astype(float)
                shape = Shape(flags={})
                shape.add_point(QtCore.QPointF(x1, y1))
                shape.add_point(QtCore.QPointF(x2, y1))
                shape.add_point(QtCore.QPointF(x2, y2))
                shape.add_point(QtCore.QPointF(x1, y2))
                shape.shape_type = "rectangle"
                shape.closed = True
                shape.label = str(self.classes[int(class_id)])
                shape.score = float(score)
                shape.selected = False
                if self.task == "pose":
                    shape.group_id = int(i)
                if self.tracker and track_id:
                    shape.group_id = int(track_id)
                shapes.append(shape)
            # 处理分割任务（绘制多边形）
            if self.task == "seg":
                # 确保多边形至少有 3 个点
                if len(point) < 3:
                    continue
                shape = Shape(flags={})
                for p in point:
                    shape.add_point(QtCore.QPointF(int(p[0]), int(p[1])))
                shape.shape_type = "polygon"
                shape.closed = True
                shape.label = str(self.classes[int(class_id)])
                shape.score = float(score)
                shape.selected = False
                if self.tracker and track_id:
                    shape.group_id = int(track_id)
                shapes.append(shape)
            # 处理关键点（姿态估计）
            if self.task == "pose":
                label = str(self.classes[int(class_id)])
                keypoint_name = self.keypoint_name[label]
                for j, kpt in enumerate(keypoint):
                    if len(kpt) == 2:
                        x, y, s = *kpt, 1.0
                    else:
                        x, y, s = kpt
                    inside_flag = point_in_bbox((x, y), box)
                    if (
                        (x == 0 and y == 0)
                        or not inside_flag
                        or s < self.kpt_thres
                    ):
                        continue
                    shape = Shape(flags={})
                    shape.add_point(QtCore.QPointF(int(x), int(y)))
                    shape.shape_type = "point"
                    shape.difficult = False
                    if self.tracker and track_id:
                        shape.group_id = int(track_id)
                    else:
                        shape.group_id = int(i)
                    shape.closed = True
                    shape.label = keypoint_name[j]
                    shape.score = float(s)
                    shape.selected = False
                    shapes.append(shape)
            # 处理旋转框任务（OBB）
            if self.task == "obb":
                poly = xywhr2xyxyxyxy(box)
                x0, y0 = poly[0]
                x1, y1 = poly[1]
                x2, y2 = poly[2]
                x3, y3 = poly[3]
                direction = self.calculate_rotation_theta(poly)
                shape = Shape(flags={})
                shape.add_point(QtCore.QPointF(x0, y0))
                shape.add_point(QtCore.QPointF(x1, y1))
                shape.add_point(QtCore.QPointF(x2, y2))
                shape.add_point(QtCore.QPointF(x3, y3))
                shape.shape_type = "rotation"
                shape.closed = True
                shape.direction = direction
                shape.label = str(self.classes[int(class_id)])
                shape.score = float(score)
                shape.selected = False
                if self.tracker and track_id:
                    shape.group_id = int(track_id)
                shapes.append(shape)
        result = AutoLabelingResult(shapes, replace=self.replace)

        return result

    """
    创建一个 (nx, ny) 网格坐标矩阵，返回每个点的坐标。
    Args:
        nx (int): 网格的行数。
        ny (int): 网格的列数。
    Returns:
        np.ndarray: 网格坐标矩阵，每个点的坐标为 (x, y)。
    """
    @staticmethod
    def make_grid(nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    """
    计算多边形旋转角度（弧度），根据多边形的第一个和第二个顶点来推算。
    Args:
        poly (np.ndarray): 包含多边形顶点的数组，至少包含两个顶点。
    Returns:
        float: 多边形旋转的弧度角度。
    """
    @staticmethod
    def calculate_rotation_theta(poly):
        x1, y1 = poly[0]
        x2, y2 = poly[1]

        # 计算对角线向量的 x 和 y 分量
        # Calculate one of the diagonal vectors (after rotation)
        diagonal_vector_x = x2 - x1
        diagonal_vector_y = y2 - y1

        # 使用 atan2 计算旋转角度（弧度）
        # Calculate the rotation angle in radians
        rotation_angle = math.atan2(diagonal_vector_y, diagonal_vector_x)

        # 将弧度转换为度
        # Convert radians to degrees
        rotation_angle_degrees = math.degrees(rotation_angle)

        # 确保旋转角度为正值
        if rotation_angle_degrees < 0:
            rotation_angle_degrees += 360

        # 返回归一化后的旋转角度（范围 [0, 2π]）
        return rotation_angle_degrees / 360 * (2 * math.pi)

    """
    调整网格的坐标，并应用缩放操作，适配模型输出。
    Args:
        outs (np.ndarray): 模型的输出。
    Returns:
        np.ndarray: 处理过的输出，形状适应输入尺寸。
    """
    def scale_grid(self, outs):
        outs = outs[0]
        row_ind = 0
        for i in range(self.nl):
            # 根据网络步幅计算网格的高度和宽度
            h = int(self.input_shape[0] / self.stride[i])
            w = int(self.input_shape[1] / self.stride[i])
            length = int(self.na * h * w)
            # 检查当前网格尺寸是否匹配，不匹配则重新创建
            if self.grid[i].shape[2:4] != (h, w):
                self.grid[i] = self.make_grid(w, h)
            # 缩放坐标并应用网格偏移
            outs[row_ind : row_ind + length, 0:2] = (
                outs[row_ind : row_ind + length, 0:2] * 2.0
                - 0.5
                + np.tile(self.grid[i], (self.na, 1))
            ) * int(self.stride[i])
            # 缩放锚点框
            outs[row_ind : row_ind + length, 2:4] = (
                outs[row_ind : row_ind + length, 2:4] * 2
            ) ** 2 * np.repeat(self.anchor_grid[i], h * w, axis=0)
            row_ind += length
        return outs[np.newaxis, :]

    """
    使用 mask 头的输出将掩码应用于边界框。
    Args:
        protos (np.ndarray): [mask_dim, mask_h, mask_w]。
        masks_in (np.ndarray): [n, mask_dim]，NMS 后的掩码。
        bboxes (np.ndarray): [n, 4]，NMS 后的边界框。
        shape (tuple): 输入图像的大小 (h, w)。
        upsample (bool): 是否上采样掩码到原图大小，默认为 False。
    Returns:
        np.ndarray: 应用边界框的二值掩码 [n, h, w]。
    """
    def process_mask(self, protos, masks_in, bboxes, shape, upsample=False):
        """
        Apply masks to bounding boxes using the output of the mask head.

        Args:
            protos (np.ndarray): A tensor of shape [mask_dim, mask_h, mask_w].
            masks_in (np.ndarray): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
            bboxes (np.ndarray): A tensor of shape [n, 4], where n is the number of masks after NMS.
            shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
            upsample (bool): A flag to indicate whether to upsample the mask to the original image size. Default is False.

        Returns:
            (np.ndarray): A binary mask tensor of shape [n, h, w],
            where n is the number of masks after NMS, and h and w
            are the height and width of the input image.
            The mask is applied to the bounding boxes.
        """
        c, mh, mw = protos.shape
        ih, iw = shape
        # 对掩码输入进行 Sigmoid 激活函数变换
        masks = 1 / (
            1
            + np.exp(
                -np.dot(masks_in, protos.reshape(c, -1).astype(float)).astype(
                    float
                )
            )
        )
        # 重新调整掩码形状
        masks = masks.reshape(-1, mh, mw)

        # 缩放边界框坐标
        downsampled_bboxes = bboxes.copy()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih
        # 裁剪掩码
        masks = self.crop_mask_np(masks, downsampled_bboxes)  # CHW
        # 如果需要，进行上采样
        if upsample:
            if masks.shape[0] == 1:
                masks_np = np.squeeze(masks, axis=0)
                masks_resized = cv2.resize(
                    masks_np,
                    (shape[1], shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                masks = np.expand_dims(masks_resized, axis=0)
            else:
                masks_np = np.transpose(masks, (1, 2, 0))
                masks_resized = cv2.resize(
                    masks_np,
                    (shape[1], shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                masks = np.transpose(masks_resized, (2, 0, 1))
        # 将掩码二值化
        masks[masks > 0.5] = 1
        masks[masks <= 0.5] = 0

        return masks

    """
    后处理函数，基于置信度阈值对模型预测结果进行筛选和分类。
    Args:
        prediction (np.ndarray): 模型的预测结果。
        task (str): 当前任务类型，默认 "det"。
        conf_thres (float): 置信度阈值，低于此值的预测将被丢弃。
        classes (list, optional): 要保留的类别，默认为 None（表示保留所有类别）。
    Returns:
        list: 经筛选后的预测结果。
    """
    def postprocess_v10(
        self, prediction, task="det", conf_thres=0.25, classes=None
    ):
        x = prediction[prediction[:, 4] >= conf_thres]
        x[:, -1] = x[:, -1].astype(int)
        if classes is not None:
            x = x[np.isin(x[:, -1], classes)]
        return [x]

    """
    根据边界框裁剪掩码。
    Args:
        masks (np.ndarray): [n, h, w] 形状的掩码数组。
        boxes (np.ndarray): [n, 4] 形状的边界框坐标数组。
    Returns:
        np.ndarray: 裁剪后的掩码数组。
    """
    @staticmethod
    def crop_mask_np(masks, boxes):
        """
        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.

        Args:
        masks (np.ndarray): [n, h, w] array of masks.
        boxes (np.ndarray): [n, 4] array of bbox coordinates in relative point form.

        Returns:
        (np.ndarray): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.hsplit(boxes[:, :, None], 4)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
        c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

        return masks * ((r >= x1) & (r < x2) & (c >= y1) & (c < y2))

    """
    将边界框坐标从输入大小重新缩放到图像大小。
    Args:
        boxes (np.ndarray): [n, 4] 边界框。
        image_shape (tuple): 图像的高和宽。
        input_shape (tuple): 输入大小的高和宽。
    Returns:
        np.ndarray: 重新缩放的边界框。
    """
    @staticmethod
    def rescale_coords_v10(boxes, image_shape, input_shape):
        image_height, image_width = image_shape
        input_height, input_width = input_shape

        scale = min(input_width / image_width, input_height / image_height)

        pad_w = (input_width - image_width * scale) / 2
        pad_h = (input_height - image_height * scale) / 2

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_height)

        return boxes

    """
    清理网络对象，释放资源。
    """
    def unload(self):
        del self.net
