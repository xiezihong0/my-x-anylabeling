import re
import math
import textwrap
import platform
import subprocess
from typing import Iterator, Tuple
from importlib_metadata import version as get_package_version

# 加粗文本格式化
def format_bold(text):
    return f"\033[1m{text}\033[0m"

# 颜色格式化
def format_color(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

# 渐变色文本渲染
def gradient_text(text: str, 
                  start_color: Tuple[int, int, int] = (0, 0, 255), 
                  end_color: Tuple[int, int, int] = (255, 0, 255), 
                  frequency: float = 1.0) -> str:
    # 计算渐变色
    def color_function(t: float) -> Tuple[int, int, int]:
        def interpolate(start: float, end: float, t: float) -> float:
            # Use a sine wave for smooth, periodic interpolation
            # 使用正弦波进行平滑插值
            return start + (end - start) * (math.sin(math.pi * t * frequency) + 1) / 2

        return tuple(round(interpolate(s, e, t)) for s, e in zip(start_color, end_color))

    # 生成颜色梯度
    def gradient_gen(length: int) -> Iterator[Tuple[int, int, int]]:
        return (color_function(i / (length - 1)) for i in range(length))

    gradient = gradient_gen(len(text))
    return ''.join(f"\033[38;2;{r};{g};{b}m{char}\033[0m" for char, (r, g, b) in zip(text, gradient))  # noqa: E501

# 将十六进制颜色转换为 RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

# 文本缩进
def indent_text(text, indent=4):
    return textwrap.indent(text, ' ' * indent)

# 判断字符串是否包含中文字符
def is_chinese(s="人工智能"):
    # Is string composed of any Chinese characters?
    return bool(re.search("[\u4e00-\u9fff]", str(s)))

# 判断四个点是否构成矩形
def is_possible_rectangle(points):
    if len(points) != 4:
        return False

    # Check if four points form a rectangle
    # The points are expected to be in the format:
    # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    dists = [square_dist(points[i], points[(i + 1) % 4]) for i in range(4)]
    dists.sort()

    # For a rectangle, the two smallest distances
    # should be equal and the two largest should be equal
    return dists[0] == dists[1] and dists[2] == dists[3]

# 计算每两个点之间的距离平方
def square_dist(p, q):
    # Calculate the square distance between two points
    # 只有当两组距离相等时才是矩形
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

# 收集系统信息
def collect_system_info():
    os_info = platform.platform()
    cpu_info = platform.processor()
    gpu_info = get_gpu_info()
    cuda_info = get_cuda_version()
    python_info = platform.python_version()
    pyqt5_info = get_installed_package_version("PyQt5")
    onnx_info = get_installed_package_version("onnx")
    ort_info = get_installed_package_version("onnxruntime")
    ort_gpu_info = get_installed_package_version("onnxruntime-gpu")
    opencv_contrib_info = get_installed_package_version(
        "opencv-contrib-python-headless"
    )

    system_info = {
        "Operating System": os_info,
        "CPU": cpu_info,
        "GPU": gpu_info,
        "CUDA": cuda_info,
        "Python Version": python_info,
    }
    pkg_info = {
        "PyQt5 Version": pyqt5_info,
        "ONNX Version": onnx_info,
        "ONNX Runtime Version": ort_info,
        "ONNX Runtime GPU Version": ort_gpu_info,
        "OpenCV Contrib Python Headless Version": opencv_contrib_info,
    }

    return system_info, pkg_info

# 获取已安装的软件包版本
def get_installed_package_version(package_name):
    try:
        return get_package_version(package_name)
    except Exception:
        return None

# 获取 CUDA 版本
def get_cuda_version():
    try:
        nvcc_output = subprocess.check_output(["nvcc", "--version"]).decode(
            "utf-8"
        )
        version_line = next(
            (line for line in nvcc_output.split("\n") if "release" in line),
            None,
        )
        if version_line:
            return version_line.split()[-1]
    except Exception:
        return None

# 获取 GPU 信息
def get_gpu_info():
    try:
        smi_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        return ", ".join(smi_output.strip().split("\n"))
    except Exception:
        return None
