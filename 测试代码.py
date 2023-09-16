import os           # 导入系统环境变量
import sys          # 导入 python环境变量相关的函数
import cv2
import torch        # 导入机器学习库 pytorch
import torch.backends.cudnn as cudnn    # 导入nvdia cudnn 相关库
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from models.experimental import attempt_load    # attempt_load用于加载模型权重文件并构建模型(可以构造普通模型或者集成模型)
from utils.datasets import LoadImages        # cv2图片、视频读取函数
from utils.general import non_max_suppression, scale_coords, set_logging
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


# 1. 路径处理
FILE = Path(__file__).resolve()     # 解析当前文件的绝对路径
ROOT = FILE.parents[0]              # 解析文件root 目录
if str(ROOT) not in  sys.path:      # 如果当前目前不在python 环境变量中，则添加环境变量
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 计算当前路径距根目录的相对路径

source = str(ROOT / "data/images/4.jpeg")          # 要检测的文件路径

set_logging()   # 日志

# 2. 配置显卡
device = select_device(device='')           # 使用默认的CUDA 显卡
half = device.type != 'cpu'                 # 半精度浮点数仅GPU支持

# 3. 加载 yolov5s.pt 模型,开始建模
weights = str(ROOT / 'yolov5s.pt')               # 模理权重文件路径
model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())                # 获取当前模型中支持的最大模型数量
names = model.modules.names if hasattr(model, 'module') else model.names    # 获取当前模型中各模型的name

if half:                # 如果支持半精度浮点
    model.half()        # 修改所有参数和buffer 为half半精度浮点类型

imgsz = [640,640]

# 4. 加载图片、视频资源
dataset = LoadImages(source, img_size=640,stride=stride, auto=True)

# 5. 开始检测
if device.type != 'cpu':
    model(torch.zeros(1,3,*imgsz).to(device).type_as(next(model.parameters())))

for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img / 255.0
    if len(img.shape) == 3:
            img = img[None]

    # 6. 配置特征图可视化
    pred = model(img, augment=False, visualize=False)[0]

    # NMS
    pred = non_max_suppression(pred, 0.5, 0.45, None, False, multi_label=False, labels=(), max_det=300)

    # 7.开始预测
    for i, det in enumerate(pred):
        p, s, im0, frame = Path(path), '', im0s.copy(), getattr(dataset, 'frame', 0)
        s += '%gx%g, ' % img.shape[2:]                # print string

        annotator = Annotator(im0, line_width=3, example=str(names))

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4]  = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            print(s)

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = (f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
                print(label,xyxy)

        # Stream results
        im0 = annotator.result()
        cv2.imshow(str(p), im0)
        cv2.waitKey(5000)  # 1 millisecond
