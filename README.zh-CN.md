[English](README.md) | [简体中文](README.zh-CN.md)
<br>

## <div align="center">文档</div>

有关训练、测试和部署的完整文档见[YOLOv8 Docs](https://docs.ultralytics.com)。请参阅下面的快速入门示例。

<details open>
<summary>安装</summary>

Pip 安装包含所有 [requirements.txt](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) 的
ultralytics 包，环境要求 [**Python>=3.7**](https://www.python.org/)，且 [\*\*PyTorch>=1.7
\*\*](https://pytorch.org/get-started/locally/)。

```bash
pip install ultralytics
```

</details>

<details open>
<summary>使用方法</summary>

YOLOv8 可以直接在命令行界面（CLI）中使用 `yolo` 命令运行：

```bash
yolo predict model=yolov8n.pt source="https://ultralytics.com/images/bus.jpg"
```

`yolo`可以用于各种任务和模式，并接受额外的参数，例如 `imgsz=640`。参见 YOLOv8 [文档](https://docs.ultralytics.com)
中可用`yolo`[参数](https://docs.ultralytics.com/cfg/)的完整列表。

```bash
yolo task=detect    mode=train    model=yolov8n.pt        args...
          classify       predict        yolov8n-cls.yaml  args...
          segment        val            yolov8n-seg.yaml  args...
                         export         yolov8n.pt        format=onnx  args...
```

YOLOv8 也可以在 Python 环境中直接使用，并接受与上面 CLI 例子中相同的[参数](https://docs.ultralytics.com/cfg/)：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("yolov8n.pt")  # 加载预训练模型（推荐用于训练）

# Use the model
results = model.train(data="coco128.yaml", epochs=3)  # 训练模型
results = model.val()  # 在验证集上评估模型性能
results = model("https://ultralytics.com/images/bus.jpg")  # 预测图像
success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
```

[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models) 会从
Ultralytics [发布页](https://github.com/ultralytics/ultralytics/releases) 自动下载。

</details>

### S2TLD 数据集


- [OpenDataLab/S2TLD](https://opendatalab.org.cn/OpenDataLab/S2TLD)
- [Huggingface/S2TLD](https://huggingface.co/datasets/yangxue/S2TLD/tree/main)

```
目录结构

S2TLD
    - S2TLD（720x1280）
        - normal_1
            - Annotations
            - JPEGImages 779 files
        - normal_2
            - Annotations
            - JPEGImages 3785 files
        - class.txt
    - S2TLD（1080x1920）
        - Annotations
        - JPEGImages 1222 files
```
`total: 5786 img files`

`8:1:1 == 4631:578:577`

