from ultralytics import YOLO

# model = YOLO('yolov8n.yaml')
model = YOLO('yolov8n.pt')  # for pretrained
results = model.train(data='S2TLD.yaml', epochs=150, imgsz=640, pretrained=True, cache=True, name='v8n_150', batch=32, workers=0)

# -- resume train --
# model = YOLO('/home/moonjay/codes/2023/ultralytics-yolov8/runs/detect/swg23_v8n/weights/last.pt')
# results = model.train(data='S2TLD.yaml', epochs=200, imgsz=640, pretrained=False, cache=False, name='swg23_v8n', batch=24, resume=True)
