import ultralytics
from ultralytics import YOLO

import torch

model = YOLO("yolov8l-seg.yaml")  # build a new model from YAML
# give the mdoel link to the weights
# model.load("/home/mohamed/repos/3d_Vision/runs/segment/train2/weights/best.pt")  # load a model from a .pt file

# results = model.train(data="yolo_ds_rgbd/dataset.yaml", epochs=50, imgsz=512, batch=8, dropout=0.3, workers = 1)

random_tensor = torch.rand(1, 4, 512, 512)
results = model(random_tensor)
print(results)