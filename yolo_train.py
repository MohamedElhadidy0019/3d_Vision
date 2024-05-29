import ultralytics
from ultralytics import YOLO



model = YOLO("yolov8x-seg.yaml")  # build a new model from YAML
# give the mdoel link to the weights
model.load("/home/mohamed/repos/3d_Vision/yolo_rgb/train15/weights/best.pt")  # load a model from a .pt file

results = model.train(data="yolo_ds/dataset.yaml",
                    epochs=200,
                    imgsz=512,
                    batch=2,
                    dropout=0.3,
                    workers = 4,
                    cos_lr = True,
                    amp = False,
                    lr0	= 1e-4,
                    project = 'yolo_rgb',
                    name = 'train',
                    cache = True,
                    hsv_h = 0,
                    hsv_s = 0,
                    hsv_v = 0,
                    degrees = 10,
                    # translate = 0.1,
                    scale = 0.1,
                    mosaic =0
                    )

model.export(format='onnx', imgsz = 512)
