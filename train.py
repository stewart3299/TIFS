from ultralytics import YOLO
import torch

def main():

    # model = YOLO("yaml/yolov8-obb.yaml")
    model = YOLO("yaml/tstream3_yolov8s.yaml")
    results = model.train(data='data/drone2.yaml', batch=2, epochs=3, workers=0, device=0,amp=False)


if __name__ == '__main__':
    main()  # 确保多进程启动时不会递归执行