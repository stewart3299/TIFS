
from ultralytics import YOLO


def main():
    model = YOLO("best.pt")
    metrics = model.val(data='data/drone2.yaml', split='test',imgsz=640, batch=16, device=0)
    # metrics = model.val(data='data/VEDAI.yaml', split='test',imgsz=1024, batch=4, device=0)

if __name__ == '__main__':
    main()  # 确保多进程启动时不会递归执行

