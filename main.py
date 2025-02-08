from ultralytics import YOLO

model = YOLO('C:/Users/Admin/Desktop/7th semi Labs/Computer vision/Mini project/Fire and smoke detector/Fire_and_smoke_detector/Fire_and_smoke_detector.pt')
model.predict(source=0, imgsz=640, conf = 0, show=True)