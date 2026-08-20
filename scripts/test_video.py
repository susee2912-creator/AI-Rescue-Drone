from ultralytics import YOLO

model = YOLO("../models/best_v2.pt")

results = model.predict(
    source="../videos/test.mp4",
    save=True,
    conf=0.25
)