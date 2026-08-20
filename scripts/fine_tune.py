from ultralytics import YOLO

# Load your previously trained model
model = YOLO("../models/best.pt")

# Continue training on the merged dataset
model.train(
    data="../dataset/HumanDataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    workers=4,
    project="../runs/detect",
    name="Human_Detection_V2",
    exist_ok=True
)