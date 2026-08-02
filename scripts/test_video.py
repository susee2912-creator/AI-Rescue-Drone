from ultralytics import YOLO

# Load trained model
model = YOLO("../models/best.pt")

# Input video
video_path = "../videos/test.mp4"

# Run prediction
results = model.predict(
    source=video_path,
    save=True,
    conf=0.25
)

print("Video prediction completed!")