from ultralytics import YOLO

# Load the trained model
model = YOLO("../models/best.pt")

# Run prediction on all test images
results = model.predict(
    source="../dataset/search-and-rescue/test/images",
    conf=0.25,
    save=True,
    show=False
)

print("Prediction completed successfully!")