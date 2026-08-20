from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("../models/best.pt")

print(model.names)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO on the frame
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Show live video
    cv2.imshow("AI Rescue Drone - Human Detection", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()