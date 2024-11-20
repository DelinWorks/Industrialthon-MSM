import torch
import cv2

# You need to have GPU acceleration enabled for faster peformance, otherwise CPU is used
# look up a guide on how to install nvidia cuda drivers on your machine if it has an nvidia gpu.
# for cuda ensure you have the graphics drivers installed along with Nvidia CUDA toolkit
# https://developer.nvidia.com/cuda-downloads
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the pre-trained YOLOv5 models
safety_equipment_model = torch.hub.load('ultralytics/yolov5', 'custom', path="models/construction-v2.pt", force_reload=True)
factory_waste          = torch.hub.load('ultralytics/yolov5', 'custom', path="models/garbage-v2.pt", force_reload=True)
ground_fluids          = torch.hub.load('ultralytics/yolov5', 'custom', path="models/leaks-v3.pt", force_reload=True)

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

def parseResults(results):
    detections = results.pandas().xyxy[0]  # Bounding boxes and labels
    for index, row in detections.iterrows():
        label = row['name']
        confidence = row['confidence']
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        # Draw bounding box on the image
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img, f"{label} ({confidence:.2f})", (x_min, y_min -40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 240), 2)

while True:
    success, img = cap.read()

    # Convert the image to a torch tensor and move it to the GPU
    img_tensor = torch.from_numpy(img).to(device) if device.type == 'cuda' else img

    # Perform inference
    results1 = safety_equipment_model(img_tensor)
    results2 = factory_waste(img_tensor)
    results3 = ground_fluids(img_tensor)

    parseResults(results1)
    parseResults(results2)
    parseResults(results3)

    cv2.imshow('Detected Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
