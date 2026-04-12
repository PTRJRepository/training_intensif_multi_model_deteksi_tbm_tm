import os
from ultralytics import YOLO
import cv2

def test_inference():
    # Paths
    model_path = r"Train\runs\exp1_yolo11n_640\weights\best.pt"
    test_image_dir = r"Dataset\dataset_pakai_ini\images\val"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Get a test image
    images = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f"Error: No images found in {test_image_dir}")
        return
    
    test_image_path = os.path.join(test_image_dir, images[0])
    print(f"Running inference on {test_image_path}...")
    
    # Run inference
    results = model.predict(
        source=test_image_path,
        imgsz=640,
        conf=0.25,
        iou=0.7,
        device='cpu' # Using CPU as default for testing
    )
    
    # Check results
    for result in results:
        boxes = result.boxes
        print(f"Detected {len(boxes)} objects.")
        for box in boxes:
            c = box.cls
            conf = box.conf
            print(f"Class: {model.names[int(c)]}, Confidence: {conf.item():.2f}")

    print("Test completed successfully!")

if __name__ == "__main__":
    test_inference()
