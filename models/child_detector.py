import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO

# Detects children in images using a pre-trained YOLO model

class ChildDetector:

    def __init__(self, model_path=None):


        # Use pre-trained YOLOv8 model (can detect 'person' class)
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Use small YOLOv8 model
            self.model = YOLO('yolov8n.pt')  # This downloads automatically
        
        self.child_class_id = 0  # 'person' class in COCO
        self.confidence_threshold = 0.5
        
    def detect_in_image(self, image_path):
        """Detect children in a single image"""
        results = self.model(image_path, conf=self.confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id == self.child_class_id:
                        # Bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Calculate if in car seat region (heuristic)
                        in_car_seat = self._is_in_car_seat_region(x1, y1, x2, y2)
                        
                        detection = {
                            'class': 'child',
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'in_car_seat': in_car_seat,
                            'image_path': image_path
                        }
                        detections.append(detection)
        
        return detections
    
    def _is_in_car_seat_region(self, x1, y1, x2, y2):
        """Heuristic to check if detection is in car seat area"""
        # Car seats are typically in bottom half of image
        image_center_y = 480 / 2  # Assuming 640x480 image
        bbox_center_y = (y1 + y2) / 2
        
        # Also consider size - children are smaller than adults
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        bbox_area = bbox_height * bbox_width
        
        # Heuristic rules
        is_lower_half = bbox_center_y > image_center_y
        is_small_size = bbox_area < (640 * 480 * 0.3)  # Less than 30% of image
        
        return is_lower_half and is_small_size
    
    def process_video_stream(self, video_path, output_path="output.mp4"):
        """Process video and save results"""
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Define video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        all_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect children
            results = self.model(frame, conf=self.confidence_threshold)
            
            child_count = 0
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls[0]) == self.child_class_id:
                            child_count += 1
                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"Child: {box.conf[0]:.2f}", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (0, 0, 255), 2)
            
            # Add child count overlay
            cv2.putText(frame, f"Children detected: {child_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add timestamp
            cv2.putText(frame, f"Frame: {frame_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            
            # Save detection data
            all_detections.append({
                'frame': frame_count,
                'child_count': child_count,
                'timestamp': frame_count / fps
            })
            
            frame_count += 1
            
            # Early exit for testing
            if frame_count > 100:  # Process first 100 frames
                break
        
        cap.release()
        out.release()
        
        # Save detection log
        pd.DataFrame(all_detections).to_csv("data/processed/video_detections.csv", index=False)
        
        return output_path

# Create a simple training script to fine-tune on car interior images
def create_training_data():
    """Create annotated training data for fine-tuning"""
    import json
    
    # Create COCO format annotations
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "child_in_carseat"}]
    }
    
    # Load our synthetic images
    image_dir = "data/raw"
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    annotation_id = 1
    for i, img_file in enumerate(image_files):
        # Add image info
        coco_format["images"].append({
            "id": i,
            "file_name": img_file,
            "width": 640,
            "height": 480
        })
        
        # Check metadata for annotations
        metadata_file = f"{image_dir}/image_metadata.csv"
        if os.path.exists(metadata_file):
            metadata = pd.read_csv(metadata_file)
            img_meta = metadata[metadata['image_id'] == img_file.replace('.jpg', '')]
            
            if not img_meta.empty and img_meta.iloc[0]['has_child']:
                # Add annotation (using estimated bbox for car seat)
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [100, 200, 200, 200],  # [x, y, width, height]
                    "area": 200 * 200,
                    "iscrowd": 0
                })
                annotation_id += 1
    
    # Save annotations
    with open("data/processed/annotations.json", "w") as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"Created {annotation_id-1} annotations for training")

# Main script
if __name__ == "__main__":
    print("=== Child Detection System ===")
    
    # Create training data
    create_training_data()
    
    # Test detector on a sample image
    detector = ChildDetector()
    
    # Create a test image if none exists
    test_image = "data/raw/test_detection.jpg"
    if not os.path.exists(test_image):
        # Create a simple test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a "child" in car seat
        cv2.rectangle(img, (150, 250), (250, 350), (255, 200, 200), -1)
        cv2.imwrite(test_image, img)
    
    # Run detection
    detections = detector.detect_in_image(test_image)
    
    print(f"\nDetection Results:")
    if detections:
        for det in detections:
            print(f"  Child detected: {det['confidence']:.2f} confidence")
            print(f"  In car seat: {det['in_car_seat']}")
    else:
        print("  No children detected")