import cv2
import numpy as np
import os
import logging
from ultralytics import YOLO
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Data class for detection results"""
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    in_car_seat: bool
    image_path: str
    detection_id: str


class ChildDetector:
    """
    Detects children in car seats using YOLOv8 object detection.
    
    This detector identifies persons in vehicle interiors and applies
    heuristics to determine if they are likely children in car seats.
    """
    
    # Class constants
    COCO_PERSON_CLASS_ID = 0
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    DEFAULT_MODEL = 'yolov8n.pt'
    
    # Car seat region parameters
    CAR_SEAT_REGION_LOWER_BOUND = 0.5  # Lower 50% of image
    MAX_CHILD_SIZE_RATIO = 0.3  # Max 30% of image area
    
    def __init__(self, model_path: Optional[str] = None, 
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        """
        Initialize the ChildDetector with YOLO model.
        
        Args:
            model_path (Optional[str]): Path to custom YOLO model. 
                                       Uses default if not provided.
            confidence_threshold (float): Minimum confidence score (0-1).
        
        Raises:
            FileNotFoundError: If custom model_path doesn't exist.
            ValueError: If confidence_threshold is not in valid range.
        """
        if confidence_threshold < 0 or confidence_threshold > 1:
            raise ValueError(f"Confidence threshold must be between 0 and 1, got {confidence_threshold}")
        
        self.confidence_threshold = confidence_threshold
        self.child_class_id = self.COCO_PERSON_CLASS_ID
        
        try:
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading custom model from: {model_path}")
                self.model = YOLO(model_path)
            else:
                logger.info(f"Loading default model: {self.DEFAULT_MODEL}")
                self.model = YOLO(self.DEFAULT_MODEL)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect_in_image(self, image_path: str) -> List[Detection]:
        """
        Detect children in a single image.
        
        Args:
            image_path (str): Path to the image file.
        
        Returns:
            List[Detection]: List of Detection objects with details.
        
        Raises:
            FileNotFoundError: If image file doesn't exist.
            ValueError: If image cannot be read.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            results = self.model(image_path, conf=self.confidence_threshold, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                
                for idx, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    if class_id != self.child_class_id:
                        continue
                    
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                    
                    in_car_seat = self._is_in_car_seat_region(
                        x1, y1, x2, y2, result.orig_shape
                    )
                    
                    detection = Detection(
                        class_name='person',
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        in_car_seat=in_car_seat,
                        image_path=image_path,
                        detection_id=f"{os.path.basename(image_path)}_{idx}_{datetime.now().timestamp()}"
                    )
                    detections.append(detection)
            
            logger.info(f"Found {len(detections)} detections in {image_path}")
            return detections
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def _is_in_car_seat_region(self, x1: float, y1: float, x2: float, y2: float, 
                               image_shape: Tuple[int, int]) -> bool:
        """
        Determine if detection is likely in a car seat region using heuristics.
        
        Args:
            x1, y1, x2, y2 (float): Bounding box coordinates.
            image_shape (Tuple[int, int]): Original image dimensions (height, width).
        
        Returns:
            bool: True if detection appears to be in car seat region.
        """
        image_height, image_width = image_shape[:2]
        
        # Calculate bbox properties
        bbox_center_y = (y1 + y2) / 2
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        bbox_area = bbox_height * bbox_width
        image_area = image_height * image_width
        
        # Apply heuristic rules
        is_lower_half = bbox_center_y > (image_height * self.CAR_SEAT_REGION_LOWER_BOUND)
        is_appropriate_size = bbox_area < (image_area * self.MAX_CHILD_SIZE_RATIO)
        
        return is_lower_half and is_appropriate_size
    
    def process_video_stream(self, video_path: str, 
                            output_path: str = "output.mp4",
                            max_frames: Optional[int] = None) -> Tuple[str, pd.DataFrame]:
        """
        Process video stream and save annotated output with detections.
        
        Args:
            video_path (str): Path to input video file.
            output_path (str): Path for output annotated video.
            max_frames (Optional[int]): Maximum frames to process (None = all).
        
        Returns:
            Tuple[str, pd.DataFrame]: Output video path and detection DataFrame.
        
        Raises:
            FileNotFoundError: If video file doesn't exist.
            IOError: If video cannot be read or written.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {video_path}")
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            if not out.isOpened():
                raise IOError(f"Cannot write to output video: {output_path}")
            
            frame_count = 0
            all_detections = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                child_count = 0
                
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    
                    for box in boxes:
                        if int(box.cls[0]) != self.child_class_id:
                            continue
                        
                        child_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Child: {confidence:.2f}", 
                                  (x1, max(y1 - 10, 20)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Add overlays
                self._add_frame_info(frame, frame_count, child_count, fps)
                out.write(frame)
                
                all_detections.append({
                    'frame_number': frame_count,
                    'timestamp_seconds': frame_count / fps,
                    'child_count': child_count,
                    'video_path': video_path
                })
                
                frame_count += 1
                
                if max_frames and frame_count >= max_frames:
                    logger.info(f"Reached max frame limit: {max_frames}")
                    break
            
            cap.release()
            out.release()
            
            detection_df = pd.DataFrame(all_detections)
            logger.info(f"Processed {frame_count} frames, saved to {output_path}")
            
            return output_path, detection_df
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
    
    @staticmethod
    def _add_frame_info(frame: np.ndarray, frame_num: int, 
                        child_count: int, fps: int) -> None:
        """Add informational overlays to frame."""
        cv2.putText(frame, f"Children Detected: {child_count}", 
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_num}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def create_training_annotations(image_dir: str = "data/raw", 
                               output_path: str = "data/processed/annotations.json") -> None:
    """
    Create COCO format annotations for model fine-tuning.
    
    Args:
        image_dir (str): Directory containing training images.
        output_path (str): Path to save annotations file.
    """
    import json
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    coco_format = {
        "info": {
            "description": "Child in car seat detection dataset",
            "version": "1.0",
            "year": datetime.now().year
        },
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "child_in_carseat"}]
    }
    
    if not os.path.exists(image_dir):
        logger.warning(f"Image directory not found: {image_dir}")
        return
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])
    metadata_file = os.path.join(image_dir, "image_metadata.csv")
    
    metadata = pd.read_csv(metadata_file) if os.path.exists(metadata_file) else None
    
    annotation_id = 1
    for i, img_file in enumerate(image_files):
        coco_format["images"].append({
            "id": i,
            "file_name": img_file,
            "width": 640,
            "height": 480
        })
        
        if metadata is not None:
            img_id = img_file.replace('.jpg', '').replace('.png', '')
            img_meta = metadata[metadata['image_id'] == img_id]
            
            if not img_meta.empty and img_meta.iloc[0].get('has_child', False):
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [100, 200, 200, 200],
                    "area": 40000,
                    "iscrowd": 0
                })
                annotation_id += 1
    
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    logger.info(f"Created {annotation_id - 1} annotations saved to {output_path}")


if __name__ == "__main__":
    logger.info("=== Child Detection System Initialized ===")
    
    try:
        create_training_annotations()
        detector = ChildDetector()
        logger.info("System ready for detection")
    except Exception as e:
        logger.error(f"Initialization failed: {e}")