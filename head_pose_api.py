from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
from pydantic import BaseModel
from typing import List, Optional
import io
import time

from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine

app = FastAPI(title="Head Pose Detection API")

# Initialize detectors and estimator at startup
print("Loading detection models...")
face_detector = FaceDetector("assets/face_detector.onnx")
mark_detector = MarkDetector("assets/face_landmarks.onnx")

# Add logging
@app.get("/")
async def root():
    return {"message": "Head Pose Detection API is running"}

class PoseResult(BaseModel):
    face_count: int
    faces: List[dict]
    processing_time_ms: float
    annotated_image: Optional[str] = None

@app.post("/detect-pose", response_model=PoseResult)
async def detect_pose(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    return_annotated: bool = True,
    face_expansion: float = 0.15,
    image_quality: int = 70,  # JPEG quality for return image (0-100)
    downsample: float = 1.0   # Scale factor to reduce size for processing
):
    """
    Detect faces and estimate head pose from an image.
    """
    start_time = time.time()
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Downsample image if requested (for performance)
        original_frame = frame.copy()
        if downsample < 1.0:
            frame = cv2.resize(frame, (0, 0), fx=downsample, fy=downsample)
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Initialize pose estimator with frame dimensions
        pose_estimator = PoseEstimator(frame_width, frame_height)
        
        # Detect faces
        faces, _ = face_detector.detect(frame, confidence_threshold)
        
        # Process results
        result = {
            "face_count": len(faces),
            "faces": [],
            "processing_time_ms": 0
        }
        
        # Create copy for annotation
        annotated_frame = frame.copy() if return_annotated else None
        
        # Process each face
        for face in refine(faces, frame_width, frame_height, face_expansion):
            x1, y1, x2, y2 = face[:4].astype(int)
            
            # Skip invalid face regions
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue
            
            # Extract face patch
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:  # Skip empty patches
                continue
                
            try:
                # Detect landmarks
                marks = mark_detector.detect([patch])[0].reshape(68, 2)
                
                # Convert local coordinates to global
                h, w = patch.shape[:2]
                marks_global = marks.copy()
                marks_global[:, 0] = marks[:, 0] * w + x1
                marks_global[:, 1] = marks[:, 1] * h + y1
                
                # Estimate pose
                pose = pose_estimator.solve(marks_global)
                
                # Extract rotation and translation vectors
                rotation_vector = pose[0].tolist()
                translation_vector = pose[1].tolist()
                
                # Add face data to results
                face_data = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(face[4]),
                    "rotation_vector": rotation_vector,
                    "translation_vector": translation_vector
                }
                result["faces"].append(face_data)
                
                # Visualize pose on annotated frame if requested
                if return_annotated:
                    pose_estimator.visualize(annotated_frame, pose, color=(0, 255, 0))
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    
                    # Draw some landmarks (reduced for performance)
                    for (x, y) in marks_global[::5].astype(int):  # Draw every 5th landmark
                        cv2.circle(annotated_frame, (x, y), 2, (0, 255, 255), -1)
            
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        # Add annotated image if requested
        if return_annotated and annotated_frame is not None:
            # Add text with face count and processing time
            cv2.putText(annotated_frame, f"Faces: {result['face_count']}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            # Use lower quality JPEG encoding for faster transfer
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, image_quality]
            _, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
            img_str = base64.b64encode(buffer).decode('utf-8')
            result["annotated_image"] = img_str
        
        # Calculate total processing time
        end_time = time.time()
        result["processing_time_ms"] = (end_time - start_time) * 1000
        
        return result
    
    except Exception as e:
        print(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Head Pose Detection API...")
    uvicorn.run("HeadApi:app", host="0.0.0.0", port=8000, reload=False)