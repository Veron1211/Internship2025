from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from utils import refine
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
import logging
import time
import math

app = FastAPI()

# Allow all origins (for testing with Unity, adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the pose estimation setup
face_detector = FaceDetector("assets/face_detector.onnx")
mark_detector = MarkDetector("assets/face_landmarks.onnx")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to determine head position based on yaw
def get_head_position(yaw):
    # Convert yaw from radians to degrees if needed
    if abs(yaw) < math.pi:  # If yaw is in radians
        yaw_degrees = math.degrees(yaw)
    else:
        yaw_degrees = yaw  # Assume already in degrees
    
    # Define thresholds for left/right determination
    # These thresholds can be adjusted based on testing
    if yaw_degrees < -15:
        return "left"  # Negative yaw often means looking right (from camera perspective)
    elif yaw_degrees > 15:
        return "right"   # Positive yaw often means looking left (from camera perspective)
    else:
        return "center"

@app.post("/estimate-pose/")
async def estimate_pose(file: UploadFile = File(...)):
    try:
        start_time = time.time()
        
        # Read the uploaded file
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="No file uploaded")

        # Convert bytes to numpy array and decode image
        nparr = np.frombuffer(file_content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Get the frame dimensions for pose estimator
        frame_height, frame_width = frame.shape[:2]

        # Step 1: Detect faces
        faces, _ = face_detector.detect(frame, 0.7)  # Using 0.7 as in the demo

        logger.info(f"Detected faces: {len(faces)}")

        if len(faces) > 0:
            # Use the first face for demonstration
            refined_faces = refine(faces, frame_width, frame_height, 0.15)
            if len(refined_faces) == 0:
                raise HTTPException(status_code=400, detail="No face detection after refinement")
                
            face = refined_faces[0]
            x1, y1, x2, y2 = face[:4].astype(int)

            # Log the face coordinates
            logger.info(f"Face coordinates: {(x1, y1, x2, y2)}")

            # Make sure we have valid coordinates
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
                logger.warning("Invalid face coordinates")
                raise HTTPException(status_code=400, detail="Invalid face coordinates")
                
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                logger.warning("Empty face patch")
                raise HTTPException(status_code=400, detail="Empty face patch")

            # Step 2: Detect landmarks
            try:
                marks = mark_detector.detect([patch])[0].reshape([68, 2])
                logger.info(f"Landmarks detected: {marks.shape}")
            except Exception as e:
                logger.error(f"Error detecting landmarks: {e}")
                raise HTTPException(status_code=500, detail="Error detecting landmarks")

            # Convert the locations from local face area to the global image
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Step 3: Estimate pose
            pose_estimator = PoseEstimator(frame_width, frame_height)
            try:
                pose = pose_estimator.solve(marks)
                
                # Extract yaw, pitch, roll from rotation vector
                rotation_vector = pose[0]
                yaw, pitch, roll = rotation_vector.flatten()
                
                # Determine head position based on yaw
                head_position = get_head_position(yaw)
                
                logger.info(f"Head position: {head_position}, Yaw: {yaw}")
                
                # Calculate FPS
                fps = 1.0 / (time.time() - start_time)
                
                # Return the complete pose data and head position
                result = {
                    "pose": [
                        pose[0].tolist() if isinstance(pose[0], np.ndarray) else pose[0],
                        pose[1].tolist() if isinstance(pose[1], np.ndarray) else pose[1]
                    ],
                    "head_position": head_position,
                    "yaw": float(yaw),
                    "pitch": float(pitch),
                    "roll": float(roll),
                    "fps": fps
                }
                return JSONResponse(content=result)
                
            except Exception as e:
                logger.error(f"Error in pose estimation: {e}")
                raise HTTPException(status_code=500, detail=f"Pose estimation error: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="No faces detected")

    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status codes
        raise
    except Exception as e:
        logger.error(f"Error during pose estimation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)