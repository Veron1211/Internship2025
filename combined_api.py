from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import time
import math
import logging
from concurrent.futures import ThreadPoolExecutor
from utils import refine
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
import asyncio
import os
from fastapi import FastAPI, Request

app = FastAPI()

# Middleware to allow all origins (for testing, adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track last modification time of config file
config_file_path = "head_pose_config.txt"
last_config_mod_time = 0
head_pose_config = {}

def load_head_pose_config(force_reload=False):
    """Load or reload the head pose configuration from file"""
    global head_pose_config, last_config_mod_time
    
    # Check if file has been modified
    try:
        current_mod_time = os.path.getmtime(config_file_path)
        if not force_reload and current_mod_time <= last_config_mod_time:
            return head_pose_config  # No changes, return existing config
        
        last_config_mod_time = current_mod_time
    except Exception as e:
        logger.warning(f"Error checking config file modification time: {e}")
        
    # Default configuration
    config = {
        "left_threshold": 15,
        "right_threshold": -15,
        "leftright_threshold_for_side": 5,
        "left_extreme": 40,
        "right_extreme": -40,
        "left_ratio": 0.33,
        "right_ratio": 0.33,
    }
    
    try:
        with open(config_file_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=")
                    if key in config:
                        config[key] = float(value)
        logger.info(f"Config loaded/reloaded: {config}")
    except Exception as e:
        logger.warning(f"Could not load head pose config: {e}. Using default values.")
    
    head_pose_config = config
    return config

# Initial load of configuration
head_pose_config = load_head_pose_config(force_reload=True)

# Load models for gender, age, and face detection
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
face_net = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
GENDER_LIST = ['Male', 'Female']
#AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
AGE_LIST = ['(Toddler)', '(Toddler)', '(Child)', '(Adolescent)', '(Young Adult)', '(Middle-age Adult)', '(Middle-age Adult)', '(Senior Adult)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
PADDING = 20
executor = ThreadPoolExecutor(max_workers=4)

# Initialize the pose estimation setup
face_detector = FaceDetector("assets/face_detector.onnx")
mark_detector = MarkDetector("assets/face_landmarks.onnx")

# Function to determine head position based on yaw
def get_head_position(yaw, face_center_x, frame_width):
    # Check if config needs reloading
    current_config = load_head_pose_config()
    
    yaw_degrees = math.degrees(yaw) if abs(yaw) < math.pi else yaw

    left_boundary = int(frame_width * current_config["left_ratio"])
    right_boundary = int(frame_width * (1 - current_config["right_ratio"]))

    if face_center_x < left_boundary:
        if yaw_degrees > current_config["left_extreme"]:
            return "left"
        elif yaw_degrees < current_config["leftright_threshold_for_side"]:
            return "right"
        else:
            return "center"
    elif face_center_x > right_boundary:
        if yaw_degrees < current_config["right_extreme"]:
            return "right"
        elif yaw_degrees > current_config["leftright_threshold_for_side"]:
            return "left"
        else:
            return "center"
    else:
        if yaw_degrees < current_config["right_threshold"]:
            return "right"
        elif yaw_degrees > current_config["left_threshold"]:
            return "left"
        else:
            return "center"


# Detect faces in an image
def detect_faces(frame: np.ndarray):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=True, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, x1 - PADDING)
            y1 = max(0, y1 - PADDING)
            x2 = min(w - 1, x2 + PADDING)
            y2 = min(h - 1, y2 + PADDING)
            faces.append([x1, y1, x2, y2])
    if len(faces) > 1:
        largest_face = max(faces, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        faces = [largest_face]
    return faces

# Process the face to determine gender and age
def process_face(face: np.ndarray) -> dict:
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    gender = GENDER_LIST[gender_net.forward().argmax()]
    age_net.setInput(blob)
    age = AGE_LIST[age_net.forward().argmax()]
    return {"gender": gender, "age": age}

@app.post("/detect/")
async def detect(image: UploadFile = File(...)):
    try:
        start_time = time.time()

        # Always check for config updates on each request
        current_config = load_head_pose_config()

        # Read and decode image
        data = await image.read()
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # Detect faces
        faces = await asyncio.get_event_loop().run_in_executor(executor, detect_faces, frame)
        if not faces:
            raise HTTPException(status_code=400, detail="No faces detected")

        # Process each detected face
        processed_results = []
        for x1, y1, x2, y2 in faces:
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            gender_age = await asyncio.get_event_loop().run_in_executor(executor, process_face, face_img)
            gender_age.update({"box": [int(x1), int(y1), int(x2), int(y2)]})
            processed_results.append(gender_age)

        # Pose estimation
        faces_for_pose, _ = face_detector.detect(frame, 0.7)
        logger.info(f"Detected faces for pose estimation: {len(faces_for_pose)}")
        if len(faces_for_pose) > 0:
            refined_faces = refine(faces_for_pose, frame.shape[1], frame.shape[0], 0.15)
            if len(refined_faces) == 0:
                raise HTTPException(status_code=400, detail="No face detection after refinement")

            face = refined_faces[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                raise HTTPException(status_code=400, detail="Empty face patch")

            marks = mark_detector.detect([patch])[0].reshape([68, 2])
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            pose_estimator = PoseEstimator(frame.shape[1], frame.shape[0])
            pose = pose_estimator.solve(marks)
            rotation_vector = pose[0]
            yaw, pitch, roll = rotation_vector.flatten()
            
            face_center_x = int((x1 + x2) / 2)
            head_position = get_head_position(yaw, face_center_x, frame.shape[1])

            fps = 1.0 / (time.time() - start_time)

            return JSONResponse(content={
                "results": processed_results,
                "pose": [
                    pose[0].tolist() if isinstance(pose[0], np.ndarray) else pose[0],
                    pose[1].tolist() if isinstance(pose[1], np.ndarray) else pose[1]
                ],
                "head_position": head_position,
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll),
                "fps": fps,
                "config": {
                    "left_ratio": current_config["left_ratio"],
                    "right_ratio": current_config["right_ratio"]
                }
            })

        else:
            raise HTTPException(status_code=400, detail="No faces detected for pose estimation")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/button")
async def button_event(req: Request):
    data = await req.json()
    print("Button event:", data)
    return {"status": "ok"}

@app.post("/rfid")
async def rfid_event(req: Request):
    data = await req.json()
    print("RFID scanned:", data)
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)