import cv2
import requests
import numpy as np

# API endpoint
api_url = 'http://127.0.0.1:8000/detect/'

# Open webcam
cap = cv2.VideoCapture(0)

def send_to_api(frame):
    """Send the entire frame to the API and return the processed results."""
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()
    files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
    try:
        response = requests.post(api_url, files=files)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]

    # Send the frame to the API
    result = send_to_api(frame)

    if result:
        # Get config ratios from API
        config = result.get("config", {})
        left_ratio = config.get("left_ratio", 0.3)
        right_ratio = config.get("right_ratio", 0.3)

        # Compute boundaries based on ratios
        left_boundary = int(frame_width * left_ratio)
        right_boundary = int(frame_width * (1 - right_ratio))

        # Draw LEFT zone
        cv2.rectangle(frame, (0, 0), (left_boundary, frame_height), (255, 0, 0), 2)
        cv2.putText(frame, 'LEFT', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Draw CENTER zone
        cv2.rectangle(frame, (left_boundary, 0), (right_boundary, frame_height), (0, 255, 0), 2)
        cv2.putText(frame, 'CENTER', (left_boundary + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw RIGHT zone
        cv2.rectangle(frame, (right_boundary, 0), (frame_width, frame_height), (0, 0, 255), 2)
        cv2.putText(frame, 'RIGHT', (right_boundary + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw face boxes and info
        if 'results' in result and result['results']:
            for face_data in result['results']:
                x1, y1, x2, y2 = face_data['box']
                gender = face_data.get('gender', 'Unknown')
                age = face_data.get('age', 'Unknown')
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{gender}, {age}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display head pose
        head_position = result.get('head_position', 'Unknown')
        cv2.putText(frame, f"Head Position: {head_position}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    else:
        cv2.putText(frame, "API Error or No Response", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame
    cv2.imshow('Age, Gender, and Head Pose Detection with Zones', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
