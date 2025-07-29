import cv2
import requests
import numpy as np
from argparse import ArgumentParser
from pose_estimation import PoseEstimator

SERVER_URL = "http://127.0.0.1:8000/estimate-pose/"

# Argument parser for video source
parser = ArgumentParser()
parser.add_argument("--video", type=str, help="Path to video file.")
parser.add_argument("--cam", type=int, default=0, help="Webcam index.")
args = parser.parse_args()

def send_frame_to_server(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    if not _:
        print("Error: Failed to encode frame as JPEG")

    files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
    try:
        response = requests.post(SERVER_URL, files=files, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error sending frame to server: {e}")
        return {'pose': [], 'fps': 0.0}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {'pose': [], 'fps': 0.0}

def run():
    video_src = args.cam if args.video is None else args.video
    cap = cv2.VideoCapture(video_src)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_src}")
        return
        
    print(f"Using video source: {video_src}")
    
    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize pose estimator for visualization
    pose_estimator = PoseEstimator(frame_width, frame_height)
    
    # Error counter to avoid infinite error loops
    error_count = 0
    max_errors = 5
    
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break

            if video_src == 0:  # Flip only for webcam
                frame = cv2.flip(frame, 1)

            # Send frame to server
            response = send_frame_to_server(frame)
            
            # Reset error count on successful request
            if 'pose' in response and response['pose']:
                error_count = 0
                
                # Extract the pose data (rotation vector and translation vector)
                pose_data = response['pose']
                
                # Convert lists back to numpy arrays for visualization
                rotation_vector = np.array(pose_data[0])
                translation_vector = np.array(pose_data[1])
                
                # Reconstruct the pose tuple
                pose = (rotation_vector, translation_vector)
                
                # Use the same visualization as in the demo code
                pose_estimator.visualize(frame, pose, color=(0, 255, 0))
                pose_estimator.draw_axes(frame, pose)
                
                # Get head position from response
                head_position = response.get('head_position', 'unknown')
                
                # Display head position with custom color based on position
                if head_position == 'left':
                    position_color = (0, 0, 255)  # Red for left
                elif head_position == 'right':
                    position_color = (255, 0, 0)  # Blue for right
                else:  # center
                    position_color = (0, 255, 0)  # Green for center
                
                # Create a colored box for head position indicator
                position_box_y = 110
                cv2.rectangle(frame, (10, position_box_y-20), (200, position_box_y+10), (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, f"Head Position: {head_position.upper()}", 
                           (15, position_box_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, position_color, 2)
                
                # Display yaw, pitch, roll values
                yaw = response.get('yaw', 0)
                pitch = response.get('pitch', 0)
                roll = response.get('roll', 0)
                
                cv2.putText(frame, f"Yaw: {yaw:.1f}, Pitch: {pitch:.1f}, Roll: {roll:.1f}", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw FPS box
            cv2.rectangle(frame, (0, 0), (90, 30), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, f"FPS: {response.get('fps', 0):.0f}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            # Draw face detection status
            status = "Face Detected" if 'pose' in response and response['pose'] else "No Face Detected"
            cv2.putText(frame, status, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0) if 'pose' in response and response['pose'] else (0, 0, 255), 1)

            # Show frame
            cv2.imshow("Pose Estimation Client", frame)
            
        except KeyboardInterrupt:
            print("Keyboard interrupt received, exiting...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            error_count += 1
            
            # If we've had too many consecutive errors, break
            if error_count > max_errors:
                print(f"Too many consecutive errors ({max_errors}), exiting...")
                break
                
            # Try to show frame even if there's an error
            try:
                if 'frame' in locals() and frame is not None:
                    cv2.imshow("Pose Estimation Client", frame)
            except:
                pass
        
        # Check for ESC key
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break

    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Unhandled error: {e}")