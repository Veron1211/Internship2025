from argparse import ArgumentParser
import cv2
import requests
import numpy as np
import base64
import json
import time
import threading
from queue import Queue, Empty

# Parse arguments from user input
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=0,
                    help="The webcam index.")
parser.add_argument("--api_url", type=str, default="http://localhost:8000/detect-pose",
                    help="URL of the head pose detection API.")
parser.add_argument("--downsample", type=float, default=0.5,
                    help="Scale factor to reduce image size for API (0.5 = half size)")
parser.add_argument("--quality", type=int, default=70,
                    help="JPEG quality for image transmission (0-100)")
args = parser.parse_args()

print("Head Pose Estimation Client")
print(f"OpenCV version: {cv2.__version__}")
print(f"API URL: {args.api_url}")
print(f"Downsample factor: {args.downsample}")
print(f"JPEG quality: {args.quality}")

# Global variables
latest_result = None
processing = False
result_ready = threading.Event()

def api_worker(frame_queue, result_queue):
    """Worker thread to send frames to API"""
    global processing
    
    while True:
        if frame_queue.empty():
            time.sleep(0.01)
            continue
            
        # Get the frame from queue
        frame, frame_number = frame_queue.get()
        if frame is None:  # Exit signal
            break
            
        processing = True
        
        try:
            # Compress frame to JPEG for API transfer
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]
            _, img_encoded = cv2.imencode('.jpg', frame, encode_params)
            
            # Send frame to API
            response = requests.post(
                args.api_url,
                files={"file": ("image.jpg", img_encoded.tobytes(), "image/jpeg")},
                data={
                    "confidence_threshold": 0.5, 
                    "return_annotated": True, 
                    "face_expansion": 0.15,
                    "image_quality": args.quality,
                    "downsample": args.downsample
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                result_queue.put((result, frame_number, frame))
            else:
                print(f"API Error: Status code {response.status_code}")
                result_queue.put(({"error": f"API Error: {response.status_code}"}, frame_number, frame))
                
        except requests.exceptions.RequestException as e:
            print(f"API Connection Error: {str(e)}")
            result_queue.put(({"error": f"API Connection Error: {str(e)}"}, frame_number, frame))
        
        frame_queue.task_done()
        processing = False

def result_handler(result_queue):
    """Worker thread to handle results"""
    global latest_result
    
    while True:
        try:
            result, frame_number, original_frame = result_queue.get(timeout=1)
            latest_result = (result, frame_number, original_frame)
            result_ready.set()
            result_queue.task_done()
        except Empty:
            continue
        except Exception as e:
            print(f"Result handler error: {e}")

def run():
    global latest_result
    
    # Initialize the video source from webcam or video file
    video_src = args.cam if args.video is None else args.video
    cap = cv2.VideoCapture(video_src)
    print(f"Video source: {video_src}")
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_src}")
        return
    
    # Create queues for thread communication
    frame_queue = Queue(maxsize=2)  # Limit queue size to prevent memory issues
    result_queue = Queue()
    
    # Start worker threads
    api_thread = threading.Thread(target=api_worker, args=(frame_queue, result_queue))
    api_thread.daemon = True
    api_thread.start()
    
    result_thread = threading.Thread(target=result_handler, args=(result_queue,))
    result_thread.daemon = True
    result_thread.start()
    
    # FPS calculation variables
    start_time = time.time()
    frame_count = 0
    total_fps = 0
    
    # Set lower resolution for webcam to improve performance
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_number = 0
    last_frame = None
    
    while True:
        # Read frame
        frame_got, frame = cap.read()
        if not frame_got:
            break
            
        # Mirror webcam feed
        if video_src == 0:
            frame = cv2.flip(frame, 2)
            
        # Store the frame
        last_frame = frame.copy()
        frame_number += 1
        
        # Only send frame to API if not currently processing
        if not processing and frame_queue.qsize() < frame_queue.maxsize:
            try:
                frame_queue.put_nowait((frame.copy(), frame_number))
            except:
                pass  # Queue is full
        
        # Display frame with latest results if available
        display_frame = last_frame.copy()
        
        if result_ready.is_set():
            result, res_frame_number, original_frame = latest_result
            result_ready.clear()
            
            if "error" in result:
                # Show error message
                cv2.putText(display_frame, result["error"][:50], 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                if "annotated_image" in result and result["annotated_image"]:
                    try:
                        # Decode annotated image
                        img_bytes = base64.b64decode(result["annotated_image"])
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        annotated_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if annotated_frame is not None:
                            display_frame = annotated_frame
                            
                            # Display API processing time
                            processing_time = result["processing_time_ms"]
                            cv2.putText(display_frame, f"API: {processing_time:.1f}ms", 
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    except Exception as e:
                        print(f"Error decoding annotated image: {e}")
                        cv2.putText(display_frame, "Error decoding annotated image", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "No annotated image returned", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Draw face count manually
                    face_count = result.get("face_count", 0)
                    cv2.putText(display_frame, f"Faces: {face_count}", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Show message that we're waiting for API response
            cv2.putText(display_frame, "Waiting for API response...", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        # Calculate and display client-side FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            total_fps = fps
            frame_count = 0
            start_time = current_time
        
        cv2.putText(display_frame, f"FPS: {total_fps:.1f}", 
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                  
        cv2.putText(display_frame, f"Queue: {frame_queue.qsize()}/{frame_queue.maxsize}", 
                  (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                  
        # Show output
        cv2.imshow("Preview", display_frame)
            
        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break
            
    # Clean up
    print("Shutting down...")
    frame_queue.put((None, -1))  # Signal to stop threads
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()