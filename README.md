# Head pose estimation & Gender-and-Age-Detection

Realtime human head pose estimation with ONNX Runtime and OpenCV. Gender and age detector that can approximately guess the gender and age of the person (face) in a picture or through webcam.

From Open Source:
- https://github.com/yinguobing/head-pose-estimation
- https://github.com/smahesh29/Gender-and-Age-Detection

## How it works

1. Face detection. A face detector is introduced to provide a face bounding box containing a human face. Then the face box is expanded and transformed to a square to suit the needs of later steps.
2. Show the bounding box around the largest face.
3. Age and gender classification.
2. Facial landmark detection. A pre-trained deep learning model take the face image as input and output 68 facial landmarks.
3. Pose estimation. After getting 68 facial landmarks, the pose could be calculated by a mutual PnP algorithm.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The code was tested on Windows 11 Pro with following frameworks:
- ONNX Runtime: 1.17.1
- OpenCV: 4.5.4

### Installing

Clone the repo:
```bash
git clone https://github.com/Veron1211/Internship2025.git
```

Create new Conda environment:
```bash
conda create -n your-env-name python=3.9
```

Activate the environment:
```bash
conda activate your-env-name
```

Install dependencies with pip:
```bash
pip install -r requirements.txt
```

Download the raw pre-trained models in the `assets` directory:
https://github.com/yinguobing/head-pose-estimation.git

## Running

### FastAPI
```bash
uvicorn combined_api:app --reload --port 8001
```

### Client
```bash
python combined_client.py
```

A video file or a webcam index should be assigned through arguments. If no source provided, the built in webcam will be used by default.

### Video file

For any video format that OpenCV supports (`mp4`, `avi` etc.):

```bash
python combined_client.py --video /path/to/video.mp4
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
