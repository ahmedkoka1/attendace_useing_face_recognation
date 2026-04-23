# Attendance Using Face Recognition

## Project Overview
This repository contains a face recognition based attendance system built with Python and computer vision. It includes:

- `face_recognation.ipynb`: main notebook for face-based attendance tracking
- `spoofing.ipynb`: notebook for anti-spoofing checks and live fake-face detection
- `utils.py`: utility code for face detection, preprocessing, distance estimation, event logging and model prediction
- `attendance.json`: saved attendance or event data
- `dlib_env/`: preconfigured Python virtual environment containing required packages

## Key Features
- Real-time face detection and recognition
- Spoofing detection using ResNet50-based anti-spoofing model
- Attendance logging for identified users
- Distance estimation from face bounding boxes
- Optional event reporting via a Cloudflare Worker backend

## Requirements
Recommended setup:

- Python 3.6
- OpenCV
- PyTorch
- TorchVision
- dlib
- face_recognition
- pygame
- requests

> Note: This project currently uses absolute paths in `utils.py` for the model file and resources. Update these paths before running the notebooks.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ahmedkoka1/attendace_useing_face_recognation.git
cd attendace_useing_face_recognation
```

2. Activate the provided virtual environment (Windows PowerShell):

```powershell
.\dlib_env\Scripts\Activate.ps1
```

Or, create a new virtual environment and install packages if you prefer:

```bash
python -m venv venv
source venv/Scripts/activate
pip install opencv-python torch torchvision dlib face_recognition pygame requests
```

3. If you have a `requirements.txt`, install it with:

```bash
pip install -r requirements.txt
```

## Configuration

Open `utils.py` and verify these paths:

- `resnet50.pth` model path
- `assets_alarm.mp3` alarm sound path
- `attendace.json` attendance data path
- `BACKEND_URL` for event reporting

If these files are not present, add them to the project directory or change the paths to point to existing assets.

## Usage

1. Launch Jupyter Notebook:

```bash
jupyter notebook
```

2. Open `face_recognation.ipynb` to run the attendance workflow.
3. Open `spoofing.ipynb` to test anti-spoofing and liveness detection.

## How it works

- `utils.py` loads a ResNet50-based model for spoof detection.
- `detect_face()` finds faces in each video frame.
- `preprocess_image()` prepares face crops for the model.
- `generate_event()` writes event data to `attendance.json` and can send it to a backend.

## Notes

- The project is designed to run locally with a webcam.
- Make sure the model weights and audio file exist before running the notebooks.
- Update any hard-coded absolute paths to work on your machine.

## License

This repository is provided as-is for learning and experimentation.

