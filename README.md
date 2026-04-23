# attendace_useing_face_recognation

## Overview
This project implements an attendance tracking system using face recognition technology. Key features include:
- Face recognition for attendance marking (face_recognation.ipynb)
- Spoofing detection to prevent fake attendance (spoofing.ipynb)
- Utility functions for processing (utils.py)
- Attendance data storage (attendance.json)
- Pre-trained models: resnet50.pth, resnet50.onnx, modelrgb.onnx

## Setup
`git clone https://github.com/ahmedkoka1/attendace_useing_face_recognation.git`
2. Install dependencies: `pip install -r requirements.txt` (create if needed)
3. Activate virtualenv or use dlib_env (pre-configured)
4. Run Jupyter notebooks.

## Usage
- Open `face_recognation.ipynb` for main attendance functionality.
- Use `spoofing.ipynb` for anti-spoofing checks.

## Models
- Dlib for face detection.
- ONNX/Torch models for recognition.

## Contributing
Use branches for features: `git checkout -b feature-name`, commit, push, create Pull Request.

