# Facial Expression Recognition on Video

## Overview

This repository demonstrates **facial expression recognition (FER)** on video using two different approaches in Python, and a third implementation in C++:

1. **[SeanHowFun/face-emotion-detection](ca://s?q=SeanHowFun_face_emotion_detection_model)** - a Hugging Face Vision Transformer model fine-tuned on FER2013.
2. **[Python FER package](ca://s?q=Python_fer_library)** (`from fer import FER`) - a CNN-based library for quick emotion detection.
3. **C++ with ONNX Runtime** - the SeanHowFun model exported to ONNX and integrated with OpenCV for inference.


## Features

- **[Face detection](ca://s?q=Explain_face_detection_in_OpenCV)** using Haar cascades (`haarcascade_frontalface_default.xml`).
- **[Expression classification](ca://s?q=Explain_facial_expression_classification)** into 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
- **[Python demos](ca://s?q=Python_facial_expression_detection_demo)**: two separate `main.py` files (one using Hugging Face, one using `fer`).
- **[ONNX export](ca://s?q=Export_PyTorch_model_to_ONNX)**: convert Hugging Face model to ONNX format.
- **[C++ implementation](ca://s?q=C++_ONNX_Runtime_example)**: run inference with ONNX Runtime and OpenCV, overlay predictions on video.


## Project Structure
```
├── python_seanhowfun_main.py        # Python demo using Hugging Face model
├── python_fer_main.py               # Python demo using FER package
├── export_to_onnx.py                # Script to export SeanHowFun model to ONNX
├── video_samples/                   # Input videos (shared by Python & C++)
├── output/                          # Annotated output videos (shared by Python & C++)
├── cpp/
│   ├── main.cpp                     # C++ source code using ONNX Runtime
│   ├── fer2013_vit.onnx             # Exported ONNX model (graph + metadata)
│   ├── fer2013_vit.onnx.data        # Model weights
│   ├── haarcascade_frontalface_default.xml

```

## Requirements
- Python 3.9+
  - `transformers`
  - `torch`
  - `fer`
  - `opencv-python`

- C++17 or newer
  - OpenCV 4.x
  - ONNX Runtime v1.18+


## Usage

### Python (SeanHowFun model)
```bash

python python_seanhowfun_main.py
```

### Python (FER package)
```
bash

python python_fer_main.py
```

### Export Hugging Face model to ONNX
```
bash

python export_to_onnx.py
```

### C++ (ONNX Runtime)
Compile:
```
bash

g++ main.cpp -o main \
    -I ../onnxruntime-linux-x64-1.18.0/include \
    -L ../onnxruntime-linux-x64-1.18.0/lib \
    -lonnxruntime \
    $(pkg-config --cflags --libs opencv4)

```

Run:
```
bash

export LD_LIBRARY_PATH=../onnxruntime-linux-x64-1.18.0/lib:$LD_LIBRARY_PATH
./main
```

## Example Output

At the end of processing, the program prints overall emotion percentages across the video:
```
Output:
0   Angry        9.22196%
1   Disgust      2.15318%
2   Fear         17.9929%
3   Happy        20.3572%
4   Neutral      9.56951%
5   Sad          9.99935%
6   Surprise     30.7059%
```
