## Installation

### 

### Requirements

- Python 3.3+ or Python 2.7
- macOS or Linux 

### 

### Installation Options:

#### Installing on Mac or Linux

First, make sure you have dlib already installed with Python bindings:

- [How to install dlib from source on macOS or Ubuntu](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)

Then, install this module from pypi using `pip3` (or `pip2` for Python 2):

```
pip3 install face_recognition
```

Alternatively, you can try this library with [Docker](https://www.docker.com/), see [this section](https://github.com/ageitgey/face_recognition/blob/master/README.md#deployment).

If you are having trouble with installation, you can also try out a [pre-configured VM](https://medium.com/@ageitgey/try-deep-learning-in-python-now-with-a-fully-pre-configured-vm-1d97d4c3e9b).



## Usage:

### Complete pipeline for Face Detection, Face Recognition and Emotion Detection
Refer to the notebook /src/facial_detection_recog_emotion.ipynb

We have trained an emotion detection model and put its trained weights at /emotion_detector_models

### Train your Emotion Detection Model
To train your own emotion detection model, Refer to the notebook /src/EmotionDetector_v2.ipynb


