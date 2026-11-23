# Final project

Final project - Emotion Detector application built with Watson NLP and Flask.

## Project name

Final project

## Overview

The EmotionDetection package analyses text and reports the score of five
emotions (anger, disgust, fear, joy and sadness) together with the dominant
emotion. A small Flask server exposes the analyser through a web page.

## Project structure

```
EmotionDetection/
  __init__.py            exposes emotion_detector
  emotion_detection.py   Watson NLP call and output formatting
server.py                Flask web deployment
test_emotion_detection.py unit tests
templates/index.html     web interface
static/mywebscript.js    client-side request handling
```

## Running the application

```bash
pip install flask requests
python3 server.py
```

Then open http://localhost:5000

## Running the unit tests

```bash
python3 -m unittest test_emotion_detection.py -v
```

## Author

Rahadyan Gusti
