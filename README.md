# ObjectDetectionByVideo
A real-time object recognition application using [Google's TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and [OpenCV](http://opencv.org/).

![traffic](https://raw.githubusercontent.com/JounyWang/ObjectDetectionByVideo/master/output/traffic.png)

## Requirements
- [Anaconda / Python 3.5](https://www.continuum.io/downloads)
- [TensorFlow 1.3](https://www.tensorflow.org/)
- [OpenCV 3.2](http://opencv.org/)

![person_phone](https://raw.githubusercontent.com/JounyWang/ObjectDetectionByVideo/master/output/person_phone.jpg)

## Getting Started
    `python object_detection_app.py`
    Optional arguments (default value):
    * Device index of the camera `--source=0`
    * Width of the frames in the video stream `--width=480`
    * Height of the frames in the video stream `--height=480`
    * Number of workers `--num-workers=2`
    * Size of the queue `--queue-size=5`


