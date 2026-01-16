# Object-Detector-App

A real-time object recognition application using [Google's TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and [OpenCV](http://opencv.org/).

## Getting Started

### Installation

1. Install Python 3.8+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

Run the main application:
```bash
python main.py
```

Optional arguments:
* `--source`: Device index of the camera (default: 0)
* `--width`: Width of the frames in the video stream (default: 480)
* `--height`: Height of the frames in the video stream (default: 360)
* `--num-workers`: Number of workers (default: 2)
* `--queue-size`: Size of the queue (default: 5)
* `--stream-input`: Get video from HLS stream rather than webcam (e.g., `http://somertmpserver.com/hls/live.m3u8`)

Example:
```bash
python main.py --source 0 --width 640 --height 480
```

## Structure

* `main.py`: Main entry point.
* `detector.py`: Object detection logic using TensorFlow.
* `utils/`: Utility modules for streaming, FPS tracking, and visualization.
* `object_detection/`: Legacy TensorFlow Object Detection API code.

## Tests

Run tests with pytest:
```bash
pytest
```

## Copyright

See [LICENSE](LICENSE) for details.
Copyright (c) 2017 [Dat Tran](http://www.dat-tran.com/).
