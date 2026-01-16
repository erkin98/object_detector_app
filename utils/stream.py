import cv2
import threading
import subprocess as sp
import json
import time
import numpy as np
from typing import Optional, Tuple

class HLSVideoStream:
    def __init__(self, src: str):
        # initialize the video camera stream and read the first frame
        # from the stream

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        self.src = src

        self.FFMPEG_BIN = "ffmpeg"
        self.metadata = {}
        self.WIDTH = 0
        self.HEIGHT = 0
        self.frame = None
        self.grabbed = False
        self.pipe = None

        self._initialize_stream()

    def _initialize_stream(self):
        while "streams" not in self.metadata.keys():
            print('ERROR: Could not access stream. Trying again.')

            # Use ffprobe to get metadata
            try:
                # Construct command safely
                cmd = [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    self.src
                ]
                
                info = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
                out, err = info.communicate()
                
                if info.returncode != 0:
                     print(f"ffprobe failed: {err}")
                     time.sleep(5)
                     continue

                self.metadata = json.loads(out.decode('utf-8'))
            except Exception as e:
                print(f"Exception during ffprobe: {e}")
                time.sleep(5)

        print('SUCCESS: Retrieved stream metadata.')

        # Assuming the first stream is video. In a real world scenario, checking codec_type is better.
        # But keeping closer to original logic for now, just safer.
        stream = next((s for s in self.metadata.get("streams", []) if s.get("codec_type") == "video"), None)
        if not stream:
             stream = self.metadata["streams"][0] # Fallback
        
        self.WIDTH = stream["width"]
        self.HEIGHT = stream["height"]

        self.pipe = sp.Popen([
            self.FFMPEG_BIN, "-i", self.src,
            "-loglevel", "quiet", # no text output
            "-an",   # disable audio
            "-f", "image2pipe",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo", "-"
        ], stdin = sp.PIPE, stdout = sp.PIPE)
        
        print('WIDTH: ', self.WIDTH)
        
        # Read first frame
        self._read_frame_from_pipe()

    def _read_frame_from_pipe(self):
        if self.pipe:
            raw_image = self.pipe.stdout.read(self.WIDTH * self.HEIGHT * 3)
            if len(raw_image) != self.WIDTH * self.HEIGHT * 3:
                self.grabbed = False
                return
            
            self.frame = np.frombuffer(raw_image, dtype='uint8').reshape((self.HEIGHT, self.WIDTH, 3))
            self.grabbed = True

    def start(self):
        # start the thread to read frames from the video stream
        t = threading.Thread(target=self.update, args=())
        t.daemon = True # Set as daemon thread so it dies when main program exits
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        # if the thread indicator variable is set, stop the thread
        while True:
            if self.stopped:
                return

            self._read_frame_from_pipe()

    def read(self) -> Optional[np.ndarray]:
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        if self.pipe:
             self.pipe.terminate()


class WebcamVideoStream:
    def __init__(self, src: int = 0, width: int = 480, height: int = 360):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.stream.release()
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self) -> Optional[np.ndarray]:
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
