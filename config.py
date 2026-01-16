import os

# Paths
CWD_PATH = os.getcwd()
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

# Model Parameters
NUM_CLASSES = 90
MIN_SCORE_THRESH = 0.5

# Video Stream Defaults
DEFAULT_WIDTH = 480
DEFAULT_HEIGHT = 360
DEFAULT_NUM_WORKERS = 2
DEFAULT_QUEUE_SIZE = 5
