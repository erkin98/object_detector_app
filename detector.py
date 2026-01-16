import os
import numpy as np
import tensorflow.compat.v1 as tf
from typing import Dict, Any, Tuple, Optional, List

# Ensure TF2 runs in TF1 compatibility mode
tf.disable_v2_behavior()

from object_detection.utils import label_map_util
from utils.visualization import draw_boxes_and_labels

class ObjectDetector:
    def __init__(self, model_path: str, label_path: str, num_classes: int):
        self.model_path = model_path
        self.label_path = label_path
        self.num_classes = num_classes
        self.detection_graph = tf.Graph()
        self.sess = None
        self.category_index = self._load_labels()
        self._load_model()
        
    def _load_labels(self) -> Dict[int, Dict[str, Any]]:
        label_map = label_map_util.load_labelmap(self.label_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=self.num_classes, use_display_name=True)
        return label_map_util.create_category_index(categories)

    def _load_model(self):
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.io.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            
            self.sess = tf.Session(graph=self.detection_graph)

    def detect(self, image_np: np.ndarray) -> Dict[str, Any]:
        """
        Detects objects in the image.
        Returns a dictionary with detection results.
        """
        if self.sess is None:
            raise RuntimeError("Session not initialized")

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes_out, scores_out, classes_out, num_detections_out) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        return {
            'boxes': np.squeeze(boxes_out),
            'scores': np.squeeze(scores_out),
            'classes': np.squeeze(classes_out).astype(np.int32),
            'num_detections': num_detections_out
        }

    def detect_and_visualize(self, image_np: np.ndarray, min_score_thresh: float = 0.5) -> Dict[str, Any]:
        results = self.detect(image_np)
        
        rect_points, class_names, class_colors = draw_boxes_and_labels(
            boxes=results['boxes'],
            classes=results['classes'],
            scores=results['scores'],
            category_index=self.category_index,
            min_score_thresh=min_score_thresh
        )
        
        return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)

    def close(self):
        if self.sess:
            self.sess.close()
            self.sess = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
