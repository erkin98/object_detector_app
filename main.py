import argparse
import time
import logging
import cv2
import numpy as np
from queue import Queue
from threading import Thread
import multiprocessing

from config import (
    MODEL_NAME, PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES,
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_NUM_WORKERS, DEFAULT_QUEUE_SIZE
)
from utils.fps import FPS
from utils.stream import HLSVideoStream, WebcamVideoStream
from detector import ObjectDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def worker_func(input_q: multiprocessing.Queue, output_q: multiprocessing.Queue, 
                model_path: str, label_path: str, num_classes: int):
    """
    Worker function for multiprocessing.
    """
    try:
        detector = ObjectDetector(model_path, label_path, num_classes)
        fps = FPS().start()
        while True:
            frame = input_q.get()
            if frame is None: # Sentinel
                break
            
            # frame is BGR from cv2, convert to RGB for model if needed
            # The model usually expects RGB. The original code did cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_and_visualize(frame_rgb)
            output_q.put(result)
            fps.update()
        
        fps.stop()
        detector.close()
    except Exception as e:
        logger.error(f"Worker exception: {e}")

def main():
    parser = argparse.ArgumentParser(description="Object Detection App")
    parser.add_argument('-str', '--stream-input', dest="stream_in", action='store', type=str, default=None,
                        help='HLS Stream URL (optional)')
    parser.add_argument('-src', '--source', dest='video_source', type=int, default=0,
                        help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int, default=DEFAULT_WIDTH,
                        help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int, default=DEFAULT_HEIGHT,
                        help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int, default=DEFAULT_NUM_WORKERS,
                        help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int, default=DEFAULT_QUEUE_SIZE,
                        help='Size of the queue.')
    parser.add_argument('-strout','--stream-output', dest="stream_out", help='The URL to send the livestreamed object detection to.')
    
    args = parser.parse_args()

    # Input Queue
    input_q = multiprocessing.Queue(maxsize=args.queue_size)
    output_q = multiprocessing.Queue(maxsize=args.queue_size)

    # Start Workers
    workers = []
    for _ in range(args.num_workers):
        p = multiprocessing.Process(
            target=worker_func, 
            args=(input_q, output_q, PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES)
        )
        p.daemon = True
        p.start()
        workers.append(p)

    # Start Video Stream
    if args.stream_in:
        logger.info('Reading from HLS stream.')
        video_capture = HLSVideoStream(src=args.stream_in).start()
    else:
        logger.info(f'Reading from webcam {args.video_source}.')
        video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()

    fps = FPS().start()
    
    logger.info("Starting detection loop. Press 'q' to quit.")

    try:
        while True:
            frame = video_capture.read()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Non-blocking put? Or blocking? 
            # If queue is full, we might want to skip frames to keep up with real-time
            if not input_q.full():
                input_q.put(frame)
            
            t = time.time()
            
            # Check for results
            if not output_q.empty():
                data = output_q.get()
                rec_points = data['rect_points']
                class_names = data['class_names']
                class_colors = data['class_colors']
                
                # Draw on the CURRENT frame (might be slightly out of sync with detection, but okay for real-time visualization)
                # Or we can pass the frame through the queue. Passing frame through queue adds latency.
                # The original code drew on 'frame'. But wait, existing code:
                # frame = input_q.get() -> detect -> output_q.put(result)
                # output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR) -> imshow
                # The original multiprocessing code passed the DETECTED IMAGE back. 
                # "output_q.put(detect_objects(frame_rgb, sess, detection_graph))"
                # detect_objects returns image_np.
                
                # My detector.detect_and_visualize returns dict of points/labels, NOT the image.
                # So I should draw on the current frame.
                # But if I draw on current frame, the boxes might lag behind the object if the camera moved.
                # Ideally we pass the frame through the pipeline.
                
                # Let's check my detector.detect_and_visualize
                # It calls draw_boxes_and_labels which returns coords, not image.
                # So I need to draw them here.
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                for point, name, color in zip(rec_points, class_names, class_colors):
                     # Coords are normalized? 
                     # Wait, draw_boxes_and_labels in original code:
                     # "rect_points.append(dict(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax))"
                     # Boxes are normalized [0,1].
                     
                    ymin, xmin, ymax, xmax = point['ymin'], point['xmin'], point['ymax'], point['xmax']
                    left = int(xmin * args.width)
                    top = int(ymin * args.height)
                    right = int(xmax * args.width)
                    bottom = int(ymax * args.height)

                    cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                    
                    # Draw label background
                    label_size = len(name[0]) * 10 # Approx
                    cv2.rectangle(frame, (left, top - 20), (left + label_size, top), color, -1)
                    
                    cv2.putText(frame, name[0], (left, top - 5), font, 0.5, (255, 255, 255), 1)

            if args.stream_out:
                print('Streaming elsewhere (not implemented)!')
            else:
                cv2.imshow('Video', frame)
            
            fps.update()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        fps.stop()
        logger.info(f'[INFO] elapsed time (total): {fps.elapsed():.2f}')
        logger.info(f'[INFO] approx. FPS: {fps.fps():.2f}')

        # Cleanup
        video_capture.stop()
        cv2.destroyAllWindows()
        
        # Stop workers
        for _ in workers:
            input_q.put(None)
        
        for p in workers:
            p.terminate()

if __name__ == '__main__':
    main()
