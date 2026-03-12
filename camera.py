import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from connections import connect_results, CONNECTIONS
from mss import mss
from datetime import datetime as dt
import time


# -----------------
# mediapipe 0.10.32
# -----------------


def draw_landmarks_on_image(bgr_image, hand_detection_result, pose_detection_result):
  
    hand_connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    pose_connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

    mp_drawing = mp.tasks.vision.drawing_utils
    mp_drawing_styles = mp.tasks.vision.drawing_styles

    landmarks = connect_results(hand_detection_result, pose_detection_result)

    annotated_image = np.copy(bgr_image)

    mp_drawing.draw_landmarks(
        annotated_image,
        landmarks,
        CONNECTIONS,
    )

    return annotated_image



hand_base_options = python.BaseOptions(model_asset_path='models/tracking/hand_landmarker.task')
hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.7,
    num_hands=2
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

pose_base_options = python.BaseOptions(model_asset_path='models/tracking/pose_landmarker_full.task')
pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)





def camera():
    width_full = 1920
    height_full = 1080

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_full)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_full)

    while cap.isOpened():
        
        _x, _y, width, height = cv2.getWindowImageRect(window_name)

        success, image = cap.read()
        if not success:
            break

        # cropped_image = image[(height_full-height) // 2 : (height_full-height) // 2 + height,
        #                       (width_full - width) // 2 : (width_full - width) // 2 + width]
        cropped_image = image

        image_rgb_data = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        image_rgb = mp.Image(mp.ImageFormat.SRGB, image_rgb_data)
        hand_result = hand_detector.detect(image_rgb)
        pose_result = pose_detector.detect(image_rgb)
        annotated_image = draw_landmarks_on_image(cropped_image, hand_result, pose_result)

        cv2.imshow(window_name, cv2.flip(annotated_image, 1))

        if cv2.waitKey(5) & 0xFF == 27: # Выход на ESC
            break

    cap.release()


def video(file_name):
    cap = cv2.VideoCapture("E:/slovo/"+file_name)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break


        image_rgb_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = mp.Image(mp.ImageFormat.SRGB, image_rgb_data)
        hand_result = hand_detector.detect_for_video(image_rgb, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        pose_result = pose_detector.detect_for_video(image_rgb, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        annotated_image = draw_landmarks_on_image(image, hand_result, pose_result)

        cv2.imshow(window_name, annotated_image)

        if cv2.waitKey(5) & 0xFF == 27: # Выход на ESC
            break

        time.sleep(0.1)

    cap.release()

def monitor():

    monitor_params = {"top": 0, "left": 0, "width": 960, "height": 1080}

    ts = dt.now()
    frame_time = 0
    with mss() as sct:
        while True:
            delta = (dt.now() - ts)
            frame_time += delta.seconds*1000 + delta.microseconds
            ts = dt.now()

            screenshot = sct.grab(monitor_params)
            frame = np.array(screenshot)
            image_rgb_data = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            image_rgb = mp.Image(mp.ImageFormat.SRGB, image_rgb_data)
            hand_result = hand_detector.detect_for_video(image_rgb, frame_time)
            pose_result = pose_detector.detect(image_rgb)
            annotated_image = draw_landmarks_on_image(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR), hand_result, pose_result)

            cv2.imshow(window_name, annotated_image)

            if cv2.waitKey(5) & 0xFF == 27: # Выход на ESC
                break
            
window_name = 'Tracking'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)           
            
video("train/023f5f2b-a75c-4395-8070-2404c7a835c2.mp4")        
            
cv2.destroyAllWindows()
