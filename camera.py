import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from connections import connect_results, CONNECTIONS


def draw_landmarks_on_image(rgb_image, hand_detection_result, pose_detection_result):
  
    hand_connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    pose_connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

    mp_drawing = mp.tasks.vision.drawing_utils
    mp_drawing_styles = mp.tasks.vision.drawing_styles

    landmarks = connect_results(hand_detection_result, pose_detection_result)

    annotated_image = np.copy(rgb_image)

    mp_drawing.draw_landmarks(
        annotated_image,
        landmarks,
        CONNECTIONS,
    )

    return annotated_image

hand_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
hand_options = vision.HandLandmarkerOptions(base_options=hand_base_options,
                                       num_hands=2)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

pose_base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
pose_options = vision.PoseLandmarkerOptions(base_options=pose_base_options)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

window_name = 'Tracking'
width_full = 1920
height_full = 1080

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_full)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_full)

cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN)

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
cv2.destroyAllWindows()