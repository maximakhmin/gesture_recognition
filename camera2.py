import cv2
import mediapipe as mp
import numpy as np
import time


mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False, 
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    refine_face_landmarks=False,
)


def draw_landmarks_on_image(bgr_image, detection_results):
  
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    annotated_image = np.copy(bgr_image)

    if detection_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, 
            detection_results.pose_landmarks, 
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    if detection_results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, 
            detection_results.left_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,
            # landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )

    if detection_results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, 
            detection_results.right_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,
            # landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )

    return annotated_image


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


        image_rgb_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = mp.Image(mp.ImageFormat.SRGB, image_rgb_data)
        result = holistic.process(image_rgb_data)
        annotated_image = draw_landmarks_on_image(image, result)

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
        result = holistic.process(image_rgb_data)
        annotated_image = draw_landmarks_on_image(image, result)

        cv2.imshow(window_name, annotated_image)

        if cv2.waitKey(5) & 0xFF == 27: # Выход на ESC
            break

        time.sleep(0.2)

    cap.release()


window_name = 'Tracking'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

video("train/0e265c95-92f0-44fd-8ace-63693e461d1d.mp4")    
# camera()

cv2.destroyAllWindows()