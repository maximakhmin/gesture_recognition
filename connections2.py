import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark, NormalizedLandmarkList
from matplotlib import pyplot as plt


DATA_SHAPE = (55, 3)

def get_data(image, model, pose_min_visibility=0):
    detection_results = model.process(image)
    width = image.shape[1]
    height = image.shape[0]

    data = np.ndarray(shape=DATA_SHAPE)

    if detection_results.pose_landmarks:
        pose_landmarks_num = [0, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
        for i in range(len(pose_landmarks_num)):
            ind = pose_landmarks_num[i]
            pose_landmark = detection_results.pose_landmarks.landmark[ind]
            if pose_landmark.visibility > pose_min_visibility:
                data[i] = np.array([pose_landmark.x * width, 
                                    pose_landmark.y * height,
                                    pose_landmark.z * width])
            else:
                data[i] = np.array([None, None, None])  
    else:
        for i in range(13):
            data[i] = np.array([None, None, None]) 


    offset = 13
    if detection_results.left_hand_landmarks:
        for i in range(21):
            hand_landmark = detection_results.left_hand_landmarks.landmark[i]
            data[i+offset] = np.array([hand_landmark.x * width,
                                       hand_landmark.y * height,
                                       hand_landmark.z * width])
    else:
        for i in range(21):
            data[i+offset] = np.array([None, None, None]) 


    offset = 34
    if detection_results.right_hand_landmarks:
        for i in range(21):
            hand_landmark = detection_results.right_hand_landmarks.landmark[i]
            data[i+offset] = np.array([hand_landmark.x * width,
                                       hand_landmark.y * height,
                                       hand_landmark.z * width])
    else:
        for i in range(21):
            data[i+offset] = np.array([None, None, None]) 


    return data


def get_null_data():
    null_data = np.ndarray(shape=DATA_SHAPE)
    for i in range(len(null_data)):
        null_data[i] = np.array([None, None, None]) 
    return null_data


def normalize_data(data):
    if None in data[3] or None in data[4]:
        return get_null_data()
    
    dot0 = (data[3] + data[4]) / 2
    norm = np.linalg.norm(data[3] - data[4])

    normalized_data = data.copy()
    for i in range(len(data)):
        normalized_data[i] = (data[i] - dot0) / norm

    # if normalized_data[7][2] and normalized_data[13][2]:
    #     dist = normalized_data[7][2] - normalized_data[13][2]
    #     for i in range(21):
    #         normalized_data[i+13][2] += dist
    # if normalized_data[8][2] and normalized_data[34][2]:
    #     dist = normalized_data[8][2] - normalized_data[34][2]
    #     for i in range(21):
    #         normalized_data[i+34][2] += dist

    return normalized_data


def draw_data_on_image(bgr_image, data):
    width = bgr_image.shape[1]
    height = bgr_image.shape[0]
  
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    annotated_image = np.copy(bgr_image)


    mp_drawing.draw_landmarks(
        annotated_image, 
        NormalizedLandmarkList(landmark=[
            NormalizedLandmark(x=landmark_data[0]/width, 
                               y=landmark_data[1]/height, 
                               z=landmark_data[2]/width) 
            for landmark_data in data]), 
        [],
    )

    return annotated_image


def process_video(video_file_name, output_file_name, model, buffer=50):

    cap = cv2.VideoCapture(video_file_name)

    dataset = np.ndarray(shape=(0, buffer, DATA_SHAPE[0], DATA_SHAPE[1]))

    queue = np.ndarray(shape=(buffer, DATA_SHAPE[0], DATA_SHAPE[1]))
    ind = 0

    for i in range(len(queue)):
        queue[i] = get_null_data()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        d=get_data(image, model)
        norm_d = normalize_data(d)
        if ind<buffer:
            queue[ind] = norm_d
            ind += 1
        else:
            dataset = np.append(dataset, [fill_none(queue, buffer)], axis=0)

            for i in range(len(queue)-1):
                queue[i] = queue[i+1]
            queue[len(queue)-1] = norm_d

    dataset = np.append(dataset, [fill_none(queue, buffer)], axis=0)

    np.save(output_file_name, dataset)

    return queue

        
def fill_none(data, buffer):

    filled_data = data.copy()

    for landmark in range(DATA_SHAPE[0]):
        for coord in range(DATA_SHAPE[1]):
            row = data[:, landmark, coord]
            xp = []
            fp = []
            for i in range(len(row)):
                if not np.isnan(row[i]):
                    xp.append(i)
                    fp.append(row[i])
            if len(xp)==0:
                filled_data[:, landmark, coord] = np.zeros(shape=buffer)
            else:
                filled_data[:, landmark, coord] = np.interp(x=range(buffer), xp=xp, fp=fp)

    return filled_data   




