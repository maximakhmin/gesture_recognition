import mediapipe as mp
from mediapipe.tasks.python.components.containers import landmark

hand_landmark_default = [landmark.NormalizedLandmark(
    x=0, y=0, z=0, visibility=0, presence=None
) for i in range(21)]

def connect_results(hand_detection_result, pose_detection_result):
    landmarks = []
    if len(pose_detection_result.pose_landmarks) != 1:
        return landmarks
    pose_landmarks = pose_detection_result.pose_landmarks[0]
    right_hand_landmarks = left_hand_landmarks = hand_landmark_default
    landmarks += [pose_landmarks[0], 
                  pose_landmarks[9],
                  pose_landmarks[10],
                  pose_landmarks[11],
                  pose_landmarks[12],
                  pose_landmarks[13],
                  pose_landmarks[14],
                  pose_landmarks[23],
                  pose_landmarks[24],
                  pose_landmarks[25],
                  pose_landmarks[26],]
    
    for ind, handedness in enumerate(hand_detection_result.handedness):
        if handedness[0].category_name == 'Left':
            left_hand_landmarks = hand_detection_result.hand_landmarks[ind]
        if handedness[0].category_name == 'Right':
            right_hand_landmarks = hand_detection_result.hand_landmarks[ind]

    landmarks += left_hand_landmarks
    landmarks += right_hand_landmarks

    return landmarks


conn = ((0,1), (1,2), (0,2), (3,4), (5,3), (4,6), 
        (4,8), (7, 8), (3,7), (8,10), (7,9), 
        (6,32), (5,11))

CONNECTIONS = [mp.tasks.vision.PoseLandmarksConnections.Connection(start=c[0], end=c[1]) for c in conn]

for c in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
    CONNECTIONS += [mp.tasks.vision.PoseLandmarksConnections.Connection(start=c.start+11, end=c.end+11),
                    mp.tasks.vision.PoseLandmarksConnections.Connection(start=c.start+32, end=c.end+32)]
    
