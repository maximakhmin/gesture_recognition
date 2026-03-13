from connections2 import process_video
from multiprocessing import Pool
import mediapipe as mp
import time
import pandas as pd


def extract_data_fun(param):

    path = "E:/slovo/"
    df = pd.read_csv(path + "annotations.csv", sep="\t")

    start = 0
    end = len(df)

    offset = param[0]
    num_processors = param[1]

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False, 
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        refine_face_landmarks=False,
    )

    for i in range(start+offset, end, num_processors):
        file_name = df.loc[i, "attachment_id"]
        y = df.loc[i, "text"]
        folder = "train/" if df.loc[i, "train"] else "test/"
        process_video(path+folder+file_name + ".mp4",
                    path+"tracking/"+folder+file_name + ".npy",
                    model = holistic,
                    buffer = 40)
        
        print(f"{i}\t{file_name}", flush=True)



if __name__ == '__main__':
   
    path = "E:/slovo/"
    df = pd.read_csv(path + "annotations.csv", sep="\t")


    params = []
    num_processors = 6
    for i in range(num_processors):
        params.append((i, num_processors))


    with Pool(num_processors) as p:
        p.map(extract_data_fun, params)
    