import cv2
import tifffile as tf
import os

def read_frames_from_video(video_path, root, name):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_number = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_number += 1
        print(f"Reading frame {frame_number}")

        save = os.path.join(root, name+'-origin')
        if not os.path.exists(save):
            os.mkdir(save)

        tf.imwrite(os.path.join(save, 'frame%04d.tif'%int(frame_number)), frame)
    cap.release()


root = 'data_process/EB1'
name = 'EB1'
video_path = 'data_process/EB1/video.mov'
read_frames_from_video(video_path, root, name)
