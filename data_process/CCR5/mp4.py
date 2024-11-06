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

        if frame_number % 3 == 0:
                tf.imwrite(os.path.join(save, 'frame%04d.tif'%int(frame_number/3)), frame)

    cap.release()

root = 'data_process/CCR5'
name = 'CCR5'
video_path = 'data_process/CCR5/video.mp4'
read_frames_from_video(video_path, root, name)
