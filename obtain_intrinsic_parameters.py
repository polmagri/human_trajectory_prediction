import cv2
import numpy as np
import pyrealsense2 as rs

# Inizializzazione dei lettori video per i file .avi
depth_cap = cv2.VideoCapture("1_depth.avi")
rgb_cap = cv2.VideoCapture("1_rgb.avi")

# Configura il contesto del pipeline RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipeline.start(config)

# Definisci gli intrinsic parameters della telecamera RGB
intrinsics = rs.intrinsics()


while depth_cap.isOpened() and rgb_cap.isOpened():
    # Leggi i frame dai video
    ret_depth, depth_frame = depth_cap.read()
    ret_rgb, rgb_frame = rgb_cap.read()

    if not ret_depth or not ret_rgb:
        break

    # Converte il frame RGB in scala di grigi
    gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

    # Ottieni la distanza dal frame di profondità
    frames = pipeline.wait_for_frames()
    depth_frame_rs = frames.get_depth_frame()
    depth_value = depth_frame_rs.get_distance(x_pixel, y_pixel)





    # Attendere il tasto ESC per uscire
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Rilascia le risorse
depth_cap.release()
rgb_cap.release()
cv2.destroyAllWindows()
