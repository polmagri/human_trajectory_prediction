import os
import cv2
import numpy as np
import pyrealsense2 as rs
import matplotlib
import time

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def calculate_3d(x, y, depth_data, intrinsics):
    depth = depth_data[y, x]
    if depth == 0:
        return 0.0, 0.0, 0.0, 0.0

    point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth / 1000.0)
    return point[0], point[1], point[2], depth

def read_depth_raw(path):
    with open(path, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint16).reshape(720, 1280)

# Ask the user for a number to replace the file names
file_number = input("Enter a number to name the files: ")

# Define paths to the directories
color_path = f'/home/mohamed/ros2_ws/dataset/rgb/{file_number}_rgb'
depth_raw_path = f'/home/mohamed/ros2_ws/dataset/raw/{file_number}_depth'

# Initialize the RealSense pipeline for intrinsics only
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

profile = pipeline.start(config)
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
pipeline.stop()

# Get list of files
color_files = sorted([f for f in os.listdir(color_path) if f.endswith('.jpg')])
depth_raw_files = sorted([f for f in os.listdir(depth_raw_path) if f.endswith('.raw')])

# Create a window
cv2.namedWindow('Depth Visualization')

# Initialize the figure and axis for plotting
fig, ax = plt.subplots()

# Mouse callback function to get the coordinates and plot the distance
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        depth_frame = param
        x_3d, y_3d, z_3d, min_depth = calculate_3d(x, y, depth_frame, color_intrinsics)
        ax.clear()
        ax.bar(['Distance'], [min_depth])
        plt.draw()

try:
    for color_file, depth_raw_file in zip(color_files, depth_raw_files):
        color_image = cv2.imread(os.path.join(color_path, color_file))
        depth_frame = read_depth_raw(os.path.join(depth_raw_path, depth_raw_file))

        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Visualization', (depth_frame * (255.0 / depth_frame.max())).astype(np.uint8))

        # Set the mouse callback function
        cv2.setMouseCallback('Depth Visualization', mouse_callback, depth_frame)

        plt.draw()
        plt.pause(0.001)

        # Add a delay to slow down the frame rate
        time.sleep(0.1)  # Adjust the delay (in seconds) to slow down the playback

        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    cv2.destroyAllWindows()
