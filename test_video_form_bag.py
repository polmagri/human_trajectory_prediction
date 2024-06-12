import numpy as np
import cv2
import os

# Paths to the directories containing the raw images
rgb_input_path = "/home/mohamed/ros2_ws/dataset/bag_data/20240607_154313_raw_rgb"
depth_input_path = "/home/mohamed/ros2_ws/dataset/bag_data/20240607_154313_raw_depth"

# List all files in the directories
rgb_files = sorted([os.path.join(rgb_input_path, f) for f in os.listdir(rgb_input_path) if f.endswith('.npy')])
depth_files = sorted([os.path.join(depth_input_path, f) for f in os.listdir(depth_input_path) if f.endswith('.npy')])

# Ensure the number of files match
if len(rgb_files) != len(depth_files):
    print("The number of RGB and depth files do not match.")
    exit()

# Create windows for display
cv2.namedWindow('RGB Video', cv2.WINDOW_NORMAL)
cv2.namedWindow('Depth Video', cv2.WINDOW_NORMAL)

# Set windows to the highest resolution
cv2.resizeWindow('RGB Video', 1280, 720)
cv2.resizeWindow('Depth Video', 1280, 720)

# Define the codec and create VideoWriter objects to save the videos
fourcc = cv2.VideoWriter_fourcc(*'XVID')
rgb_video_writer = cv2.VideoWriter('rgb_video.avi', fourcc, 30.0, (1280, 720))
depth_video_writer = cv2.VideoWriter('depth_video.avi', fourcc, 30.0, (1280, 720))

# Loop through all files and display the images
for rgb_file, depth_file in zip(rgb_files, depth_files):
    # Load images
    rgb_image = np.load(rgb_file)
    depth_image = np.load(depth_file)

    # Resize images to 1280x720
    rgb_image = cv2.resize(rgb_image, (1280, 720), interpolation=cv2.INTER_AREA)
    depth_image = cv2.resize(depth_image, (1280, 720), interpolation=cv2.INTER_AREA)

    # Normalize depth image for visualization
    depth_colormap = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_colormap = np.uint8(depth_colormap)
    depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)

    # Display images
    cv2.imshow('RGB Video', rgb_image)
    cv2.imshow('Depth Video', depth_colormap)

    # Write frames to video files
    rgb_video_writer.write(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    depth_video_writer.write(depth_colormap)

    # Wait for a key event for a short time to create a video effect
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video writers and windows
rgb_video_writer.release()
depth_video_writer.release()
cv2.destroyAllWindows()
