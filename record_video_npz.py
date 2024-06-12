import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Ask the user for a number to replace the file names
file_number = input("Enter a number to name the files: ")

# Create directories to save the frames if they don't exist
color_path = f'/home/mohamed/ros2_ws/dataset/rgb/{file_number}_rgb'
depth_raw_path = f'/home/mohamed/ros2_ws/dataset/raw/{file_number}_depth'

os.makedirs(color_path, exist_ok=True)
os.makedirs(depth_raw_path, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()

# Enable depth and color streams at 1280x720 resolution
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start the pipeline
profile = pipeline.start(config)
# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)
# Get the depth sensor and set its options
depth_sensor = profile.get_device().first_depth_sensor()
#depth_sensor.set_option(rs.option.exposure, 8500)  # microseconds
#depth_sensor.set_option(rs.option.gain, 16)
depth_sensor.set_option(rs.option.frames_queue_size, 5)

frame_count = 0

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
                        # Align the depth frame to the color frame
        aligned_frames = align.process(frames)
        # Get the aligned depth and color frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Save color frame as PNG
        color_image_path = os.path.join(color_path, f"{file_number}_rgb_{frame_count:06d}.png")
        cv2.imwrite(color_image_path, color_image)

        # Save depth data as NPZ
        depth_image_path = os.path.join(depth_raw_path, f"{file_number}_depth_{frame_count:06d}.npz")
        np.savez_compressed(depth_image_path, depth_image=depth_image)

        print(f"Saved {color_image_path} and {depth_image_path}")

        frame_count += 1

        # Display the color and depth frames
        cv2.imshow('Color Frame', color_image)
        cv2.imshow('Depth Frame', depth_image)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
