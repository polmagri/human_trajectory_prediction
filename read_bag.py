import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os

# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Extract frames from recorded bag file and save as images.")
parser.add_argument("-i", "--input", type=str, default="/home/mohamed/ros2_ws/dataset/bag_data/20240607_154313.bag",
                    help="Path to the bag file")
parser.add_argument("-o", "--output", type=str, default="/home/mohamed/ros2_ws/dataset/bag_data",
                    help="Directory to save the extracted frames")
args = parser.parse_args()

# Check if the given file has bag extension
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()

# Create output directories
bag_name = os.path.splitext(os.path.basename(args.input))[0]
depth_output_dir = os.path.join(args.output, bag_name + '_depth')
rgb_output_dir = os.path.join(args.output, bag_name + '_rgb')
depth_raw_output_dir = os.path.join(args.output, bag_name + '_raw_depth')
rgb_raw_output_dir = os.path.join(args.output, bag_name + '_raw_rgb')

os.makedirs(depth_output_dir, exist_ok=True)
os.makedirs(rgb_output_dir, exist_ok=True)
os.makedirs(depth_raw_output_dir, exist_ok=True)
os.makedirs(rgb_raw_output_dir, exist_ok=True)

# Create a pipeline
pipeline = rs.pipeline()

# Create a config object
config = rs.config()

# Tell config that we will use a recorded device from file to be used by the pipeline through playback.
rs.config.enable_device_from_file(config, args.input)

# Start streaming from file
profile = pipeline.start(config)

frame_count = 0

try:
    while True:
        # Get frameset of depth and color
        frames = pipeline.wait_for_frames()

        # Get color and depth frames
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Save images
        color_image_path = os.path.join(rgb_output_dir, f"color_frame_{frame_count:06d}.png")
        depth_image_path = os.path.join(depth_output_dir, f"depth_frame_{frame_count:06d}.png")
        depth_raw_path = os.path.join(depth_raw_output_dir, f"depth_raw_{frame_count:06d}.npy")
        color_raw_path = os.path.join(rgb_raw_output_dir, f"color_raw_{frame_count:06d}.npy")

        # Save depth as 16-bit PNG to maintain precision
        cv2.imwrite(color_image_path, color_image)
        cv2.imwrite(depth_image_path, depth_image.astype(np.uint16))
        np.save(depth_raw_path, depth_image)
        np.save(color_raw_path, color_image)

        print(f"Saved {color_image_path}, {depth_image_path}, {depth_raw_path}, and {color_raw_path}")

        frame_count += 1

        # Break the loop after extracting a fixed number of frames, e.g., 1000 frames
        if frame_count >= 1000:
            break

finally:
    pipeline.stop()
