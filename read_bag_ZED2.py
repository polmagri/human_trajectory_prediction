import pyzed.sl as sl
import numpy as np
import cv2
import argparse
import os

# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Extract frames from recorded SVO file and save as images.")
parser.add_argument("-i", "--input", type=str, default="/home/mohamed/ros2_ws//dataset/bag_data/02152023_front_avoid_passenger_1.bag",
                    help="Path to the SVO file")
parser.add_argument("-o", "--output", type=str, default="/home/mohamed/ros2_ws/dataset/bag_data",
                    help="Directory to save the extracted frames")
args = parser.parse_args()

# Check if the given file has the correct extension
if os.path.splitext(args.input)[1] not in [".svo", ".bag"]:
    print("The given file is not of correct file format.")
    print("Only .svo or .bag files are accepted")
    exit()

# Check if the file exists
if not os.path.exists(args.input):
    print(f"The file {args.input} does not exist.")
    exit()

# Print the absolute path for debugging
print(f"Attempting to open SVO file at: {os.path.abspath(args.input)}")

# Create output directories
bag_name = os.path.splitext(os.path.basename(args.input))[0]
depth_output_dir = os.path.join(args.output, bag_name + '_depth')
rgb_output_dir = os.path.join(args.output, bag_name + '_rgb')
raw_output_dir = os.path.join(args.output, bag_name + '_raw')

os.makedirs(depth_output_dir, exist_ok=True)
os.makedirs(rgb_output_dir, exist_ok=True)
os.makedirs(raw_output_dir, exist_ok=True)

# Create a ZED camera object
zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.set_from_svo_file(args.input)
init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units (default)
init_params.svo_real_time_mode = False  # Don't skip frames

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(f"Error opening ZED camera: {err}")
    exit(1)

# Manually set intrinsic parameters
fx = 700.819
fy = 700.819
cx = 665.465
cy = 371.953
distortion = [-0.174318, 0.0261121]

print("Intrinsic parameters:")
print(f"Focal length (fx, fy): ({fx}, {fy})")
print(f"Principal point (cx, cy): ({cx}, {cy})")
print(f"Distortion coefficients: {distortion}")

runtime_params = sl.RuntimeParameters()
mat_rgb = sl.Mat()
mat_depth = sl.Mat()
frame_count = 0

try:
    while True:
        # Grab an image
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image
            zed.retrieve_image(mat_rgb, sl.VIEW.LEFT)
            # Retrieve the depth map. Depth is aligned with the left image
            zed.retrieve_measure(mat_depth, sl.MEASURE.DEPTH)

            # Convert images to numpy arrays
            rgb_image = mat_rgb.get_data()
            depth_image = mat_depth.get_data()

            # Save images
            color_image_path = os.path.join(rgb_output_dir, f"color_frame_{frame_count:06d}.png")
            depth_image_path = os.path.join(depth_output_dir, f"depth_frame_{frame_count:06d}.png")
            depth_raw_path = os.path.join(raw_output_dir, f"depth_raw_{frame_count:06d}.npy")
            color_raw_path = os.path.join(raw_output_dir, f"color_raw_{frame_count:06d}.npy")

            cv2.imwrite(color_image_path, rgb_image)
            cv2.imwrite(depth_image_path, depth_image)
            np.save(depth_raw_path, depth_image)
            np.save(color_raw_path, rgb_image)

            print(f"Saved {color_image_path}, {depth_image_path}, {depth_raw_path}, and {color_raw_path}")

            frame_count += 1

            # Break the loop after extracting a fixed number of frames, e.g., 1000 frames
            if frame_count >= 1000:
                break
finally:
    zed.close()
