import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Ask the user for a number to replace the file names
file_number = input("Enter a number to name the files: ")

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable depth and color streams at 1280x720 resolution
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start the pipeline with the specified configuration
profile = pipeline.start(config)

# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)

# Create directories to save frames
color_path = f'/home/mohamed/ros2_ws/dataset/rgb/{file_number}_rgb'
depth_raw_path = f'/home/mohamed/ros2_ws/dataset/raw/{file_number}_depth'
depth_png_path = f'/home/mohamed/ros2_ws/dataset/depth_png/{file_number}_depth'
os.makedirs(color_path, exist_ok=True)
os.makedirs(depth_raw_path, exist_ok=True)
os.makedirs(depth_png_path, exist_ok=True)

# Create video writers
color_video_path = f'/home/mohamed/ros2_ws/dataset/rgb/{file_number}_rgb.avi'
depth_video_path = f'/home/mohamed/ros2_ws/dataset/depth/{file_number}_depth.avi'

fourcc = cv2.VideoWriter_fourcc(*'XVID')
color_video_writer = cv2.VideoWriter(color_video_path, fourcc, 30, (1280, 720))
depth_video_writer = cv2.VideoWriter(depth_video_path, fourcc, 30, (1280, 720), isColor=False)

frame_number = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Save raw depth data
        depth_image_path = os.path.join(depth_raw_path, f"{file_number}_depth_{frame_number:04d}.raw")
        with open(depth_image_path, 'wb') as depth_file:
            depth_file.write(depth_image.tobytes())

        # Save depth data as PNG
        depth_png_path = os.path.join(depth_png_path, f"{file_number}_depth_{frame_number:04d}.png")
        cv2.imwrite(depth_png_path, depth_image)

        # Normalize the depth image to 16-bit for visualization
        depth_image_vis = cv2.normalize(depth_image, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)

        # Convert the 16-bit depth image to 8-bit for saving as video
        depth_image_vis_8bit = (depth_image_vis / 256).astype(np.uint8)

        # Save color image
        color_image_path = os.path.join(color_path, f"{file_number}_rgb_{frame_number:04d}.jpg")
        cv2.imwrite(color_image_path, color_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Write the frames to video files
        color_video_writer.write(color_image)
        depth_video_writer.write(depth_image_vis_8bit)

        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Image', depth_image_vis_8bit)

        if cv2.waitKey(1) == ord('q'):
            break

        frame_number += 1

except Exception as e:
    print(f"Error: {e}")

finally:
    pipeline.stop()
    color_video_writer.release()
    depth_video_writer.release()
    cv2.destroyAllWindows()
