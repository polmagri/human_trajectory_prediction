import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import csv
import glob
import re

def numerical_sort(value):
    parts = re.split(r'(\d+)', value)
    return [int(part) if part.isdigit() else part for part in parts]

def calculate_3d(x, y, depth_data, intrinsics, w, h, window_size=10):
    if x == 0 and y == 0:
        return 0.0, 0.0, 0.0, 0.0

    min_depth = float('inf')

    for i in range(-window_size // 2, window_size // 2 + 1):
        for j in range(-window_size // 2, window_size // 2 + 1):
            x_pixel = int(x) + i
            y_pixel = int(y) + j

            if 0 <= x_pixel < w and 0 <= y_pixel < h:
                depth = depth_data[y_pixel, x_pixel]
                if depth != 0 and depth < min_depth:
                    min_depth = depth

    if min_depth == float('inf'):
        return 0.0, 0.0, 0.0, 0.0

    pixel = [x, y]
    depth = min_depth

    point = rs.rs2_deproject_pixel_to_point(intrinsics, pixel, depth / 1000.0)  # Convert depth to meters

    x_3d, z_3d, y_3d = point

    return x_3d, y_3d, -z_3d, min_depth

def camera_to_world(camera_coords, transformation_matrix):
    camera_coords_h = np.append(camera_coords, 1)
    world_coords_h = np.dot(transformation_matrix, camera_coords_h)
    world_coords = world_coords_h[:3] / world_coords_h[3]
    return world_coords

def calculate_plane_and_arrow(p1, p2, p3, p4, p5, arrow_length):
    v1 = np.subtract(p2, p1)
    v2 = np.subtract(p3, p1)

    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    normal_xy = np.array([normal[0], normal[1], 0])
    normal_xy = normal_xy / np.linalg.norm(normal_xy)

    arrow_start = p4
    arrow_end = p4 + arrow_length * normal_xy

    return arrow_start, arrow_end, normal_xy

def save_camera_csv(frame_number, timestamp, keypoints_data, csv_writer, alpha, beta, alphaV, betaV):
    row = [frame_number, timestamp]
    
    keypoint_names = [
        'Nose', 'Eye.L', 'Eye.R', 'Ear.L', 'Ear.R',
        'Shoulder.L', 'Shoulder.R', 'Elbow.L', 'Elbow.R',
        'Wrist.L', 'Wrist.R', 'Hip.L', 'Hip.R',
        'Knee.L', 'Knee.R', 'Ankle.L', 'Ankle.R', 'Neck', 'Pelvis'
    ]
    
    keypoints_dict = {kp: (None, None, None) for kp in keypoint_names}
    
    for keypoint in keypoints_data:
        keypoint_name = keypoint[0]
        if keypoint_name in keypoints_dict:
            keypoints_dict[keypoint_name] = keypoint[1:]
    
    for keypoint in keypoint_names:
        row.extend(keypoints_dict[keypoint])
    row.extend([alpha, beta, alphaV, betaV])

    csv_writer.writerow(row)

def save_global_csv(frame_number, timestamp, keypoints_data, csv_writer, alpha_glob, beta_glob, alphaV_glob, betaV_glob):
    row = [frame_number, timestamp]
    
    keypoint_names = [
        'Nose', 'Eye.L', 'Eye.R', 'Ear.L', 'Ear.R',
        'Shoulder.L', 'Shoulder.R', 'Elbow.L', 'Elbow.R',
        'Wrist.L', 'Wrist.R', 'Hip.L', 'Hip.R',
        'Knee.L', 'Knee.R', 'Ankle.L', 'Ankle.R', 'Neck', 'Pelvis'
    ]

    keypoints_dict = {kp: (None, None, None) for kp in keypoint_names}

    for keypoint in keypoints_data:
        keypoint_name = keypoint[0]
        if keypoint_name in keypoints_dict:
            keypoints_dict[keypoint_name] = keypoint[1:]

    for keypoint in keypoint_names:
        row.extend(keypoints_dict[keypoint])
    row.extend([alpha_glob, beta_glob, alphaV_glob, betaV_glob])

    csv_writer.writerow(row)

# Initialize the RealSense pipeline for intrinsics only
pipeline = rs.pipeline()
config = rs.config()

# Enable only RGB stream to get the intrinsics
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start the pipeline with the specified configuration
profile = pipeline.start(config)
# Get the depth sensor and set its options
depth_sensor = profile.get_device().first_depth_sensor()

depth_sensor.set_option(rs.option.frames_queue_size, 5)
# Get the intrinsic parameters of the RGB sensor
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# Stop the pipeline as we don't need it for further processing
pipeline.stop()

# Initialize the YOLO detector
model = YOLO('yolov8n-pose')

# Map of body part indices to their names
index_to_label = {
    0: 'Nose', 1: 'Eye.L', 2: 'Eye.R', 3: 'Ear.L', 4: 'Ear.R',
    5: 'Shoulder.L', 6: 'Shoulder.R', 7: 'Elbow.L', 8: 'Elbow.R',
    9: 'Wrist.L', 10: 'Wrist.R', 11: 'Hip.L', 12: 'Hip.R',
    13: 'Knee.L', 14: 'Knee.R', 15: 'Ankle.L', 16: 'Ankle.R'
}

# Define keypoint connections for drawing lines
keypoint_connections = [
    (0, 1), (0, 2), (5, 6), (11, 12), (2, 4), (1, 3), (5, 7),
    (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

# Create the figure and 3D axis outside the loop
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax_global = fig.add_subplot(122, projection='3d')

# Set fixed limits for the axes
for axis in [ax, ax_global]:
    axis.set_xlim(-4, 4)
    axis.set_ylim(0, 4)
    axis.set_zlim(-1, 3)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')

# Initialize lists for plots
scatter_plots = []
line_plots = []
text_labels = []
arrow_plots = []
sector_plots = []
circle_plots = []

# Define paths for saving data
test_number = int(input("Enter the test number: "))

color_path = f'dataset/test{test_number}/rgb'
depth_path = f'dataset/test{test_number}/depth'
camera_csv_path = f'/home/mohamed/ros2_ws/pre_processing/test{test_number}/camera_coordinates_pp.csv'
global_csv_path = f'/home/mohamed/ros2_ws/pre_processing/test{test_number}/global_coordinates_pp.csv'

# Ensure the directories exist
os.makedirs(os.path.dirname(camera_csv_path), exist_ok=True)
os.makedirs(os.path.dirname(global_csv_path), exist_ok=True)

# Open CSV files for writing
camera_csv_file = open(camera_csv_path, 'w', newline='')
global_csv_file = open(global_csv_path, 'w', newline='')
camera_csv_writer = csv.writer(camera_csv_file)
global_csv_writer = csv.writer(global_csv_file)

# Write headers for camera coordinates CSV
keypoints_header_camera = [f'{kp}.{axis}' for kp in [
    'Nose', 'Eye.L', 'Eye.R', 'Ear.L', 'Ear.R',
    'Shoulder.L', 'Shoulder.R', 'Elbow.L', 'Elbow.R',
    'Wrist.L', 'Wrist.R', 'Hip.L', 'Hip.R',
    'Knee.L', 'Knee.R', 'Ankle.L', 'Ankle.R', 'Neck', 'Pelvis'
] for axis in ['x', 'y', 'z']]
camera_csv_writer.writerow(['Frame', 'Time'] + keypoints_header_camera + ['alpha', 'beta', 'alphaV', 'betaV'])

# Write headers for global coordinates CSV
keypoints_header_global = [f'{kp}.{axis}' for kp in [
    'Nose', 'Eye.L', 'Eye.R', 'Ear.L', 'Ear.R',
    'Shoulder.L', 'Shoulder.R', 'Elbow.L', 'Elbow.R',
    'Wrist.L', 'Wrist.R', 'Hip.L', 'Hip.R',
    'Knee.L', 'Knee.R', 'Ankle.L', 'Ankle.R', 'Neck', 'Pelvis'
] for axis in ['x', 'y', 'z']]
global_csv_writer.writerow(['Frame', 'Time'] + keypoints_header_global + ['alpha_glob', 'beta_glob', 'alphaV_glob', 'betaV_glob'])

frame_number = 0

# Load frames from paths
rgb_frames = sorted(glob.glob(os.path.join(color_path, '*.png')), key=numerical_sort)
depth_frames = sorted(glob.glob(os.path.join(depth_path, '*.npz')), key=numerical_sort)

try:
    for rgb_frame_path, depth_frame_path in zip(rgb_frames, depth_frames):
        frame_number += 1

        color_image = cv2.imread(rgb_frame_path)
        depth_data = np.load(depth_frame_path)['arr_0']

        # Run the YOLO model on the frames
        persons = model(color_image)

        # Clear previous plots
        for plots in [scatter_plots, line_plots, text_labels, arrow_plots, sector_plots, circle_plots]:
            for plot in plots:
                plot.remove()
            plots.clear()

        alpha = alpha_glob = beta = beta1 = betac = beta_glob = beta1_glob = betac_glob = alphaV = betaV = alphaV_glob = betaV_glob = 0

        for results in persons:
            for result in results:
                if hasattr(result, 'keypoints'):
                    kpts = result.keypoints.xy.cpu().numpy()
                    keypoints_list = kpts.flatten().tolist()
                    labels = [index_to_label.get(i, '') for i in range(len(keypoints_list) // 2)]

                    keypoints_2d = {}
                    keypoints_3d_camera = []
                    keypoints_3d_global = []

                    for i, (x, y) in enumerate(zip(keypoints_list[::2], keypoints_list[1::2])):
                        label = labels[i]
                        if label:
                            x_3d_camera, y_3d_camera, z_3d_camera, min_depth = calculate_3d(int(x), int(y), depth_data, color_intrinsics, depth_data.shape[1], depth_data.shape[0])
                            if not np.isnan(z_3d_camera) and (x_3d_camera != 0 or y_3d_camera != 0 or z_3d_camera != 0):
                                keypoints_2d[label] = (int(x), int(y))
                                keypoints_3d_camera.append((label, x_3d_camera, y_3d_camera, z_3d_camera))

                                # Convert to global coordinates
                                transformation_matrix = np.array([
                                    [1, 0, 0, 0],  
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]
                                ])

                                x_3d_global, y_3d_global, z_3d_global = camera_to_world([x_3d_camera, y_3d_camera, z_3d_camera], transformation_matrix)
                                keypoints_3d_global.append((label, x_3d_global, y_3d_global, z_3d_global))

                    if 'Hip.L' in keypoints_2d and 'Hip.R' in keypoints_2d:
                        hip_l = keypoints_2d['Hip.L']
                        hip_r = keypoints_2d['Hip.R']
                        if hip_l != (0, 0) and hip_r != (0, 0):
                            pelvis_x = (hip_l[0] + hip_r[0]) // 2
                            pelvis_y = (hip_l[1] + hip_r[1]) // 2
                            keypoints_2d['Pelvis'] = (pelvis_x, pelvis_y)

                            x_3d_camera, y_3d_camera, z_3d_camera, min_depth = calculate_3d(pelvis_x, pelvis_y, depth_data, color_intrinsics, depth_data.shape[1], depth_data.shape[0])
                            if not np.isnan(z_3d_camera) and (x_3d_camera != 0 or y_3d_camera != 0 or z_3d_camera != 0):
                                keypoints_3d_camera.append(('Pelvis', x_3d_camera, y_3d_camera, z_3d_camera))

                                x_3d_global, y_3d_global, z_3d_global = camera_to_world([x_3d_camera, y_3d_camera, z_3d_camera], transformation_matrix)
                                keypoints_3d_global.append(('Pelvis', x_3d_global, y_3d_global, z_3d_global))

                                if x_3d_camera != 0 or y_3d_camera != 0:
                                    alpha = np.degrees(np.arctan2(y_3d_camera, x_3d_camera))
                                else:
                                    alpha = 0

                                if x_3d_global != 0 or y_3d_global != 0:
                                    alpha_glob = np.degrees(np.arctan2(y_3d_global, x_3d_global))
                                else:
                                    alpha_glob = 0

                    if 'Shoulder.L' in keypoints_2d and 'Shoulder.R' in keypoints_2d:
                        shoulder_l = keypoints_2d['Shoulder.L']
                        shoulder_r = keypoints_2d['Shoulder.R']
                        if shoulder_l != (0, 0) and shoulder_r != (0, 0):
                            neck_x = (shoulder_l[0] + shoulder_r[0]) // 2
                            neck_y = (shoulder_l[1] + shoulder_r[1]) // 2
                            keypoints_2d['Neck'] = (neck_x, neck_y)

                            x_3d_camera, y_3d_camera, z_3d_camera, min_depth = calculate_3d(neck_x, neck_y, depth_data, color_intrinsics, depth_data.shape[1], depth_data.shape[0])
                            if not np.isnan(z_3d_camera) and (x_3d_camera != 0 or y_3d_camera != 0 or z_3d_camera != 0):
                                keypoints_3d_camera.append(('Neck', x_3d_camera, y_3d_camera, z_3d_camera))

                                x_3d_global, y_3d_global, z_3d_global = camera_to_world([x_3d_camera, y_3d_camera, z_3d_camera], transformation_matrix)
                                keypoints_3d_global.append(('Neck', x_3d_global, y_3d_global, z_3d_global))

                                if x_3d_camera != 0 or y_3d_camera != 0:
                                    alpha = np.degrees(np.arctan2(y_3d_camera, x_3d_camera))
                                else:
                                    alpha = 0

                                if x_3d_global != 0 or y_3d_global != 0:
                                    alpha_glob = np.degrees(np.arctan2(y_3d_global, x_3d_global))
                                else:
                                    alpha_glob = 0

 

                    for keypoint in keypoints_3d_camera:
                        label, x_3d, y_3d, z_3d = keypoint
                        scatter = ax.scatter(x_3d, y_3d, z_3d, label=label)
                        scatter_plots.append(scatter)
                        text = ax.text(x_3d, y_3d, z_3d, label, color='red')
                        text_labels.append(text)

                    for (start, end) in keypoint_connections:
                        start_label = index_to_label.get(start, '')
                        end_label = index_to_label.get(end, '')
                        if start_label and end_label:
                            start_point = next((kp for kp in keypoints_3d_camera if kp[0] == start_label), None)
                            end_point = next((kp for kp in keypoints_3d_camera if kp[0] == end_label), None)
                            if start_point and end_point:
                                line, = ax.plot([start_point[1], end_point[1]], [start_point[2], end_point[2]], [start_point[3], end_point[3]], 'b')
                                line_plots.append(line)

                    for keypoint in keypoints_3d_global:
                        label, x_3d, y_3d, z_3d = keypoint
                        scatter = ax_global.scatter(x_3d, y_3d, z_3d, label=label)
                        scatter_plots.append(scatter)
                        text = ax_global.text(x_3d, y_3d, z_3d, label, color='blue')
                        text_labels.append(text)

                    for (start, end) in keypoint_connections:
                        start_label = index_to_label.get(start, '')
                        end_label = index_to_label.get(end, '')
                        if start_label and end_label:
                            start_point = next((kp for kp in keypoints_3d_global if kp[0] == start_label), None)
                            end_point = next((kp for kp in keypoints_3d_global if kp[0] == end_label), None)
                            if start_point and end_point:
                                line, = ax_global.plot([start_point[1], end_point[1]], [start_point[2], end_point[2]], [start_point[3], end_point[3]], 'b')
                                line_plots.append(line)

                    if 'Pelvis' in [kp[0] for kp in keypoints_3d_camera] and 'Neck' in [kp[0] for kp in keypoints_3d_camera]:
                        pelvis = next(kp for kp in keypoints_3d_camera if kp[0] == 'Pelvis')
                        neck = next(kp for kp in keypoints_3d_camera if kp[0] == 'Neck')
                        pelvis_point = np.array([pelvis[1], pelvis[2], pelvis[3]])
                        pelvis_point_global = np.array([kp[1:] for kp in keypoints_3d_global if kp[0] == 'Pelvis'])[0]

                        if 'Shoulder.L' in [kp[0] for kp in keypoints_3d_camera] and 'Shoulder.R' in [kp[0] for kp in keypoints_3d_camera]:
                            shoulder_l = next(kp for kp in keypoints_3d_camera if kp[0] == 'Shoulder.L')
                            shoulder_r = next(kp for kp in keypoints_3d_camera if kp[0] == 'Shoulder.R')
                            shoulder_l_global = next(kp for kp in keypoints_3d_global if kp[0] == 'Shoulder.L')
                            shoulder_r_global = next(kp for kp in keypoints_3d_global if kp[0] == 'Shoulder.R')

                            p1, p2, p3, p4, p5 = np.array(pelvis[1:]), np.array(shoulder_l[1:]), np.array(shoulder_r[1:]), np.array(pelvis[1:]), np.array(neck[1:])
                            p1_global, p2_global, p3_global, p4_global, p5_global = pelvis_point_global, np.array(shoulder_l_global[1:]), np.array(shoulder_r_global[1:]), pelvis_point_global, np.array(neck[1:])

                            arrow_start, arrow_end, arrow_vector = calculate_plane_and_arrow(p1, p2, p3, p4, p5, arrow_length=3)
                            arrow_start_global, arrow_end_global, arrow_vector_global = calculate_plane_and_arrow(p1_global, p2_global, p3_global, p4_global, p5_global, arrow_length=3)

                            arrow = ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
                                            arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], arrow_end[2] - arrow_start[2],
                                            color='g', length=1.0, arrow_length_ratio=0.1)
                            arrow_plots.append(arrow)

                            pelvis_xy = pelvis_point[:2]
                            arrow_vector_xy = arrow_vector[:2]

                            beta = (np.degrees(np.arctan2(arrow_vector_xy[1], arrow_vector_xy[0])) + 90) % 360
                            
                            print(f'Angle beta: {beta} degrees')

                            pelvis_xy_global = pelvis_point_global[:2]
                            arrow_vector_xy_global = arrow_vector_global[:2]

                            beta_glob = (np.degrees(np.arctan2(arrow_vector_xy_global[1], arrow_vector_xy_global[0])) + 90) % 360
                            
                            print(f'Angle beta_glob: {beta_glob} degrees')

                        if 'Eye.L' in [kp[0] for kp in keypoints_3d_camera] and 'Eye.R' in [kp[0] for kp in keypoints_3d_camera] and 'Neck' in [kp[0] for kp in keypoints_3d_camera]:
                            eye_l = next(kp for kp in keypoints_3d_camera if kp[0] == 'Eye.L')
                            eye_r = next(kp for kp in keypoints_3d_camera if kp[0] == 'Eye.R')
                            neck = next(kp for kp in keypoints_3d_camera if kp[0] == 'Neck')
                            neck_global = next(kp for kp in keypoints_3d_global if kp[0] == 'Neck')

                            p1, p2, p3, p4, p5 = np.array(eye_l[1:]), np.array(eye_r[1:]), np.array(neck[1:]), np.array(neck[1:]), np.array(neck[1:])
                            p1_global, p2_global, p3_global, p4_global, p5_global = np.array(eye_l[1:]), np.array(eye_r[1:]), np.array(neck_global[1:]), np.array(neck_global[1:]), np.array(neck_global[1:])

                            arrow_neck_start, arrow_neck_end, arrow_vector_view = calculate_plane_and_arrow(p1, p2, p3, p4, p5, arrow_length=4)
                            arrow_neck_start_global, arrow_neck_end_global, arrow_vector_view_global = calculate_plane_and_arrow(p1_global, p2_global, p3_global, p4_global, p5_global, arrow_length=4)

                            arrow = ax.quiver(arrow_neck_start[0], arrow_neck_start[1], arrow_neck_start[2],
                                            arrow_neck_end[0] - arrow_neck_start[0], arrow_neck_end[1] - arrow_neck_start[1], arrow_neck_end[2] - arrow_neck_start[2],
                                            color='r', length=0.5, arrow_length_ratio=0.1)
                            arrow_plots.append(arrow)

                            angle_main = np.arctan2(arrow_neck_end[1] - arrow_neck_start[1], arrow_neck_end[0] - arrow_neck_start[0])
                            angles_additional = [angle_main + np.radians(100), angle_main - np.radians(100)]
                            arrow_end_additional = []
                            for angle in angles_additional:
                                arrow_end_add = arrow_neck_start + 0.5 * np.array([np.cos(angle), np.sin(angle), 0])
                                arrow_end_additional.append(arrow_end_add)
                                arrow = ax.quiver(arrow_neck_start[0], arrow_neck_start[1], arrow_neck_start[2],
                                                arrow_end_add[0] - arrow_neck_start[0], arrow_end_add[1] - arrow_neck_start[1], arrow_end_add[2] - arrow_neck_start[2],
                                                color='r', length=0.5, arrow_length_ratio=0.1)
                                arrow_plots.append(arrow)

                            theta = np.linspace(angles_additional[1], angles_additional[0], 100)
                            x_circle = arrow_neck_start[0] + 0.5 * np.cos(theta)
                            y_circle = arrow_neck_start[1] + 0.5 * np.sin(theta)
                            z_circle = np.full_like(x_circle, arrow_neck_start[2])
                            circle = ax.plot(x_circle, y_circle, z_circle, color='r')
                            circle_plots.append(circle[0])

                            verts = [list(zip(x_circle, y_circle, z_circle))]
                            sector = Poly3DCollection(verts, color='red', alpha=0.2)
                            ax.add_collection3d(sector)
                            sector_plots.append(sector)

                            if x_3d_camera != 0 or y_3d_camera != 0:
                                alphaV = np.degrees(np.arctan2(y_3d_camera, x_3d_camera))
                            else:
                                alphaV = 0

                            neck_xy = np.array([neck[1], neck[2]])
                            arrow_vector_xy_view = arrow_vector_view[:2]

                            betaV = (np.degrees(np.arctan2(arrow_vector_xy_view[1], arrow_vector_xy_view[0])) + 90) % 360
                            
                            print(f'AlphaV: {alphaV} degrees')
                            print(f'BetaV: {betaV} degrees')

                            if x_3d_global != 0 or y_3d_global != 0:
                                alphaV_glob = np.degrees(np.arctan2(y_3d_global, x_3d_global))
                            else:
                                alphaV_glob = 0

                            neck_xy_global = np.array([neck_global[1], neck_global[2]])
                            arrow_vector_xy_view_global = arrow_vector_view_global[:2]

                            betaV_glob = (np.degrees(np.arctan2(arrow_vector_xy_view_global[1], arrow_vector_xy_view_global[0])) + 90) % 360
                            
                            print(f'AlphaV_glob: {alphaV_glob} degrees')
                            print(f'BetaV_glob: {betaV_glob} degrees')

                    save_camera_csv(frame_number, frame_number, keypoints_3d_camera, camera_csv_writer, alpha, beta, alphaV, betaV)
                    save_global_csv(frame_number, frame_number, keypoints_3d_global, global_csv_writer, alpha_glob, beta_glob, alphaV_glob, betaV_glob)
        plt.draw()
        plt.pause(0.001)

        cv2.imshow('YOLO Keypoints', color_image)
        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    camera_csv_file.close()
    global_csv_file.close()
    cv2.destroyAllWindows()
