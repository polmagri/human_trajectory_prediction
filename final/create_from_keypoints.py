import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import csv

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

def save_camera_csv(frame_number, timestamp, keypoints_data, csv_writer, alpha_calculated, beta_calculated, alphaV_calculated, betaV_calculated):
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
    row.extend([alpha_calculated, beta_calculated, alphaV_calculated, betaV_calculated])

    csv_writer.writerow(row)

def save_global_csv(frame_number, timestamp, keypoints_data, csv_writer, alpha_glob_calculated, beta_glob_calculated, alphaV_glob_calculated, betaV_glob_calculated):
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
    row.extend([alpha_glob_calculated, beta_glob_calculated, alphaV_glob_calculated, betaV_glob_calculated])

    csv_writer.writerow(row)


# Initialize plot
fig = plt.figure()
ax_camera = fig.add_subplot(121, projection='3d')
ax_global = fig.add_subplot(122, projection='3d')

# Set fixed limits for the axes
for axis in [ax_camera, ax_global]:
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

# Define input paths
camera_csv_path = f'/home/mohamed/ros2_ws/dataset/test{test_number}/camera_coord.csv'
global_csv_path = f'/home/mohamed/ros2_ws/dataset/test{test_number}/global_coord.csv'

# Load the CSV files
camera_df = pd.read_csv(camera_csv_path)
global_df = pd.read_csv(global_csv_path)

# Define output paths
output_camera_csv_path = f'/home/mohamed/ros2_ws/pre_processing/test{test_number}/camera_coordinates_from_keypoint_calculated.csv'
output_global_csv_path = f'/home/mohamed/ros2_ws/pre_processing/test{test_number}/global_coordinates_from_keypoint_calculated.csv'

# Ensure the directories exist
os.makedirs(os.path.dirname(output_camera_csv_path), exist_ok=True)
os.makedirs(os.path.dirname(output_global_csv_path), exist_ok=True)

# Open CSV files for writing
camera_csv_file = open(output_camera_csv_path, 'w', newline='')
global_csv_file = open(output_global_csv_path, 'w', newline='')
camera_csv_writer = csv.writer(camera_csv_file)
global_csv_writer = csv.writer(global_csv_file)

# Write headers for camera coordinates CSV
keypoints_header_camera = [f'{kp}.{axis}' for kp in [
    'Nose', 'Eye.L', 'Eye.R', 'Ear.L', 'Ear.R',
    'Shoulder.L', 'Shoulder.R', 'Elbow.L', 'Elbow.R',
    'Wrist.L', 'Wrist.R', 'Hip.L', 'Hip.R',
    'Knee.L', 'Knee.R', 'Ankle.L', 'Ankle.R', 'Neck', 'Pelvis'
] for axis in ['x', 'y', 'z']]
camera_csv_writer.writerow(['Frame', 'Time'] + keypoints_header_camera + ['alpha_calculated', 'beta_calculated', 'alphaV_calculated', 'betaV_calculated'])

# Write headers for global coordinates CSV
keypoints_header_global = [f'{kp}.{axis}' for kp in [
    'Nose', 'Eye.L', 'Eye.R', 'Ear.L', 'Ear.R',
    'Shoulder.L', 'Shoulder.R', 'Elbow.L', 'Elbow.R',
    'Wrist.L', 'Wrist.R', 'Hip.L', 'Hip.R',
    'Knee.L', 'Knee.R', 'Ankle.L', 'Ankle.R', 'Neck', 'Pelvis'
] for axis in ['x', 'y', 'z']]
global_csv_writer.writerow(['Frame', 'Time'] + keypoints_header_global + ['alpha_glob_calculated', 'beta_glob_calculated', 'alphaV_glob_calculated', 'betaV_glob_calculated'])

keypoint_connections = [
    (0, 1), (0, 2), (5, 6), (11, 12), (2, 4), (1, 3), (5, 7),
    (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

try:
    frame_groups_camera = camera_df.groupby('Frame')
    frame_groups_global = global_df.groupby('Frame')

    common_frames = set(frame_groups_camera.groups.keys()).intersection(frame_groups_global.groups.keys())

    for frame_number in common_frames:
        camera_rows = frame_groups_camera.get_group(frame_number)
        global_rows = frame_groups_global.get_group(frame_number)

        timestamp = camera_rows.iloc[0]['Time']

        # Clear previous plots
        for plots in [scatter_plots, line_plots, text_labels, arrow_plots, sector_plots, circle_plots]:
            for plot in plots:
                plot.remove()
            plots.clear()

        for camera_row, global_row in zip(camera_rows.iterrows(), global_rows.iterrows()):
            keypoints_3d_camera = []
            keypoints_3d_global = []

            keypoint_names = [
                'Nose', 'Eye.L', 'Eye.R', 'Ear.L', 'Ear.R',
                'Shoulder.L', 'Shoulder.R', 'Elbow.L', 'Elbow.R',
                'Wrist.L', 'Wrist.R', 'Hip.L', 'Hip.R',
                'Knee.L', 'Knee.R', 'Ankle.L', 'Ankle.R', 'Neck', 'Pelvis'
            ]

            for kp in keypoint_names:
                keypoints_3d_camera.append((kp, camera_row[1][f'{kp}.x'], camera_row[1][f'{kp}.y'], camera_row[1][f'{kp}.z']))
                keypoints_3d_global.append((kp, global_row[1][f'{kp}.x'], global_row[1][f'{kp}.y'], global_row[1][f'{kp}.z']))

            alpha_calculated = camera_row[1]['alpha']
            beta_calculated = camera_row[1]['beta']
            alphaV_calculated = camera_row[1]['alphaV']
            betaV_calculated = camera_row[1]['betaV']

            alpha_glob_calculated = global_row[1]['alpha_glob']
            beta_glob_calculated = global_row[1]['beta_glob']
            alphaV_glob_calculated = global_row[1]['alphaV_glob']
            betaV_glob_calculated = global_row[1]['betaV_glob']

            for keypoint in keypoints_3d_camera:
                label, x_3d, y_3d, z_3d = keypoint
                scatter = ax_camera.scatter(x_3d, y_3d, z_3d, label=label)
                scatter_plots.append(scatter)
                text = ax_camera.text(x_3d, y_3d, z_3d, label, color='red')
                text_labels.append(text)
            for (start, end) in keypoint_connections:
                start_label = keypoint_names[start]
                end_label = keypoint_names[end]
                start_point = next((kp for kp in keypoints_3d_camera if kp[0] == start_label), None)
                end_point = next((kp for kp in keypoints_3d_camera if kp[0] == end_label), None)
                if start_point and end_point:
                    line, = ax_camera.plot([start_point[1], end_point[1]], [start_point[2], end_point[2]], [start_point[3], end_point[3]], 'b')
                    line_plots.append(line)

            for keypoint in keypoints_3d_global:
                label, x_3d, y_3d, z_3d = keypoint
                scatter = ax_global.scatter(x_3d, y_3d, z_3d, label=label)
                scatter_plots.append(scatter)
                text = ax_global.text(x_3d, y_3d, z_3d, label, color='blue')
                text_labels.append(text)

            for (start, end) in keypoint_connections:
                start_label = keypoint_names[start]
                end_label = keypoint_names[end]
                start_point = next((kp for kp in keypoints_3d_global if kp[0] == start_label), None)
                end_point = next((kp for kp in keypoints_3d_global if kp[0] == end_label), None)
                if start_point and end_point:
                    line, = ax_global.plot([start_point[1], end_point[1]], [start_point[2], end_point[2]], [start_point[3], end_point[3]], 'b')
                    line_plots.append(line)

            # Calculate the angles for camera coordinates
            pelvis_camera = next((kp for kp in keypoints_3d_camera if kp[0] == 'Pelvis'), None)
            neck_camera = next((kp for kp in keypoints_3d_camera if kp[0] == 'Neck'), None)

            if pelvis_camera and neck_camera:
                pelvis_point = np.array([pelvis_camera[1], pelvis_camera[2], pelvis_camera[3]])
                neck_point = np.array([neck_camera[1], neck_camera[2], neck_camera[3]])

                # Body direction vector
                if 'Shoulder.L' in [kp[0] for kp in keypoints_3d_camera] and 'Shoulder.R' in [kp[0] for kp in keypoints_3d_camera]:
                    shoulder_l = next(kp for kp in keypoints_3d_camera if kp[0] == 'Shoulder.L')
                    shoulder_r = next(kp for kp in keypoints_3d_camera if kp[0] == 'Shoulder.R')

                    p1, p2, p3, p4, p5 = pelvis_point, np.array(shoulder_l[1:]), np.array(shoulder_r[1:]), pelvis_point, neck_point
                    arrow_start, arrow_end, arrow_vector = calculate_plane_and_arrow(p1, p2, p3, p4, p5, arrow_length=3)

                    arrow = ax_camera.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
                                             arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], arrow_end[2] - arrow_start[2],
                                             color='g', length=1.0, arrow_length_ratio=0.1)
                    arrow_plots.append(arrow)

                    pelvis_xy = pelvis_point[:2]
                    arrow_vector_xy = arrow_vector[:2]
                    beta_calculated = (np.degrees(np.arctan2(arrow_vector_xy[1], arrow_vector_xy[0])) + 90) % 360

                # View direction vector
                if 'Eye.L' in [kp[0] for kp in keypoints_3d_camera] and 'Eye.R' in [kp[0] for kp in keypoints_3d_camera]:
                    eye_l = next(kp for kp in keypoints_3d_camera if kp[0] == 'Eye.L')
                    eye_r = next(kp for kp in keypoints_3d_camera if kp[0] == 'Eye.R')

                    p1, p2, p3, p4, p5 = np.array(eye_l[1:]), np.array(eye_r[1:]), neck_point, neck_point, neck_point
                    arrow_neck_start, arrow_neck_end, arrow_vector_view = calculate_plane_and_arrow(p1, p2, p3, p4, p5, arrow_length=4)

                    arrow = ax_camera.quiver(arrow_neck_start[0], arrow_neck_start[1], arrow_neck_start[2],
                                             arrow_neck_end[0] - arrow_neck_start[0], arrow_neck_end[1] - arrow_neck_start[1], arrow_neck_end[2] - arrow_neck_start[2],
                                             color='r', length=0.5, arrow_length_ratio=0.1)
                    arrow_plots.append(arrow)

                    angle_main = np.arctan2(arrow_neck_end[1] - arrow_neck_start[1], arrow_neck_end[0] - arrow_neck_start[0])
                    angles_additional = [angle_main + np.radians(100), angle_main - np.radians(100)]
                    arrow_end_additional = []
                    for angle in angles_additional:
                        arrow_end_add = arrow_neck_start + 0.5 * np.array([np.cos(angle), np.sin(angle), 0])
                        arrow_end_additional.append(arrow_end_add)
                        arrow = ax_camera.quiver(arrow_neck_start[0], arrow_neck_start[1], arrow_neck_start[2],
                                                 arrow_end_add[0] - arrow_neck_start[0], arrow_end_add[1] - arrow_neck_start[1], arrow_end_add[2] - arrow_neck_start[2],
                                                 color='r', length=0.5, arrow_length_ratio=0.1)
                        arrow_plots.append(arrow)

                    theta = np.linspace(angles_additional[1], angles_additional[0], 100)
                    x_circle = arrow_neck_start[0] + 0.5 * np.cos(theta)
                    y_circle = arrow_neck_start[1] + 0.5 * np.sin(theta)
                    z_circle = np.full_like(x_circle, arrow_neck_start[2])
                    circle = ax_camera.plot(x_circle, y_circle, z_circle, color='r')
                    circle_plots.append(circle[0])

                    verts = [list(zip(x_circle, y_circle, z_circle))]
                    sector = Poly3DCollection(verts, color='red', alpha=0.2)
                    ax_camera.add_collection3d(sector)
                    sector_plots.append(sector)

                    neck_xy = neck_point[:2]
                    arrow_vector_xy_view = arrow_vector_view[:2]
                    betaV_calculated = (np.degrees(np.arctan2(arrow_vector_xy_view[1], arrow_vector_xy_view[0])) + 90) % 360

            # Calculate the angles for global coordinates
            pelvis_global = next((kp for kp in keypoints_3d_global if kp[0] == 'Pelvis'), None)
            neck_global = next((kp for kp in keypoints_3d_global if kp[0] == 'Neck'), None)

            if pelvis_global and neck_global:
                pelvis_point_global = np.array([pelvis_global[1], pelvis_global[2], pelvis_global[3]])
                neck_point_global = np.array([neck_global[1], neck_global[2], neck_global[3]])

                # Body direction vector
                if 'Shoulder.L' in [kp[0] for kp in keypoints_3d_global] and 'Shoulder.R' in [kp[0] for kp in keypoints_3d_global]:
                    shoulder_l_global = next(kp for kp in keypoints_3d_global if kp[0] == 'Shoulder.L')
                    shoulder_r_global = next(kp for kp in keypoints_3d_global if kp[0] == 'Shoulder.R')

                    p1, p2, p3, p4, p5 = pelvis_point_global, np.array(shoulder_l_global[1:]), np.array(shoulder_r_global[1:]), pelvis_point_global, neck_point_global
                    arrow_start_global, arrow_end_global, arrow_vector_global = calculate_plane_and_arrow(p1, p2, p3, p4, p5, arrow_length=3)

                    arrow = ax_global.quiver(arrow_start_global[0], arrow_start_global[1], arrow_start_global[2],
                                             arrow_end_global[0] - arrow_start_global[0], arrow_end_global[1] - arrow_start_global[1], arrow_end_global[2] - arrow_start_global[2],
                                             color='g', length=1.0, arrow_length_ratio=0.1)
                    arrow_plots.append(arrow)

                    pelvis_xy_global = pelvis_point_global[:2]
                    arrow_vector_xy_global = arrow_vector_global[:2]
                    beta_glob_calculated = (np.degrees(np.arctan2(arrow_vector_xy_global[1], arrow_vector_xy_global[0])) + 90) % 360

                if 'Eye.L' in [kp[0] for kp in keypoints_3d_global] and 'Eye.R' in [kp[0] for kp in keypoints_3d_global]:
                    eye_l_global = next(kp for kp in keypoints_3d_global if kp[0] == 'Eye.L')
                    eye_r_global = next(kp for kp in keypoints_3d_global if kp[0] == 'Eye.R')

                    p1, p2, p3, p4, p5 = np.array(eye_l_global[1:]), np.array(eye_r_global[1:]), neck_point_global, neck_point_global, neck_point_global
                    arrow_neck_start_global, arrow_neck_end_global, arrow_vector_view_global = calculate_plane_and_arrow(p1, p2, p3, p4, p5, arrow_length=4)

                    arrow = ax_global.quiver(arrow_neck_start_global[0], arrow_neck_start_global[1], arrow_neck_start_global[2],
                                             arrow_neck_end_global[0] - arrow_neck_start_global[0], arrow_neck_end_global[1] - arrow_neck_start_global[1], arrow_neck_end_global[2] - arrow_neck_start_global[2],
                                             color='r', length=0.5, arrow_length_ratio=0.1)
                    arrow_plots.append(arrow)

                    angle_main_global = np.arctan2(arrow_neck_end_global[1] - arrow_neck_start_global[1], arrow_neck_end_global[0] - arrow_neck_start_global[0])
                    angles_additional_global = [angle_main_global + np.radians(100), angle_main_global - np.radians(100)]
                    arrow_end_additional_global = []
                    for angle in angles_additional_global:
                        arrow_end_add_global = arrow_neck_start_global + 0.5 * np.array([np.cos(angle), np.sin(angle), 0])
                        arrow_end_additional_global.append(arrow_end_add_global)
                        arrow = ax_global.quiver(arrow_neck_start_global[0], arrow_neck_start_global[1], arrow_neck_start_global[2],
                                                 arrow_end_add_global[0] - arrow_neck_start_global[0], arrow_end_add_global[1] - arrow_neck_start_global[1], arrow_end_add_global[2] - arrow_neck_start_global[2],
                                                 color='r', length=0.5, arrow_length_ratio=0.1)
                        arrow_plots.append(arrow)

                    theta_global = np.linspace(angles_additional_global[1], angles_additional_global[0], 100)
                    x_circle_global = arrow_neck_start_global[0] + 0.5 * np.cos(theta_global)
                    y_circle_global = arrow_neck_start_global[1] + 0.5 * np.sin(theta_global)
                    z_circle_global = np.full_like(x_circle_global, arrow_neck_start_global[2])
                    circle = ax_global.plot(x_circle_global, y_circle_global, z_circle_global, color='r')
                    circle_plots.append(circle[0])

                    verts_global = [list(zip(x_circle_global, y_circle_global, z_circle_global))]
                    sector_global = Poly3DCollection(verts_global, color='red', alpha=0.2)
                    ax_global.add_collection3d(sector_global)
                    sector_plots.append(sector_global)

                    neck_xy_global = neck_point_global[:2]
                    arrow_vector_xy_view_global = arrow_vector_view_global[:2]
                    betaV_glob_calculated = (np.degrees(np.arctan2(arrow_vector_xy_view_global[1], arrow_vector_xy_view_global[0])) + 90) % 360

            save_camera_csv(frame_number, timestamp, keypoints_3d_camera, camera_csv_writer, alpha_calculated, beta_calculated, alphaV_calculated, betaV_calculated)
            save_global_csv(frame_number, timestamp, keypoints_3d_global, global_csv_writer, alpha_glob_calculated, beta_glob_calculated, alphaV_glob_calculated, betaV_glob_calculated)

        plt.draw()
        plt.pause(0.001)

except Exception as e:
    print(f"Error: {e}")

finally:
    camera_csv_file.close()
    global_csv_file.close()
    plt.show()
