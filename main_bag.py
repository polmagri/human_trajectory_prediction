import cv2
import numpy as np
import rosbag2_py
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ultralytics import YOLO
import os

def calculate_3d(x, y, depth_image, camera_params, w, h, window_size=10):
    if x == 0 and y == 0:
        return 0.0, 0.0, 0.0, 0.0

    min_depth = float('inf')

    for i in range(-window_size // 2, window_size // 2 + 1):
        for j in range(-window_size // 2, window_size // 2 + 1):
            x_pixel = int(x) + i
            y_pixel = int(y) + j

            if 0 <= x_pixel < w and 0 <= y_pixel < h:
                depth = depth_image[y_pixel, x_pixel]
                if depth != 0 and depth < min_depth:
                    min_depth = depth

    if min_depth == float('inf'):
        return 0.0, 0.0, 0.0, 0.0

    # Conversione delle coordinate 2D e della profondità in coordinate 3D considerando la distorsione
    D = np.array(camera_params['D'])
    K = np.array(camera_params['K']).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Uso delle coordinate distorte per ottenere le coordinate non distorte
    undistorted_points = cv2.undistortPoints(np.array([[x, y]], dtype=np.float32), K, D)
    x_undist, y_undist = undistorted_points[0, 0]

    z = min_depth
    x_3d = (x_undist - cx) * z / fx
    y_3d = (y_undist - cy) * z / fy

    return x_3d, y_3d, z, min_depth

def calculate_plane_and_arrow(p1, p2, p3, p4, p5, arrow_length):
    v1 = np.subtract(p2, p1)
    v2 = np.subtract(p3, p1)

    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    normal_xy = np.array([normal[0], normal[1], 0])
    normal_xy = normal_xy / np.linalg.norm(normal_xy)

    arrow_start = p4
    arrow_end = p4 + arrow_length * normal_xy

    p5_xy = np.array([p5[0], p5[1], 0])
    angleX_camera_neck = np.arctan2(p5[1], p5[0])
    angleX_camera_neck_degrees = np.degrees(angleX_camera_neck)

    def line_intersection(p1, p2, p3, p4):
        a1, b1 = p1[:2]
        a2, b2 = p2[:2]
        a3, b3 = p3[:2]
        a4, b4 = p4[:2]
        
        denominator = (a1 - a2) * (b3 - b4) - (b1 - b2) * (a3 - a4)
        if denominator == 0:
            return None
        
        intersection_x = ((a1 * b2 - b1 * a2) * (a3 - a4) - (a1 - a2) * (a3 * b4 - b3 * a4)) / denominator
        intersection_y = ((a1 * b2 - b1 * a2) * (b3 - b4) - (b1 - b2) * (a3 * b4 - b3 * a4)) / denominator
        return np.array([intersection_x, intersection_y, 0])

    intersection = line_intersection([0, 0, 0], p5, arrow_start, arrow_end)
    
    if intersection is not None:
        orientation = np.arctan2(intersection[1], intersection[0])
        orientation_degrees = np.degrees(orientation)
    else:
        orientation_degrees = None

    return arrow_start, arrow_end, angleX_camera_neck_degrees, orientation_degrees

# Percorso alla directory contenente i file rosbag
rosbag_dir = '/home/mohamed/Downloads/train_rosbags/rosbags'

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
    (0, 1), (0, 2), (5 , 6), (11, 12), (2, 4), (1, 3), (5, 7),
    (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

# Create the figure and 3D axis outside the loop
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set fixed limits for the axes
ax.set_xlim(-4, 4)
ax.set_ylim(0 , 4)
ax.set_zlim(-1, 3)

# Set the axis labels to match the new orientation
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Define the vertices of a 1x1x1 cube centered at (0, 0, 0)
cube_vertices = np.array([[-0.05, -0.05, -0.05],
                          [0.05, -0.05, -0.05],
                          [0.05, 0.05, -0.05],
                          [-0.05, 0.05, -0.05],
                          [-0.05, -0.05, 0.05],
                          [0.05, -0.05, 0.05],
                          [0.05, 0.05, 0.05],
                          [-0.05, 0.05, 0.05]])

# Define the 12 edges of the cube
cube_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
              (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]

# Plot the cube
cube_lines = []
for edge in cube_edges:
    start, end = edge
    line, = ax.plot([cube_vertices[start][0], cube_vertices[end][0]],
                    [cube_vertices[start][1], cube_vertices[end][1]],
                    [cube_vertices[start][2], cube_vertices[end][2]], 'k')
    cube_lines.append(line)

# Store scatter, plot, arrow, sector, and circle objects
scatter_plots = []
line_plots = []
text_labels = []  # Store text labels
arrow_plots = []  # Store arrow plots
sector_plots = []  # Store sector plots
circle_plots = []  # Store circle plots

# Parametri della fotocamera
camera_params = {
    'width': 752,
    'height': 480,
    'D': [-0.336591, 0.159742, 0.00012697, -7.22557e-05, -0.0461953],
    'K': [
        476.71, 0, 350.738,
        0, 479.505, 209.532,
        0, 0, 1
    ],
    'R': [
        0.999994, 0.000654539, 0.00340293,
        -0.000654519, 1, -6.81963e-06,
        -0.00340293, 4.59231e-06, 0.999994
    ],
    'T': [-0.0104242, -3.70974, -56.9177]
}

bridge = CvBridge()

for filename in os.listdir(rosbag_dir):
    if filename.endswith(".bag"):
        bag_path = os.path.join(rosbag_dir, filename)
        
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr')

        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        while reader.has_next():
            (topic, data, t) = reader.read_next()
            if topic == '/camera/rgb/image_raw':
                msg = rosbag2_py.deserialize_message(data, type_map[topic])
                color_image = bridge.imgmsg_to_cv2(msg, 'bgr8')
            elif topic == '/camera/depth/image_raw':
                msg = rosbag2_py.deserialize_message(data, type_map[topic])
                depth_image = bridge.imgmsg_to_cv2(msg, '16UC1')

            if 'color_image' in locals() and 'depth_image' in locals():
                persons = model(color_image)

                # Remove previous scatter plots, lines, text labels, arrows, sectors, and circles
                for scatter in scatter_plots:
                    scatter.remove()
                scatter_plots.clear()
                for line in line_plots:
                    line.remove()
                line_plots.clear()
                for text in text_labels:
                    text.remove()
                text_labels.clear()
                for arrow in arrow_plots:
                    arrow.remove()
                arrow_plots.clear()
                for sector in sector_plots:
                    sector.remove()
                sector_plots.clear()
                for circle in circle_plots:
                    circle.remove()
                circle_plots.clear()

                for results in persons:
                    for result in results:
                        if hasattr(result, 'keypoints'):
                            kpts = result.keypoints.xy.cpu().numpy()
                            keypoints_list = kpts.flatten().tolist()
                            labels = [index_to_label.get(i, '') for i in range(len(keypoints_list) // 2)]

                            keypoints_2d = {}
                            keypoints_3d = []

                            for i, (x, y) in enumerate(zip(keypoints_list[::2], keypoints_list[1::2])):
                                cv2.circle(color_image, (int(x), int(y)), 5, (0, 255, 0), -1)
                                label = labels[i]
                                if label:
                                    cv2.putText(color_image, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                    cv2.putText(color_image, f"({int(x)}, {int(y)})", (int(x) + 10, int(y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                    keypoints_2d[label] = (int(x), int(y))
                                    x_3d, y_3d, z_3d, min_depth = calculate_3d(int(x), int(y), depth_image, camera_params, depth_image.shape[1], depth_image.shape[0])
                                    print(f"Keypoint: {label} - 2D: ({x}, {y}), 3D: ({x_3d}, {y_3d}, {z_3d}), Min Depth: {min_depth}")
                                    if not np.isnan(z_3d) and (x_3d != 0 or y_3d != 0 or z_3d != 0):
                                        keypoints_3d.append((label, x_3d, y_3d, z_3d))

                            if 'Hip.L' in keypoints_2d and 'Hip.R' in keypoints_2d:
                                hip_l = keypoints_2d['Hip.L']
                                hip_r = keypoints_2d['Hip.R']
                                if hip_l != (0, 0) and hip_r != (0, 0):
                                    pelvis_x = (hip_l[0] + hip_r[0]) // 2
                                    pelvis_y = (hip_l[1] + hip_r[1]) // 2
                                    keypoints_2d['Pelvis'] = (pelvis_x, pelvis_y)

                                    x_3d, y_3d, z_3d, min_depth = calculate_3d(pelvis_x, pelvis_y, depth_image, camera_params, depth_image.shape[1], depth_image.shape[0])
                                    print(f"Keypoint: Pelvis - 2D: ({pelvis_x}, {pelvis_y}), 3D: ({x_3d}, {y_3d}, {z_3d}), Min Depth: {min_depth}")
                                    if not np.isnan(z_3d) and (x_3d != 0 or y_3d != 0 or z_3d != 0):
                                        keypoints_3d.append(('Pelvis', x_3d, y_3d, z_3d))

                                    cv2.circle(color_image, (pelvis_x, pelvis_y), 5, (0, 0, 255), -1)
                                    cv2.putText(color_image, 'Pelvis', (pelvis_x, pelvis_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                    cv2.putText(color_image, f"({pelvis_x}, {pelvis_y})", (pelvis_x + 10, pelvis_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                            if 'Shoulder.L' in keypoints_2d and 'Shoulder.R' in keypoints_2d:
                                shoulder_l = keypoints_2d['Shoulder.L']
                                shoulder_r = keypoints_2d['Shoulder.R']
                                if shoulder_l != (0, 0) and shoulder_r != (0, 0):
                                    neck_x = (shoulder_l[0] + shoulder_r[0]) // 2
                                    neck_y = (shoulder_l[1] + shoulder_r[1]) // 2
                                    keypoints_2d['Neck'] = (neck_x, neck_y)

                                    x_3d, y_3d, z_3d, min_depth = calculate_3d(neck_x, neck_y, depth_image, camera_params, depth_image.shape[1], depth_image.shape[0])
                                    print(f"Keypoint: Neck - 2D: ({neck_x}, {neck_y}), 3D: ({x_3d}, {y_3d}, {z_3d}), Min Depth: {min_depth}")
                                    if not np.isnan(z_3d) and (x_3d != 0 or y_3d != 0 or z_3d != 0):
                                        keypoints_3d.append(('Neck', x_3d, y_3d, z_3d))

                                    cv2.circle(color_image, (neck_x, neck_y), 5, (255, 0, 0), -1)
                                    cv2.putText(color_image, 'Neck', (neck_x, neck_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                    cv2.putText(color_image, f"({neck_x}, {neck_y})", (neck_x + 10, neck_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                            for (start, end) in keypoint_connections:
                                if labels[start] in keypoints_2d and labels[end] in keypoints_2d:
                                    start_point = keypoints_2d[labels[start]]
                                    end_point = keypoints_2d[labels[end]]
                                    if start_point != (0, 0) and end_point != (0, 0):
                                        cv2.line(color_image, start_point, end_point, (255, 0, 0), 2)

                            for keypoint in keypoints_3d:
                                label, x_3d, y_3d, z_3d = keypoint
                                scatter = ax.scatter(x_3d, y_3d, z_3d, label=label)
                                scatter_plots.append(scatter)
                                text = ax.text(x_3d, y_3d, z_3d, label, color='red')
                                text_labels.append(text)

                            for (start, end) in keypoint_connections:
                                start_label = index_to_label.get(start, '')
                                end_label = index_to_label.get(end, '')
                                if start_label and end_label:
                                    start_point = next((kp for kp in keypoints_3d if kp[0] == start_label), None)
                                    end_point = next((kp for kp in keypoints_3d if kp[0] == end_label), None)
                                    if start_point and end_point:
                                        line, = ax.plot([start_point[1], end_point[1]], [start_point[2], end_point[2]], [start_point[3], end_point[3]], 'b')
                                        line_plots.append(line)

                            if 'Shoulder.L' in [kp[0] for kp in keypoints_3d] and 'Shoulder.R' in [kp[0] for kp in keypoints_3d] and 'Pelvis' in [kp[0] for kp in keypoints_3d] and 'Neck' in [kp[0] for kp in keypoints_3d]:
                                shoulder_l = next(kp for kp in keypoints_3d if kp[0] == 'Shoulder.L')
                                shoulder_r = next(kp for kp in keypoints_3d if kp[0] == 'Shoulder.R')
                                pelvis = next(kp for kp in keypoints_3d if kp[0] == 'Pelvis')
                                neck = next(kp for kp in keypoints_3d if kp[0] == 'Neck')

                                p1 = np.array(shoulder_l[1:])
                                p2 = np.array(shoulder_r[1:])
                                p3 = np.array(pelvis[1:])
                                p4 = p3  # The arrow should start at the pelvis point
                                p5 = np.array(neck[1:])

                                arrow_start, arrow_end, angleX_camera_neck_degrees, orientation_degrees = calculate_plane_and_arrow(p1, p2, p3, p4, p5, arrow_length=3)

                                print(f"angleX_camera_neck: {angleX_camera_neck_degrees} degrees")
                                if orientation_degrees is not None:
                                    print(f"orientation: {orientation_degrees} degrees")
                                else:
                                    print("orientation: No intersection found")

                                arrow = ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
                                                  arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], arrow_end[2] - arrow_start[2],
                                                  color='g', length=1.0, arrow_length_ratio=0.1)
                                arrow_plots.append(arrow)

                            if 'Eye.L' in [kp[0] for kp in keypoints_3d] and 'Eye.R' in [kp[0] for kp in keypoints_3d] and 'Neck' in [kp[0] for kp in keypoints_3d]:
                                eye_l = next(kp for kp in keypoints_3d if kp[0] == 'Eye.L')
                                eye_r = next(kp for kp in keypoints_3d if kp[0] == 'Eye.R')
                                neck = next(kp for kp in keypoints_3d if kp[0] == 'Neck')

                                p1 = np.array(eye_l[1:])
                                p2 = np.array(eye_r[1:])
                                p3 = np.array(neck[1:])
                                p4 = p3  # The arrow should start at the neck point
                                p5 = np.array(neck[1:])

                                arrow_start, arrow_end, angleX_camera_neck_degrees, orientation_degrees = calculate_plane_and_arrow(p1, p2, p3, p4, p5, arrow_length=4)

                                print(f"angleX_camera_neck (head): {angleX_camera_neck_degrees} degrees")
                                if orientation_degrees is not None:
                                    print(f"orientation (head): {orientation_degrees} degrees")
                                else:
                                    print("orientation (head): No intersection found")

                                arrow = ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
                                                  arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], arrow_end[2] - arrow_start[2],
                                                  color='r', length=0.5, arrow_length_ratio=0.1)
                                arrow_plots.append(arrow)

                                angle_main = np.arctan2(arrow_end[1] - arrow_start[1], arrow_end[0] - arrow_start[0])
                                angles_additional = [angle_main + np.radians(100), angle_main - np.radians(100)]
                                arrow_end_additional = []
                                for angle in angles_additional:
                                    arrow_end_add = arrow_start + 0.5 * np.array([np.cos(angle), np.sin(angle), 0])
                                    arrow_end_additional.append(arrow_end_add)
                                    arrow = ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
                                                      arrow_end_add[0] - arrow_start[0], arrow_end_add[1] - arrow_start[1], arrow_end_add[2] - arrow_start[2],
                                                      color='r', length=0.5, arrow_length_ratio=0.1)
                                    arrow_plots.append(arrow)

                                theta = np.linspace(angles_additional[1], angles_additional[0], 100)
                                x_circle = arrow_start[0] + 0.5 * np.cos(theta)
                                y_circle = arrow_start[1] + 0.5 * np.sin(theta)
                                z_circle = np.full_like(x_circle, arrow_start[2])
                                circle = ax.plot(x_circle, y_circle, z_circle, color='r')
                                circle_plots.append(circle[0])

                                verts = [list(zip(x_circle, y_circle, z_circle))]
                                sector = Poly3DCollection(verts, color='red', alpha=0.2)
                                ax.add_collection3d(sector)
                                sector_plots.append(sector)

                            # Check if the person is turned around
                            if 'Shoulder.L' in keypoints_2d and 'Shoulder.R' in keypoints_2d and \
                                (keypoints_2d['Shoulder.L'][0] > keypoints_2d['Shoulder.R'][0]) and \
                                (('Ear.L' in keypoints_2d and 'Ear.R' in keypoints_2d) and \
                                not ('Eye.L' in keypoints_2d and 'Eye.R' in keypoints_2d)):

                                print("la persona è girata")  # Aggiungi questa linea

                                if 'Ear.L' in keypoints_2d and 'Ear.R' in keypoints_2d:
                                    ear_l = next(kp for kp in keypoints_3d if kp[0] == 'Ear.L')
                                    ear_r = next(kp for kp in keypoints_3d if kp[0] == 'Ear.R')
                                    neck = next(kp for kp in keypoints_3d if kp[0] == 'Neck')

                                    p1 = np.array(ear_l[1:])
                                    p2 = np.array(ear_r[1:])
                                    p3 = np.array(neck[1:])
                                    p4 = p3  # The arrow should start at the neck point
                                    p5 = np.array(neck[1:])

                                    arrow_start, arrow_end, angleX_camera_neck_degrees, orientation_degrees = calculate_plane_and_arrow(p1, p2, p3, p4, p5, arrow_length=4)

                                    print(f"angleX_camera_neck (turned): {angleX_camera_neck_degrees} degrees")
                                    if orientation_degrees is not None:
                                        print(f"orientation (turned): {orientation_degrees} degrees")


                                    arrow = ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
                                                      arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], arrow_end[2] - arrow_start[2],
                                                      color='r', length=0.5, arrow_length_ratio=0.1)
                                    arrow_plots.append(arrow)

                                    angle_main = np.arctan2(arrow_end[1] - arrow_start[1], arrow_end[0] - arrow_start[0])
                                    angles_additional = [angle_main + np.radians(100), angle_main - np.radians(100)]
                                    arrow_end_additional = []
                                    for angle in angles_additional:
                                        arrow_end_add = arrow_start + 0.5 * np.array([np.cos(angle), np.sin(angle), 0])
                                        arrow_end_additional.append(arrow_end_add)
                                        arrow = ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
                                                          arrow_end_add[0] - arrow_start[0], arrow_end_add[1] - arrow_start[1], arrow_end_add[2] - arrow_start[2],
                                                          color='r', length=0.5, arrow_length_ratio=0.1)
                                        arrow_plots.append(arrow)

                                    theta = np.linspace(angles_additional[1], angles_additional[0], 100)
                                    x_circle = arrow_start[0] + 0.5 * np.cos(theta)
                                    y_circle = arrow_start[1] + 0.5 * np.sin(theta)
                                    z_circle = np.full_like(x_circle, arrow_start[2])
                                    circle = ax.plot(x_circle, y_circle, z_circle, color='r')
                                    circle_plots.append(circle[0])

                                    verts = [list(zip(x_circle, y_circle, z_circle))]
                                    sector = Poly3DCollection(verts, color='red', alpha=0.2)
                                    ax.add_collection3d(sector)
                                    sector_plots.append(sector)

                plt.draw()
                plt.pause(0.001)

                cv2.imshow('YOLO Keypoints', color_image)
                
                if cv2.waitKey(1) == ord('q'):
                    break
cv2.destroyAllWindows()