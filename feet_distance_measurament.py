import cv2
import math
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import pyrealsense2 as rs
import numpy as np

# Inizializzation  RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

model = YOLO("yolov8s.pt")
#model = YOLO('yolov8s-pose.pt') 

cap = cv2.VideoCapture(4) 
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('visioneye-distance-calculation.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

center_point = (0, h)
txt_color, txt_background, bbox_clr = ((0, 0, 0), (255, 255, 255), (255, 0, 255))

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Il frame video è vuoto o l'elaborazione del video è stata completata con successo.")
        break

    annotator = Annotator(im0, line_width=2)

    results = model.track(im0, persist=True)
    boxes = results[0].boxes.xyxy.cpu()

    # Attendiamo i frame del sensore di profondità RealSense
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            annotator.box_label(box, label=str(track_id), color=bbox_clr)

            # Calcola le coordinate per la distanza
            x_center = int((box[0] + box[2]) // 2)  # Coordinata x del centro del bounding box
            y_bottom = int(box[3])                   # Coordinata y del punto inferiore destro del bounding box

            # Controllo dei limiti dell'immagine
            y_bottom = min(y_bottom, im0.shape[0] - 1)  # Assicura che y1 sia all'interno dell'immagine

            # Calcolo della distanza dal sensore di profondità
            dist = depth_frame.get_distance(x_center, y_bottom)

            # Disegna il testo sulla frame
            text_size, _ = cv2.getTextSize(f"Distance: {dist:.2f} m", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(im0, (x_center, y_bottom - text_size[1] - 10), (x_center + text_size[0] + 10, y_bottom), txt_background, -1)
            cv2.putText(im0, f"Distance: {dist:.2f} m", (x_center, y_bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, txt_color, 3)

        # Disegna un solo pallino nel punto più in basso di ogni bounding box
        for box in boxes:
            x_center = int((box[0] + box[2]) // 2)
            y_bottom = int(box[3])
            y_bottom = min(y_bottom, im0.shape[0] - 1)
            annotator.visioneye(box, (x_center, y_bottom))

    out.write(im0)
    cv2.imshow("visioneye-distance-calculation", im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()

