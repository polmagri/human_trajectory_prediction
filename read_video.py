import cv2

cap = cv2.VideoCapture("1_depth.avi")
while (1 == 1):
    ret, colormap_frame = cap.read()
    if ret:
        cv2.imshow("res", colormap_frame)
    if  cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()